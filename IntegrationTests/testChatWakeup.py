"""
First integration test: wake word -> Gemini Live conversation -> back to idle.

Joins the two halves that have worked separately until now --
BasicFunctionTests/testWordDetection.py (detects "johnny" but only prints) and
BasicFunctionTests/testGeminiLiveAPI.py (holds a real conversation, but
push-to-talk on the Enter key). No keyboard in the loop here.

  IDLE   detector runs on a rolling 2.2 s window looking for "johnny", while a
         ring buffer quietly keeps the last PRE_ROLL_SECONDS of decimated mic
         audio.
  CONVO  wake word fires -> open the Live socket, flush the ring into it so the
         command that followed the wake word is not truncated, then stream live
         with server VAD doing the turn taking.
  exit   the user says goodbye (regex or the end_conversation tool), or says
         nothing for EXIT_DURATION -> close the socket, re-arm the detector.

Two things about this file that are deliberate and easy to "fix" wrongly:

1. SERVER VAD IS ON and the mic stays live while the speaker plays, so the speaker
   leaks into the mic and Gemini's VAD can read Johnny as an interruption. That was
   measured as harmless at VOLUME = 0.1 on the original mic gain, and stopped being
   harmless the moment the capture gain went up +23 dB -- the leakage rose by exactly
   the same amount. The fix was module-echo-cancel, not backing the gain off, and
   USE_AEC routes both directions through it. `interrupted` therefore means a real
   barge-in, and stopping playback on it is the right response.
   BasicFunctionTests/testAEC.py measures that path on its own.

2. THE DETECTOR STOPS while a session is open. CLAUDE.md's invariant is that the
   detector reads AEC output, never the raw mic. Even with the canceller in the path
   the residual is not worth gambling a false wake on, so it does not predict at all
   while Jarvis is talking. It also gives the conversation a free core.
"""

import asyncio
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import firwin, lfilter, resample_poly
from google import genai
from google.genai import types
from livekit.wakeword import WakeWordModel


# Action.py / ActionManager.py live at the repo root, one level up. Resolved from
# __file__ rather than the CWD, for the same reason MODEL_PATH is.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ESP32DeviceManager import ESP32DeviceManager
from Action import Action                      # noqa: E402
from ActionManager import ActionManager, builtin_actions   # noqa: E402

# --- config ---------------------------------------------------------------

MIC = "USB PnP Sound Device"
SPEAKER = "USB Composite Device"

DEVICE_RATE = 48000               # the PnP mic is 48 kHz only, so capture high
SEND_RATE = 16000                 # and decimate; the Live API accepts nothing else
MODEL_RATE = 16000                # what the wake word model wants, same decimation
BLOCK = int(DEVICE_RATE * 0.08)   # 80 ms capture blocks

RECV_RATE = 24000                 # what the Live API sends back
PLAYBACK_RATE = 48000             # the speaker accepts nothing else
OUT_BLOCK = 1920                  # 40 ms of output


# Keep this modest even with the canceller. Every dB out of the speaker is a dB into
# the mic, cancellation is never perfect, and clipping is non-linear -- an overdriven
# speaker produces echo no adaptive filter can model.
# Measured 2026-09-07, 440 Hz at amplitude 0.3, read at the raw mic:
#
#   raw hw: speaker, as testGeminiLiveAPI.py plays        -21.8 dBFS
#   through the pulse route, sink at 50%                  -36.0 dBFS
#   through the pulse route, sink at 100%                 -23.1 dBFS
#
# That 14 dB is the whole reason this file sounded quieter than testGeminiLiveAPI.py
# despite VOLUME being 10x higher. It was never headroom for anything: the speaker
# card exposes NO playback volume control (`amixer -c 0 contents` has only a `PCM
# Playback Switch`), so PulseAudio's 50% was applied in *software* to everything
# crossing the sink -- and the raw hw: path in testGeminiLiveAPI.py never crosses it.
# Fixed with `pactl set-sink-volume 0 100%`, which module-device-restore persists.
#
# Gain staging, measured 2026-09-10 (RMS over everything above -60 dBFS):
#   Gemini's own reply audio      peak  -1.6 dBFS   rms -16.9 dBFS
#   the repo mp3 (mastered music) peak   0.1 dBFS   rms -12.1 dBFS
# So at VOLUME = 0.5 Johnny lands at -22.9 dBFS rms and music, once librespot stopped
# attenuating it by 30 dB, lands at -12.1 -- Johnny would be 10.8 dB *under* the music
# he is talking over. 0.5 is kept anyway, because it is the level this room is already
# calibrated to and raising it eats the peak headroom the two need to sum into; the
# music is what moves, via DUCK_LEVEL. Change this and DUCK_LEVEL together.
VOLUME = 0.2

# Chimes. Played through Player like everything else, on purpose: the same stream, the
# same sink, so they are cancelled out of the mic instead of being heard as a wake word
# or read by Gemini's VAD as the user speaking.
SOUNDS = Path(__file__).resolve().parent.parent / "SoundEffects"
WAKE_SOUND = SOUNDS / "wakeUp.wav"
EXIT_SOUND = SOUNDS / "ExitConvoMode.wav"
# As recorded these two are 11 dB apart in RMS (wakeUp peaks at 0.205, ExitConvoMode at
# 0.777), which reads as one chime being broken. Both are normalised to this peak so the
# pair is consistent; None plays them as they are.
CHIME_PEAK = 0.6

# Everything else on the speaker is turned down to this percent for the length of a
# conversation, and put back afterwards. See MusicDucker for why this is not VOLUME.
#
# pactl's percent is cubic, not linear -- verified on this Pi: 50% = -18.06 dB,
# 40% = -23.88 dB, 25% = -36.12 dB. 25 was chosen while librespot's default log
# softvol was already holding music 30 dB down, so it was a mute, not a duck, and the
# ducker read as broken because there was nothing audible to bring back. Against music
# at its real -12.1 dBFS, 40% puts it at -36.0 dBFS -- 13 dB under Johnny's -22.9,
# which is the usual broadcast spread and leaves the sum peaking near -3 dBFS.
DUCK_LEVEL = 60
DUCK_APP = "librespot"

MODEL = "gemini-3.1-flash-live-preview"
VOICE = "Sadachbia"                    # 30 prebuilt voices; see testGeminiLiveAPI.py

# --- echo cancellation ----------------------------------------------------

# PulseAudio's module-echo-cancel (webrtc), loaded from ~/.config/pulse/default.pa.
#
# Raising the mic capture gain by +23 dB raised speaker leakage into the mic by the
# same +23 dB, which is what turned echo from "not a problem" into one. Measured on
# this Pi with pink noise at amplitude 0.3:
#
#   without the canceller   echo sits 10.5 dB ABOVE the mic noise floor
#   with it                 echo sits  3.6 dB BELOW it   (26.8 dB less in the mic)
#
# Set False to A/B against the raw hw: path; expect Gemini's VAD to start hearing
# Johnny and cutting him off.
USE_AEC = True
AEC_SOURCE = "jarvis_aec_source"
AEC_SINK = "jarvis_aec_sink"

# Device buffer, in seconds, and NOT latency="high" -- that is what broke this.
#
# "high" is relative to whatever device PortAudio resolved, and the ALSA "pulse"
# plugin advertises default_high_output_latency = 0.035 s. Ask for "high" there and
# you get an 80 ms playback buffer, which PulseAudio's timer-based scheduling cannot
# keep fed: measured 620 ALSA "underrun occurred" recoveries in 12 s from a callback
# that did nothing but copy a preallocated sine. The raw hw: speaker survives 80 ms
# happily -- there is no server in the path -- so this only appeared when USE_AEC
# routed playback through Pulse.
#
# Measured on this Pi, output-only, blocksize 1920, per 8 s:
#
#   0.10 -> granted 0.120   1285 underruns
#   0.12 -> granted 0.120   1336 underruns
#   0.16 -> granted 0.160      0
#   0.24 -> granted 0.240      0
#
# The cliff is between 0.12 and 0.16. 0.24 is the first value with real margin, and
# it costs only responsiveness: Player.tail folds it into quiet_since() correctly, and
# a barge-in leaves up to that much already-committed audio playing after flush().
OUT_LATENCY = 0.24 if USE_AEC else "high"
IN_LATENCY = 0.16 if USE_AEC else "high"

# The player is fed from the network, so a gap is normal and not an error. What is
# not normal is `queued` sitting above zero while the callback stops consuming it --
# that means the output stream died, and idle_watchdog would otherwise wait on it
# forever. Callbacks run every OUT_BLOCK / PLAYBACK_RATE = 40 ms.
STALL_SECONDS = 1.5

# A send to a half-open websocket can hang indefinitely. Generous -- a normal send is
# sub-millisecond -- but finite, so the pump reports and recovers instead of wedging.
SEND_TIMEOUT = 5.0

# --- wake word ------------------------------------------------------------

# Relative to this file, not the working directory. testWordDetection.py uses a bare
# relative path and already breaks when run from inside its own folder.
MODEL_PATH = Path(__file__).resolve().parent.parent / "wakeup_models" / "johnny.onnx"
MODEL_NAME = MODEL_PATH.stem      # scores are keyed by the file stem, not a chosen name

# 2.0 s yields exactly 16 embeddings -- the bare minimum -- so a window even slightly
# short silently scores 0.0 without running the classifier. 2.2 s gives 18.
WINDOW_SECONDS = 2.2
WINDOW_SAMPLES = int(DEVICE_RATE * WINDOW_SECONDS)

# predict() costs ~70-95 ms on this Pi, so a 1-block (80 ms) hop would saturate a core
# and back the queue up permanently. 2 blocks = every 160 ms, roughly half duty.
HOP_BLOCKS = 2
HOP_SECONDS = HOP_BLOCKS * BLOCK / DEVICE_RATE

THRESHOLD = 0.1              # working band on this device is 0.1-0.2, not 0.5
CONSECUTIVE = 2                   # predictions above threshold required to fire
REFRACTORY = 2.0                  # seconds ignored after a detection

VERBOSE = False                   # print every score; a tuning aid, floods the log

# --- conversation ---------------------------------------------------------

# Seconds of silence that ends the conversation, measured from the moment the room
# goes quiet -- which means after *Johnny's* audio finishes, not after the user's.
# See idle_watchdog: the gap between those two is a whole round trip wide.
EXIT_DURATION = 10

# Hard cap on holding the clock while waiting for a reply that never comes (the
# model decided the audio was not addressed to it, or the socket stalled). Without
# this, "waiting for a reply" could wedge the session open forever.
REPLY_TIMEOUT = 8.0

# How long one reason may hold a session open before it is said out loud. A session
# that will not close is otherwise completely invisible: health() reports the audio
# path, not the conversation, so "it stalled going back to idle" has no evidence
# attached to it at all. Purely diagnostic -- it changes no timing.
HOLD_REPORT = 3.0

# Longest the farewell may take before the session closes anyway. "Leave after this
# turn finishes" was the old rule and it deadlocks: if the model's turn never
# completes, session.receive() never returns, model_busy stays True, idle_watchdog
# holds forever and nothing is printed. Measured shape of the bug: Johnny says
# goodbye, then it hangs until Ctrl-C.
FAREWELL_TIMEOUT = 8.0

# A turn that has produced no audio for this long is treated as over. Same defect
# class as above, for the case where no goodbye was ever said -- without it a turn
# that never completes wedges the session with no way back to idle.
TURN_TIMEOUT = 12.0

# Audio kept before the wake word fires. Deliberately larger than CLAUDE.md's
# "1-2 s": detection cannot fire until "johnny" sits inside a 2.2 s window, and
# CONSECUTIVE=2 adds another hop on top, so the word is up to ~2 s in the past by
# the time the socket opens. Without this the command arrives truncated.
PRE_ROLL_SECONDS = 2.0
PRE_ROLL_FRAMES = max(1, int(PRE_ROLL_SECONDS / (BLOCK / DEVICE_RATE)))

# The fast path out. Deterministic and instant -- no round trip, no model judgment.
# Anything this misses is caught by the end_conversation tool below.
GOODBYE = re.compile(
    r"\b(bye|goodbye|good bye|see you|good ?night|"
    r"stop talking|stop listening|shut up|be quiet|"
    r"that'?s all|that'?s it|that will be all|"
    r"never ?mind|forget it|go to sleep|dismissed|we'?re done|i'?m done)\b",
    re.IGNORECASE,
)

def end_conversation(reason: str = ""):
    """Just records why. The session teardown is the caller's job -- see on_call in
    _handle_tool_call, which sets convo.ending so playback can drain first."""
    return {"status": "ok", "reason": reason}


END_CONVERSATION = Action(
    "End the conversation and return to idle listening. Call this whenever the user "
    "signals they are finished -- saying goodbye, telling you to stop, dismissing you, "
    "or otherwise indicating they no longer want to talk.",
    end_conversation,
    describe={"reason": "briefly, what the user said that ended the conversation"},
)

SYSTEM_INSTRUCTION = (
    "You are Johnny, a voice assistant in a college dorm room. The user woke you by "
    "saying your name, so the first thing you hear will usually start with it. "
    "Answer in one or two short sentences. You are speaking out loud, so no markdown, "
    "no lists, and no emoji. "
    "When the user signals they are done -- saying goodbye, telling you to stop, or "
    "dismissing you -- say a short goodbye and call the end_conversation function."
)

def build_config(manager):
    """The session config, built after the manager is populated.

    It cannot stay a module-level constant any more: both `tools` and the tail of the
    system instruction come from the registered actions.
    """
    return types.LiveConnectConfig(
        # AUDIO is the only modality these models support.
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_INSTRUCTION + manager.briefing(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ),
        tools=manager.tools(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # Server VAD ON -- the opposite of testGeminiLiveAPI.py, and the whole point of
        # this file. Gemini decides where a turn ends, so there is no key to press.
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                # Audio kept from before speech starts, so the first word is not clipped.
                prefix_padding_ms=300,
                # Silence that ends a user turn. Distinct from EXIT_DURATION: this ends a
                # *turn*, EXIT_DURATION ends the whole *conversation*.
                silence_duration_ms=800,
            ),
        ),
    )


# --- resampling -----------------------------------------------------------


class Decimator:
    """48 kHz -> 16 kHz uplink, carrying filter state across blocks.

    Resampling each 80 ms block independently zero-pads the filter at both edges,
    leaving a transient at every block boundary -- 12.5 a second into an ASR front
    end. Holding lfilter's state makes the stream continuous.
    """

    def __init__(self, factor=DEVICE_RATE // SEND_RATE, taps=129):
        self.factor = factor
        # Cutoff is relative to Nyquist, so 1/factor lands on the target's Nyquist.
        self.b = firwin(taps, 1.0 / factor)
        self.zi = np.zeros(taps - 1)

    def __call__(self, frame):
        y, self.zi = lfilter(self.b, [1.0], frame, zi=self.zi)
        return y[:: self.factor]


class Upsampler:
    """24 kHz -> 48 kHz downlink, also stateful.

    Same reasoning in reverse: per-chunk resampling clicks at every seam, and Gemini
    sends many small chunks per reply.
    """

    def __init__(self, factor=PLAYBACK_RATE // RECV_RATE, taps=129):
        self.factor = factor
        # Zero-stuffing divides energy by `factor`, so scale the filter back up.
        self.b = firwin(taps, 1.0 / factor) * factor
        self.zi = np.zeros(taps - 1)

    def __call__(self, samples):
        stuffed = np.zeros(len(samples) * self.factor)
        stuffed[:: self.factor] = samples
        y, self.zi = lfilter(self.b, [1.0], stuffed, zi=self.zi)
        return y.astype(np.float32)


# --- capture --------------------------------------------------------------

# Two queues, one producer. The consumers cannot share one: the detector is a thread
# doing 70-95 ms of blocking inference, the uplink lives on the asyncio loop.
detect_q = queue.Queue(maxsize=50)
uplink_q = queue.Queue(maxsize=50)

overflow_flag = threading.Event()
stop_flag = threading.Event()
listening = threading.Event()     # set = idle, detector predicting

# Counted per queue, not together. One number cannot say *which* consumer stopped, and
# that is the whole diagnosis: detector drops mean the wake word thread fell behind,
# uplink drops mean the pump is not sending to Gemini any more.
detect_drops = 0
uplink_drops = 0


def callback(indata, frames, time_info, status):
    """Realtime thread. One copy, two enqueues, nothing else. Never print here."""
    global detect_drops, uplink_drops

    if status:
        overflow_flag.set()

    frame = indata[:, 0].copy()
    try:
        detect_q.put_nowait(frame)
    except queue.Full:
        detect_drops += 1         # drop rather than block the callback
    try:
        uplink_q.put_nowait(frame)
    except queue.Full:
        uplink_drops += 1


# --- playback -------------------------------------------------------------


class Player:
    """48 kHz output stream drained by its callback, so the reply starts playing
    while the rest of it is still arriving.

    The callback copies out of queued chunks piecewise into a preallocated scratch
    buffer rather than concatenating, so it allocates nothing and does no work
    beyond memcpy.
    """

    def __init__(self, device, channels):
        self.channels = channels
        self.q = queue.Queue()
        self.scratch = np.zeros(OUT_BLOCK, dtype=np.float32)
        self.head = None
        self.pos = 0
        self.queued = 0          # samples not yet played
        self.underruns = 0
        # When the callback last had anything to play. `queued` alone is not enough:
        # it hits zero when the callback *consumes* the last sample, while the device
        # buffer still has `tail` seconds of audio left to actually emit.
        self.last_sound_at = 0.0
        # Every callback, sound or not -- this is the liveness signal, and silence
        # while nothing is queued is normal.
        self.last_callback_at = time.perf_counter()
        self.stream = sd.OutputStream(
            device=device,
            samplerate=PLAYBACK_RATE,
            blocksize=OUT_BLOCK,
            dtype="float32",
            channels=channels,
            callback=self._callback,
            latency=OUT_LATENCY,
        )
        # A large device buffer is exactly what makes `queued == 0` an early signal.
        # Fold it back in when asking "has Johnny actually stopped talking?".
        try:
            self.tail = float(self.stream.latency)
        except (TypeError, ValueError):
            self.tail = 0.2

    def quiet_since(self):
        """When the speaker last went silent. 0.0 if it has never played."""
        return self.last_sound_at + self.tail if self.last_sound_at else 0.0

    def stalled(self):
        """Audio is queued but the callback has stopped draining it.

        The output stream is the one component whose death is completely silent from
        in here: PortAudio simply stops calling back, `queued` freezes above zero, and
        every consumer that waits on "is Johnny still talking?" waits forever. That is
        what a hung session looks like -- no error, no output, nothing.
        """
        if self.queued <= 0:
            return False
        if not self.stream.active:
            return True
        return time.perf_counter() - self.last_callback_at > STALL_SECONDS

    def _callback(self, outdata, frames, time_info, status):
        if status:
            overflow_flag.set()
        self.last_callback_at = time.perf_counter()

        scratch = self.scratch[:frames]
        filled = 0

        while filled < frames:
            if self.head is None:
                try:
                    self.head = self.q.get_nowait()
                    self.pos = 0
                except queue.Empty:
                    break
            take = min(frames - filled, len(self.head) - self.pos)
            scratch[filled : filled + take] = self.head[self.pos : self.pos + take]
            self.pos += take
            filled += take
            self.queued -= take
            if self.pos >= len(self.head):
                self.head = None

        if filled:
            # A bare timestamp, not printing -- the callback still does no work.
            self.last_sound_at = time.perf_counter()

        if filled < frames:
            scratch[filled:] = 0.0
            if filled:
                self.underruns += 1

        # Broadcast mono across the device's channels without allocating.
        outdata[:] = scratch[:, np.newaxis]

    def push(self, samples):
        # Scaled here rather than in the callback, which must stay copy-only.
        samples = np.clip(samples * VOLUME, -1.0, 1.0)
        self.queued += len(samples)
        self.q.put(samples)

    def flush(self):
        """Drop everything pending -- used when the model is interrupted."""
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.head = None
        self.queued = 0

    async def drain(self, timeout=30.0):
        """Wait for queued audio to play out. Bounded -- a dead stream never drains."""
        deadline = time.perf_counter() + timeout
        while self.queued > 0 and time.perf_counter() < deadline:
            if self.stalled():
                print("  (playback stalled -- abandoning drain)")
                self.flush()
                return
            await asyncio.sleep(0.05)
        # Let the last block reach the DAC before returning.
        await asyncio.sleep(OUT_BLOCK / PLAYBACK_RATE)

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *exc):
        return self.stream.__exit__(*exc)


def resolve_devices():
    """Choose capture/playback devices, routing through the canceller if enabled.

    PortAudio has no way to name a PulseAudio source or sink, so the ALSA "pulse"
    device is addressed instead and steered with PULSE_SOURCE / PULSE_SINK, which
    libpulse reads when the stream is opened.

    Both directions have to go through it. If playback went to the raw hw: speaker
    the canceller would never see the reference signal and would remove nothing --
    and the failure is silent, which is why a missing module exits here rather than
    quietly falling back to the raw mic.

    Returns (in_device, out_device, out_channels).
    """
    if not USE_AEC:
        return MIC, SPEAKER, min(
            sd.query_devices(SPEAKER, "output")["max_output_channels"], 2
        )

    try:
        listed = subprocess.run(
            ["pactl", "list", "short", "sources"], capture_output=True,
            text=True, timeout=5,
        ).stdout + subprocess.run(
            ["pactl", "list", "short", "sinks"], capture_output=True,
            text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        listed = ""

    missing = [n for n in (AEC_SOURCE, AEC_SINK) if n not in listed]
    if missing:
        sys.exit(
            f"echo canceller not loaded -- missing: {', '.join(missing)}\n"
            "Load it with:\n"
            "  pactl load-module module-echo-cancel aec_method=webrtc \\\n"
            "    source_master=alsa_input.usb-C-Media_Electronics_Inc."
            "_USB_PnP_Sound_Device-00.analog-mono \\\n"
            "    sink_master=alsa_output.usb-Jieli_Technology_USB_Composite"
            "_Device_433135383532342E-00.analog-stereo \\\n"
            f"    source_name={AEC_SOURCE} sink_name={AEC_SINK} \\\n"
            "    rate=48000 channels=1 use_volume_sharing=no \\\n"
            "    'aec_args=\"analog_gain_control=0 digital_gain_control=0 "
            "noise_suppression=1 high_pass_filter=1 voice_detection=0\"'\n"
            "\nIt is also in ~/.config/pulse/default.pa, so `systemctl --user "
            "restart pulseaudio` should bring it back.\n"
            "Or set USE_AEC = False to run on the raw devices and hear the echo."
        )

    # Read by libpulse when the stream opens, so this must happen before any
    # sd.InputStream / sd.OutputStream is constructed.
    os.environ["PULSE_SOURCE"] = AEC_SOURCE
    os.environ["PULSE_SINK"] = AEC_SINK
    # The canceller's sink is mono, and Player broadcasts mono anyway.
    return "pulse", "pulse", 1


def load_chime(path, peak=CHIME_PEAK):
    """One .wav from SoundEffects/, converted once into what Player.push expects.

    Player's queue holds mono float32 at PLAYBACK_RATE; these files are 44.1 kHz stereo.
    Both conversions happen here, at startup, so nothing resamples on the audio path --
    44100 -> 48000 is 160/147, not a decimation the Decimator/Upsampler pair can do.

    A missing or unreadable file is a warning, not an exit: a chime is a nicety, and
    losing the wake word over it would be a bad trade.
    """
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        print(f"  (no chime -- {path.name}: {exc})")
        return None

    mono = data.mean(axis=1)
    if rate != PLAYBACK_RATE:
        step = math.gcd(int(rate), PLAYBACK_RATE)
        mono = resample_poly(mono, PLAYBACK_RATE // step, int(rate) // step)

    loudest = float(np.abs(mono).max())
    if peak and loudest > 0:
        mono = mono * (peak / loudest)

    print(f"  chime {path.name}: {len(mono) / PLAYBACK_RATE:.2f}s, "
          f"peak {loudest:.3f} -> {peak if peak else loudest:.3f}")
    return np.ascontiguousarray(mono, dtype="float32")


class MusicDucker:
    """Turn the music down for the length of a conversation, then put it back.

    This is the honest version of "leave headroom when music is playing". Baking the
    headroom into VOLUME would attenuate Johnny exactly when he is competing with music
    -- the wrong half. Lowering the *music* stream instead keeps the sum away from the
    sink's clipping point and leaves the voice at full scale, which is what the mixing
    desk convention (ducking) has always done.

    It works on PulseAudio sink-inputs rather than through the Spotify API because it
    must be instant and must not depend on Spotify being authorised; it also then covers
    anything else playing, not just librespot.

    The saved volumes are per sink-input index, and an index is only valid while that
    stream exists -- but a restore against a stream that has since gone is *not* the
    harmless no-op it looks like. module-stream-restore remembers volume per
    *application*, not per stream, so duck() has already written the ducked level into
    librespot's remembered volume. If the stream then disappears while ducked (paused,
    handed off to another Connect device, librespot restarted), restore() addresses a
    dead index and every future librespot stream is *born* at DUCK_LEVEL -- across
    reboots, with the Spotify slider maxed and unable to help, because softvol is
    upstream of the sink-input fader. Observed: a stream sitting at 25% long after
    DUCK_LEVEL had moved to 40. Hence unduck_strays(), on both paths.
    """

    def __init__(self, level=DUCK_LEVEL, app=DUCK_APP):
        self.level = level
        self.app = app
        self.saved = {}

    def _pactl(self, *args):
        try:
            return subprocess.run(
                ["pactl", *args], capture_output=True, text=True, timeout=5
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    def _streams(self):
        """[(index, percent)] for the sink-inputs belonging to self.app."""
        listed = self._pactl("list", "sink-inputs")
        found = []
        for block in re.split(r"(?=^Sink Input #)", listed, flags=re.M):
            index = re.match(r"Sink Input #(\d+)", block)
            if not index or self.app.lower() not in block.lower():
                continue
            volume = re.search(r"Volume:.*?/\s*(\d+)%", block, re.S)
            if volume:
                found.append((int(index.group(1)), int(volume.group(1))))
        return found

    def duck(self):
        """Turn down anything not already low. Returns how many, for the log line.

        Idempotent, and that is not decoration: an earlier version reset `saved` on
        every call, so a second duck() while already ducked recorded the *ducked*
        volume as the original -- or nothing at all -- and restore() then silently
        left the music at 25% forever. Only streams that appear after this runs are
        missed, so music started mid-conversation plays at full volume until the
        next wake.
        """
        if self.level is None:
            return 0
        ducked = 0
        for index, percent in self._streams():
            # Already recorded, or already quieter than the target: leave it alone.
            # Restoring such a stream *up* would be a change the user did not ask for.
            if index in self.saved or percent <= self.level:
                continue
            self.saved[index] = percent
            self._pactl("set-sink-input-volume", str(index), f"{self.level}%")
            ducked += 1
        return ducked

    def unduck_strays(self, target=100):
        """Put back any stream of this app left turned down by a run that never restored.

        The test is "below target", not "exactly DUCK_LEVEL", and the difference is not
        academic: the duck actually observed in the wild was at 25, the value DUCK_LEVEL
        held *before* it moved to 40, so an exact match walks straight past the case this
        exists for. Below-target is safe here specifically because nothing else ever
        writes this fader -- Spotify's own volume slider is librespot's softvol, applied
        to the samples upstream of the sink-input, so a librespot sink-input under 100%
        can only be this class's doing. That reasoning is tied to DUCK_APP; point this at
        an app whose own volume control *is* the sink-input fader and it would fight the
        user. Returns how many, so a stranded duck is reported, not silently corrected.
        """
        if self.level is None:
            return 0
        strays = [i for i, pct in self._streams() if i not in self.saved and pct < target]
        for index in strays:
            self._pactl("set-sink-input-volume", str(index), f"{target}%")
        return len(strays)

    def restore(self):
        for index, percent in self.saved.items():
            self._pactl("set-sink-input-volume", str(index), f"{percent}%")
        # The stream may have gone and come back under a new index while ducked, which
        # carries the ducked volume forward through module-stream-restore. Sweep it to
        # the loudest thing we saved, which is where this conversation found the music.
        target = max(self.saved.values(), default=100)
        self.saved = {}
        self.unduck_strays(target)


def mic_capture_gain():
    """(percent, dB) for the mic's ALSA capture control, or None if unreadable.

    Worth printing at startup. This control was found sitting at 0 of 16 once, and
    the only symptom was having to talk with your nose against the mic -- nothing in
    the wake word scores or the transcripts pointed at it. PulseAudio owns the
    control on this source (HW_VOLUME_CTRL), so a stray 0% there writes straight
    through to the value the hw: path reads, bypassing nothing.
    """
    try:
        cards = Path("/proc/asound/cards").read_text()
    except OSError:
        return None

    index = None
    for line in cards.splitlines():
        head = re.match(r"\s*(\d+)\s+\[", line)
        if head and MIC in line:
            index = head.group(1)
            break
    if index is None:
        return None

    try:
        out = subprocess.run(
            ["amixer", "-c", index, "sget", "Mic"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    level = re.search(r"Capture \d+ \[(\d+)%\] \[(-?[\d.]+)dB\]", out)
    return (int(level.group(1)), float(level.group(2))) if level else None


# --- wake word detector ---------------------------------------------------


def detector(loop, wake_event):
    """Worker thread. Rolling window -> predict() -> set an asyncio event.

    Runs for the life of the program but only predicts while `listening` is set.
    While a session is open it still drains its queue -- otherwise the queue fills
    and the drop counter stops meaning anything -- but never scores, so Jarvis
    cannot wake himself on his own voice.
    """
    model = WakeWordModel(models=[MODEL_PATH])

    # Ring buffer at the device rate. The whole window is decimated at once per
    # prediction, not per 80 ms block, so resampler edge effects are not spliced
    # into the window every block.
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    primed = 0
    hits = 0
    cooldown = 0
    since_predict = 0
    was_listening = False

    while not stop_flag.is_set():
        try:
            frame = detect_q.get(timeout=0.5)
        except queue.Empty:
            continue

        active = listening.is_set()

        # Re-arming after a conversation: the window still holds Jarvis's farewell
        # bleeding in through the mic, which would score immediately. Start over.
        if active and not was_listening:
            window[:] = 0.0
            primed = 0
            hits = 0
            since_predict = 0
        was_listening = active

        if not active:
            continue

        n = len(frame)
        window[:-n] = window[n:]
        window[-n:] = frame
        primed = min(primed + n, WINDOW_SAMPLES)

        since_predict += 1
        if since_predict < HOP_BLOCKS or primed < WINDOW_SAMPLES:
            continue
        since_predict = 0

        chunk = resample_poly(window, MODEL_RATE, DEVICE_RATE)
        # int16 explicitly: float32 is accepted but the expected scaling is
        # ambiguous, and int16 is divided by 32768.0 internally anyway.
        pcm = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)

        score = model.predict(pcm)[MODEL_NAME]

        if cooldown > 0:
            cooldown -= 1
            continue

        if VERBOSE:
            print(f"    score {score:.3f}")

        if score > THRESHOLD:
            hits += 1
            if hits >= CONSECUTIVE:
                hits = 0
                cooldown = int(REFRACTORY / HOP_SECONDS)
                loop.call_soon_threadsafe(wake_event.set)
        else:
            hits = 0


# --- conversation state ---------------------------------------------------


class Conversation:
    """Shared state for one wake-to-idle cycle, plus the always-on pre-roll ring."""

    def __init__(self):
        self.session = None
        self.primed = False         # has the pre-roll been flushed into this session
        self.pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
        self.exit_event = asyncio.Event()
        # Set the moment a goodbye is known, from anywhere -- the regex, the tool, a
        # future caller. Acting on it is close_when_ending's job, NOT receive_loop's,
        # which is the whole point: receive_loop can be blocked inside session.receive()
        # forever, and that is exactly when the ending needs to happen.
        self.ending_event = asyncio.Event()
        # Counts completed turns. A counter and not an Event, because an Event is an
        # edge: receive_loop would set it at turn_complete and clear it at the top of
        # the next turn, and a closer that was not scheduled in between would miss it
        # entirely and wait out the whole FAREWELL_TIMEOUT -- turning a normal goodbye
        # into the very stall this is meant to fix. `ending_turn` is captured inside
        # begin_ending(), so which turn to wait for is decided by the writer, not by
        # whenever the reader happens to wake up.
        self.turn_index = 0
        self.ending_turn = 0
        self.reason = ""
        self.last_voice_at = 0.0    # when the user was last heard speaking
        self.model_busy = False     # a model turn is in flight
        # The user has spoken and a reply is owed but has not started arriving yet.
        # This covers the round trip -- server VAD's silence_duration_ms, then the
        # model thinking -- during which nothing else marks the session as active.
        self.awaiting_reply = False
        self.asked_at = 0.0
        self.ending = False         # goodbye seen; leave after this turn finishes
        # Actions in flight. While one runs, model_busy is False and the player is
        # quiet, so idle_watchdog would otherwise count that as silence and drop the
        # session mid-action. Same class of hole as the round-trip window below.
        self.action_busy = 0
        self.interruptions = 0      # echo leakage read as an interruption
        self.last_data_at = 0.0     # when the model last sent audio, for TURN_TIMEOUT

    def begin(self, session):
        self.session = session
        self.primed = False
        self.exit_event.clear()
        self.ending_event.clear()
        self.turn_index = 0
        self.ending_turn = 0
        self.reason = ""
        self.model_busy = False
        self.awaiting_reply = False
        self.ending = False
        self.interruptions = 0
        self.action_busy = 0
        # Start the clock, but do NOT go through heard_voice(): waking is not a
        # question, so no reply is owed yet. Otherwise "say the wake word and then
        # nothing" would hold the session open for the whole REPLY_TIMEOUT instead
        # of dropping back to idle after EXIT_DURATION.
        self.last_voice_at = time.perf_counter()

    def end(self):
        self.session = None
        self.primed = False

    def heard_voice(self):
        """The user is talking. Also means a reply is now owed."""
        self.last_voice_at = time.perf_counter()
        if not self.awaiting_reply:
            self.awaiting_reply = True
            self.asked_at = self.last_voice_at

    def begin_ending(self, reason):
        """A goodbye has been recognised. The session closes once the farewell is out.

        Deliberately separate from finish(): this says "start leaving", finish() says
        "leave now". Everything that spots a goodbye calls this and nothing else, so
        there is exactly one place that decides how long to wait for the farewell.
        """
        if not self.ending:
            self.ending = True
            self.reason = self.reason or reason
            self.ending_turn = self.turn_index
            self.ending_event.set()

    def finish(self, reason):
        if not self.exit_event.is_set():
            self.reason = reason
            self.exit_event.set()


# --- tasks ----------------------------------------------------------------


def _next_frame(q):
    try:
        return q.get(timeout=0.2)
    except queue.Empty:
        return None


async def uplink_pump(convo):
    """Mic -> 16 kHz -> either the pre-roll ring (idle) or the socket (convo).

    One long-lived task rather than one per conversation, for two reasons: the
    Decimator's lfilter state has to stay continuous across the idle/convo boundary
    or the first block of every conversation carries a transient, and the ring has
    to be filling *before* the wake word fires or there is nothing to pre-roll.
    """
    loop = asyncio.get_running_loop()
    decimator = Decimator()
    complained_for = None      # the session already reported, so it is said once

    while True:
        frame = await loop.run_in_executor(None, _next_frame, uplink_q)
        if frame is None:
            continue

        # Always filter, even while idle, so the state stays continuous.
        chunk = decimator(frame)
        pcm = np.clip(chunk * 32767, -32768, 32767).astype(np.int16).tobytes()

        session = convo.session
        if session is None:
            convo.pre_roll.append(pcm)
            continue

        try:
            if not convo.primed:
                # First frame of a new conversation. Everything the ring holds is the
                # wake word and whatever the user said straight after it, so it goes in
                # ahead of live audio -- this is what keeps the command from arriving
                # truncated.
                backlog = b"".join(convo.pre_roll)
                convo.pre_roll.clear()
                convo.primed = True
                if backlog:
                    await _send(session, backlog)
                    print(f"  (pre-roll: {len(backlog) // 2 / SEND_RATE:.1f}s)")

            await _send(session, pcm)
        except Exception as exc:
            # THE bug this guard exists for: an unguarded raise here killed the task
            # outright. Nothing awaits the pump until shutdown, so the failure printed
            # nothing, the mic simply stopped reaching Gemini for the rest of the run,
            # and the only visible symptom was uplink_q filling and dropping every
            # frame. The socket closing under a send is routine -- at the end of a
            # conversation the pump can be mid-await when the session tears down.
            if convo.session is not session:
                continue          # stale session, already replaced: not an error
            if complained_for is not session:
                # Once per session. Teardown takes a moment and frames keep arriving at
                # 12.5 a second; the same line that often is noise, not information.
                complained_for = session
                print(f"  (uplink send failed: {type(exc).__name__}: {exc})")
            # The socket is gone. Ending the conversation drops us back to idle, where
            # the wake word can start a fresh one -- far better than staying deaf.
            convo.finish("uplink failed")


async def _send(session, pcm):
    """One realtime chunk, bounded. A half-open socket can hang a send forever."""
    await asyncio.wait_for(
        session.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={SEND_RATE}")
        ),
        timeout=SEND_TIMEOUT,
    )


async def receive_loop(session, convo, player, manager):
    """One pass of session.receive() per model turn -- it stops at turn_complete."""
    while True:
        upsampler = Upsampler()
        heard = []
        said = []

        async for response in session.receive():
            if response.tool_call:
                await _handle_tool_call(session, convo, manager, response.tool_call)

            # The user barged in over the turn that asked for the call. Sending a result
            # afterwards would push a stale answer into a conversation that moved on.
            if response.tool_call_cancellation:
                manager.cancel(response.tool_call_cancellation.ids)

            content = response.server_content
            if not content:
                continue

            if content.interrupted:
                # A real barge-in: the user talked over Johnny. Echo leakage was
                # the worry here and it measured fine on this hardware at
                # VOLUME = 0.1, so treat this as the user meaning it and stop.
                convo.interruptions += 1
                player.flush()

            # Streams *while* the user speaks, so it is the low-latency signal for
            # "someone is still there" -- far better than waiting for turn commit.
            interim = content.interim_input_transcription
            if interim and interim.text:
                convo.heard_voice()

            if content.input_transcription and content.input_transcription.text:
                convo.heard_voice()
                heard.append(content.input_transcription.text)
                # Checked here, as the words arrive, rather than only at turn_complete.
                # A goodbye recognised mid-turn still waits for the farewell (that is
                # close_when_ending's bounded wait), but it no longer *depends* on a
                # turn_complete that may never come.
                if GOODBYE.search("".join(heard)):
                    convo.begin_ending("goodbye phrase")

            if content.output_transcription and content.output_transcription.text:
                said.append(content.output_transcription.text)

            if response.data:
                convo.model_busy = True
                convo.last_data_at = time.perf_counter()
                convo.awaiting_reply = False    # it arrived; model_busy holds now
                pcm = np.frombuffer(response.data, dtype=np.int16)
                player.push(upsampler(pcm.astype(np.float32) / 32768.0))

            # The model says it is deliberately not answering and expects the user
            # to keep talking. Nothing is owed, so release the hold and let the
            # silence clock run normally.
            if content.waiting_for_input:
                convo.awaiting_reply = False

        # --- turn complete ---
        convo.model_busy = False
        convo.awaiting_reply = False

        text = "".join(heard).strip()
        if text:
            print(f"  [heard]  {text}")
            if GOODBYE.search(text):
                convo.begin_ending("goodbye phrase")
        if said:
            print(f"  johnny>  {''.join(said).strip()}")

        # Whoever is leaving does not leave from here. close_when_ending owns the
        # teardown, and it is a separate task precisely so that it still runs when
        # this loop is stuck inside session.receive() waiting for a turn_complete
        # that is not coming.
        convo.turn_index += 1

        # Deliberately no heard_voice() here. The clock is anchored in
        # idle_watchdog on max(user last spoke, speaker went quiet), so the
        # countdown starts on its own once Johnny's audio finishes -- and
        # heard_voice() would now wrongly mark another reply as owed.


async def _handle_tool_call(session, convo, manager, tool_call):
    """Hand the calls to the ActionManager, and pick up the one bit of state it cannot own.

    Everything about running a function -- the executor, the timeout, turning an exception
    into something Gemini can say out loud -- belongs to the manager. Ending the session
    does not: only this file knows that playback has to drain before the socket closes, or
    the farewell is cut off mid-word.
    """
    def on_call(name, args, response):
        if name == END_CONVERSATION.name:
            convo.begin_ending("end_conversation tool")

    convo.action_busy += 1
    try:
        await manager.dispatch(session, tool_call, on_call=on_call)
    finally:
        convo.action_busy -= 1


async def idle_watchdog(convo, player):
    """Leave after EXIT_DURATION of silence in the room.

    The anchor is the later of "the user last spoke" and "Johnny last made a sound",
    so the countdown starts when the *room* goes quiet -- which after a reply means
    when Johnny stops talking, not when the user stopped.

    That distinction is the whole point. Anchoring on the user's audio alone leaves
    three separate windows where nothing marks the session as active:

      * the round trip -- server VAD waits silence_duration_ms, then the model
        thinks, and only then does the first chunk arrive. `awaiting_reply` covers
        this one, capped by REPLY_TIMEOUT so a reply that never comes cannot wedge
        the session open;
      * the reply itself -- covered by `model_busy` and `player.queued`;
      * the device buffer -- `queued` hits zero when the callback consumes the last
        sample, but latency="high" means audio is still coming out. `quiet_since()`
        folds that tail back in.

    Get any of them wrong and the conversation drops to idle while Johnny is still
    answering, and the user has to say the wake word again to continue.
    """
    holding = None            # which of the four reasons is holding the session
    holding_since = 0.0
    announced = False
    counting = False

    while True:
        await asyncio.sleep(0.25)
        now = time.perf_counter()

        # A reply is in flight or still playing: not silence by any definition.
        # `stalled()` is the escape hatch -- without it a dead output stream leaves
        # `queued` pinned above zero and this loop never exits, which is exactly how
        # a session hangs with no error printed anywhere.
        if convo.model_busy and now - convo.last_data_at > TURN_TIMEOUT:
            # The other half of the same deadlock, for a turn with no goodbye in it.
            # model_busy is only ever cleared at turn_complete, so a turn that stops
            # arriving without completing would hold this loop open forever.
            print(f"  (no audio for {TURN_TIMEOUT:.0f}s and the turn never completed "
                  "-- treating it as over)")
            convo.model_busy = False
            hold = None
        elif convo.model_busy:
            hold = "the model's turn has not completed"
        elif convo.action_busy:
            hold = "an action is still running"
        elif player.queued > 0 and not player.stalled():
            hold = f"{player.queued / PLAYBACK_RATE:.1f}s of audio still queued"
        else:
            hold = None

        # Owed a reply that has not started arriving. Hold, but not forever.
        if hold is None and convo.awaiting_reply:
            if now - convo.asked_at < REPLY_TIMEOUT:
                hold = "waiting for a reply that has not started"
            else:
                convo.awaiting_reply = False

        if hold != holding:
            holding, holding_since, announced = hold, now, False

        if hold is not None:
            counting = False
            if not announced and now - holding_since > HOLD_REPORT:
                print(f"  (session still open: {hold})")
                announced = True
            continue

        quiet_for = now - max(convo.last_voice_at, player.quiet_since())
        if quiet_for < HOLD_REPORT:
            counting = False          # someone spoke; the countdown restarted
        elif not counting:
            # The single most likely explanation for "it stalled going back to idle":
            # the goodbye was not recognised, so the only thing left to end the session
            # is this countdown -- and it used to run in complete silence. Announced
            # once, after HOLD_REPORT, so ordinary gaps between turns stay quiet.
            counting = True
            print(f"  (room quiet -- back to idle in "
                  f"{EXIT_DURATION - quiet_for:.0f}s unless you speak)")
        if quiet_for > EXIT_DURATION:
            convo.finish(f"{EXIT_DURATION:.0f}s silence")
            return


async def close_when_ending(convo, player):
    """Close the session once a goodbye is known, farewell first, but not forever.

    This exists because "leave after this turn finishes" is not a promise the server
    keeps. If turn_complete never arrives, receive_loop stays inside session.receive(),
    model_busy stays True, idle_watchdog holds on it, and the session hangs with nothing
    printed -- the farewell plays and then it is stuck until Ctrl-C.

    So the wait for the farewell is bounded, and every path out of here closes.
    """
    await convo.ending_event.wait()
    deadline = time.perf_counter() + FAREWELL_TIMEOUT
    while convo.turn_index == convo.ending_turn and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)
    if convo.turn_index == convo.ending_turn:
        print(f"  (the farewell turn never completed in {FAREWELL_TIMEOUT:.0f}s "
              "-- closing anyway)")
    await player.drain(timeout=FAREWELL_TIMEOUT)
    convo.finish(convo.reason or "goodbye")


async def converse(session, convo, player, manager):
    """Run one conversation until something ends it."""
    recv = asyncio.create_task(receive_loop(session, convo, player, manager))
    watch = asyncio.create_task(idle_watchdog(convo, player))
    closer = asyncio.create_task(close_when_ending(convo, player))

    done, pending = await asyncio.wait(
        [recv, watch, closer, asyncio.create_task(convo.exit_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    # Surface a crash in a task rather than silently returning to idle.
    for task in done:
        if task.exception():
            raise task.exception()


def supervise(task, name):
    """Make a background task's death loud.

    create_task + never awaiting it means an exception vanishes: no traceback, no exit,
    just a capability that silently stopped. Both long-lived tasks here are exactly that
    shape, so both get this.
    """
    def done(finished):
        if finished.cancelled():
            return                      # shutdown, expected
        exc = finished.exception()
        if exc is not None:
            print(f"\n  *** {name} DIED: {type(exc).__name__}: {exc}")
            print(f"  *** {name} is gone for the rest of this run. Restart the script.")
    task.add_done_callback(done)
    return task


async def health(player, tasks=()):
    """Audio health, printed from outside the callbacks."""
    last_detect = 0
    last_uplink = 0
    warned = False
    while True:
        await asyncio.sleep(1.0)
        if overflow_flag.is_set():
            print("  (xrun)")
            overflow_flag.clear()

        for name, task in tasks:
            if task.done():
                print(f"  ({name} is not running -- audio is only half working)")
        # A silent death otherwise. Repeated ALSA "underrun occurred" lines on stderr
        # with playback stuck is the signature of too small a device buffer -- see
        # OUT_LATENCY.
        if player.stalled():
            if not warned:
                print(f"  (PLAYBACK STALLED: {player.queued} samples queued, "
                      f"stream.active={player.stream.active}. Recovering.)")
                warned = True
        else:
            warned = False
        # A stateful window fed spliced audio scores garbage, and spliced uplink
        # confuses the VAD. Both numbers must stay at zero. Reported separately because
        # which one moves is the diagnosis: sustained uplink drops at ~12/s (one per
        # 80 ms block) mean the pump has stopped consuming, not that the Pi is slow.
        if detect_drops != last_detect:
            print(f"  (detector dropped {detect_drops - last_detect}, "
                  f"total {detect_drops})")
            last_detect = detect_drops
        if uplink_drops != last_uplink:
            print(f"  (uplink dropped {uplink_drops - last_uplink}, "
                  f"total {uplink_drops}) -- Gemini is not hearing the mic")
            last_uplink = uplink_drops


# --- run ------------------------------------------------------------------


def load_dotenv():
    """Read KEY=VALUE lines from Dormy/.env into the environment, if it exists.

    Hand-rolled rather than pulling in python-dotenv for six lines. The environment
    always wins, so an explicit `GEMINI_API_KEY=... python ...` still overrides.
    """
    path = Path(__file__).resolve().parent.parent / ".env"
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


async def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "GEMINI_API_KEY is not set.\n"
            "Get a key at https://aistudio.google.com/apikey, then either put\n"
            f"  GEMINI_API_KEY=...\n"
            f"in {Path(__file__).resolve().parent.parent / '.env'}, or export it.\n"
        )
    if not MODEL_PATH.exists():
        sys.exit(f"wake word model not found: {MODEL_PATH}")

    client = genai.Client(api_key=api_key)

    device_manager = ESP32DeviceManager("/dev/ttyUSB0")

    # Built before the config, since both `tools` and the briefing come from it.
    manager = ActionManager([END_CONVERSATION, *builtin_actions()])
    manager.add_all(device_manager.actions())

    config = build_config(manager)

    in_device, out_device, channels = resolve_devices()

    print(f"model    {MODEL}  (voice {VOICE})")
    print(f"wakeword '{MODEL_NAME}'  threshold {THRESHOLD}  window {WINDOW_SECONDS}s")
    if USE_AEC:
        print(f"aec      on -- {AEC_SOURCE} / {AEC_SINK} (webrtc)")
        print(f"buffers  in {IN_LATENCY}s / out {OUT_LATENCY}s "
              "(explicit: \"high\" is only 80 ms through pulse and underruns)")
        print(f"mic      {MIC} via pulse @ {DEVICE_RATE} Hz -> {SEND_RATE} Hz")
        print(f"speaker  {SPEAKER} via pulse @ {RECV_RATE} Hz -> "
              f"{PLAYBACK_RATE} Hz, {channels} ch")
    else:
        print("aec      OFF -- raw hw: devices, expect echo at high mic gain")
        print(f"mic      {MIC} @ {DEVICE_RATE} Hz -> {SEND_RATE} Hz")
        print(f"speaker  {SPEAKER} @ {RECV_RATE} Hz -> {PLAYBACK_RATE} Hz, "
              f"{channels} ch")
    print(f"exit     {EXIT_DURATION:.0f}s silence, a goodbye, or Ctrl-C")
    print(f"pre-roll {PRE_ROLL_SECONDS}s")
    print(f"actions  {len(manager)} registered")
    print(manager.describe())

    gain = mic_capture_gain()
    if gain is None:
        print("mic gain unknown (could not read the ALSA capture control)")
    else:
        percent, db = gain
        print(f"mic gain {percent}%  ({db:+.1f} dB)")
        if percent < 50:
            # Not fatal -- it still runs, it just makes you lean in.
            print(
                f"  WARNING: capture gain is low. Expect to have to talk close.\n"
                f"  Raise it with:  pactl set-source-volume "
                f"alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00"
                f".analog-mono 95%"
            )
    print()

    loop = asyncio.get_running_loop()
    wake_event = asyncio.Event()
    convo = Conversation()
    wake_chime = load_chime(WAKE_SOUND)
    exit_chime = load_chime(EXIT_SOUND)
    ducker = MusicDucker()
    # A duck that outlived its stream is remembered per-application by
    # module-stream-restore, so the music comes back ducked on the next run, and the
    # next reboot, with no symptom except "Spotify is quiet even at full volume".
    # Clear it at startup and say so, because it is otherwise invisible.
    stranded = ducker.unduck_strays()
    if stranded:
        print(f"un-ducked {stranded} music stream(s) left at {DUCK_LEVEL}% by an earlier run")

    with sd.InputStream(
        device=in_device,
        samplerate=DEVICE_RATE,
        blocksize=BLOCK,
        dtype="float32",
        channels=1,
        callback=callback,
        latency=IN_LATENCY,
    ), Player(out_device, channels) as player:
        worker = threading.Thread(target=detector, args=(loop, wake_event), daemon=True)
        worker.start()

        pump = supervise(asyncio.create_task(uplink_pump(convo)), "uplink pump")
        monitor = asyncio.create_task(health(player, [("uplink pump", pump)]))

        try:
            while True:
                listening.set()
                # Clear *after* re-arming, not after waiting. The detector can fire
                # twice on one long utterance (REFRACTORY is shorter than a held
                # sound), and a set that lands between the wait and the clear would
                # open the next session with no wake word at all. Re-arming zeroes
                # the detector's window, so it cannot score for a full WINDOW_SECONDS
                # after this point and no real wake can be lost here.
                wake_event.clear()
                print(f"idle -- say \"{MODEL_NAME}\"")
                await wake_event.wait()
                listening.clear()      # stop scoring before Jarvis makes any sound
                print("*** wake ***")

                # Not awaited. Opening the socket takes a few hundred ms of silence,
                # and the chime is what that silence is for -- JARVIS_PLAN.md:236's
                # "chime covering the gap". It also holds idle_watchdog off while the
                # session comes up, because queued audio counts as Johnny talking.
                if wake_chime is not None:
                    player.push(wake_chime)
                ducked = await asyncio.to_thread(ducker.duck)
                if ducked:
                    print(f"    (music ducked to {DUCK_LEVEL}%: {ducked} stream(s))")

                try:
                    async with client.aio.live.connect(
                        model=MODEL, config=config
                    ) as session:
                        convo.begin(session)
                        await converse(session, convo, player, manager)
                finally:
                    convo.end()
                    player.flush()
                    # After the flush, so it is not thrown away with the model's
                    # audio, and before listening.set() at the top of the loop, so the
                    # detector's window is zeroed over it and Johnny cannot wake
                    # himself on his own goodbye.
                    if exit_chime is not None:
                        player.push(exit_chime)
                        await player.drain(timeout=5.0)
                    await asyncio.to_thread(ducker.restore)

                note = ""
                if convo.interruptions:
                    note = f", {convo.interruptions} barge-in(s)"
                print(f"--- back to idle ({convo.reason}{note})\n")
        finally:
            stop_flag.set()
            # Ctrl-C in the middle of a conversation must not leave the music at 25%.
            ducker.restore()
            for task in (pump, monitor):
                task.cancel()
            await asyncio.gather(pump, monitor, return_exceptions=True)
            if player.underruns:
                print(f"playback underruns: {player.underruns}")
            if detect_drops or uplink_drops:
                print(f"dropped frames: detector {detect_drops}, uplink {uplink_drops}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print()
