"""
Phase 4, standalone: Gemini Live API, microphone in -> spoken reply out.

Independent of the wake word work -- no detector, no ring buffer, no chime.
Push-to-talk instead: Enter opens a turn, Enter closes it.

Why audio out rather than text out: the Live API's current models are all
native-audio and support only the AUDIO response modality. Asking for TEXT is
rejected outright with

    1007 ... The requested combination of response modalities (TEXT) is not
    supported by the model

so a text-out + local-TTS design is not available on this API today. Gemini
speaks the reply itself here; output transcription is logged so there is still
a text record of what it said.

Input is strict and does not degrade gracefully: raw 16-bit PCM, 16 kHz, mono,
little-endian. Output arrives as 24 kHz PCM, which this speaker cannot play --
see PLAYBACK_RATE.
"""

import asyncio
import os
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from scipy.signal import firwin, lfilter
from google import genai
from google.genai import types

# --- config ---------------------------------------------------------------

MIC = "USB PnP Sound Device"
SPEAKER = "USB Composite Device"

DEVICE_RATE = 48000               # the PnP mic is 48 kHz only, so capture high
SEND_RATE = 16000                 # and decimate; the API accepts nothing else
BLOCK = int(DEVICE_RATE * 0.08)   # 80 ms capture blocks

RECV_RATE = 24000                 # what the Live API sends back
# The speaker accepts 48 kHz and nothing else (probed: 24 kHz is rejected).
# Addressing it by name maps to hw:0,0 and bypasses ALSA's plug layer, so there
# is no automatic resampling -- 24 kHz raises paInvalidSampleRate. Upsample x2.
PLAYBACK_RATE = 48000
OUT_BLOCK = 1920                  # 40 ms of output; small enough to stay responsive

# Playback volume, 0.0-1.0. This speaker exposes no hardware volume control --
# card 0 has only a mute switch ('PCM Playback Switch'), so alsamixer cannot
# turn it down and the level has to be scaled in software. Roughly logarithmic
# to the ear: 0.5 is about -6 dB, 0.25 about -12 dB.
# Worth keeping low for more than comfort: clipping is non-linear and defeats
# AEC, so a quieter speaker buys echo suppression that no tuning can. That
# matters once the wake word drives this and the mic is live during playback.
VOLUME = 0.1

# Model IDs in this family rotate frequently. Read off
# ai.google.dev/gemini-api/docs/models on 2026-09-06 -- re-check rather than
# trusting this constant, and do not copy one out of a tutorial.
MODEL = "gemini-3.1-flash-live-preview"

# 30 prebuilt voices, and this is the only voice control the Live API offers --
# there is no cloning here (see the note in testGeminiLiveAPIwithTTS.py). Pick by
# character: Sulafat warm, Charon informative, Achird friendly, Vindemiatrix
# gentle, Schedar even, Kore firm, Puck upbeat, Zephyr bright, Aoede breezy,
# Enceladus breathy, Gacrux mature, Achernar soft, Algenib gravelly. The rest:
# Fenrir, Leda, Orus, Callirrhoe, Autonoe, Iapetus, Umbriel, Algieba, Despina,
# Erinome, Rasalgethi, Laomedeia, Alnilam, Pulcherrima, Zubenelgenubi,
# Sadachbia, Sadaltager. Audition them by re-running with the same question.
#
# Do not add speech_config.language_code alongside this: native-audio models
# choose the language themselves and reject an explicit code.
VOICE = "Puck"

SYSTEM_INSTRUCTION = (
    "You are Johnny, a voice assistant in a college dorm room. "
    "Answer in one or two short sentences. You are speaking out loud, so no "
    "markdown, no lists, and no emoji."
)

CONFIG = types.LiveConnectConfig(
    # AUDIO is the only modality these models support.
    response_modalities=["AUDIO"],
    system_instruction=SYSTEM_INSTRUCTION,
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
        )
    ),
    # Both on from the start: logging what Gemini heard against what it said
    # removes most of the guesswork when a command misfires.
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    # Server VAD off, explicit turn boundaries -- the plan's Phase 4 position
    # until AEC is proven. Server VAD reads echo leakage as an interruption and
    # makes the model talk over itself. Push-to-talk also means the mic never
    # sends while the speaker is playing, so there is no echo path here at all.
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
    ),
)

# --- resampling -----------------------------------------------------------


class Decimator:
    """48 kHz -> 16 kHz for the uplink, carrying filter state across blocks.

    Resampling each 80 ms block independently zero-pads the filter at both
    edges, leaving a transient at every block boundary -- 12.5 a second into an
    ASR front end. Holding lfilter's state makes the stream continuous.
    """

    def __init__(self, factor=DEVICE_RATE // SEND_RATE, taps=129):
        self.factor = factor
        # Cutoff is relative to Nyquist, so 1/factor lands on the target rate's
        # Nyquist (8 kHz here).
        self.b = firwin(taps, 1.0 / factor)
        self.zi = np.zeros(taps - 1)

    def __call__(self, frame):
        y, self.zi = lfilter(self.b, [1.0], frame, zi=self.zi)
        return y[:: self.factor]


class Upsampler:
    """24 kHz -> 48 kHz for the downlink, also stateful.

    Same reasoning in reverse: per-chunk resampling would click at every chunk
    seam, and Gemini sends many small chunks per reply.
    """

    def __init__(self, factor=PLAYBACK_RATE // RECV_RATE, taps=129):
        self.factor = factor
        # Zero-stuffing divides signal energy by `factor`, so scale the filter
        # back up to preserve level.
        self.b = firwin(taps, 1.0 / factor) * factor
        self.zi = np.zeros(taps - 1)

    def __call__(self, samples):
        stuffed = np.zeros(len(samples) * self.factor)
        stuffed[:: self.factor] = samples
        y, self.zi = lfilter(self.b, [1.0], stuffed, zi=self.zi)
        return y.astype(np.float32)


# --- capture --------------------------------------------------------------

audio_q = queue.Queue(maxsize=50)
overflow_flag = threading.Event()
drops = 0


def callback(indata, frames, time_info, status):
    """Realtime thread. Copy and enqueue only -- no work, no printing."""
    global drops

    if status:
        overflow_flag.set()

    try:
        audio_q.put_nowait(indata[:, 0].copy())
    except queue.Full:
        drops += 1


# --- playback -------------------------------------------------------------


class Player:
    """48 kHz output stream drained by its callback, so the reply starts
    playing while the rest of it is still arriving.

    The callback copies out of queued chunks piecewise into a preallocated
    scratch buffer rather than concatenating, so it allocates nothing and does
    no work beyond memcpy.
    """

    def __init__(self, device, channels):
        self.channels = channels
        self.q = queue.Queue()
        self.scratch = np.zeros(OUT_BLOCK, dtype=np.float32)
        self.head = None
        self.pos = 0
        self.queued = 0          # samples not yet played
        self.underruns = 0
        # When the first sample of this turn actually reached the DAC. Set in
        # the callback as a bare timestamp -- a flag, not printing.
        self.first_sound_at = None
        self.stream = sd.OutputStream(
            device=device,
            samplerate=PLAYBACK_RATE,
            blocksize=OUT_BLOCK,
            dtype="float32",
            channels=channels,
            callback=self._callback,
            latency="high",
        )

    def _callback(self, outdata, frames, time_info, status):
        if status:
            overflow_flag.set()

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

        if filled and self.first_sound_at is None:
            self.first_sound_at = time.perf_counter()

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

    async def drain(self):
        while self.queued > 0:
            await asyncio.sleep(0.05)
        # Let the last block reach the DAC before returning.
        await asyncio.sleep(OUT_BLOCK / PLAYBACK_RATE)

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *exc):
        return self.stream.__exit__(*exc)


# --- session --------------------------------------------------------------


class Turn:
    """Push-to-talk state shared between the console and the capture pump."""

    def __init__(self):
        self.open = False
        self.done = asyncio.Event()
        self.sent_at = None      # when activity_end went out; t0 for the timings


def _next_frame():
    try:
        return audio_q.get(timeout=0.2)
    except queue.Empty:
        return None


async def capture_pump(session, turn):
    """Drain the mic queue and forward audio while a turn is open."""
    loop = asyncio.get_running_loop()
    decimator = Decimator()

    while True:
        frame = await loop.run_in_executor(None, _next_frame)
        if frame is None:
            continue

        # Always filter, even while idle, so the decimator's state stays
        # continuous and the first block of a turn carries no transient.
        chunk = decimator(frame)

        if not turn.open:
            continue

        pcm = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
        await session.send_realtime_input(
            audio=types.Blob(data=pcm.tobytes(), mime_type=f"audio/pcm;rate={SEND_RATE}")
        )


async def receive_loop(session, turn, player):
    """One pass of session.receive() per model turn -- it stops at turn_complete."""
    while True:
        upsampler = Upsampler()
        heard = []
        said = []
        first_data_at = None         # first audio byte off the socket

        async for response in session.receive():
            content = response.server_content

            if content and content.interrupted:
                player.flush()

            if content and content.input_transcription and content.input_transcription.text:
                heard.append(content.input_transcription.text)
            if content and content.output_transcription and content.output_transcription.text:
                said.append(content.output_transcription.text)

            if response.data:
                if first_data_at is None:
                    first_data_at = time.perf_counter()
                pcm = np.frombuffer(response.data, dtype=np.int16)
                player.push(upsampler(pcm.astype(np.float32) / 32768.0))

        if heard:
            print(f"  [heard]  {''.join(heard).strip()}")
        if said:
            print(f"  jarvis>  {''.join(said).strip()}")

        await player.drain()

        # The numbers that make this file comparable to testGeminiLiveAPIwithTTS.py,
        # measured from the same t0 (activity_end) so the two can sit in one table.
        # Here Gemini synthesizes, so FIRST SOUND is round trip only -- there is no
        # local synthesis term at all.
        t0 = turn.sent_at
        if t0 is not None:
            print("\n  timings from end of your turn:")
            if first_data_at:
                print(f"    first audio from Gemini {first_data_at - t0:5.2f}s")
            if player.first_sound_at:
                print(f"    FIRST SOUND             {player.first_sound_at - t0:5.2f}s"
                      f"  <-- the wait")
            print(f"    finished speaking       {time.perf_counter() - t0:5.2f}s")

        player.first_sound_at = None
        turn.done.set()


async def console(session, turn):
    """Enter to start a turn, Enter again to close it."""
    print("\nEnter = start talking, Enter again = send, Ctrl-C = quit\n")

    while True:
        await asyncio.to_thread(input, "[press Enter to speak] ")

        if overflow_flag.is_set():
            print("  (xrun since last turn)")
            overflow_flag.clear()
        if drops:
            print(f"  ({drops} frames dropped since start)")

        # Discard whatever was captured while idle so the turn opens clean.
        while not audio_q.empty():
            audio_q.get_nowait()

        await session.send_realtime_input(activity_start=types.ActivityStart())
        turn.open = True
        print("  listening... (Enter to send)")

        await asyncio.to_thread(input)
        turn.open = False
        await session.send_realtime_input(activity_end=types.ActivityEnd())
        turn.sent_at = time.perf_counter()

        turn.done.clear()
        await turn.done.wait()
        print()


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "GEMINI_API_KEY is not set.\n"
            "Get a key at https://aistudio.google.com/apikey, then:\n"
            "  export GEMINI_API_KEY=...\n"
        )

    client = genai.Client(api_key=api_key)
    channels = min(sd.query_devices(SPEAKER, "output")["max_output_channels"], 2)

    print(f"model   {MODEL}  (voice {VOICE})")
    print(f"mic     {MIC} @ {DEVICE_RATE} Hz -> {SEND_RATE} Hz")
    print(f"speaker {SPEAKER} @ {RECV_RATE} Hz -> {PLAYBACK_RATE} Hz, {channels} ch")

    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
        turn = Turn()

        with sd.InputStream(
            device=MIC,
            samplerate=DEVICE_RATE,
            blocksize=BLOCK,
            dtype="float32",
            channels=1,
            callback=callback,
            latency="high",
        ), Player(SPEAKER, channels) as player:
            pump = asyncio.create_task(capture_pump(session, turn))
            recv = asyncio.create_task(receive_loop(session, turn, player))
            try:
                await console(session, turn)
            finally:
                for task in (pump, recv):
                    task.cancel()
                await asyncio.gather(pump, recv, return_exceptions=True)
                if player.underruns:
                    print(f"playback underruns: {player.underruns}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print()
