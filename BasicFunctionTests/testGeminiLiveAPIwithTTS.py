"""
Phase 4 variant: Gemini Live API for understanding, pocket-tts for the voice.

Same premise as testGeminiLiveAPI.py -- push-to-talk, Enter opens a turn, Enter
closes it -- but Gemini's own audio is discarded and the reply is spoken locally
by pocket-tts instead. The point is to measure whether that delay is tolerable.

The Live API cannot return text (its models are all native-audio and reject
response_modalities=["TEXT"] with a 1007), so the text driving pocket-tts comes
from output_audio_transcription. Gemini still synthesizes audio we throw away;
that is the price of this route.

pocket-tts runs at RTF ~1.6 on this Pi, so it dominates the total latency. The
one lever that helps is pipelining: each sentence is handed to pocket-tts as
soon as its transcript is complete, so speaking starts after the first sentence
is synthesized rather than the whole reply. Every stage is timed and printed.
"""

import asyncio
import os
import queue
import re
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import firwin, lfilter
from google import genai
from google.genai import types
from pocket_tts import TTSModel
from pocket_tts.default_parameters import (
    DEFAULT_FRAMES_AFTER_EOS,
    MAX_TOKEN_PER_CHUNK,
    get_default_voice_for_language,
)

# --- config ---------------------------------------------------------------

MIC = "USB PnP Sound Device"
SPEAKER = "USB Composite Device"

DEVICE_RATE = 48000               # the PnP mic is 48 kHz only
SEND_RATE = 16000                 # Live API input accepts nothing else
BLOCK = int(DEVICE_RATE * 0.08)

TTS_RATE = 24000                  # pocket-tts output, same as testTTS.py
PLAYBACK_RATE = 48000             # the speaker accepts nothing else
OUT_BLOCK = 1920                  # 40 ms of output

# No hardware volume control on this speaker -- only a mute switch -- so level
# is scaled in software. 0.5 is about -6 dB.
VOLUME = 0.5

MODEL = "gemini-3.1-flash-live-preview"

# pocket-tts voice: None uses the built-in, or point at a .wav to clone (needs
# the gated weights -- see testTTS.py).
VOICE_WAV = None
BUILTIN_VOICE = "alba"
LANGUAGE = "english"
QUANTIZE = True                   # RTF 3.19x -> 1.61x; fp32 is unusable here

SYSTEM_INSTRUCTION = (
    "You are Jarvis, a voice assistant in a college dorm room. "
    "Answer in one or two short sentences. Your reply is read aloud by a "
    "speech synthesizer, so no markdown, no lists, and no emoji."
)

CONFIG = types.LiveConnectConfig(
    # AUDIO is the only modality these models support. We discard it and use the
    # transcription instead.
    response_modalities=["AUDIO"],
    system_instruction=SYSTEM_INSTRUCTION,
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
    ),
)

# Split on sentence enders so a sentence can be synthesized while the rest of
# the reply is still arriving.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# FIRST SOUND is gated entirely by the *first* hand-off: nothing can play until
# segment one is fully synthesized, and at RTF ~1.6 a four-second opening
# sentence costs almost six seconds of silence. So before anything has been
# spoken, break at a clause boundary as well -- measured on this Pi, that takes
# the first hand-off from 5.7 s to 2.7 s, more than every other lever combined.
# Only the opening segment is split this way; once audio is flowing, later
# segments keep the sentence split, where a pause belongs.
CLAUSE_END = re.compile(r"(?<=,)\s+|\s+(?=and\s|but\s|so\s|then\s)")

# Below this, pocket-tts generates poorly (it has a pad_with_spaces_for_short_inputs
# path for exactly this), and the clause is too short to be worth a seam.
MIN_CLAUSE_WORDS = 4

# --- resampling -----------------------------------------------------------


class Decimator:
    """48 kHz -> 16 kHz uplink, carrying filter state across blocks so block
    seams do not put a transient into the ASR front end every 80 ms."""

    def __init__(self, factor=DEVICE_RATE // SEND_RATE, taps=129):
        self.factor = factor
        self.b = firwin(taps, 1.0 / factor)
        self.zi = np.zeros(taps - 1)

    def __call__(self, frame):
        y, self.zi = lfilter(self.b, [1.0], frame, zi=self.zi)
        return y[:: self.factor]


class Upsampler:
    """24 kHz -> 48 kHz downlink, also stateful. Zero-stuffing divides energy by
    `factor`, so the filter is scaled back up to preserve level."""

    def __init__(self, factor=PLAYBACK_RATE // TTS_RATE, taps=129):
        self.factor = factor
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
    """48 kHz output stream drained by its callback, so a sentence starts
    playing while later ones are still being synthesized.

    The callback copies piecewise out of queued chunks into a preallocated
    scratch buffer, so it allocates nothing and does no work beyond memcpy.
    """

    def __init__(self, device, channels):
        self.channels = channels
        self.q = queue.Queue()
        self.scratch = np.zeros(OUT_BLOCK, dtype=np.float32)
        self.head = None
        self.pos = 0
        self.queued = 0
        self.starved = 0             # output blocks with nothing to play
        self.expecting = False       # True while a reply is still being synthesized
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
            self.first_sound_at = time.perf_counter()   # a flag, not printing

        if filled < frames:
            scratch[filled:] = 0.0
            # Count any shortfall once playback has started and more is coming.
            # Only counting partial fills would miss total starvation, which is
            # exactly the case that tears a reply apart.
            if self.expecting and self.first_sound_at is not None:
                self.starved += 1

        outdata[:] = scratch[:, np.newaxis]

    def push(self, samples):
        # Scaled here rather than in the callback, which stays copy-only.
        samples = np.clip(samples * VOLUME, -1.0, 1.0)
        self.queued += len(samples)
        self.q.put(samples)

    async def drain(self):
        while self.queued > 0:
            await asyncio.sleep(0.05)
        await asyncio.sleep(OUT_BLOCK / PLAYBACK_RATE)


# --- synthesis ------------------------------------------------------------


class Speaker:
    """One worker thread turning sentences into audio, in order.

    A single thread keeps sentences sequential (so the reply is not scrambled)
    and keeps torch to one generation at a time.
    """

    END_OF_TURN = object()

    def __init__(self, model, voice_state, player):
        self.model = model
        self.voice_state = voice_state
        self.player = player
        self.q = queue.Queue()
        self.idle = threading.Event()
        self.idle.set()
        self.samples = 0            # audio produced this turn
        self.synth_time = 0.0       # wall time spent synthesizing this turn
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        upsampler = Upsampler()
        while True:
            item = self.q.get()
            if item is self.END_OF_TURN:
                self.idle.set()
                continue

            t0 = time.perf_counter()
            # Generate the whole sentence before queueing any of it. pocket-tts
            # runs at RTF ~1.6, so pushing chunks as they appear lets playback
            # outrun synthesis and tears silence into the middle of a word --
            # measured at 2.9 s of gaps in a 4.6 s reply. Holding the sentence
            # keeps every gap at a sentence boundary, where a pause belongs.
            pieces = [
                chunk.squeeze().cpu().numpy().astype(np.float32)
                for chunk in self.model.generate_audio_stream(
                    model_state=self.voice_state,
                    text_to_generate=item,
                    max_tokens=MAX_TOKEN_PER_CHUNK,
                    frames_after_eos=DEFAULT_FRAMES_AFTER_EOS,
                )
            ]
            self.synth_time += time.perf_counter() - t0

            if pieces:
                audio = np.concatenate(pieces)
                self.samples += len(audio)
                self.player.push(upsampler(audio))

    def say(self, sentence):
        self.idle.clear()
        self.q.put(sentence)

    def end_turn(self):
        self.q.put(self.END_OF_TURN)

    def reset(self):
        self.samples = 0
        self.synth_time = 0.0

    async def wait_idle(self):
        while not self.idle.is_set() or not self.q.empty():
            await asyncio.sleep(0.05)


def _split_first_clause(pending):
    """Return (clause, sep, rest) if `pending` holds a settled opening clause.

    Settled means a boundary has actually arrived in the transcript, so the text
    before it will not change. Returns ("", "", pending) when there is nothing
    worth handing off yet.
    """
    parts = CLAUSE_END.split(pending, maxsplit=1)
    if len(parts) != 2:
        return "", "", pending
    head, tail = parts[0].strip(), parts[1]
    if len(head.split()) < MIN_CLAUSE_WORDS:
        return "", "", pending
    return head, " ", tail


# --- session --------------------------------------------------------------


class Turn:
    def __init__(self):
        self.open = False
        self.done = asyncio.Event()
        self.sent_at = None          # when activity_end was sent


def _next_frame():
    try:
        return audio_q.get(timeout=0.2)
    except queue.Empty:
        return None


async def capture_pump(session, turn):
    loop = asyncio.get_running_loop()
    decimator = Decimator()

    while True:
        frame = await loop.run_in_executor(None, _next_frame)
        if frame is None:
            continue

        # Always filter so the decimator's state stays continuous.
        chunk = decimator(frame)

        if not turn.open:
            continue

        pcm = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
        await session.send_realtime_input(
            audio=types.Blob(data=pcm.tobytes(), mime_type=f"audio/pcm;rate={SEND_RATE}")
        )


async def receive_loop(session, turn, speaker, player):
    """One pass of session.receive() per model turn -- it stops at turn_complete."""
    while True:
        heard = []
        pending = ""                 # transcript not yet split into a sentence
        spoken = []
        first_text_at = None
        first_sentence_at = None

        async for response in session.receive():
            content = response.server_content
            if not content:
                continue

            # response.data carries Gemini's own audio. Deliberately unused --
            # pocket-tts speaks instead.

            if content.input_transcription and content.input_transcription.text:
                heard.append(content.input_transcription.text)

            if content.output_transcription and content.output_transcription.text:
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                pending += content.output_transcription.text

                # Nothing spoken yet: take the first settled clause rather than
                # waiting for the whole sentence. The cost is a slightly worse
                # prosodic seam and, because playback starts earlier against an
                # RTF > 1 pipeline, a little more total gap -- both bought for
                # roughly three seconds off the wait.
                if first_sentence_at is None:
                    head, _, tail = _split_first_clause(pending)
                    if head:
                        first_sentence_at = time.perf_counter()
                        spoken.append(head)
                        player.expecting = True
                        speaker.say(head)
                        pending = tail

                # Hand off every complete sentence immediately; keep the tail.
                parts = SENTENCE_END.split(pending)
                for sentence in parts[:-1]:
                    if sentence.strip():
                        if first_sentence_at is None:
                            first_sentence_at = time.perf_counter()
                        spoken.append(sentence.strip())
                        player.expecting = True
                        speaker.say(sentence.strip())
                pending = parts[-1]

        if pending.strip():
            if first_sentence_at is None:
                first_sentence_at = time.perf_counter()
            spoken.append(pending.strip())
            player.expecting = True
            speaker.say(pending.strip())

        speaker.end_turn()

        if heard:
            print(f"  [heard]  {''.join(heard).strip()}")
        if spoken:
            print(f"  jarvis>  {' '.join(spoken)}")

        await speaker.wait_idle()
        player.expecting = False
        await player.drain()

        # --- the numbers this file exists for --------------------------------
        t0 = turn.sent_at
        audio_secs = speaker.samples / TTS_RATE if speaker.samples else 0.0
        print("\n  timings from end of your turn:")
        if first_text_at:
            print(f"    first transcript text   {first_text_at - t0:5.2f}s")
        if first_sentence_at:
            print(f"    first full sentence     {first_sentence_at - t0:5.2f}s")
        if player.first_sound_at:
            print(f"    FIRST SOUND             {player.first_sound_at - t0:5.2f}s  <-- the wait")
        print(f"    finished speaking       {time.perf_counter() - t0:5.2f}s")
        if audio_secs:
            print(
                f"    synthesized {audio_secs:.1f}s of speech in "
                f"{speaker.synth_time:.1f}s (RTF {speaker.synth_time / audio_secs:.2f}x)"
            )
        gap = player.starved * OUT_BLOCK / PLAYBACK_RATE
        if gap > 0.05:
            print(f"    silence gaps            {gap:5.2f}s  "
                  f"(pocket-tts fell behind; gaps sit between sentences)")

        speaker.reset()
        player.first_sound_at = None
        player.starved = 0
        turn.done.set()


async def console(session, turn):
    print("\nEnter = start talking, Enter again = send, Ctrl-C = quit\n")

    while True:
        await asyncio.to_thread(input, "[press Enter to speak] ")

        if overflow_flag.is_set():
            print("  (xrun since last turn)")
            overflow_flag.clear()
        if drops:
            print(f"  ({drops} frames dropped since start)")

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

    torch.set_num_threads(4)

    print("loading pocket-tts...")
    t0 = time.perf_counter()
    tts = TTSModel.load_model(language=LANGUAGE, quantize=QUANTIZE)
    tts.to("cpu")

    if VOICE_WAV is None:
        voice = get_default_voice_for_language(LANGUAGE) or BUILTIN_VOICE
    else:
        if not tts.has_voice_cloning:
            sys.exit(
                "This build cannot clone voices -- the gated weights were not "
                "downloaded.\n"
                "  1. Accept the terms at https://huggingface.co/kyutai/pocket-tts\n"
                "  2. Log in locally:  .venv/bin/hf auth login\n"
                "Or set VOICE_WAV = None to use a built-in voice."
            )
        voice = VOICE_WAV
    voice_state = tts.get_state_for_audio_prompt(voice, truncate=VOICE_WAV is not None)
    print(f"  ready in {time.perf_counter() - t0:.1f}s"
          f"  ({'int8' if QUANTIZE else 'fp32'}, voice {voice})")

    client = genai.Client(api_key=api_key)
    channels = min(sd.query_devices(SPEAKER, "output")["max_output_channels"], 2)

    print(f"model   {MODEL}  (its audio is discarded)")
    print(f"mic     {MIC} @ {DEVICE_RATE} Hz -> {SEND_RATE} Hz")
    print(f"speaker {SPEAKER} @ {TTS_RATE} Hz -> {PLAYBACK_RATE} Hz, {channels} ch")

    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
        turn = Turn()
        player = Player(SPEAKER, channels)
        speaker = Speaker(tts, voice_state, player)

        with sd.InputStream(
            device=MIC,
            samplerate=DEVICE_RATE,
            blocksize=BLOCK,
            dtype="float32",
            channels=1,
            callback=callback,
            latency="high",
        ), player.stream:
            pump = asyncio.create_task(capture_pump(session, turn))
            recv = asyncio.create_task(receive_loop(session, turn, speaker, player))
            try:
                await console(session, turn)
            finally:
                for task in (pump, recv):
                    task.cancel()
                await asyncio.gather(pump, recv, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print()
