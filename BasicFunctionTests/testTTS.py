"""
pocket-tts standalone test: speak TEXT out loud, optionally in a cloned voice.

Independent of the rest of the project -- no wake word, no Gemini. This exists to
answer two questions before pocket-tts is wired into anything:

  1. Does it sound right, in the voice we want?
  2. Is it fast enough on this Pi to speak a reply without a long pause?

Question 2 is the one that decides the design, so generation is timed and the
real-time factor is printed. RTF < 1 means the model produces audio faster than
it plays, which is what streaming playback needs.

First run downloads model weights from HuggingFace (nothing is cached yet).
"""

import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly
from pocket_tts import TTSModel
from pocket_tts.default_parameters import (
    DEFAULT_FRAMES_AFTER_EOS,
    MAX_TOKEN_PER_CHUNK,
    get_default_voice_for_language,
)

# --- config ---------------------------------------------------------------

TEXT = (
    "Good evening. The lights are at forty percent, "
    "and it is nineteen degrees in the room."
)

# Voice cloning. Point this at your own recording to clone it; set it to None to
# use a built-in voice instead. Any format soundfile can read works (.wav, .mp3)
# -- it is resampled to 24 kHz and mixed to mono automatically, so the file does
# not have to match the mic's format.
#
# For a good clone: roughly 10-30 seconds of clean, continuous speech, one
# speaker, no music or background noise. Anything past 30 s is ignored.
#
# Cloning needs the gated weights -- see the setup note in load_voice() if this
# errors. Built-in voices work without any of that.
VOICE_WAV = None                 # e.g. Path("voices/johnny.wav")

# Used only when VOICE_WAV is None. Built-ins include alba, jean, anna, charles,
# paul, george, mary, michael, eve.
BUILTIN_VOICE = "alba"

LANGUAGE = "english"

# int8 quantization. Measured on this Pi (4 threads, same sentence):
#   fp32  RTF 3.19x   int8  RTF 1.61x
# Roughly half the generation time for some loss of fidelity. fp32 is too slow
# to be usable here, so this defaults on -- flip it to False to hear the
# difference and judge the quality cost yourself.
QUANTIZE = True

SPEAKER = "USB Composite Device"

# pocket-tts emits 24 kHz, but this speaker accepts 48 kHz and nothing else
# (probed: 8k/16k/22.05k/24k/32k/44.1k all rejected). Addressing it by name maps
# to hw:0,0 and bypasses ALSA's plug layer, so there is no automatic resampling
# to fall back on -- 24 kHz raises paInvalidSampleRate. Upsample before playing.
PLAYBACK_RATE = 48000

# Playback volume, 0.0-1.0. This speaker exposes no hardware volume control --
# card 0 has only a mute switch ('PCM Playback Switch'), so alsamixer cannot
# turn it down and the level has to be scaled in software. Roughly logarithmic
# to the ear: 0.5 is about -6 dB, 0.25 about -12 dB.
# Worth keeping low for more than comfort: clipping is non-linear and defeats
# AEC, so a quieter speaker buys echo suppression that no tuning can.
VOLUME = 0.5
SAVE_TO = "tts_output.wav"       # None to skip writing a file

# --- voice ----------------------------------------------------------------


def describe_wav(path):
    """Report the conditioning clip, since clone quality depends on it."""
    info = sf.info(str(path))
    print(
        f"voice file : {path}  "
        f"{info.duration:.1f}s  {info.samplerate} Hz  {info.channels} ch"
    )
    if info.duration < 5:
        print("  warning: under 5 s of audio usually gives a weak clone")
    elif info.duration > 30:
        print("  note: only the first 30 s is used")


def load_voice(model):
    """Resolve the configured voice into a conditioned model state."""
    if VOICE_WAV is None:
        voice = get_default_voice_for_language(LANGUAGE) or BUILTIN_VOICE
        print(f"voice      : built-in '{voice}'")
        # Built-ins ship as precomputed states, so this is a load, not an encode.
        return model.get_state_for_audio_prompt(voice)

    path = Path(VOICE_WAV)
    if not path.exists():
        raise SystemExit(f"VOICE_WAV does not exist: {path}")

    if not model.has_voice_cloning:
        # The cloning weights live in a gated repo. Without access, load_model
        # quietly falls back to the no-cloning build and the failure only
        # surfaces deep inside the encoder, so check it here instead.
        raise SystemExit(
            "This model build cannot clone voices -- the gated weights were not "
            "downloaded.\n"
            "  1. Accept the terms at https://huggingface.co/kyutai/pocket-tts\n"
            "  2. Log in locally:  .venv/bin/hf auth login\n"
            "Then re-run. Built-in voices work meanwhile: set VOICE_WAV = None."
        )

    describe_wav(path)
    t0 = time.perf_counter()
    # truncate=True caps conditioning at 30 s -- the same guard the pocket-tts
    # server applies to uploaded files, and what keeps a long recording from
    # eating memory.
    state = model.get_state_for_audio_prompt(path, truncate=True)
    print(f"  encoded voice in {time.perf_counter() - t0:.1f}s")
    # Encoding is repeated on every run. To skip it later:
    #   pocket-tts export-voice <wav> voice.safetensors
    # then point VOICE_WAV at the .safetensors instead.
    return state


# --- playback -------------------------------------------------------------


def play(samples, rate):
    """Play mono float32 through the USB speaker, matching its rate and channels."""
    if rate != PLAYBACK_RATE:
        samples = resample_poly(samples, PLAYBACK_RATE, rate)

    samples = np.clip(samples * VOLUME, -1.0, 1.0)

    channels = sd.query_devices(SPEAKER, "output")["max_output_channels"]
    audio = np.repeat(samples[:, None], min(channels, 2), axis=1)
    sd.play(audio, samplerate=PLAYBACK_RATE, device=SPEAKER)
    sd.wait()


# --- run ------------------------------------------------------------------


def main():
    print(f"text       : {TEXT!r}\n")

    t0 = time.perf_counter()
    model = TTSModel.load_model(language=LANGUAGE, quantize=QUANTIZE)
    model.to("cpu")
    print(f"model loaded in {time.perf_counter() - t0:.1f}s"
          f"{' (int8)' if QUANTIZE else ' (fp32)'}")

    rate = model.config.mimi.sample_rate
    state = load_voice(model)

    print("\ngenerating...")
    t0 = time.perf_counter()
    first_chunk_at = None
    chunks = []

    for chunk in model.generate_audio_stream(
        model_state=state,
        text_to_generate=TEXT,
        max_tokens=MAX_TOKEN_PER_CHUNK,
        frames_after_eos=DEFAULT_FRAMES_AFTER_EOS,
    ):
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter() - t0
        chunks.append(chunk.squeeze().cpu().numpy())

    elapsed = time.perf_counter() - t0
    if not chunks:
        raise SystemExit("no audio generated -- check TEXT is not empty")

    samples = np.concatenate(chunks).astype(np.float32)
    duration = len(samples) / rate

    print(f"  first chunk after {first_chunk_at:.2f}s")
    print(f"  generated {duration:.1f}s of audio in {elapsed:.1f}s")
    rtf = elapsed / duration
    if rtf < 1:
        verdict = "faster than realtime, streaming playback is viable"
    elif QUANTIZE:
        verdict = "slower than realtime; shorter replies are the remaining lever"
    else:
        verdict = "slower than realtime -- try QUANTIZE = True"
    print(f"  RTF {rtf:.2f}x -- {verdict}")

    if SAVE_TO:
        sf.write(SAVE_TO, samples, rate)
        print(f"  wrote {SAVE_TO}")

    print(f"\nplaying through {SPEAKER} at {PLAYBACK_RATE} Hz...")
    play(samples, rate)
    print("done")


if __name__ == "__main__":
    main()
