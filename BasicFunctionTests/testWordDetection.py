"""
Loopback + wake word detection test.

Audio callback stays lightweight: copy in, copy out, queue a frame.
All inference happens on a worker thread so the realtime deadline is never at risk.

The detector holds a rolling ~2 s window. livekit-wakeword's WakeWordModel is
stateless: it mels the whole chunk it is handed, slides a 76-frame window at
stride 8, and needs >= 16 embeddings before any classifier runs. Feed it less and
it returns a hardcoded 0.0 without touching the model. See
livekit/wakeword/inference/model.py:95.
"""

import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from livekit.wakeword import WakeWordModel

# --- config ---------------------------------------------------------------

MIC = "USB PnP Sound Device"
SPEAKER = "USB Composite Device"

# The PnP mic is 48 kHz only, so capture at 48k and decimate for the detector.
DEVICE_RATE = 48000
MODEL_RATE = 16000

BLOCK = int(DEVICE_RATE * 0.08)   # 80 ms blocks

# Rolling window fed to predict(). 2.0 s yields exactly 16 embeddings -- the bare
# minimum -- so a window even slightly short silently scores 0.0. 2.2 s gives 18,
# a real margin.
WINDOW_SECONDS = 2.2
WINDOW_SAMPLES = int(DEVICE_RATE * WINDOW_SECONDS)

# Blocks between predictions. predict() measures ~70-95 ms on this Pi, so a
# 1-block (80 ms) hop would saturate a core and back the queue up permanently.
# 2 blocks = predict every 160 ms at roughly half duty.
HOP_BLOCKS = 2
HOP_SECONDS = HOP_BLOCKS * BLOCK / DEVICE_RATE

THRESHOLD = 0.2
CONSECUTIVE = 2                    # predictions above threshold required to fire
REFRACTORY = 2.0                   # seconds to ignore after a detection

MODEL_PATH = Path("wakeup_models/johnny.onnx")
MODEL_NAME = MODEL_PATH.stem       # scores are keyed by the file stem, not a chosen name

# --- setup ----------------------------------------------------------------

model = WakeWordModel(models=[MODEL_PATH])

audio_q = queue.Queue(maxsize=50)
overflow_flag = threading.Event()
stop_flag = threading.Event()
drops = 0                          # frames discarded by the callback


def callback(indata, outdata, frames, time_info, status):
    global drops

    if status:
        overflow_flag.set()          # never print from the audio thread

    outdata[:] = np.repeat(indata, 2, axis=1)

    try:
        audio_q.put_nowait(indata[:, 0].copy())
    except queue.Full:
        drops += 1                   # drop rather than block the callback


def detector():
    # Ring buffer at the device rate. The whole window is decimated at once per
    # prediction rather than per 80 ms block, so resampler edge effects don't get
    # spliced into the window every block.
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    primed = 0

    hits = 0
    cooldown = 0
    since_predict = 0

    while not stop_flag.is_set():
        try:
            frame = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue

        n = len(frame)
        window[:-n] = window[n:]
        window[-n:] = frame
        primed = min(primed + n, WINDOW_SAMPLES)

        since_predict += 1
        if since_predict < HOP_BLOCKS or primed < WINDOW_SAMPLES:
            continue
        since_predict = 0

        if DEVICE_RATE != MODEL_RATE:
            chunk = resample_poly(window, MODEL_RATE, DEVICE_RATE)
        else:
            chunk = window

        # Convert to int16 explicitly. float32 is accepted but the expected
        # scaling is ambiguous; int16 removes the guesswork.
        pcm = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)

        t0 = time.perf_counter()
        scores = model.predict(pcm)
        elapsed = (time.perf_counter() - t0) * 1000
        score = scores[MODEL_NAME]

        if cooldown > 0:
            cooldown -= 1
            continue

        print(f"score {score:.3f}  ({elapsed:.0f} ms)")   # comment out once tuning is done

        if score > THRESHOLD:
            hits += 1
            if hits >= CONSECUTIVE:
                print(f"*** DETECTED ({score:.2f}) ***")
                hits = 0
                cooldown = int(REFRACTORY / HOP_SECONDS)
        else:
            hits = 0


# --- run ------------------------------------------------------------------

print(sd.query_devices())
print(
    f"model '{MODEL_NAME}'  window {WINDOW_SECONDS}s  "
    f"predict every {HOP_SECONDS * 1000:.0f}ms"
)

worker = threading.Thread(target=detector, daemon=True)
worker.start()

try:
    with sd.Stream(
        device=(MIC, SPEAKER),
        samplerate=DEVICE_RATE,
        blocksize=BLOCK,
        dtype="float32",
        channels=(1, 2),
        callback=callback,
        latency="high",
    ):
        last_drops = 0
        while True:
            sd.sleep(1000)
            if overflow_flag.is_set():
                print("xrun (overflow/underflow)")
                overflow_flag.clear()
            if drops != last_drops:
                # A stateful window fed spliced audio scores garbage; this number
                # must stay at zero.
                print(f"dropped {drops - last_drops} frames (total {drops})")
                last_drops = drops
except KeyboardInterrupt:
    pass
finally:
    stop_flag.set()
