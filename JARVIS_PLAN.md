# Dormy — Jarvis Dorm Assistant

Implementation plan. Written to be handed to a Claude Code session as project context.

---

## 1. What this is

A always-listening voice assistant running on a Raspberry Pi in a college dorm room.
Wake word activation, conversational AI via Google Gemini's Live API, and control of
ESP32-driven devices around the room (LED strips first, more later).

**Design priority: understanding over speed.** The owner is learning this stack
deliberately. Explain architectural reasoning before implementation. Prefer showing
why a design is correct over producing working code quickly.

---

## 2. Hardware

| Component | Detail |
|---|---|
| Compute | Raspberry Pi, Python 3.13, venv at `~/Dormy/.venv` |
| Mic | USB PnP Sound Device — `hw:1,0`, 1 in / 0 out, **48 kHz only, rejects 16 kHz** |
| Speaker | USB Composite Device — `hw:0,0`, 1 in / 2 out |
| Room devices | ESP32 microcontrollers, WS2812B LED strips |

### Open hardware question (resolve early — high payoff)

The speaker device reports **1 input channel**. If that is a real capture device and
not a monitor loopback, use it for both capture and playback.

Why this matters enormously: two separate USB audio devices have two independent
crystals. They drift apart over minutes, which causes periodic buffer
overflow/underflow and progressively destroys AEC filter alignment. A single device
shares one clock and eliminates both problems permanently.

Test: open a duplex stream with both endpoints set to `USB Composite Device`,
`channels=(1, 2)`, and check whether loopback audio is audible. If yes, retire the
PnP mic.

---

## 3. Current state

### Working
- Duplex audio loopback via `sounddevice.Stream` (mic → speaker passthrough, verified good)
- Acoustic echo cancellation (reported working)

### Blocked — this is the active task
Wake word detection returns **exactly 0.000** on every frame using
`livekit-wakeword` with openWakeWord's pretrained `hey_jarvis.onnx`.

Constant zero (as opposed to small fluctuating values) indicates a discrete bug,
not a tuning problem. Diagnostic ladder, in order:

1. **Swap to LiveKit's own `hey_livekit.onnx`** and say "hey LiveKit." Nonzero scores
   mean the audio path is fine and `hey_jarvis.onnx` is not truly compatible with
   LiveKit's runtime. LiveKit replaced openWakeWord's flat DNN head with a
   conv-attention architecture; they claim backward compatibility, but a classifier
   trained on subtly different mel/embedding features will saturate to zero through
   its sigmoid. **This is the leading hypothesis.**
2. **Offline replay.** `arecord -D hw:1,0 -f S16_LE -r 48000 -c 1 test.wav`, then loop
   `predict()` over decimated 1280-sample chunks. Removes the callback, queue,
   threading, and frame drops from the equation in one step.
3. **Inspect model output shape:** `model.predict(np.zeros(1280, dtype=np.int16))`
   at startup — confirm dict keys and that values are scalars.
4. **Warm-up.** The embedding stage accumulates a sliding window and needs ~1.5 s of
   audio before output is meaningful. Let it run 30 s before judging.
5. **Frame drops.** The callback discards on `queue.Full`. Count and log discards —
   a stateful sliding-window model fed discontinuous audio produces garbage.

If `hey_livekit.onnx` works and `hey_jarvis.onnx` does not, the fix is to train a
custom "hey jarvis" model with livekit-wakeword's own pipeline
(`pip install livekit-wakeword[train,eval,export]`), not to keep debugging the
openWakeWord artifact.

Fallback if livekit-wakeword stays unreliable: **Picovoice Porcupine**. Type the
phrase into a console, get a tuned model back. Wake word detection is one component
of a much larger project and is not worth unbounded time.

---

## 4. Architecture

```
USB mic (48 kHz)
    ↓
sd.Stream callback  ──────────────────────►  outdata (playback)
    │  (realtime thread — copy only, no work)
    ↓
AEC  (mic frame − speaker reference)
    ↓
ring buffer (last ~2 s, always retained)
    ↓
queue  ──►  worker thread
                ↓
            decimate 48k → 16k
                ↓
            VAD gate (Silero)
                ↓
            wake word model.predict()
                ↓  threshold crossed
            flush ring buffer → Gemini Live socket
                ↓
            stream live 16 kHz PCM ⇄ receive 24 kHz PCM
                ↓
            function calls  ──►  MQTT / ESP-NOW  ──►  ESP32 nodes
```

### Non-negotiable rules

- **The audio callback does no work.** Copy in, copy out, enqueue. All inference,
  resampling, and network I/O happens on worker threads. ONNX inference in the
  callback blows the realtime deadline every time.
- **Never print from the audio callback.** Set a flag; print elsewhere. Printing does
  I/O under a lock and is self-reinforcing — one xrun triggers a print that causes
  the next xrun.
- **The wake word detector reads AEC output, never the raw mic.** Otherwise Jarvis's
  own speech triggers Jarvis. This was the root cause of an earlier failure:
  `WakeWordListener` opens its own PortAudio stream on the default device, silently
  bypassing the entire cleaned pipeline. Use `WakeWordModel.predict()` directly.
- **Explicit int16 conversion before `predict()`.** float32 is accepted but the
  expected scaling (±1.0 vs ±32768) is undocumented. Ambiguous scaling produces
  erratic scores without erroring.
- **Pre-roll ring buffer is mandatory.** By the time the wake word is recognized, the
  following words are already in the past. Flush ~1–2 s of retained audio into the
  Gemini socket before streaming live, or the first command arrives truncated.

---

## 5. Repo layout

```
Dormy/
├── jarvis/
│   ├── audio/
│   │   ├── stream.py       # sd.Stream lifecycle, callback, xrun accounting
│   │   ├── aec.py          # echo cancellation + ERLE measurement
│   │   ├── ring.py         # pre-roll buffer
│   │   └── playback.py     # 24 kHz output queue from Gemini
│   ├── wake/
│   │   ├── detector.py     # predict() loop, threshold, debounce, refractory
│   │   └── vad.py          # Silero gate
│   ├── gemini/
│   │   ├── session.py      # Live API websocket lifecycle
│   │   └── tools.py        # function declarations + dispatch
│   ├── devices/
│   │   ├── transport.py    # MQTT or ESP-NOW serial gateway
│   │   └── lights.py       # LED zone abstraction
│   ├── config.py           # single source of truth for rates, block sizes, devices
│   └── main.py             # state machine
├── firmware/               # ESP32 sketches
├── tests/
│   ├── positives/          # 50+ wake word recordings, varied conditions
│   └── negatives/          # 1 hour of normal room audio, no wake word
├── tools/
│   ├── replay.py           # offline predict() over a WAV
│   └── sweep.py            # threshold sweep → miss rate + false accepts/hour
└── JARVIS_PLAN.md
```

---

## 6. Phases

Each phase ends with something demonstrable. Do not start the next phase until the
acceptance criterion passes.

### Phase 1 — Audio foundation (mostly done)

Duplex stream, explicit `blocksize`, `latency="high"`, xrun counter surfaced outside
the callback. Config centralized so sample rate and block size are defined once.

**Accept:** 10 minutes of continuous loopback with zero xruns.

### Phase 2 — AEC verification (mostly done)

AEC in the pipeline between callback and queue.

Measure rather than judge by ear: play a known file with nobody speaking, compute
energy ratio between canceller input and output. That ratio is **ERLE**. Above ~20 dB
is workable for recognition. Track convergence time from cold start.

Test order: silence → speech with speaker silent (confirm voice isn't damaged) →
double-talk (both at once, the hard case).

Keep a **half-duplex mute flag** as a runtime option — mute the mic during playback.
If AEC misbehaves, one switch keeps everything working.

**Accept:** ERLE > 20 dB sustained, voice intact through the canceller.

### Phase 3 — Wake word (current blocker)

Work the diagnostic ladder in §3. Then:

- Threshold, N consecutive frames above threshold, refractory period after firing
- VAD gate in front so the detector isn't scored against silence
- Ring buffer wired

**Tuning is measured, not guessed.** Build the test sets in `tests/`. Sweep thresholds
across both sets. Positives give miss rate; the hour of negatives gives false accepts
per hour — that second number predicts whether the thing stays plugged in. Bias toward
missing occasionally rather than waking during a movie.

Record test sets **against the AEC-processed stream**, not the raw mic.

**Accept:** < 1 false accept/hour on the negative set, > 90% detection on positives.

### Phase 4 — Gemini Live API

Stateful WebSocket. Input: raw 16-bit PCM, 16 kHz, mono, little-endian, base64 in
`realtimeInput.audio` with mimeType `audio/pcm;rate=16000`. Output: raw 16-bit PCM,
24 kHz, base64 in `serverContent.modelTurn.parts[].inlineData.data`. Chunks ~100 ms.

Use the `google-genai` Python SDK, not raw websockets — it handles framing,
reconnection, and session resumption. Start from the official examples repo.

Format is strict. Wrong format closes the socket with error 1007 and a message naming
`16khz s16le pcm, mono channel`. It does not degrade gracefully.

**Start with server-side VAD disabled** and send explicit turn boundaries, combined
with mic mute during playback. Server VAD interprets echo leakage as user interruption
and will make the model talk over itself. Enable it only once AEC is proven.

Turn on **input transcription** immediately — logging what Gemini heard versus what it
did removes most guesswork when a command misfires.

Model names in this family rotate frequently. Pull the current one from the docs; do
not copy from a tutorial.

Playback: `sd.RawOutputStream` at 24 kHz with a callback draining a queue, so audio
starts before the response finishes generating.

Play a short chime the instant the wake word fires, before contacting Gemini. Preload
it into memory. Costs nothing and hides network latency.

**Accept:** say the wake word, ask a question, hear a spoken answer.

### Phase 5 — First ESP32, no AI

Single ESP32, WS2812B strip, controlled by publishing messages from a laptop. Prove
the hardware path while the variable count is low.

**Power warning:** WS2812B draws up to 60 mA each at full white. 300 LEDs is 18 A worst
case. Dedicated 5 V supply, power injection at both ends on long runs, never from an
ESP32 pin.

**Accept:** a message from the Pi changes the strip color.

### Phase 6 — Bridge

Gemini **function calling** is the joint, and it is supported natively in the Live API —
same socket, nothing separate to build. Declare functions like
`set_light_color(zone, color, brightness)`; the model returns a structured call; the
handler dispatches to the transport.

Track device state so Jarvis knows the lights are already on.

**Accept:** "Jarvis, make the lights blue" works end to end.

### Phase 7 — Expand

Additional ESP32 nodes (desk lamp relay, temperature sensor, door sensor). Persistent
memory across sessions. Scenes and multi-device commands.

### Phase 8 — Make it live there

systemd unit with restart-on-failure, watchdog for hung API calls, WiFi reconnect
logic, enclosure. The difference between a demo and something used daily is whether it
survives a power cycle without SSH.

---

## 7. The networking risk

**Campus WiFi will likely break MQTT.** University networks run WPA2-Enterprise (painful
but possible on ESP32) and typically enable **client isolation** — the Pi and the ESP32s
sit on the same network and cannot see each other.

Two routes, decide before buying more ESP32s:

- **ESP-NOW.** ESP32s talk directly, no access point involved. One ESP32 wired to the Pi
  over serial acts as gateway. Cheap, robust, sidesteps campus IT entirely.
  **Recommended.**
- **Private router.** Travel router creating a private subnet. Check housing policy —
  many schools prohibit personal APs.

Also worth confirming early: dorm rules on soldering, and whether a 5 V supply of the
size the LEDs need raises fire-code concerns with an RA.

---

## 8. Accumulated gotchas

Things already learned the hard way in this project. Do not re-derive them.

- The Pi has **no ADC**. Analog mic breakouts (VDD/GND/OUT, e.g. MAX4466) cannot work
  directly. Those are still useful on an ESP32 for sound-reactive LED effects.
- **Named ALSA devices map to `hw:`**, bypassing the plug layer — no automatic
  resampling. Requesting an unsupported rate throws `paInvalidSampleRate`. Routing via
  `pulse` or `default` resamples transparently.
- **PulseAudio `module-echo-cancel` requires going through PulseAudio.** Opening `hw:`
  directly means it never sees the streams and silently does nothing.
- **Clock drift between two USB audio devices** is unfixable by tuning. Only
  resampling, a drift compensator, or a single shared device eliminates it.
- **Clipping defeats AEC.** Adaptive filters are linear; clipping is not. Lowering
  playback volume can buy more echo suppression than any parameter tuning.
- **Short wake phrases are fundamentally harder.** Three or more syllables with
  distinctive phonemes outperforms tuning. "hey jarvis" > "jarvis".
- **Library wrappers can silently isolate pipelines.** `WakeWordListener` looked like a
  convenience and was actually a second, unclean audio path. Understand what a wrapper
  does before integrating it.
- **openWakeWord's training pipeline has broken dependencies** (pinned to 2022-era
  torch/TF). Inference still works. Do not attempt to fix the training pipeline.

---

## 9. Working agreement for Claude Code

- Lead with the architectural reasoning, then the code.
- Change one variable at a time. Do not debug two layers simultaneously.
- Prefer a measurable number (ERLE, false accepts/hour, xrun count) over "sounds better."
- When a component is a time sink, name the escape hatch rather than grinding.
- Keep the half-duplex fallback and the offline replay tool working — they are the
  diagnostic floor when something regresses.
