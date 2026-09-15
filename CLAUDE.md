# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dormy is an always-listening voice assistant for a Raspberry Pi dorm room: wake word →
Gemini Live API conversation → function calls that drive ESP32/WS2812B devices.

**`JARVIS_PLAN.md` is the authoritative design document.** Read it before non-trivial work.
It carries the phase plan, the non-negotiable audio rules, and §8 "Accumulated gotchas" —
hardware and library traps already paid for once. Do not re-derive them.

Its §9 working agreement governs how to respond here: lead with architectural reasoning
before code, change one variable at a time, and prefer a measured number (ERLE, false
accepts/hour, xrun count) over "sounds better." The owner is learning this stack
deliberately — understanding is the priority, not speed.

## Commands

Everything runs in the venv at `.venv` (Python 3.13). There is no build, no linter, and no
test framework configured — the `test*.py` files at the root are manual hardware probes run
one at a time, not an automated suite.

```bash
# Probes live in BasicFunctionTests/, integration tests in IntegrationTests/.
# Run them from the repo root.
.venv/bin/python BasicFunctionTests/testAEC.py            # ERLE A/B, raw vs canceller
.venv/bin/python BasicFunctionTests/testAEC.py --live     # double-talk meter
.venv/bin/python BasicFunctionTests/test.py               # record 5 s, play it back
.venv/bin/python BasicFunctionTests/testWordDetection.py  # loopback + wake word worker
GEMINI_API_KEY=... .venv/bin/python BasicFunctionTests/testGeminiLiveAPI.py
.venv/bin/python BasicFunctionTests/testTTS.py            # pocket-tts, optionally cloned voice

# The one that ties it together: wake word -> conversation -> idle, no keyboard.
GEMINI_API_KEY=... .venv/bin/python IntegrationTests/testChatWakeup.py

.venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
arecord -D hw:1,0 -f S16_LE -r 48000 -c 1 out.wav   # capture for offline replay

# Spotify Connect endpoint (user service, not the packaged raspotify one)
systemctl --user status librespot
pactl list sink-inputs | grep -i sink:                # must say jarvis_aec_sink
.venv/bin/python Spotify.py                           # auth + device diagnostic
.venv/bin/python Spotify.py login                     # one-time OAuth, headless
```

`testGeminiLiveAPI.py` needs `GEMINI_API_KEY` in the environment (aistudio.google.com/apikey);
it exits with instructions if unset. `testTTS.py` downloads model weights from
HuggingFace on first run (nothing was cached).

## Hardware layout

Two separate USB audio devices, addressed **by name** in code (PortAudio matches substrings):

| Role | Name | ALSA | Channels |
|---|---|---|---|
| Mic | `USB PnP Sound Device` | `hw:1,0` | 1 in, 0 out — **48 kHz only, rejects 16 kHz** |
| Speaker | `USB Composite Device` | `hw:0,0` | 1 in, 2 out — **48 kHz only** (probed) |

Hence the standing shape of every stream: `sd.Stream(device=(MIC, SPEAKER), channels=(1, 2))`
at 48 kHz, with `np.repeat(indata, 2, axis=1)` to fan mono mic into stereo out, and
decimation to 16 kHz on a worker thread for the models.

**Both devices are 48 kHz only.** The speaker rejects 8k/16k/22.05k/24k/32k/44.1k — addressing
it by name maps to `hw:0,0` and bypasses ALSA's plug layer, so nothing resamples for you and an
unsupported rate raises `paInvalidSampleRate`. This contradicts `JARVIS_PLAN.md:236`, which
specifies `sd.RawOutputStream` at 24 kHz for Gemini playback: Gemini's 24 kHz output and
pocket-tts's 24 kHz output must both be upsampled to 48 kHz before they reach this speaker.

**Check the mic's capture gain before debugging sensitivity.** It was found sitting at
**0 of 16 — the floor** (`amixer -c 1 sget Mic`), and the only symptom was having to talk
with your nose against the mic: wake word scores, transcripts and xrun counts all looked
normal. PulseAudio owns that control on this source (`HW_VOLUME_CTRL`), so its stored 0%
writes straight through to the value the `hw:1,0` path reads — addressing the card
directly does *not* bypass it. Set it through Pulse so the value sticks
(`module-device-restore` persists it across reboots); setting it with `amixer` alone can
be clobbered the next time Pulse touches the source:

```bash
pactl set-source-volume alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-mono 95%
amixer -c 1 sget Mic          # expect: Capture 15 [94%] [22.32dB]
```

Measured, ambient room noise: step 0 → **-67.9 dBFS** RMS, step 15 → **-44.5 dBFS**, so the
control is worth **+23.4 dB** end to end, leaving 28.8 dB of headroom on ambient peaks.
It costs the wake word nothing — 12 s of ambient at the raised gain scored mean 0.0029 /
max 0.0064, still 23x below the 0.15 threshold and matching the ~0.003 recorded above.
`testChatWakeup.py` prints the gain at startup and warns under 50%.

`Auto Gain Control` (the card's other control) is **on**, and was left on deliberately —
changing it is a separate variable. It is a plausible thing to turn *off* later: AGC pumps
the gain around, which makes wake word scores less repeatable and breaks the linear echo
path that any future AEC depends on.

Two devices means two crystals and unavoidable clock drift, which degrades AEC alignment
over minutes. `JARVIS_PLAN.md` §2 has an open question worth resolving early: the speaker
reports 1 input channel, so it may work as a single shared-clock duplex device.

## Audio pipeline invariants

These are load-bearing; violating them produces failures that look like something else.

- **The `sd.Stream` callback does no work.** Copy in, copy out, `put_nowait` onto a queue,
  drop on `queue.Full`. Inference or resampling in the callback misses the realtime deadline.
- **Never print from the callback.** Set a `threading.Event` and print from the main loop.
  Printing does I/O under a lock, so one xrun causes the next.
- **The detector must read AEC output, never the raw mic** — otherwise Jarvis wakes itself.
  Never use `livekit.wakeword.WakeWordListener`: it opens its own PortAudio stream on the
  *default* device, silently bypassing the whole cleaned pipeline. Use `WakeWordModel.predict()`.
- A pre-roll ring buffer (~1–2 s) is mandatory before the Gemini socket, or the first command
  arrives truncated — by the time the wake word scores, the following words are already past.

## `WakeWordModel` contract (source of the current blocker)

Read `.venv/lib/python3.13/site-packages/livekit/wakeword/inference/model.py` — the docstring
is the real spec:

- **The model is stateless and needs ~2 s of 16 kHz audio per call.** It mels the whole chunk,
  slides a 76-frame window at stride 8, and needs ≥16 embeddings. Anything shorter hits an
  early `return {name: 0.0 ...}`.
- Scores are keyed by **the model file's stem**, not a name you choose.
- int16 and float32 are both accepted; int16 is divided by 32768.0 internally.

`testWordDetection.py` previously violated all three and could not run at all. Fixed
2026-09-06 — the history is worth keeping because the symptom was misleading:

1. `MODEL_PATH` pointed at `hey_livekit.onnx`, which exists nowhere on this machine, so
   construction raised `FileNotFoundError`. Now `wakeup_models/hey_jarvis.onnx`.
2. `MODEL_NAME` was a chosen string (`"johnny"`) rather than the file stem, so the score
   lookup would `KeyError`. Now derived via `MODEL_PATH.stem`.
3. It fed one 80 ms block per call. **This, not model incompatibility, was the cause of the
   "exactly 0.000 on every frame" symptom in `JARVIS_PLAN.md` §3** — measured, 80 ms yields
   5 mel frames and 0 embeddings, so `predict()` returned a hardcoded zero before any
   classifier ran. The plan's leading hypothesis (openWakeWord/LiveKit feature mismatch) was
   not the problem, and §3's ladder can be skipped. It now holds a rolling 2.2 s window.

Measured on this Pi, worth knowing before changing the window or hop:

- 2.0 s yields **exactly 16** embeddings, the bare minimum — a window even slightly short
  silently returns 0.0. 2.2 s yields 18. Do not trim the window to save CPU.
- `predict()` costs **~70-95 ms**. The hop is therefore 2 blocks (160 ms), about half duty on
  one core; a per-block 80 ms hop would saturate the core, back up the queue, and cause the
  frame drops that corrupt a rolling window.
- The window is decimated 48k->16k **whole, once per prediction**, not per 80 ms block, so
  resampler edge artifacts aren't spliced in every block.
- The detector warms up for one full window (~2.2 s, 27 blocks) before its first score.
- Broadband noise scores ~0.003. Small fluctuating values mean the classifier is running;
  a constant 0.000 means it is short-circuiting again.
- **Working threshold is 0.1-0.2** (owner-confirmed on-device; the file is set to 0.2), not the 0.5 that
  reads as a natural default. Phase 3's accept criteria still need the `tests/` sets swept.

## Gemini Live API (`testGeminiLiveAPI.py`)

Phase 4 standing alone — no wake word, no ring buffer. Push-to-talk: Enter opens a turn, Enter
closes it. Gemini speaks the reply itself.

- **The Live API cannot return text.** Every current Live model is native-audio and supports
  only the AUDIO modality; `response_modalities=["TEXT"]` is rejected with
  `1007 ... combination of response modalities (TEXT) is not supported`. The half-cascade models
  that allowed TEXT (`gemini-2.0-flash-live-001`, `gemini-live-2.5-flash-preview`) are no longer
  listed. **A text-out + local-TTS design is not available on this API** — the documented
  workaround is output audio transcription, which is what this file logs. Do not "fix" this by
  setting TEXT again.
- **`MODEL = "gemini-3.1-flash-live-preview"`**, read off the docs 2026-09-06. `JARVIS_PLAN.md:230`
  is right that these rotate — the 2.x live IDs are gone. Re-check; never copy from a tutorial.
- **Playback is 24 kHz → 48 kHz.** The API returns 24 kHz PCM and the speaker accepts only
  48 kHz, so `JARVIS_PLAN.md:236`'s "`sd.RawOutputStream` at 24 kHz" cannot work as written.
  `Player` runs at 48 kHz and `Upsampler` does a stateful 2×.
- Both **`Decimator` (48k→16k up) and `Upsampler` (24k→48k down) carry `lfilter` state across
  blocks.** Per-chunk resampling would zero-pad the filter edges and click at every seam, and
  Gemini sends many small chunks per reply. Verified on hardware: seam steps match the typical
  sample step, level preserved, zero underruns.
- `Player`'s callback copies piecewise into a preallocated scratch buffer and broadcasts mono
  across channels, so it allocates nothing — the plan's "callback does no work" rule.
- **Server VAD is disabled** with explicit `ActivityStart`/`ActivityEnd`. Push-to-talk also means
  the mic never sends while the speaker plays, so this file has no echo path and needs no AEC.
  That changes once the wake word drives it.
- `session.receive()` **terminates at `turn_complete`** (verified in `live.AsyncSession.receive`),
  so it must sit inside an outer `while True` — one pass per model turn, not per session.
- Input is strict: 16-bit PCM, 16 kHz, mono, LE. Wrong format closes the socket with 1007.
- **The voice is selectable, but not clonable.** `speech_config.voice_config.
  prebuilt_voice_config.voice_name` works on the Live API and picks one of **30** prebuilt
  voices (Sulafat warm, Charon informative, Achird friendly, Vindemiatrix gentle, Schedar
  even, Kore firm, Puck upbeat...) at **zero latency cost** -- Gemini was synthesizing
  anyway. `types.ReplicatedVoiceConfig` (voice sample + consent audio + consent signature)
  exists in `google/genai/types.py:5475`, but neither the Live guide nor the
  speech-generation docs list it and custom voice is still an open feature request; treat
  it as Vertex/Cloud-TTS surface that leaked into the shared type tree. Probe script:
  `probe_replicated_voice.py` (scratchpad) -- run it before assuming either way.
  Do **not** also set `speech_config.language_code`: native-audio models choose the
  language themselves and reject an explicit code.
- The file now prints `first audio from Gemini` / `FIRST SOUND` / `finished speaking` from
  the same t0 (`activity_end`) as `testGeminiLiveAPIwithTTS.py`, so the two routes can be
  compared in one table.

## pocket-tts (`testTTS.py`)

Standalone: speaks `TEXT` through the USB speaker, in a built-in voice or one cloned from
`VOICE_WAV`. Verified end to end on-device.

- **Output is 24 kHz mono** (`model.config.mimi.sample_rate`); `play()` upsamples to 48 kHz
  because the speaker accepts nothing else.
- **pocket-tts is not on the reply path.** Phase 4 plays Gemini's own voice; this file is now a
  standalone TTS bench (and the route to a cloned voice for other uses).
- **Measured RTF on this Pi**, same sentence, 4 threads: **fp32 3.19x, int8 1.61x**. The
  `english_2026-01` variant measures identically (1.61x) and `sampler_decode_steps` is already
  at its minimum of 1. **RTF ~1.6 is the floor here, and this was re-tested properly on
  2026-09-06** — a full ladder moved it almost not at all:

  | Rung | RTF | Verdict |
  |---|---|---|
  | baseline, 4 threads, `torch.ao`/qnnpack | 1.63–1.65 | — |
  | **torchao backend** | **1.58–1.59** | ~4%, the only real gain |
  | 3 threads / 2 threads | 1.64 / 1.61–1.65 | **no effect** |
  | `frames_after_eos=1` | 1.59–1.63 | noise |
  | `performance` governor | not run | CPU already at 2.4 GHz max under load |

  Two things worth not re-deriving. **Thread count does nothing**: generation is
  autoregressive and per-token latency-bound, not throughput-bound, so 2 threads matches 4 —
  which means two cores are free for AEC and the wake word at no TTS cost. And
  `copy.deepcopy(model_state)`, done per chunk in `_generate_audio_stream_short_text`,
  measures **5.4 ms** — noise against seconds of generation, not worth attacking.
  On torchao: `pocket_tts/quantization.py:_get_backend()` returns `"torchao"` whenever the
  package is importable, because it guards on a `_SKIPPED_CPP_EXTENSIONS` attribute torchao
  never sets. The aarch64 wheel is `py3-none-any` with **no `_C*.so`**, so this is the ATen
  path, not the "optimized C++ kernels" the docstring promises. 4% is all it buys.
- Because RTF cannot be moved, **hiding** the latency is the only remaining lever.
  Per-sentence *buffered* synthesis is that lever — see
  `testGeminiLiveAPIwithTTS.py` for the measured tradeoff. It hides latency rather than reducing
  it, and only works if each sentence is fully generated before it is queued. Both are
  slower than realtime, so streaming playback would underrun and a reply cannot start speaking
  before it is fully generated. `QUANTIZE = True` is the default for that reason — it roughly
  halves generation time for some loss of fidelity. Time to first chunk is ~0.5-0.8 s.
  **This is a real constraint on Phase 4's feel**: a 5 s reply costs ~7 s to synthesize, so
  keep replies short (the system instruction asks for one or two sentences) and expect to want
  the chime from `JARVIS_PLAN.md:236` covering the gap.
- **Voice cloning needs gated weights.** `kyutai/pocket-tts` requires accepting terms on
  HuggingFace plus `.venv/bin/hf auth login`; without that, `load_model` silently falls back to
  the no-cloning build and `has_voice_cloning` is False. `load_voice()` checks that up front
  rather than letting it fail inside the encoder. Built-in voices need no auth.
- Conditioning audio is resampled and mixed to mono automatically, so the reference clip need
  not match the mic. `truncate=True` caps it at 30 s. Re-encoding the clip costs time on every
  run; `pocket-tts export-voice <wav> voice.safetensors` precomputes it and `VOICE_WAV` can
  point at the `.safetensors`.

## Live API + pocket-tts (`testGeminiLiveAPIwithTTS.py`)

Same push-to-talk premise as `testGeminiLiveAPI.py`, but Gemini's audio is **discarded** and the
reply is spoken by pocket-tts, driven by `output_audio_transcription` (the Live API cannot return
text). Exists to answer whether the pocket-tts delay is tolerable. Measured, 4.6 s reply:

| Approach | First sound | Finished | Gaps |
|---|---|---|---|
| chunks pushed as generated | 0.66 s | 8.25 s | **2.9 s torn through the middle of words** |
| per-sentence, buffered *(chosen)* | 2.78 s | 9.45 s | 2.0 s, at sentence boundaries |
| whole reply, buffered | 8.30 s | 13.38 s | none |

- **The first hand-off is the whole latency, and shortening it is the only lever that
  works.** Nothing plays until segment one is fully synthesized, so a four-second opening
  sentence costs ~5.7 s of silence. Breaking the *opening* segment at a clause boundary
  (`CLAUSE_END`, `MIN_CLAUSE_WORDS = 4`, applied only while `first_sentence_at is None`)
  measured **5.7 s -> 2.7 s**, more than every RTF lever combined. It is not free: playback
  starts earlier against an RTF > 1 pipeline, so projected gaps rise ~2.5 s -> ~3.4 s. That
  tradeoff is inherent — with RTF > 1 you cannot both start early and stay ahead.
- **Never push pocket-tts chunks as they arrive.** RTF ~1.6 means playback outruns synthesis and
  the player starves mid-word. `Speaker` generates a full sentence before queueing it, so gaps
  land where a pause belongs. This corrects an earlier note here: per-sentence chunking helps
  only *with* buffering — pocket-tts already streams chunks internally, so splitting alone buys
  nothing (0.66 s vs 0.54 s to first sound) and costs RTF (1.76x vs 1.65x).
- Per-sentence beats whole-reply on **both** first sound and total time, because synthesis of
  sentence N+1 overlaps playback of sentence N.
- `Player.starved` counts *any* shortfall once playback starts, not just partial fills — counting
  only partial fills misses total starvation, which is the case that matters.
- Budget ~2.8 s of local synthesis before the first word, plus Gemini's own round trip on top.
  Compare against `testGeminiLiveAPI.py`, where Gemini speaks and there is no synthesis wait.

## Wake word -> conversation (`IntegrationTests/testChatWakeup.py`)

The first file that joins two halves: `testWordDetection.py`'s detector and
`testGeminiLiveAPI.py`'s session. Idle listens for "johnny" at `THRESHOLD = 0.15`; a hit opens a
Live session, flushes a pre-roll ring into it, and converses hands-free until a goodbye or
`EXIT_DURATION = 3` s of user silence, then closes the socket and re-arms.

- **Server VAD is ON here**, the opposite of `testGeminiLiveAPI.py`, so the mic is live while the
  speaker plays. Echo history, because the order matters: it was measured as *not* a problem at
  the original mic gain, then became one the moment the capture gain went up +23.4 dB — the
  leakage rose by exactly the same +23.4 dB. **The fix was `module-echo-cancel`, not backing the
  gain off.** `USE_AEC = True` routes both directions through it; `interrupted` now means a real
  barge-in.
- **Routing goes through the ALSA `pulse` device, steered by `PULSE_SOURCE`/`PULSE_SINK`.**
  PortAudio cannot name a PulseAudio source or sink, so `resolve_devices()` sets those env vars
  (libpulse reads them when the stream opens, so it must happen before any stream is constructed)
  and returns `"pulse"` for both. **Both directions must go through it** — if playback went to the
  raw `hw:` speaker the canceller would never see the reference signal and would cancel nothing,
  silently. A missing module therefore **exits** rather than falling back to the raw mic.
- **The detector stops predicting while a session is open** (`listening` Event). `CLAUDE.md`'s rule
  is that the detector reads AEC output, never the raw mic; with no AEC the only honest version is
  to not score at all while Jarvis talks, or he wakes himself. It still drains its queue so the
  drop counter stays meaningful.
- **Re-arming zeroes the detector window.** After a conversation the rolling window still holds
  Jarvis's farewell bleeding through the mic, which scores immediately. Resetting forces a fresh
  `WINDOW_SECONDS` before the next score. `wake_event` is cleared at re-arm rather than after the
  wait, because one held utterance can fire twice (`REFRACTORY` is shorter than a long sound) and
  a late set would otherwise open the next session with no wake word.
- **`PRE_ROLL_SECONDS = 2.0`**, larger than the "1–2 s" above on purpose: detection cannot fire
  until the wake word sits inside a 2.2 s window, and `CONSECUTIVE = 2` adds a hop, so the word is
  ~2 s in the past when the socket opens. The uplink pump is one long-lived task that *always*
  decimates — idle included — so the `Decimator`'s `lfilter` state stays continuous across the
  idle/convo boundary and the ring is already full when the wake fires.
- **`EXIT_DURATION` is anchored on when the *room* goes quiet, not on when the user stopped.**
  `idle_watchdog` counts from `max(user last spoke, speaker went quiet)`. Getting this wrong drops
  the session to idle while Johnny is still answering, and the user has to say the wake word again
  to continue — which is exactly what the first version did. There are **three** windows where
  nothing else marks the session as active, and all three must be held:
    1. **The round trip.** Server VAD waits `silence_duration_ms` (800 ms), then the model thinks,
       then the first chunk arrives. Through all of it `model_busy` is False and `queued` is 0, so
       a 3 s clock started at the user's last word often expires *before Johnny speaks at all*.
       `Conversation.awaiting_reply` covers it, capped by `REPLY_TIMEOUT = 8` so a reply that never
       comes cannot wedge the session open. `server_content.waiting_for_input` releases it early
       when the model deliberately isn't answering.
    2. **The reply itself** — `model_busy` plus `player.queued`.
    3. **The device buffer.** `queued` hits zero when the callback *consumes* the last sample, but
       the large `OUT_LATENCY` buffer means audio is still leaving the speaker. `Player.quiet_since()` adds
       `stream.latency` back on.
  `interim_input_transcription` ("low latency transcription updated while the user is speaking")
  is what resets the user side — it streams *during* speech, unlike `input_transcription`.
- **`Conversation.begin()` must not call `heard_voice()`.** `heard_voice()` now also means "a reply
  is owed", so waking with no follow-up speech would hold the session for the whole `REPLY_TIMEOUT`
  instead of dropping back to idle after `EXIT_DURATION`. Likewise there is deliberately no
  `heard_voice()` at `turn_complete` — the `max()` anchor starts the countdown on its own.
- **Two exits, on purpose.** A `GOODBYE` regex over the input transcription is instant and
  deterministic; an `end_conversation` function declaration catches what the regex misses, which is
  the general case. Both set `ending` and let playback drain first, so the farewell is heard. The
  regex will false-positive on "goodbye is a nice song" — that is the accepted cost of the fast
  path, and why the tool exists.
- The callback fans out to **two queues** (`detect_q`, `uplink_q`). One queue cannot serve both: the
  detector is a thread doing 70–95 ms of blocking inference, the uplink lives on the asyncio loop.
- **Never pass `latency="high"` to a stream on the `pulse` device.** "High" is relative to the
  device PortAudio resolved, and the ALSA `pulse` plugin advertises
  `default_high_output_latency = 0.035 s` — so it grants an **80 ms** playback buffer, which
  PulseAudio's timer-based scheduling cannot keep fed. Measured: **620 ALSA `underrun occurred`
  recoveries in 12 s** from a callback that only copied a preallocated sine. The raw `hw:` speaker
  survives 80 ms fine (no server in the path), so this appeared only when `USE_AEC` moved playback
  through Pulse. Sweep, output-only, blocksize 1920, per 8 s: `0.10 → 1285`, `0.12 → 1336`,
  `0.16 → 0`, `0.24 → 0`. The cliff is between 0.12 and 0.16; `OUT_LATENCY = 0.24` is the first
  value with margin, `IN_LATENCY = 0.16`. Both revert to `"high"` when `USE_AEC` is off.
- **A dead output stream hangs the whole session, silently.** PortAudio just stops calling back;
  `player.queued` freezes above zero; `idle_watchdog` (`queued > 0` means "still talking") and
  `Player.drain()` then wait forever, with nothing printed. That is what the first underrun storm
  looked like from the outside: ALSA spam on stderr, one `(xrun)`, then a stall. `Player.stalled()`
  (`queued > 0` and either `stream.active` false or no callback for `STALL_SECONDS`) is now checked
  in both places, `drain()` takes a timeout, and `health()` prints it.
- **A `hw:` device vanishes from `sd.query_devices()` while PulseAudio holds it.** PortAudio's ALSA
  enumeration skips devices it cannot open, so right after an AEC run `USE_AEC = False` can fail
  with `ValueError: No output device matching 'USB Composite Device'`. It is transient — Pulse
  releases the card a moment later. Not a broken config.
- **A `create_task` you never await is a capability that can vanish in silence.** This cost a
  live debugging session. `uplink_pump` had an unguarded `await session.send_realtime_input()`;
  the socket closing under a send is routine (the pump is often mid-await when a conversation
  tears down), the exception killed the task, and because nothing awaited it until shutdown,
  **nothing was printed at all**. The mic simply stopped reaching Gemini for the rest of the
  run. The only visible symptom was `uplink_q` filling and dropping every frame. Reproduced
  with a fake session that raises on its fourth send. Three separate fixes, all needed:
    1. Sends are wrapped. A stale session (`convo.session is not session`) is ignored; a live
       one reports **once** — 12.5 frames/s makes a per-frame message noise — and calls
       `convo.finish()`, so it drops back to idle where the wake word can start a fresh
       session. Deaf-but-running is the worst possible state.
    2. `_send()` wraps every send in `asyncio.wait_for(SEND_TIMEOUT)`, because a half-open
       websocket can hang a send forever, which would wedge the pump just as thoroughly.
    3. `supervise(task, name)` adds a done-callback that prints loudly if a long-lived task
       dies (staying quiet on cancellation, which is normal shutdown), and `health()` reports
       any supervised task that has stopped.
- **Drops are counted per queue, and that *is* the diagnosis.** One combined counter cannot say
  which consumer stalled. `detect_drops` moving means the wake-word thread fell behind;
  `uplink_drops` climbing at ~12/s (one per 80 ms block) means the pump is not sending — not
  that the Pi is slow. The health line for uplink says "Gemini is not hearing the mic" outright.
- **Playback level: the Pulse sink was at 50%, and that 50% is applied in software.** This file
  sounded much quieter than `testGeminiLiveAPI.py` despite `VOLUME = 1` against its `0.1`.
  Measured 2026-09-07, 440 Hz at amplitude 0.3, read at the raw mic: raw `hw:` speaker
  **−21.8 dBFS**, pulse route with sink #0 at 50% **−36.0 dBFS**, pulse route at 100%
  **−23.1 dBFS**. So the 14 dB gap was entirely the sink volume, and nothing to do with
  headroom, mixing or the canceller. The reason it bites only here is that the speaker card
  exposes **no playback volume control at all** — `amixer -c 0 contents` has a `PCM Playback
  Switch` and nothing else — so PulseAudio has to apply its volume in software to everything
  crossing the sink, and `testGeminiLiveAPI.py`'s raw `hw:0,0` path never crosses it. Pulse
  still reports the sink `Flags: HARDWARE`, which is what makes this misleading. Fixed with
  `pactl set-sink-volume 0 100%` (persisted by `module-device-restore`); revert with `50%`.
- **Music is ducked, speech is not.** With the sink at 100%, speech and music now sum closer to
  clipping, which is a real concern — but paying for it out of `VOLUME` would attenuate Johnny
  exactly when he is competing with music. `MusicDucker` lowers the *music* sink-input to
  `DUCK_LEVEL = 40%` for the length of a conversation and puts it back, matched on
  `application.name = librespot` (so it does not need Spotify authorised, and does not duck
  the assistant's own stream, which is `ALSA plug-in [python3.13]` on the same sink).
  Three things it must keep doing: `duck()` is **idempotent** — an earlier
  version reset its saved map on each call, so a second duck recorded the ducked volume as the
  original and `restore()` left the music ducked forever; a stream already below the target is
  left alone rather than restored *up*; and `restore()` also runs in the outer `finally`, or
  Ctrl-C mid-conversation leaves the music quiet.
- **A duck that outlives its stream is permanent, and this was the real "music is always
  quiet" bug.** `module-stream-restore` remembers volume per **application**, not per stream,
  so once `duck()` writes the level, a stream that then disappears while ducked (paused,
  handed to another Connect device, librespot restarted) leaves `restore()` addressing a dead
  index — and every future librespot stream is *born* ducked, across reboots. The class's own
  docstring used to call that no-op "the correct outcome anyway"; it is not. Found live on
  2026-09-10 at **25%**, the value `DUCK_LEVEL` held *before* it moved to 40 — which is why
  `unduck_strays()` sweeps anything **below 100%** rather than matching `DUCK_LEVEL`. That is
  safe only because nothing else writes this fader: Spotify's slider is librespot's *softvol*,
  applied to the samples upstream of the sink-input, so a maxed slider cannot undo it and the
  symptom has no other explanation. It runs at startup (reporting what it found) and at the
  end of `restore()`. Verified against a `paplay` stream killed mid-duck.
- **Chimes play through `Player`, not a second stream.** `wakeUp.wav` on wake, `ExitConvoMode.wav`
  on exit, from `SoundEffects/`. Going through the same stream and the same `jarvis_aec_sink`
  is the point: they are cancelled out of the mic, so they cannot score as a wake word or read
  to Gemini's VAD as the user speaking. `load_chime()` does 44.1 kHz stereo → 48 kHz mono once
  at startup (160/147 via `resample_poly` — not a ratio the `Decimator`/`Upsampler` pair can
  do), and normalises both to `CHIME_PEAK`, because as recorded they are 11 dB apart in RMS
  (peaks 0.205 and 0.777) and the quieter one reads as broken. Placement is load-bearing: the
  wake chime is **not awaited**, so it covers the socket-opening gap (`JARVIS_PLAN.md:236`) and
  holds `idle_watchdog` off while the session comes up; the exit chime is pushed **after**
  `player.flush()` (or it is discarded with the model's audio) and **before** `listening.set()`
  (so the detector's window is zeroed over it and Johnny cannot wake himself on his own
  goodbye). A missing file warns and is skipped — losing the wake word over a nicety would be a
  bad trade. Each chime ends mid-block, so expect `playback underruns` to read 2 higher per
  conversation.
- **"Leave after this turn finishes" is not a promise the server keeps, and assuming it was
  deadlocked the session.** Symptom: say goodbye, Johnny says goodbye back, then it hangs forever
  with nothing printed until Ctrl-C. `convo.ending` was only ever *acted on* after
  `session.receive()`'s `async for` ended, so if `turn_complete` never arrived, `receive_loop`
  stayed inside receive(), `model_busy` stayed True (it is only cleared at turn end),
  `idle_watchdog` held on `model_busy`, and nothing could close the session. The `end_conversation`
  tool made it likelier, because it sets the flag *mid-turn* and then has to wait for a turn end
  that may not come. Three fixes, all needed:
    1. `close_when_ending` is its own task, in `converse`'s wait set. It waits for a goodbye, gives
       the farewell a **bounded** `FAREWELL_TIMEOUT` to finish, drains, and closes. Being a separate
       task is the whole point: it still runs while `receive_loop` is blocked in `receive()`.
    2. The wait is on a **turn counter, not an Event**. An Event is an edge, and `receive_loop`
       would set it at `turn_complete` and clear it at the top of the next turn — a closer not
       scheduled in between misses it and waits out the full timeout, turning a *normal* goodbye
       into the same stall. `begin_ending()` captures `ending_turn = turn_index` at the moment the
       goodbye is seen, so the writer decides which turn to wait for, not whenever the reader wakes.
    3. `TURN_TIMEOUT` covers the same deadlock with no goodbye in it: a turn that stops producing
       audio and never completes used to wedge the session just as permanently.
  Verified against fake sessions (no key, no hardware): normal goodbye 1.0 s, goodbye with a turn
  that never completes 1.5 s, no goodbye with a stuck turn 3.0 s, plain silence 3.3 s — all four
  previously either hung forever or were untested.
- **`idle_watchdog` now says why a session is still open.** It held four different reasons silently,
  so "it stalled going back to idle" had no evidence attached to it at all — `health()` reports the
  audio path, not the conversation. After `HOLD_REPORT` seconds it prints the reason once, and the
  silence countdown announces itself. Purely diagnostic; it changes no timing.
- `MODEL_PATH` resolves relative to `__file__`, not the CWD. `testWordDetection.py` uses a bare
  relative path and already breaks when run from inside its own folder — do not copy that.

## Actions (`Action.py`, `ActionManager.py`, `Spotify.py`)

Phase 6 of `JARVIS_PLAN.md`: Gemini does things, not just talks. Three files at the repo
root, imported by `IntegrationTests/testChatWakeup.py` via a `sys.path` insert resolved from
`__file__`.

**The one thing to understand before changing any of this: speech and arguments are already
two separate channels.** `LiveServerMessage.tool_call` is a *sibling* field of
`server_content`, not nested in it (`google/genai/types.py:20688-20800`), and carries
`function_calls[].args` as structured JSON. Function calls never appear in
`output_audio_transcription`. So Gemini says "Okay, desk lights blue" on the audio channel
while `set_light_color(zone="desk", color="blue")` arrives on the tool channel.

- **Never parse the transcript to find a call.** It was the obvious-looking design and it is
  the wrong one — it would force the model to speak its own arguments, and it would break the
  moment the model phrased a sentence differently. The structured channel is exact.
- **Automatic function calling (AFC) does not exist on the Live API.** The engine in
  `models.py` (`invoke_function_from_dict_args`, `automatic_function_calling_history`) has no
  counterpart in `live.py`; `_t_live_connect_config` (`live.py:1149-1211`) special-cases only
  MCP tools and passes anything else through untouched. Passing bare Python callables in
  `config.tools` therefore does nothing useful — always pass
  `types.Tool(function_declarations=[...])`. Dispatch is yours to write.
- **Schema comes from the Python signature.** `Action` reads `inspect.signature` +
  `typing.get_type_hints`: `str/int/float/bool` map to scalar types, `Literal[...]` becomes an
  **enum** (the highest-value case — without it the model invents zone names), `list[X]` an
  array, no default means `required`. A no-argument function sends `parameters=None` rather
  than an empty object.
- **`purpose` is the load-bearing string.** It is the only thing the model reads when deciding
  *whether* to call an action, so it is written as an instruction about when to use it. The
  `describe={}` overrides exist for wording a type hint cannot carry.
- **`briefing()` is policy, not documentation.** The model learns the tools from the
  declarations; prose repeating them is redundant. What the text is for is the rule that keeps
  the spoken half human — never read function names, argument names or values aloud, never
  announce a tool call. It is appended to `SYSTEM_INSTRUCTION` in `build_config()`.
- **Return values go back under the `"output"` key.** `FunctionResponse.response` documents
  '"output" key to specify function output and "error" key to specify error details'
  (`types.py:2025-2027`). `Action.normalise` used `{"result": ...}` first, and in one run the
  model was handed 60.6 and said "31 degrees" out loud. After switching to `"output"`, four
  consecutive temperature readings came back exact (63.4, 61.1, 60.6, 60.6). One run is not
  proof of causation, but there is no reason to use a key the API does not document.
- **Handlers run in `loop.run_in_executor`, never inline.** `uplink_pump` lives on the same
  event loop and feeds the mic to Gemini, so a handler blocking two seconds on a subprocess
  stops the microphone for two seconds. It presents as an audio fault, not a slow action.
- **An action that raises must not kill the conversation.** `Action.invoke` catches everything
  and returns `{"error": ...}`, which reaches Gemini as text it can say out loud ("Spotify
  isn't linked yet"). `ACTION_TIMEOUT = 15 s` bounds a hung one — note the worker thread is
  not killed, only abandoned.
- **`convo.action_busy`** holds `idle_watchdog` while an action runs. Without it `model_busy`
  is False and the player is quiet, so a slow action reads as silence and drops the session
  mid-call — the same hole as the round-trip window, and the same fix.
- **`tool_call_cancellation`** (`types.py:20451-20461`) arrives when the user barges in over
  the turn that asked. `ActionManager.cancel()` records the ids and `dispatch` then withholds
  those responses rather than pushing a stale result into a moved-on conversation.
- `end_conversation` is now a registered `Action` like any other. The one thing the manager
  cannot own is the teardown: only `testChatWakeup.py` knows playback must drain before the
  socket closes, or the farewell is cut off. That is what `dispatch(on_call=...)` is for.
- All actions are **blocking** by choice: Gemini waits for the return value, so every reply is
  truthful. `types.Behavior.NON_BLOCKING` plus `FunctionResponseScheduling` (`types.py:340-352`,
  `types.py:198-212`) is the escape hatch if an action ever gets slow enough to hurt —
  `SILENT` delivers a result without making the model speak at all.

`.venv/bin/python ActionManager.py` self-tests the whole framework with no API key and no
hardware: inference, the construction guards, dispatch, an unknown function, a handler that
raises, cancellation, and the timeout.

**Spotify plays on the Pi itself**, via librespot from the raspotify package, run as a
**user** systemd service — see the section below. Control goes through **spotipy**, not
`spotify-cli`. The two halves fail independently, which is useful: you can play to the Pi from a
phone before the API half is authorised at all.

**`spotify-cli` was removed, and the reason is worth keeping.** Its `auth login` is dead on
Python 3.13 — PyInquirer → prompt_toolkit 1.0.14 → `from collections import Mapping`, gone since
3.10 — but that alone was fixable, and only the optional *scope picker* touched it (all 18 of its
command modules import fine). The disqualifying part is `cli/utils/Spotify.py:44-88`: it POSTs to
`https://asia-east2-spotify-cli-283006.cloudfunctions.net/auth-refresh` on **every token
refresh**, and forwards your **client secret** there if you supply your own app. That is the
author's personal cloud function, untouched since 2020 — and it still answers with HTTP 200,
which makes it a worse trap, not a safer one. Do not reinstall it. `spotify-cli`, `PyInquirer`
and the `prompt_toolkit` 1.0.14 pin (which would block any modern prompt_toolkit) are all
uninstalled.

## Echo cancellation (`module-echo-cancel`)

Loaded from `~/.config/pulse/default.pa`, so it survives a reboot. Measured on this Pi, pink
noise at amplitude 0.3, playback and capture on separate streams:

| | mic noise floor | during playback | echo vs. floor |
|---|---|---|---|
| raw `hw:` devices | −46.0 dBFS | −35.5 dBFS | **+10.5 dB** (audible) |
| through the canceller | −58.7 dBFS | −62.3 dBFS | **−3.6 dB** (below the floor) |

Absolute level in the mic during playback fell **26.8 dB**. Roughly 14 dB of that is echo
cancellation proper (echo went from 10.5 dB above its own floor to 3.6 dB below it) and ~13 dB is
the noise suppression enabled in `aec_args`. It also *helps* the wake word: 12 s of ambient scored
max 0.0028 / mean 0.0025 through the canceller versus max 0.0064 / mean 0.0029 raw — 54x below the
0.15 threshold instead of 23x.

Four traps, each of which cost a debugging cycle:

- **`aec_args` must carry literal double quotes.** `pactl` and the module parser split the whole
  argument string on whitespace, so an unquoted multi-word value is read as extra (invalid) module
  parameters. Any single option loads; any two fail with nothing but `Module initialization
  failed`. Pass it as `'aec_args="a=0 b=1"'`.
- **`use_volume_sharing=no` is mandatory.** Without it the new virtual source shares volume with
  its master, and its own fresh 0% is written straight through to the hardware capture control —
  silently undoing the mic gain the moment the module loads. Observed.
- **`analog_gain_control=0`** for the same reason: webrtc's analog AGC drives that same control.
- **`.pa` files have no line continuation.** A `load-module` split across lines with `\` parses as
  nothing and the module simply never appears, with no error anywhere. One line only.

`~/.config/pulse/default.pa` **replaces** `/etc/pulse/default.pa` rather than extending it, so it
must begin with `.include /etc/pulse/default.pa` or there is no audio at all.

Reload after editing with `systemctl --user restart pulseaudio`. To re-measure any of the above,
run `BasicFunctionTests/testAEC.py`, which does both routes in one pass. To A/B in the real
script, set `USE_AEC = False` in `testChatWakeup.py`. Note `sd.playrec()` and `sd.play()`/`sd.rec()` are unreliable here — duplex
through the ALSA `pulse` plugin throws `ContinuePoll` errors, and `play`/`rec` share one
module-level stream so they cannot run concurrently. Use explicit `sd.InputStream` +
`sd.OutputStream`, which is what the real code does anyway.

## Spotify Connect on the Pi (`librespot`, user service)

`~/.config/systemd/user/librespot.service` runs `/usr/bin/librespot` (from the raspotify
package) with `PULSE_SINK=jarvis_aec_sink`. `spotify_play` in `Spotify.py` switches to it with
`spotify devices --switch-to Johnny` and retries when nothing is playing anywhere.

- **The packaged `raspotify.service` cannot work here and is disabled.** It runs as root with
  `ProtectHome=true` and `PrivateUsers=true` (`/lib/systemd/system/raspotify.service`), so it
  cannot reach this user's PulseAudio socket in `/run/user/1000`. It would fall back to its
  default `LIBRESPOT_BACKEND="alsa"` and take `hw:0,0` directly — the same speaker
  `testChatWakeup.py` plays through, which ALSA gives to one process at a time. A **user** unit
  inherits `XDG_RUNTIME_DIR` and finds Pulse for free. Leaving it enabled also advertises a
  second Connect device that would fight for the card if picked.
- **`PULSE_SINK=jarvis_aec_sink` is load-bearing.** `pactl get-default-sink` is the raw
  speaker, so without it music bypasses the canceller entirely and leaks into the mic
  uncancelled — with no symptom other than a wake word that stops working. Check with
  `pactl list sink-inputs | grep -i sink:`.
- **librespot's softvol default is `--volume-ctrl log` over a 60 dB range**, so the unit's
  original `--initial-volume 50` put music at **-30 dB** before it ever reached the sink --
  24 dB under Johnny at `VOLUME = 0.5`, which is the whole of the "music is always quiet, voice
  is always loud" symptom. `MusicDucker` was not at fault (verified against a live `paplay`
  stream: 100% -> 25% -> 100%); it only *looked* broken because it took already-inaudible music
  another 36 dB down. The unit now passes `--volume-ctrl cubic --initial-volume 100`. Measured
  program levels, 2026-09-10: Gemini's reply audio peak -1.6 / rms **-16.9 dBFS**, the repo mp3
  peak 0.1 / rms **-12.1 dBFS**. Note a comment cannot go inside the `ExecStart` line
  continuation -- same trap as the `.pa` files, one directive per logical line.
- **Music is mono**, because `module-echo-cancel` takes one `channels=` for its source and its
  sink alike (confirmed against the module's own argument string). `channels=2` in
  `~/.config/pulse/default.pa` is the experiment, gated on `testAEC.py` still showing the mono
  ERLE baseline.
- Auth is **zeroconf** — no credentials in the unit. `--cache` keeps them after the first login
  from a phone so it reconnects itself; `--disable-audio-cache` stops it filling the SD card.
  librespot logs `Mixing with softvol`, so volume is applied to the samples *before* the sink,
  which keeps the canceller's reference matching what is actually emitted.
- `Linger=no` for this user, so it stops with your last session — as does PulseAudio, and
  therefore the whole assistant. `sudo loginctl enable-linger johnny` if it should survive
  logout.

**Never diagnose a 403 as "needs Premium".** `_call` used to append `(playback control needs
Premium)` to every 403, so a Premium account was told out loud that it was not one — the error
text goes straight to Gemini, which reads it as fact. Measured 2026-09-07 with nothing playing:
`PUT /me/player/pause` → `403 reason='UNKNOWN'`, `Player command failed: Restriction violated`.
That is Spotify's "invalid in the current player state" (nothing is playing, or the player is
already in the state you asked for) and says nothing about the account tier; Premium has its own
reason, `PREMIUM_REQUIRED`. Note the tier cannot be read back either — `me()['product']` is
`None` without the `user-read-private` scope, so do not try to check it. `_speakable()` also
strips the request URL spotipy puts in `exc.msg`, which was otherwise being read aloud.

**Search asks for a page, never `limit=1`.** Spotify's ranker returns a *different and worse*
order at `limit=1`: measured on this account, 6 trials each and deterministic both ways,
`q="Moog City"` gives "Aria Math" at `limit=1` and "Moog City" at `limit=10` (`q="Sweden C418"`
gives "Mice on Venus" vs "Sweden"). That was the whole of the `spotify_play(query='Moog City')`
-> "Aria Math" bug — not a bad query, an unlucky page size. `SEARCH_LIMIT = 10` is also the
practical ceiling: 12, 15, 20, 25 and 50 all answer `400 Invalid limit` here despite the
documented 50. The page is re-ranked locally by `_match_score`, and the score doubles as the
mood/name test — a query that names a title plays that one track, a hedged one ("some jazz")
queues the whole page so playback does not stop after one song. `.venv/bin/python Spotify.py
selftest` checks the whole chooser against recorded pages, with no key and no network.

**`spotify_playlist` needs two more scopes, and adding a scope silently invalidates the cached
token.** `/me/playlists` requires `playlist-read-private` (even for your own private ones) plus
`playlist-read-collaborative`. spotipy's `validate_token()` rejects a cached token whose scope
is not a superset of `SCOPES`, and the symptom is indistinguishable from never having
authorised — re-run `Spotify.py login`. The library list is cached for `PLAYLIST_TTL` (300 s)
because a voice turn cannot afford four HTTP round trips before the music starts; a name that
misses re-fetches once, so a playlist made a minute ago is still findable. `/me/playlists`
returns `null` rows for playlists deleted out from under the index, hence the filter.

**Spotify control lives in `Spotify.py` via spotipy.** Credentials come from
`SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` in `Dormy/.env` (same file as `GEMINI_API_KEY`);
the refresh token caches to `~/.config/dormy/spotify-token.json` and never leaves the Pi. The
redirect URI must be `http://127.0.0.1:8888/callback` — the literal loopback IP, since Spotify
rejects `localhost` and allows plain http only for loopback. `python Spotify.py login` runs the
flow with `open_browser=False`, printing the URL and taking the redirected URL back by hand,
because the browser is on your laptop and the Pi is over SSH. The client is built **lazily**, so
an unconfigured Spotify never stops the assistant starting — those actions just report why they
cannot run. `spotify_play` transfers to the Pi and retries on `NO_ACTIVE_DEVICE`; the other verbs
do not, because "pause" with nothing playing should say so rather than hunt for a speaker.

**Measured: music does not break the wake word.** Real music (the repo's mp3) into
`jarvis_aec_sink`, scoring `johnny.onnx` on the AEC source:

| music level | mic during music | vs. floor | wake word max | margin to 0.15 |
|---|---|---|---|---|
| amp 0.3 | −60.0 dBFS | **−7.8 dB** (buried) | 0.0028 | 54x |
| amp 0.7 (loud) | −52.5 dBFS | +4.0 dB | 0.0105 | 14x |

No false accepts at either level, and 0.0028 is the same figure a silent room scores — at
moderate volume the canceller removes music so completely the detector cannot tell it is
playing. The margin falls to 14x when it is loud, so that is the level to watch. Two clients on
the sink at once (librespot plus the assistant's own playback) measured 0 ALSA underruns and 0
dropped frames. **Not yet tested: whether a person can be *heard* over music** — that needs a
voice at the mic, and it is the other half of the question.

## State of the code vs. the plan

The plan's §5 repo layout (`jarvis/`, `firmware/`, `tests/`, `tools/`) is the target, not what
exists. Today it is `BasicFunctionTests/` (single-capability probes) plus `IntegrationTests/`
(things wired together), with `wakeup_models/` at the root.

**AEC now exists, at the OS level.** `module-echo-cancel` (webrtc) is loaded from
`~/.config/pulse/default.pa` and `IntegrationTests/testChatWakeup.py` routes through it —
see the AEC section below. What follows still holds for `testAEC.py` itself.

**`testAEC.py` now measures the canceller** — it used to be an 18-line loopback
(`outdata[:] = np.repeat(indata, 2, axis=1)`) with no canceller of any kind, which tests nothing:
a loopback has no far-end signal to cancel, so hearing no howl only means acoustic loop gain is
under unity. It now plays pink noise out of the speaker, records the mic, and reports level
against the mic's own noise floor, for both routes in one run (`--route raw|aec|both`).

Reading it: **the margin over the floor is the number, not the absolute level**, because the
absolute moves with capture gain. The two floors are not comparable across routes either — the
canceller's noise suppressor lowers the floor as well — which is why it prints both an absolute
reduction and a relative one. The per-second trace exists so convergence is visible; the webrtc
filter settles in ~2–3 s, and a level that never drops means it is not receiving a reference
signal at all (almost always: playback went somewhere other than `jarvis_aec_sink`). Hence
`--secs` defaults to 12 and the level is read from the back 45% only. `--live` plays continuously
and meters the mic once a second, for the double-talk check — speech must still get through, or
the thing is a gate, not a canceller, and Johnny goes deaf while he talks.

The OS route is `/usr/lib/pulse-17.0+dfsg1/modules/module-echo-cancel.so` with the `webrtc` method
and drift compensation (`pa_webrtc_ec_set_drift`), reached through PortAudio's `pulse` device.

**`JARVIS_PLAN.md` §2's shared-clock question is answered: the drift is real and large, and the
speaker card's input cannot fix it.** Measured 2026-09-07 by playing 440.000 Hz out of the
speaker and recovering the tone's frequency from each candidate source (20–30 s window,
parabolic-interpolated FFT peak):

| source | ambient floor | 440 Hz SNR | recovered | clock error |
|---|---|---|---|---|
| C-Media PnP (current) | −38.7 dBFS | **76.2 dB** | 440.95 Hz | **+2150 ppm** (repeat: +2253) |
| Jieli speaker-card input | −63.6 dBFS | **9.6 dB** | 439.96 Hz | −99 ppm (≈0 within error) |

So the two crystals really are ~**0.22 % apart** — about **103 samples per second** of slip at
48 kHz, which is why drift compensation is load-bearing rather than decorative. And the same-card
input really is shared-clock, as the theory predicts.

It is still the wrong mic, by a wide margin. It is **66 dB less sensitive to the same acoustic
signal**, and at most ~22 dB of that is the gain difference (its `Mic Capture Volume` is already
140/147, near 0 dB, against the C-Media's +22 dB analog boost). Its ambient floor is 25 dB
quieter than the room mic's — it is not hearing the room. The 440 Hz it does show is at 0 ppm
and 66 dB down, which is what **electrical crosstalk from the DAC on the same chip** looks like,
not acoustic pickup; the first run, reading broadband level rather than a single bin, put the
speaker *below* that source's own noise floor. Its port advertises `analog-input-mic`
("availability unknown") with nothing evidently plugged into it.

The shared-clock path is therefore only worth revisiting with a **real microphone element on the
speaker's card** — plugging the electret into that card's mic jack is the cheap experiment, and
`scratchpad/narrow.py`-style tone recovery is how to check it: a working mic there should show a
high 440 Hz SNR *and* ~0 ppm. Until then the two-card setup plus `pa_webrtc_ec_set_drift` is the
right configuration, and the measured result it produces (echo 3.6 dB below the mic floor) says
drift is not currently the binding constraint.
