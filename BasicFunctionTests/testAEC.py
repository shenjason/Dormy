"""Measure the echo canceller, on its own, with a number at the end.

This file used to be an 18-line loopback -- `outdata[:] = np.repeat(indata, 2, axis=1)`
-- which does not test echo cancellation at all. A loopback has no far-end signal in
it, so there is nothing to cancel: hearing no howl only means the acoustic loop gain is
under unity. The real test plays a *known* far-end signal out of the speaker, records
what comes back into the mic, and compares that against the mic's own noise floor.

It runs the same route testChatWakeup.py runs, so a result here transfers:

  raw   MIC / SPEAKER by name -> hw:1,0 / hw:0,0, straight past PulseAudio, no
        canceller of any kind in the path.
  aec   the ALSA "pulse" device steered with PULSE_SOURCE / PULSE_SINK at the
        module-echo-cancel source and sink. PortAudio cannot name a Pulse source, so
        the env vars are the only handle -- and libpulse reads them when the stream
        is opened, which is why they are set before any stream is constructed.

  .venv/bin/python BasicFunctionTests/testAEC.py            # both routes, A/B table
  .venv/bin/python BasicFunctionTests/testAEC.py --route aec
  .venv/bin/python BasicFunctionTests/testAEC.py --live     # talk over it, watch the meter
  .venv/bin/python BasicFunctionTests/testAEC.py --amp 0.5  # louder far end

Two measurements per route, back to back so the room does not change between them:

  floor      3 s of capture with the speaker silent -- the room plus the mic's own noise.
  playback   pink noise out of the speaker for --secs, capture running throughout.

What matters is *playback minus floor*. An absolute mic level says nothing on its own,
because it moves with capture gain; the margin over the floor is what decides whether
Gemini's VAD hears Johnny and cuts him off. Above the floor means audible echo. Below
means the echo has been buried in the noise, which is as good as this gets.

Pink noise on purpose: it is broadband and continuous, so it excites the whole response
and gives the adaptive filter nothing to hide behind. Speech is an easier signal.

Reference, measured on this Pi at --amp 0.3 (also in CLAUDE.md):

    raw   floor -46.0   playback -35.5   echo +10.5 dB   (audible)
    aec   floor -58.7   playback -62.3   echo  -3.6 dB   (buried)

Caveats worth not re-deriving: sd.playrec() throws ContinuePoll errors through the ALSA
pulse plugin, and sd.play()/sd.rec() share one module-level stream so they deadlock when
run together. Explicit InputStream + OutputStream is the only thing that works here --
and it is the topology the real code uses anyway.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import sounddevice as sd

# --- config, kept in step with IntegrationTests/testChatWakeup.py ----------

MIC = "USB PnP Sound Device"
SPEAKER = "USB Composite Device"

AEC_SOURCE = "jarvis_aec_source"
AEC_SINK = "jarvis_aec_sink"

RATE = 48000          # both USB cards are 48 kHz only
BLOCKSIZE = 2048      # ~43 ms; this file has no realtime deadline to hit

FLOOR_SECONDS = 3.0   # silent capture, for the noise floor
SETTLE = 0.3          # discard after opening a stream -- the first blocks are junk

# The adaptive filter needs time to converge, so the level is read from the back half
# of the playback window only. Anything shorter than ~8 s measures the convergence,
# not the result.
CONVERGE_FRACTION = 0.55


def db(x):
    return -99.0 if (x <= 1e-9 or not np.isfinite(x)) else 20.0 * np.log10(x)


def rms_db(frames, skip_seconds):
    """dBFS of a list of mono blocks, discarding the first skip_seconds."""
    if not frames:
        return float("nan")
    a = np.concatenate(frames)[int(RATE * skip_seconds):].astype(np.float64)
    a = a[np.isfinite(a)]
    return db(np.sqrt((a ** 2).mean())) if len(a) else float("nan")


def pink(seconds, amp):
    """Pink noise, amplitude-normalised. Fixed seed so runs are comparable."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal(int(RATE * seconds))
    spec = np.fft.rfft(w)
    f = np.maximum(np.arange(len(spec)), 1.0)   # 1/f amplitude -> -3 dB/octave
    out = np.fft.irfft(spec / np.sqrt(f), n=len(w))
    return (out / np.abs(out).max() * amp).astype(np.float32)


def aec_loaded():
    """(ok, what is missing). Cheaper than letting PortAudio fail obscurely."""
    try:
        listed = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True, text=True, timeout=5,
        ).stdout + subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False, [AEC_SOURCE, AEC_SINK]
    missing = [n for n in (AEC_SOURCE, AEC_SINK) if n not in listed]
    return not missing, missing


LOAD_HINT = f"""\
Load it with (note the literal double quotes -- pactl splits the argument string on
whitespace, so an unquoted multi-word aec_args parses as extra invalid parameters and
fails with nothing but "Module initialization failed"):

  pactl load-module module-echo-cancel aec_method=webrtc \\
    source_master=alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-mono \\
    sink_master=alsa_output.usb-Jieli_Technology_USB_Composite_Device_433135383532342E-00.analog-stereo \\
    source_name={AEC_SOURCE} sink_name={AEC_SINK} \\
    rate=48000 channels=1 use_volume_sharing=no \\
    'aec_args="analog_gain_control=0 digital_gain_control=0 noise_suppression=1 high_pass_filter=1 voice_detection=0"'

It is also in ~/.config/pulse/default.pa, so `systemctl --user restart pulseaudio`
should bring it back."""


def route_devices(route):
    """(in_device, out_device, out_channels) and the env side effects for a route.

    Both directions must go through the canceller. If playback went to the raw hw:
    speaker the module would never see the reference signal and would remove nothing
    -- silently, which is the failure this whole file exists to catch.
    """
    if route == "aec":
        os.environ["PULSE_SOURCE"] = AEC_SOURCE
        os.environ["PULSE_SINK"] = AEC_SINK
        return "pulse", "pulse", 1

    # Addressing the cards by name maps to hw:1,0 / hw:0,0 and bypasses Pulse
    # entirely, but clear these anyway so a previous route cannot leak in.
    os.environ.pop("PULSE_SOURCE", None)
    os.environ.pop("PULSE_SINK", None)
    channels = min(sd.query_devices(SPEAKER, "output")["max_output_channels"], 2)
    return MIC, SPEAKER, channels


def capture(in_device, seconds, far=None, out_device=None, out_channels=1,
            on_second=None):
    """Record `seconds` of mic, optionally playing `far` out of the speaker at once.

    Returns the captured blocks. The callbacks only copy and index -- no work, no
    printing -- because that rule holds here too even without a deadline to miss.
    """
    frames = []

    def in_cb(indata, n, t, status):
        frames.append(indata[:, 0].copy())

    pos = [0]

    def out_cb(outdata, n, t, status):
        i = pos[0]
        chunk = far[i:i + n]
        if len(chunk) < n:
            outdata[:len(chunk)] = chunk[:, None]
            outdata[len(chunk):] = 0
            raise sd.CallbackStop
        outdata[:] = chunk[:, None]     # mono broadcast across however many channels
        pos[0] = i + n

    mic = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                         device=in_device, blocksize=BLOCKSIZE, callback=in_cb)
    if far is None:
        streams = [mic]
    else:
        streams = [sd.OutputStream(samplerate=RATE, channels=out_channels,
                                   dtype="float32", device=out_device,
                                   blocksize=BLOCKSIZE, callback=out_cb), mic]

    for s in streams:
        s.start()
    try:
        deadline = time.perf_counter() + seconds
        mark = time.perf_counter() + 1.0
        seen = 0
        while time.perf_counter() < deadline:
            time.sleep(0.05)
            if on_second and time.perf_counter() >= mark:
                on_second(frames[seen:])
                seen = len(frames)
                mark += 1.0
    finally:
        for s in streams:
            s.stop()
            s.close()
    return frames


def measure(route, secs, amp):
    print(f"\n--- {route.upper()} " + "-" * (60 - len(route)))
    in_dev, out_dev, out_ch = route_devices(route)
    print(f"    capture {in_dev!r}   playback {out_dev!r} x{out_ch}")

    print(f"    floor:    {FLOOR_SECONDS:.0f} s, speaker silent ... ", end="", flush=True)
    floor = rms_db(capture(in_dev, FLOOR_SECONDS), SETTLE)
    print(f"{floor:6.1f} dBFS")

    far = pink(secs, amp)
    print(f"    playback: {secs:.0f} s of pink noise at amp {amp} ...")

    # Per-second trace, so convergence is visible rather than inferred. The webrtc
    # filter usually settles inside 2-3 s; a level that never drops means it is not
    # getting the reference signal at all.
    def tick(new_blocks):
        level = rms_db(new_blocks, 0.0)
        bar = "#" * max(0, int((level + 80) / 2))
        print(f"      {level:6.1f} dBFS  {bar}")

    frames = capture(in_dev, secs, far=far, out_device=out_dev,
                     out_channels=out_ch, on_second=tick)
    echo = rms_db(frames, secs * CONVERGE_FRACTION)
    print(f"    playback: {echo:6.1f} dBFS  (last {(1 - CONVERGE_FRACTION) * 100:.0f}%, "
          "after convergence)")
    return floor, echo


def live(route, amp):
    """Play pink noise forever and meter the mic, for the double-talk check.

    The number to watch is what happens when you speak. Echo suppressed and speech
    passing through is the whole point -- a canceller that also removes the near end
    is just a gate, and would make Johnny deaf while he talks.
    """
    in_dev, out_dev, out_ch = route_devices(route)
    print(f"\n--- {route.upper()} live meter, Ctrl-C to stop " + "-" * 20)
    print(f"    capture {in_dev!r}   playback {out_dev!r} x{out_ch}")
    print("    Speaker is playing. Stay quiet to read the echo, then talk over it:")
    print("    the level should jump for your voice and settle back down.\n")

    far = np.tile(pink(10.0, amp), 60)   # 10 min, seamless enough for a meter
    try:
        capture(in_dev, len(far) / RATE, far=far, out_device=out_dev,
                out_channels=out_ch,
                on_second=lambda b: print(
                    f"      {rms_db(b, 0.0):6.1f} dBFS  "
                    + "#" * max(0, int((rms_db(b, 0.0) + 80) / 2))))
    except KeyboardInterrupt:
        print("\n    stopped")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--route", choices=("raw", "aec", "both"), default="both")
    ap.add_argument("--secs", type=float, default=12.0,
                    help="playback window; under ~8 s measures convergence, not result")
    ap.add_argument("--amp", type=float, default=0.3,
                    help="far-end amplitude 0-1 (testChatWakeup.py plays at VOLUME=0.1)")
    ap.add_argument("--live", action="store_true",
                    help="continuous playback + mic meter, for a double-talk check")
    args = ap.parse_args()

    routes = ("raw", "aec") if args.route == "both" else (args.route,)

    if "aec" in routes:
        ok, missing = aec_loaded()
        if not ok:
            sys.exit(f"echo canceller not loaded -- missing: {', '.join(missing)}\n\n"
                     + LOAD_HINT)

    if args.live:
        live(routes[-1], args.amp)
        return

    results = {}
    for route in routes:
        results[route] = measure(route, args.secs, args.amp)

    print("\n" + "=" * 64)
    print(f"{'route':<8}{'floor':>10}{'playback':>12}{'echo vs floor':>16}")
    for route, (floor, echo) in results.items():
        print(f"{route:<8}{floor:>9.1f} {echo:>11.1f} {echo - floor:>+15.1f} dB")

    if len(results) == 2:
        raw_floor, raw_echo = results["raw"]
        _, aec_echo = results["aec"]
        # Two different numbers, and they answer two different questions.
        # Absolute is what the mic actually sees, and includes the noise suppressor.
        # Relative is cancellation proper -- how far the echo moved against its own
        # floor -- and is the honest figure for the adaptive filter.
        print(f"\nabsolute reduction in the mic: {raw_echo - aec_echo:.1f} dB")
        print(f"echo relative to its own floor: "
              f"{raw_echo - raw_floor:+.1f} dB -> {aec_echo - results['aec'][0]:+.1f} dB")
        # +3 dB over the floor is roughly a doubling of power and about where the
        # echo stops being separable from the room -- close enough to "buried".
        if aec_echo - results["aec"][0] < 3.0:
            print("verdict: echo is at or under the mic noise floor. Working.")
        elif raw_echo - aec_echo > 10:
            print("verdict: reduced but still audible. Lower VOLUME, or check that "
                  "playback really goes through the sink.")
        else:
            print("verdict: little or no cancellation. The usual cause is playback "
                  "not going through jarvis_aec_sink, so the module never sees a "
                  "reference signal.")
    print("=" * 64)


if __name__ == "__main__":
    main()
