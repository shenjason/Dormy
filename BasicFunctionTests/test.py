import sounddevice as sd
import numpy as np


sd.default.samplerate = 48000
sd.default.dtype = "float64"
sd.default.device = ("USB PnP Sound Device", "USB Composite Device") 


print(sd.query_devices())


recording = sd.rec(int(5 * sd.default.samplerate), channels=1)

print("Recording...")
sd.wait()

recording = np.repeat(recording, 2, axis=1)

print("Recording finished.")

print("Playing back the recording...")
sd.play(recording)
sd.wait()
print("Playback finished.")