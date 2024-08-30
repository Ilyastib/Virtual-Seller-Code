import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Parameters
device_id = 1
duration = 10  # Duration of recording in seconds
sample_rate = 44100  # Sample rate in Hz
channels=1

# Set the selected device
sd.default.device = device_id

print("Recording...")

# Record audio
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')

# Wait until recording is finished
sd.wait()

# Save as WAV file
wav.write("data/records/output.wav", sample_rate, audio)

print("Recording finished. Saved to output.wav")
