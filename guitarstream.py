import sounddevice as sd
import numpy as np
from pylsl import StreamInfo, StreamOutlet

# LSL setup
channel_count = 1
lsl_rate = 256  # Hz
info = StreamInfo('Guitar256Hz', 'Audio', channel_count, lsl_rate, 'float32')
outlet = StreamOutlet(info)

# Audio input setup
audio_rate = 44100  # or whatever your USB-C device uses
chunk_size = 1024  # number of samples per callback

buffer = []

def callback(indata, frames, time, status):
    if status:
        print(status)
    buffer.extend(indata[:, 0])  # mono input

    # downsample
    while len(buffer) >= int(audio_rate / lsl_rate):
        sample = buffer[:int(audio_rate / lsl_rate)]
        buffer[:] = buffer[int(audio_rate / lsl_rate):]
        avg = float(np.mean(sample))  # simple downsampling
        outlet.push_sample([avg])

with sd.InputStream(channels=channel_count, samplerate=audio_rate, callback=callback):
    print("Streaming guitar to LSL at 256 Hz. Ctrl+C to stop.")
    while True:
        pass
