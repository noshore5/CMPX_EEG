from pylsl import resolve_byprop, StreamInlet
import time
import signal
import sys

running = True

def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

print("Looking for an EEG stream...")

# Resolve stream (waits up to 5 seconds for stream to appear)
streams = resolve_byprop('type', 'EEG', timeout=5)

if not streams:
    print("ERROR: No EEG stream found!")
    print("Make sure test.py (provider) is running first.")
    sys.exit(1)

# Connect to the first EEG stream found
inlet = StreamInlet(streams[0])
print(f"Connected to stream: {inlet.info().name()}")
print(f"Channels: {inlet.info().channel_count()}")
print(f"Sample rate: {inlet.info().nominal_srate()} Hz")
print("\nReceiving data (press Ctrl+C to stop)...\n")

sample_count = 0
try:
    while running:
        # Get sample from stream
        sample, timestamp = inlet.pull_sample()
        if sample:
            sample_count += 1
            if sample_count % 100 == 0:  # Print every 100 samples
                print(f"Sample #{sample_count}: {[f'{x:.3f}' for x in sample]} | Time: {timestamp:.3f}")
    
    print(f"\nReceived {sample_count} samples total")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

