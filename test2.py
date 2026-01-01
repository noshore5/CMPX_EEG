from pylsl import StreamInfo, StreamOutlet
import numpy as np
import time
import gc
import signal
import sys
import psutil
import os
sys.path.insert(0, '/Users/noahshore/Documents/CoherIQs/CMPX_EEG')
from utils.signal_utils import generate_signals

sr = 128
running = True

def signal_handler(sig, frame):
    global running
    print("\nShutting down gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

info = StreamInfo(
    name='EEG2',
    type='EEG',
    channel_count=8,
    nominal_srate=sr,
    channel_format='float32',
    source_id='fake_eeg_001'
)

# Add channel labels (adjust names as needed)
labels = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4']
ch = info.desc().append_child("channels")
for lab in labels:
    ch.append_child("channel").append_child_value("label", lab)

outlet = StreamOutlet(info)

# Generate signal buffer and stream it continuously
signal_buffer_size = 256  # Seconds of pre-generated signal
signals = generate_signals(n_signals=8, length=signal_buffer_size, fs=sr).astype(np.float32)
sample_idx = 0

# Memory monitoring
process = psutil.Process(os.getpid())
max_memory_mb = 1000  # Increased threshold

try:
    while running:
        try:
            # Get one sample from each channel
            sample = [float(signals[i, sample_idx % signal_buffer_size]) for i in range(8)]
            outlet.push_sample(sample)
            sample_idx += 1
        except Exception as e:
            print(f"Error pushing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Aggressive garbage collection every 100 samples
        if sample_idx % 100 == 0:
            gc.collect()
        
        # Monitor memory every 500 samples
        if sample_idx % 500 == 0:
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"Pushed {sample_idx} samples | Memory: {mem_mb:.1f} MB")
            
            if mem_mb > max_memory_mb:
                print(f"WARNING: Memory usage ({mem_mb:.1f} MB) exceeds threshold ({max_memory_mb} MB)")
                print("Stopping to prevent crash...")
                break
        
        time.sleep(1/sr)
    
    print(f"Stream completed: {sample_idx} samples pushed")
except KeyboardInterrupt:
    print(f"Stream interrupted at {sample_idx} samples")
except Exception as e:
    print(f"Fatal error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
