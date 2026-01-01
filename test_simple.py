import multiprocessing as mp
mp.set_start_method("spawn", force=True)

print("Start method:", mp.get_start_method())

from pylsl import StreamInfo, StreamOutlet
import numpy as np
import time
import gc
import signal
import sys
import psutil
import os

sr = 64
running = True

def signal_handler(sig, frame):
    global running
    print("\nShutting down gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

info = StreamInfo(
    name='EEG_SIMPLE',
    type='EEG',
    channel_count=8,
    nominal_srate=sr,
    channel_format='float32',
    source_id='fake_eeg_simple'
)

# Add channel labels
labels = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4']
ch = info.desc().append_child("channels")
for lab in labels:
    ch.append_child("channel").append_child_value("label", lab)

outlet = StreamOutlet(info)

# Memory monitoring
process = psutil.Process(os.getpid())
max_memory_mb = 1000

t = 0
try:
    while running:
        try:
            # Simple sine wave without numpy array creation
            freq = 10
            sample = [float(np.sin(2*np.pi*freq*t/sr)) for _ in range(8)]
            outlet.push_sample(sample)
            t += 1
        except Exception as e:
            print(f"Error pushing sample {t}: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Aggressive garbage collection
        if t % 100 == 0:
            gc.collect()
        
        # Monitor memory
        if t % 500 == 0:
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"Pushed {t} samples | Memory: {mem_mb:.1f} MB")
            
            if mem_mb > max_memory_mb:
                print(f"WARNING: Memory usage ({mem_mb:.1f} MB) exceeds threshold")
                break
        
        time.sleep(1/sr)
    
    print(f"Stream completed: {t} samples pushed")
except KeyboardInterrupt:
    print(f"Stream interrupted at {t} samples")
except Exception as e:
    print(f"Fatal error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
