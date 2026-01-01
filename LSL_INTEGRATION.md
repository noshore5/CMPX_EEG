# LSL Stream Integration

## Overview
The nodes in the brain viewer now listen for and visualize LSL (Lab Streaming Layer) streams on the local network.

## Architecture

### Backend (Python/FastAPI)
- **LSL Discovery**: Scans network for available LSL streams
- **Stream Connection**: Connects to selected LSL streams and pulls real-time data
- **WebSocket Broadcasting**: Streams data to browser in real-time (~100 Hz)

### Frontend (JavaScript/WebGL)
- **Stream Discovery UI**: Button to scan for available streams
- **Stream Management**: Dropdown selector and subscribe/unsubscribe controls
- **Live Visualization**: Brain node colors update based on stream data

## Files Modified

### 1. `requirements.txt`
Added:
- `pylsl` - Lab Streaming Layer Python bindings
- `websockets` - WebSocket support for real-time data
- `mne` - MNE-Python (already needed for brain model)

### 2. `main.py`
Added:
- `LSLStreamManager` class: Handles stream discovery and data streaming
- `/api/lsl/discover` - Endpoint to discover available streams
- `/api/lsl/connect/{stream_name}` - Connect to a stream
- `/api/lsl/disconnect/{stream_name}` - Disconnect from a stream
- `/ws/lsl` - WebSocket endpoint for real-time data streaming
- Background task `broadcast_lsl_data()` - Continuously broadcasts stream data to connected clients

### 3. `templates/brain_viewer.html`
Added:
- LSL Streams control section in the control panel
- Discover Streams button to scan network
- Stream selector dropdown
- Subscribe/Unsubscribe buttons
- Active streams list
- WebSocket connection management
- Real-time node visualization updates based on stream data

## Usage

1. **Start the application**:
   ```bash
   source CMPX/bin/activate
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
   # or for development
   python main.py
   ```

2. **Open the brain viewer**: Navigate to `http://localhost:8000`

3. **Discover streams**: Click "Discover Streams" button
   - Lists all available LSL streams on the local network

4. **Connect WebSocket**: Click "Connect WebSocket"
   - Establishes connection to receive real-time data

5. **Subscribe to a stream**:
   - Select a stream from dropdown
   - Click "Subscribe"
   - Stream data will appear in "Active Streams" list
   - Brain nodes will update colors based on stream values

## Stream Data Visualization

The brain nodes respond to LSL stream data:
- First channel value is normalized to 0-1
- Node colors shift from red (high value) to green (low value) based on normalized data
- Updates occur at ~100 Hz

## Testing with Fake Data

A test LSL stream generator exists in `test.ipynb`:
```python
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import time

info = StreamInfo(
    name='FakeEEG',
    type='EEG',
    channel_count=8,
    nominal_srate=250,
    channel_format='float32',
    source_id='fake_eeg_001'
)

outlet = StreamOutlet(info)

t = 0
while True:
    sample = np.sin(2*np.pi*10*t/250) * np.ones(8)
    outlet.push_sample(sample.tolist())
    t += 1
    time.sleep(1/250)
```

Run this in one terminal while the web app runs in another to see live stream visualization.

## Connection Status

The UI displays connection status:
- ✓ Green: Connected and receiving data
- ✗ Red: Error or no connection

## Notes

- LSL discovery has a 2-second timeout by default
- WebSocket updates at ~100 Hz (0.01s interval)
- Multiple streams can be subscribed simultaneously
- Streams are automatically removed from visualization if they stop sending data
