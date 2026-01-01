import os
import json
import asyncio
import threading
import time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
import xml.etree.ElementTree as ET
from utils.fft_utils import compute_fft_multi_channel, cosine_similarity
from utils.coherence_utils import transform, coherence

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow browser requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load heavy imports to avoid fork issues with gunicorn
_brain_model_initialized = False
_vertices = None
_faces = None
_transformed_positions = None

def _init_brain_model():
    """Lazy initialization of brain model - called on first use"""
    global _brain_model_initialized, _vertices, _faces, _transformed_positions
    
    if _brain_model_initialized:
        return
    
    import mne
    import numpy as np
    from mne.transforms import apply_trans
    
    # Define the output path for the brain model
    output_dir = "static"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "brain_model.obj")

    # Fetch the sample dataset
    sample_path = mne.datasets.sample.data_path(verbose=False)

    # Define the paths to the left and right hemisphere surfaces
    lh_surface_path = os.path.join(sample_path, "subjects", "sample", "surf", "lh.pial")
    rh_surface_path = os.path.join(sample_path, "subjects", "sample", "surf", "rh.pial")

    # Read the brain surfaces
    lh_vertices, lh_faces = mne.read_surface(lh_surface_path)
    rh_vertices, rh_faces = mne.read_surface(rh_surface_path)

    # Combine the vertices and faces of both hemispheres
    rh_faces += len(lh_vertices)  # Adjust right hemisphere face indices
    _vertices = np.vstack([lh_vertices, rh_vertices])
    _faces = np.vstack([lh_faces, rh_faces])

    # Export the combined brain surface to an OBJ file
    with open(output_file, "w") as f:
        for vertex in _vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in _faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"Full brain model from sample dataset exported to {output_file}")

    # Define EEG node positions in MNI space
    mni_positions = {
        'Fp1': [-2, 10, 8],
        'Fp2': [2, 10, 8],
        'Cz': [0, 0, 12],
    }

    # Load the transformation matrix to match the model's space
    trans = mne.transforms.Transform(fro='mri', to='head')

    # Transform EEG node positions to the model's space
    transformed_positions = {
        label: apply_trans(trans, np.array(pos)) for label, pos in mni_positions.items()
    }

    # Convert transformed positions to lists for JSON serialization
    _transformed_positions = {
        label: pos.tolist() for label, pos in transformed_positions.items()
    }

    # Export transformed positions for visualization
    output_positions_file = os.path.join(output_dir, "eeg_positions.json")
    with open(output_positions_file, "w") as f:
        json.dump(_transformed_positions, f)

    print(f"Transformed EEG positions exported to {output_positions_file}")
    _brain_model_initialized = True

# Mount the static directory to serve static files (excluding brain_model.obj which has custom endpoint)
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)
# Only mount other static files, brain_model.obj is handled by custom endpoint above
# We'll still use StaticFiles for other assets

# Mount the templates directory to serve HTML files
templates_dir = "templates"
os.makedirs(templates_dir, exist_ok=True)
app.mount("/templates", StaticFiles(directory=templates_dir), name="templates")

# Mount static directory last (after our custom endpoints)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve the brain viewer HTML file
@app.get("/")
async def serve_brain_viewer():
    with open(os.path.join(templates_dir, "brain_viewer.html"), "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# Custom endpoint to serve the brain model file with proper streaming
@app.get("/static/brain_model.obj")
async def serve_brain_model():
    """Serve the brain model OBJ file with proper streaming and headers"""
    try:
        _init_brain_model()  # Lazy initialize brain model
        output_dir = "static"
        file_path = os.path.join(output_dir, "brain_model.obj")
        if not os.path.exists(file_path):
            return {"error": "Brain model not found"}, 404
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Return file with proper headers
        return FileResponse(
            file_path,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "inline; filename=brain_model.obj",
                "Content-Type": "text/plain",
                "Content-Length": str(file_size)
            }
        )
    except Exception as e:
        print(f"Error serving brain model: {e}")
        return {"error": str(e)}, 500

# LSL stream management
class LSLStreamManager:
    def __init__(self):
        # map stream_name -> {"inlet": StreamInlet, "channel_count": int, "channels": [labels]}
        self.active_inlets = {}
        self.listening = False
        self.discovery_thread = None
    
    def discover_streams(self):
        """Discover available LSL streams on the local network"""
        try:
            from pylsl import resolve_streams
            streams = resolve_streams()
            return [
                {
                    "name": stream.name(),
                    "type": stream.type(),
                    "hostname": stream.hostname(),
                    "channel_count": stream.channel_count(),
                }
                for stream in streams
            ]
        except Exception as e:
            print(f"Error discovering streams: {e}")
            return []
    
    def start_stream(self, stream_name):
        """Connect to a specific LSL stream"""
        from pylsl import resolve_streams, StreamInlet
        
        if stream_name in self.active_inlets:
            meta = {
                "channel_count": self.active_inlets[stream_name].get("channel_count"),
                "channels": self.active_inlets[stream_name].get("channels"),
            }
            return {"status": "already_connected", "stream": stream_name, **meta}
        
        try:
            streams = resolve_streams()
            matching_streams = [s for s in streams if s.name() == stream_name]
            
            if not matching_streams:
                return {"status": "error", "message": f"Stream '{stream_name}' not found"}
            
            inlet = StreamInlet(matching_streams[0])
            # extract channel info from StreamInfo XML if available
            try:
                info = inlet.info()
                channel_count = info.channel_count()
                channels = []
                try:
                    xml = info.as_xml()
                    root = ET.fromstring(xml)
                    for ch in root.findall(".//channel"):
                        lab = ch.find("label")
                        if lab is not None and lab.text:
                            channels.append(lab.text)
                except Exception:
                    channels = []
                if not channels:
                    channels = [f"Ch{i+1}" for i in range(channel_count)]
            except Exception:
                channel_count = None
                channels = []

            self.active_inlets[stream_name] = {
                "inlet": inlet,
                "channel_count": channel_count,
                "channels": channels,
            }
            return {"status": "connected", "stream": stream_name, "channel_count": channel_count, "channels": channels}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def stop_stream(self, stream_name):
        """Disconnect from a specific LSL stream"""
        if stream_name in self.active_inlets:
            # optionally close inlet resources here if needed
            del self.active_inlets[stream_name]
            return {"status": "disconnected", "stream": stream_name}
        return {"status": "not_connected", "stream": stream_name}
    
    def get_stream_data(self, stream_name, timeout=0.05):
        """Get the latest sample from a stream"""
        if stream_name not in self.active_inlets:
            return None
        
        inlet_entry = self.active_inlets[stream_name]
        inlet = inlet_entry.get("inlet")
        try:
            sample, timestamp = inlet.pull_sample(timeout=timeout)
            return {
                "data": sample,
                "timestamp": timestamp,
                "channel_count": inlet_entry.get("channel_count"),
                "channels": inlet_entry.get("channels"),
            }
        except:
            return None
    
    def get_all_stream_data(self, stream_name):
        """Get all available buffered samples from a stream"""
        if stream_name not in self.active_inlets:
            return []
        
        inlet_entry = self.active_inlets[stream_name]
        inlet = inlet_entry.get("inlet")
        samples = []
        
        # Pull all available samples in the buffer (non-blocking)
        while True:
            try:
                sample, timestamp = inlet.pull_sample(timeout=0.0)  # Non-blocking
                samples.append({
                    "data": sample,
                    "timestamp": timestamp,
                })
            except:
                break  # No more samples in buffer
        
        return samples
    
    def list_active_streams(self):
        """Return metadata for active streams"""
        out = []
        for name, entry in self.active_inlets.items():
            out.append({
                "name": name,
                "channel_count": entry.get("channel_count"),
                "channels": entry.get("channels"),
            })
        return out

# Initialize the LSL stream manager
lsl_manager = LSLStreamManager()

# API endpoint to discover available LSL streams
@app.get("/api/lsl/discover")
async def discover_lsl_streams():
    """Discover available LSL streams on the network"""
    streams = lsl_manager.discover_streams()
    return {"streams": streams}

# API endpoint to connect to a stream
@app.get("/api/lsl/connect/{stream_name}")
async def connect_to_stream(stream_name: str):
    """Connect to a specific LSL stream"""
    result = lsl_manager.start_stream(stream_name)
    return result

# Add API endpoint to list active streams and their channels
@app.get("/api/lsl/active")
async def active_lsl_streams():
    return {"active_streams": lsl_manager.list_active_streams()}

# API endpoint to disconnect from a stream
@app.get("/api/lsl/disconnect/{stream_name}")
async def disconnect_stream(stream_name: str):
    """Disconnect from a specific LSL stream"""
    result = lsl_manager.stop_stream(stream_name)
    return result

# WebSocket endpoint for streaming real-time data
connected_clients = set()

# Stream buffering for FFT computation
class StreamBuffer:
    def __init__(self, fft_window_size=256):
        self.buffers = {}  # channel_name -> list of samples
        self.fft_window_size = fft_window_size
        self.fs = 128  # Default sampling rate (can be updated per stream)
        self.last_wavelet_computation = 0  # Track last wavelet coherence computation time
        self.wavelet_window_size = 256  # Window for wavelet transform (256 samples)
    
    def add_sample(self, channel_name, value):
        """Add a sample to the buffer for a channel"""
        if channel_name not in self.buffers:
            self.buffers[channel_name] = []
        self.buffers[channel_name].append(float(value))
        
        # Keep buffer size reasonable
        if len(self.buffers[channel_name]) > self.fft_window_size * 2:
            self.buffers[channel_name] = self.buffers[channel_name][-self.fft_window_size:]
        
        # Debug: log when we're getting close to ready
        if len(self.buffers[channel_name]) == self.fft_window_size:
            print(f"StreamBuffer: {channel_name} reached FFT window size ({self.fft_window_size}), is_ready={self.is_ready()}")
    
    def is_ready(self):
        """Check if all channels have enough samples for FFT"""
        if not self.buffers:
            return False
        return all(len(buf) >= self.fft_window_size for buf in self.buffers.values())
    
    def is_ready_for_wavelet(self):
        """Check if all channels have enough samples for wavelet transform"""
        if not self.buffers:
            return False
        return all(len(buf) >= self.wavelet_window_size for buf in self.buffers.values())
    
    def get_fft_data(self):
        """Compute FFT for current buffer and return results"""
        if not self.is_ready():
            return None
        
        # Get the most recent FFT_WINDOW_SIZE samples from each channel
        fft_input = {
            ch: np.array(buf[-self.fft_window_size:], dtype=np.float32)
            for ch, buf in self.buffers.items()
        }
        
        # Compute FFT for all channels
        fft_results = compute_fft_multi_channel(fft_input, fs=self.fs, window='hann')
        print(f"FFT computed: {len(fft_results)} channels, first channel has {len(fft_results[list(fft_results.keys())[0]]['frequencies'])} frequency bins")
        return fft_results
    
    def compute_wavelet_coherence_for_highest_pair(self, fft_data):
        """
        Compute wavelet coherence for the pairing with highest cosine similarity.
        
        Args:
            fft_data: FFT results from get_fft_data()
            
        Returns:
            dict with keys: 'pair', 'ch1', 'ch2', 'coherence', 'freqs' or None
        """
        if not fft_data or len(fft_data) < 2:
            return None
        
        # Find the highest cosine similarity pair from FFT data
        channels = list(fft_data.keys())
        max_similarity = -1
        best_pair = None
        
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                ch1, ch2 = channels[i], channels[j]
                fft1 = np.array(fft_data[ch1]['magnitudes'], dtype=np.float32)
                fft2 = np.array(fft_data[ch2]['magnitudes'], dtype=np.float32)
                
                similarity = cosine_similarity(fft1, fft2)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pair = (ch1, ch2)
        
        if best_pair is None:
            return None
        
        ch1, ch2 = best_pair
        
        # Get wavelet window samples for the best pair
        if (ch1 not in self.buffers or ch2 not in self.buffers or 
            len(self.buffers[ch1]) < self.wavelet_window_size or 
            len(self.buffers[ch2]) < self.wavelet_window_size):
            return None
        
        signal1 = np.array(self.buffers[ch1][-self.wavelet_window_size:], dtype=np.float64)
        signal2 = np.array(self.buffers[ch2][-self.wavelet_window_size:], dtype=np.float64)
        
        try:
            # Compute wavelet transform for both signals
            coeffs1, freqs = transform(signal1, self.fs, highest=50, lowest=0.5, nfreqs=100)
            coeffs2, freqs = transform(signal2, self.fs, highest=50, lowest=0.5, nfreqs=100)
            
            # Compute wavelet coherence
            coh, freqs, S12 = coherence(coeffs1, coeffs2, freqs)
            
            # Handle NaN values
            coh = np.nan_to_num(coh, nan=0.0)
            
            # Convert to list format for JSON serialization
            # Replace any NaN or inf values with 0
            freqs_list = [float(f) if np.isfinite(f) else 0.0 for f in (freqs.tolist() if hasattr(freqs, 'tolist') else list(freqs))]
            coh_list = [[float(v) if np.isfinite(v) else 0.0 for v in row] for row in coh]
            
            print(f"Wavelet coherence computed successfully: shape={coh.shape}, freqs={len(freqs_list)}, coh_rows={len(coh_list)}")
            
            return {
                'pair': f"{ch1}-{ch2}",
                'ch1': ch1,
                'ch2': ch2,
                'coherence': coh_list,
                'freqs': freqs_list,
                'cosine_similarity': float(max_similarity)
            }
        except Exception as e:
            print(f"Error computing wavelet coherence: {e}")
            import traceback
            traceback.print_exc()
            return None

@app.websocket("/ws/lsl")
async def websocket_lsl_endpoint(websocket: WebSocket):
    """WebSocket endpoint to stream LSL data in real-time with FFT computation"""
    await websocket.accept()
    connected_clients.add(websocket)
    
    client_subscriptions = {}  # stream_name -> StreamBuffer
    
    async def receive_messages():
        """Handle incoming messages from client"""
        try:
            while True:
                data = await websocket.receive_json()
                command = data.get("command", "")
                
                if command == "subscribe":
                    stream_name = data.get("stream", "")
                    result = lsl_manager.start_stream(stream_name)
                    
                    if result.get("status") == "connected" or result.get("status") == "already_connected":
                        # Initialize buffer for this stream
                        client_subscriptions[stream_name] = StreamBuffer(fft_window_size=256)
                    
                    await websocket.send_json(result)
                
                elif command == "unsubscribe":
                    stream_name = data.get("stream", "")
                    result = lsl_manager.stop_stream(stream_name)
                    client_subscriptions.pop(stream_name, None)
                    await websocket.send_json(result)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"WebSocket receive error: {e}")
    
    async def send_data():
        """Continuously send data to client with FFT computation"""
        try:
            stream_fft_counters = {}  # Track FFT counter per stream
            stream_wavelet_timers = {}  # Track wavelet computation timers per stream
            while True:
                # Send data for subscribed streams
                for stream_name in list(client_subscriptions.keys()):
                    stream_data = lsl_manager.get_stream_data(stream_name)
                    
                    if stream_data:
                        channels = stream_data.get("channels", [])
                        data_values = stream_data.get("data", [])
                        
                        # Add samples to buffer
                        buffer = client_subscriptions[stream_name]
                        for i, channel_label in enumerate(channels):
                            if i < len(data_values):
                                buffer.add_sample(channel_label, data_values[i])
                        
                        # Initialize counter for this stream if needed
                        if stream_name not in stream_fft_counters:
                            stream_fft_counters[stream_name] = 0
                        
                        # Initialize wavelet timer for this stream if needed
                        if stream_name not in stream_wavelet_timers:
                            stream_wavelet_timers[stream_name] = time.time()
                        
                        # Every N samples, compute and send FFT
                        stream_fft_counters[stream_name] += 1
                        if stream_fft_counters[stream_name] >= 4:  # Send FFT every ~4 samples
                            stream_fft_counters[stream_name] = 0
                            try:
                                fft_results = buffer.get_fft_data()
                                if fft_results:
                                    # Send FFT data to client
                                    await websocket.send_json({
                                        "type": "fft",
                                        "stream": stream_name,
                                        "fft": fft_results
                                    })
                                else:
                                    # Buffer not ready yet
                                    pass
                            except Exception as e:
                                print(f"FFT computation error for stream {stream_name}: {e}")
                        
                        # Check every iteration if 5 seconds have passed for wavelet coherence computation
                        current_time = time.time()
                        if current_time - stream_wavelet_timers[stream_name] >= 5.0:
                            stream_wavelet_timers[stream_name] = current_time
                            try:
                                fft_results = buffer.get_fft_data()
                                if fft_results:
                                    # Compute wavelet coherence for highest similarity pair
                                    wavelet_result = buffer.compute_wavelet_coherence_for_highest_pair(fft_results)
                                    if wavelet_result:
                                        await websocket.send_json({
                                            "type": "wavelet_coherence",
                                            "stream": stream_name,
                                            **wavelet_result
                                        })
                                        print(f"Sent wavelet coherence for {wavelet_result['pair']}")
                                    else:
                                        print(f"Failed to compute wavelet coherence for {stream_name}")
                                else:
                                    print(f"FFT data not ready for wavelet computation for {stream_name}")
                            except Exception as e:
                                print(f"Wavelet coherence computation error for stream {stream_name}: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # Also send raw data for time-domain display
                        await websocket.send_json({
                            "type": "data",
                            "stream": stream_name,
                            **stream_data
                        })
                
                # Small sleep to avoid busy loop
                await asyncio.sleep(1/128)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"WebSocket send error: {e}")
    
    try:
        # Run both tasks concurrently
        await asyncio.gather(
            receive_messages(),
            send_data()
        )
    finally:
        connected_clients.discard(websocket)

# Start background broadcast task
@app.on_event("startup")
async def startup_event():
    pass  # No longer need background broadcast task

# Run the app with Gunicorn (if needed)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)