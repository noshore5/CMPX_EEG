from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from utils.signal_utils import live_signal_generator
from utils.coherence_utils import transform, coherence
import numpy as np
import json
import time
import uvicorn
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import sys

# Try to import Claude utils with error handling
try:
    from utils.claude_utils import claude_agent, get_ieee_paper_content
    CLAUDE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Claude AI features disabled due to import error: {e}")
    CLAUDE_AVAILABLE = False
    
    # Create dummy functions
    async def claude_agent(message, signal_data=None):
        return {"response": "Claude AI not available due to configuration error", "status": "error"}
    
    def get_ieee_paper_content():
        return "IEEE paper content not available"
def resource_path(relative_path):
    # Get absolute path to resource, works for dev and PyInstaller .exe
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def copy_resources_from_bundle():
    """Copy resources from PyInstaller bundle to current directory if needed"""
    if hasattr(sys, '_MEIPASS'):
        # We're running from PyInstaller bundle
        current_dir = os.path.dirname(os.path.abspath(sys.executable))
        
        # Copy templates if not exists
        templates_dest = os.path.join(current_dir, "templates")
        if not os.path.exists(templates_dest):
            import shutil
            templates_src = os.path.join(sys._MEIPASS, "templates")
            if os.path.exists(templates_src):
                shutil.copytree(templates_src, templates_dest)
                print(f"[main.py] Copied templates to {templates_dest}")
        
        # Copy static if not exists  
        static_dest = os.path.join(current_dir, "static")
        if not os.path.exists(static_dest):
            import shutil
            static_src = os.path.join(sys._MEIPASS, "static")
            if os.path.exists(static_src):
                shutil.copytree(static_src, static_dest)
                print(f"[main.py] Copied static to {static_dest}")
        
        return current_dir
    return os.path.abspath(".")

def get_static_path():
    """Get the correct static path for FastAPI"""
    if hasattr(sys, '_MEIPASS'):
        # Use the directory where the .exe is located
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        static_path = os.path.join(exe_dir, "static")
        if os.path.exists(static_path):
            return static_path
        # Fallback to bundle path
        return os.path.join(sys._MEIPASS, "static")
    return "static"

# Create FastAPI app instance
app = FastAPI()

# Copy resources from bundle if running as .exe
copy_resources_from_bundle()

# Mount static files using get_static_path for PyInstaller compatibility
app.mount("/static", StaticFiles(directory=get_static_path()), name="static")

print("[main.py] Starting Coherent Multiplex FastAPI app...")

# Check Claude configuration at startup
if CLAUDE_AVAILABLE:
    try:
        import os
        claude_key = os.getenv('CLAUDE_API_KEY', '')
        if claude_key:
            print("✅ Claude AI: Available with API key configured")
        else:
            print("⚠️  Claude AI: Module loaded but no API key found")
    except Exception as e:
        print(f"⚠️  Claude AI: Configuration issue - {e}")
else:
    print("❌ Claude AI: Not available due to import/configuration error")

N_SIGNALS = 8
FS = 100
CHUNK_SIZE = 100

class AgentQuery(BaseModel):
    message: str
    signal_data: dict = None

async def coherent_multiplex_agent(user_message: str, signal_data: dict = None):
    """
    AI Agent that provides expert analysis and insights about the Coherent Multiplex signal analysis system.
    """
    return await claude_agent(user_message, signal_data)


def signal_stream():
    N_SIGNALS = 8
    FS = 100
    MAX_POINTS = 256
    frame_rate = 1
    step_counter = 0  # Track steps for coherence calculation
    wavelet_coherence_data = None  # Store latest coherence data
    
    gen = live_signal_generator(N_SIGNALS, FS, frame_rate)
    # Initialize buffer with 256 points
    init_signals = []
    points_needed = MAX_POINTS
    while points_needed > 0:
        chunk = next(gen)
        signals = chunk['signals']
        if signals.shape[1] > points_needed:
            signals = signals[:, :points_needed]
        init_signals.append(signals)
        points_needed -= signals.shape[1]
    buffer = np.concatenate(init_signals, axis=1)
    signal_procedures = [None for _ in range(N_SIGNALS)]
    procedure_names = ['filter', 'normalize', 'smooth', 'detrend', 'amplify', 'clip', 'offset', 'none']
    while True:
        chunk = next(gen)
        signals = chunk['signals']
        fft = chunk['fft']
        # Ensure FFT is non-empty and modulus
        if isinstance(fft, np.ndarray):
            # If complex, take modulus
            if np.iscomplexobj(fft):
                fft_mod = np.abs(fft)
            else:
                fft_mod = fft
        else:
            fft_mod = np.array(fft)
            if np.iscomplexobj(fft_mod):
                fft_mod = np.abs(fft_mod)
        buffer = np.concatenate([buffer, signals], axis=1)
        if buffer.shape[1] > MAX_POINTS:
            buffer = buffer[:, -MAX_POINTS:]

        # Compute pairwise cosine similarities between FFT outputs
        # fft_mod shape: (N_SIGNALS, Nfft)
        distances = []
        cosine_similarities = []
        pair_indices = []
        for i in range(N_SIGNALS):
            for j in range(i + 1, N_SIGNALS):
                # Cosine similarity: 1 - cosine(fft_mod[i], fft_mod[j])
                a = fft_mod[i]
                b = fft_mod[j]
                dot = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                cosine_sim = float(dot) / (float(norm_a) * float(norm_b) + 1e-8)
                # Convert similarity to a 'distance' (higher = more different)
                dist = float(1.0 - cosine_sim)
                distances.append(dist)
                cosine_similarities.append(cosine_sim)
                pair_indices.append((i, j))
        # Every 10 steps, randomly assign a new procedure to one signal
        if step_counter % 10 == 0:
            idx = np.random.randint(0, N_SIGNALS)
            # Pick a random procedure
            proc = np.random.choice(procedure_names)
            signal_procedures[idx] = proc

        # Calculate wavelet coherence every 2 steps for highest similarity pair
        step_counter += 1
        if step_counter % 5 == 0:
            try:
                # Find pair with highest cosine similarity
                max_sim_idx = np.argmax(cosine_similarities)
                i, j = pair_indices[max_sim_idx]
                
                # Get the signals for the most similar pair
                signal1 = buffer[i, :]
                signal2 = buffer[j, :]
                
                # Calculate wavelet transform for both signals
                coeffs1, freqs = transform(signal1, FS, highest=FS/2, lowest=2.0, nfreqs=200)
                coeffs2, _ = transform(signal2, FS, highest=FS/2, lowest=2.0, nfreqs=200)

                # Calculate wavelet coherence
                coh, _, xwt = coherence(coeffs1, coeffs2, freqs)

                # Extract phase from cross-wavelet transform
                phases = np.angle(xwt)  # Phase in radians

                # Calculate cone of influence (COI)
                # For Morlet wavelet, COI is usually sqrt(2) * scale (or period) from each edge
                # Here, we approximate scale as 1/frequency
                N_time = coh.shape[1] if hasattr(coh, 'shape') else len(coh[0])
                coi = np.zeros((len(freqs), N_time))
                for t in range(N_time):
                    edge_dist = min(t, N_time - t - 1)
                    for k, f in enumerate(freqs):
                        scale = 1.0 / f
                        # COI mask: 1 if inside cone, 0 if outside
                        coi[k, t] = 1 if edge_dist >= np.sqrt(2) * scale else 0

                # Store coherence data (convert to serializable format)
                wavelet_coherence_data = {
                    "coherence": coh.tolist() if hasattr(coh, 'tolist') else coh,
                    "phases": phases.tolist() if hasattr(phases, 'tolist') else phases,
                    "freqs": freqs.tolist() if hasattr(freqs, 'tolist') else freqs,
                    "coi": coi.tolist(),
                    "pair": [int(i), int(j)],
                    "pair_labels": [chr(65 + i), chr(65 + j)],  # Convert to A, B, C, etc.
                    "similarity": float(cosine_similarities[max_sim_idx])
                }
            except Exception as e:
                print(f"Error calculating wavelet coherence: {e}")
                wavelet_coherence_data = None

        data = {
            "signals": buffer.tolist(),
            "fft": fft_mod.tolist(),
            "distances": distances,
            "wavelet_coherence": wavelet_coherence_data,
            "timestamp": time.time()
        }
        yield f"data: {json.dumps(data)}\n\n"
        time.sleep(1 / frame_rate)


@app.post("/agent")
async def ask_agent(query: AgentQuery):
    """
    Endpoint to interact with the Coherent Multiplex AI agent.
    Accepts user questions and optional signal data for analysis.
    """
    try:
        if CLAUDE_AVAILABLE:
            result = await claude_agent(query.message, query.signal_data)
        else:
            result = {
                "response": "Claude AI agent is not available. Please check the configuration and API key.",
                "status": "error"
            }
        return result
    except Exception as e:
        return {
            "response": f"Agent error: {str(e)}",
            "status": "error"
        }

@app.get("/ieee-paper")
async def get_ieee_paper():
    """
    Endpoint to retrieve the IEEE paper content.
    """
    try:
        paper_content = get_ieee_paper_content()
        return {
            "content": paper_content,
            "status": "success"
        }
    except Exception as e:
        return {
            "content": f"Error retrieving paper: {str(e)}",
            "status": "error"
        }

@app.get("/version")
async def get_version():
    """Version endpoint to verify deployment"""
    try:
        with open("version_info.json", "r") as f:
            version_info = json.load(f)
        return version_info
    except FileNotFoundError:
        return {
            "error": "version_info.json not found",
            "deployment_status": "unknown",
            "timestamp": int(time.time())
        }

@app.get("/health")
async def health_check():
    """Health check endpoint for DigitalOcean"""
    claude_status = "available" if CLAUDE_AVAILABLE else "unavailable"
    return {
        "status": "healthy",
        "claude_ai": claude_status,
        "timestamp": int(time.time())
    }

@app.get("/", response_class=HTMLResponse)
async def index():
    # Try to read from extracted location first, then from bundle
    template_paths = []
    if hasattr(sys, '_MEIPASS'):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        template_paths.append(os.path.join(exe_dir, "templates", "coherentmultiplex.html"))
        template_paths.append(os.path.join(sys._MEIPASS, "templates", "coherentmultiplex.html"))
    else:
        template_paths.append("templates/coherentmultiplex.html")
    
    for template_path in template_paths:
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                html = f.read()
            return HTMLResponse(content=html)
        except FileNotFoundError:
            continue
    
    # If we get here, no template was found
    return HTMLResponse(content="<h1>Error: Template file not found</h1>", status_code=500)

@app.get("/stream")
async def stream():
    return StreamingResponse(signal_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import sys
    import traceback
    try:
        print("=" * 60)
        print("🚀 COHERENT MULTIPLEX SERVER STARTING...")
        print("=" * 60)
        print("📡 Server will start on: http://127.0.0.1:800")
        print("🌐 Open this URL in your web browser to use the app")
        print("⚠️  KEEP THIS WINDOW OPEN while using the application")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        uvicorn.run(app, host="0.0.0.0", port=8080)
        print("[main.py] Server exited successfully.")
    except KeyboardInterrupt:
        print("\n[main.py] Server stopped by user (Ctrl+C)")
    except Exception as e:
        print("[main.py] ERROR during startup:")
        traceback.print_exc()
        input("\nPress Enter to close this window...")
        sys.exit(1)