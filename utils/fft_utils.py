import numpy as np
from scipy.fft import rfft
from scipy.signal.windows import hann, hamming, blackman


def compute_fft(samples, fs=128, window='hann'):
    """
    Compute FFT magnitude spectrum from time-domain samples.
    
    Args:
        samples: 1D array of time-domain samples
        fs: Sampling frequency in Hz
        window: Window type ('hann', 'hamming', 'blackman', or None for rectangular)
    
    Returns:
        dict with keys:
            'frequencies': frequency values in Hz
            'magnitudes': FFT magnitude spectrum
            'n_fft': number of FFT points
    """
    if samples is None or len(samples) == 0:
        return {'frequencies': [], 'magnitudes': [], 'n_fft': 0}
    
    samples = np.asarray(samples, dtype=np.float32)
    n = len(samples)
    
    # Apply window if specified
    if window == 'hann':
        windowed = samples * hann(n)
    elif window == 'hamming':
        windowed = samples * np.hamming(n)
    elif window == 'blackman':
        windowed = samples * np.blackman(n)
    else:
        windowed = samples
    
    # Compute real FFT (returns only positive frequencies)
    fft_result = rfft(windowed)
    magnitudes = np.abs(fft_result) / n  # Normalize
    
    # Frequency axis
    frequencies = np.fft.rfftfreq(n, d=1/fs)
    
    return {
        'frequencies': frequencies.tolist(),
        'magnitudes': magnitudes.tolist(),
        'n_fft': n
    }


def compute_fft_multi_channel(samples_dict, fs=128, window='hann'):
    """
    Compute FFT for multiple channels at once.
    
    Args:
        samples_dict: dict mapping channel_name -> sample_array
        fs: Sampling frequency in Hz
        window: Window type
    
    Returns:
        dict mapping channel_name -> FFT results (frequencies, magnitudes)
    """
    results = {}
    for ch_name, samples in samples_dict.items():
        results[ch_name] = compute_fft(samples, fs, window)
    return results


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    Used for coherence estimation between FFT magnitudes.
    """
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))
