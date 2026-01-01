import numpy as np
import pyfftw

def generate_signals(n_signals=8, length=120, fs=1):
    t = np.arange(length) / fs
    signals = []
    coherent_pairs = int(round(0.1 * n_signals))  # 10% of pairs will be coherent
    for i in range(n_signals):
        # Mix 2-3 sine waves with random frequencies and phases

        n_components = np.random.randint(2, 4)
        sig = np.zeros_like(t)
        for _ in range(n_components):
            freq = np.random.uniform(5, .5 * fs)  # up to 50% Nyquist
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.5, 2.0)
            # Envelope for fading in/out
            envelope = np.ones_like(t)
            if np.random.rand() < 0.5:
                center = np.random.uniform(0.2, 0.8) * length
                width = np.random.uniform(0.1, 0.3) * length
                envelope = np.exp(-0.5 * ((t*fs - center)/width)**2)
            sig += amp * envelope * np.sin(2 * np.pi * freq * t + phase)

        signals.append(sig)

    # Inject coherence into some random pairs (after all signals are created)
    for _ in range(coherent_pairs):
        i, j = np.random.choice(n_signals, 2, replace=False)
        shared_freq = np.random.uniform(2,6 )
        phase_shift = np.random.uniform(0, np.random.uniform(0, 1))  # Nearly phase-locked
        coherent_signal = np.sin(2 * np.pi * shared_freq * t + phase_shift)
        signals[i] += coherent_signal + np.random.normal(0, 0.3, size=length)
        signals[j] += coherent_signal + np.random.normal(0, 0.3, size=length)

    return np.array(signals)


def live_signal_generator(n_signals, fs, buffer_size=2):
    signal_length = 256
    
    signals_buffer = generate_signals(n_signals=n_signals, length=signal_length, fs=fs).astype(np.float32)
    fft_inputs = [pyfftw.empty_aligned(signal_length, dtype='float32') for _ in range(n_signals)]
    fft_outputs = [pyfftw.empty_aligned(signal_length//2 + 1, dtype='complex64') for _ in range(n_signals)]
    fft_plans = [
        pyfftw.FFTW(fft_inputs[i], fft_outputs[i], flags=["FFTW_MEASURE"], threads=4)
        for i in range(n_signals)
    ]
    params = [
        (np.random.uniform(5, fs),  # up to full Nyquist
         np.random.uniform(0, 2 * np.pi),
         np.random.uniform(0.5, 2.0))
        for _ in range(n_signals)
    ]
    t = signal_length / fs
    step = 0
    while True:
        if step % 60 == 0:
            # randomly choose a pair to share the same parameters
            # This will create a coherent signal between two random signals
            params = [
                (np.random.uniform(5, fs),  # up to full Nyquist
                np.random.uniform(0, 2 * np.pi),
                np.random.uniform(0.5, 2.0))
                for _ in range(n_signals)
            ]
            i, j = np.random.choice(n_signals, 2, replace=False)
            params[i] = (params[j][0], params[j][1], params[j][2])
        # Add two new points per signal
        new_points = []
        for i in range(n_signals):
            freq, phase, amp = params[i]
            envelope = 1.0
            for _ in range(buffer_size):
                val = amp * envelope * np.sin(2 * np.pi * freq * t + phase)
                new_points.append(val)
        signals_buffer = np.roll(signals_buffer, -buffer_size, axis=1)
        signals_buffer[:, -buffer_size:] = np.array(new_points).reshape(n_signals, buffer_size)
        fft_len = signal_length // 2
        fft_results = []
        for i in range(n_signals):
            fft_inputs[i][:] = signals_buffer[i]
            fft_plans[i]()
            fft_out = np.abs(fft_outputs[i])[:fft_len]
            fft_results.append(fft_out)
        fft_results = np.array(fft_results)
        yield {
            'signals': signals_buffer,
            'fft': fft_results
        }
        t += buffer_size / fs
        step += 1
