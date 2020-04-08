import numpy as np
import pandas as pd

from .io import Recording


def simulated_eeg_signal(duration, frequency=250, amplitude=40):
    """Create a synthetic EEG-like signal.

    Parameters
    ----------
    duration : float
        Duration of the recording in seconds.
    frequency : float
        Sampling frequency in Hz. Default is 250 Hz.
    amplitude : float
        Mean amplitude of the signal.

    Returns
    -------
    (times, signal) : Tuple[ndarray, ndarray]
        The `times` in seconds and the `signal`.
    """
    ts = np.arange(0, duration, 1 / frequency)
    freqs = np.fft.rfftfreq(ts.size, 1 / frequency)

    with np.errstate(divide='ignore'):
        freq_signal = 1 / freqs**np.random.normal(1, 0.1)

    freq_signal[0] = 0

    phases = np.exp(1j * 2 * np.pi * np.random.random(size=freq_signal.size))
    mul_noise = np.random.normal(1, 0.1, size=freq_signal.size)
    freq_signal = freq_signal * phases * mul_noise

    signal = np.fft.irfft(freq_signal)

    return ts, signal * amplitude / np.abs(signal).mean()


def simulated_eeg_recording(channels=2, duration=60, frequency=250,
                            amplitude=80):
    """Create a synthetic EEG-like recording.

    Parameters
    ----------
    channels : int | Sequence
        Number of channels or a list with channel names.
    duration : float
        Duration of the recording in seconds.
    frequency : float
        Sampling frequency in Hz. Default is 250 Hz.
    amplitude : float
        Mean amplitude value of the signal.
    """
    if isinstance(channels, int):
        channels = [f'EEG_{i+1}' for i in range(channels)]

    _data = {}
    for channel in channels:
        ts, sig = simulated_eeg_signal(duration, frequency, amplitude)
        _data[channel] = sig

    times = pd.to_datetime('now') + pd.to_timedelta(ts, unit='s')
    data = pd.DataFrame(_data, index=times)
    return Recording(data)
