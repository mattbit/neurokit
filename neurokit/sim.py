import numpy as np
from typing import Union, Sequence

from .core import Recording, TimeSeries


def simulated_eeg_signal(
        duration: float,
        frequency: float = 250,
        amplitude: float = 40,
        theta: float = 2,
        channels: Union[int, Sequence[str]] = 1
) -> TimeSeries:
    """Create a synthetic EEG-like signal.

    The signal power is distributed as 1/f^θ

    Parameters
    ----------
    duration : float
        Duration of the recording in seconds.
    frequency : float
        Sampling frequency in Hz. Default is 250 Hz.
    amplitude : float
        Mean amplitude of the signal.
    theta : float
        Power decay with frequency, as 1/f^θ.

    Returns
    -------
    timeseries : TimeSeries
        The simulated signal.
    """
    if isinstance(channels, int):
        channels = [f'EEG_{i+1}' for i in range(channels)]

    ts = np.arange(0, duration, 1 / frequency)
    freqs = np.fft.rfftfreq(ts.size, 1 / frequency)

    data = np.zeros((len(ts), len(channels)))
    for n_ch in range(len(channels)):
        with np.errstate(divide='ignore'):
            freq_signal = 1 / freqs**(0.5 * theta * np.random.normal(1, 0.05))
        freq_signal[0] = 0
        phases = np.exp(1j * 2 * np.pi
                        * np.random.random(size=freq_signal.size))
        mul_noise = np.random.normal(1, 0.1, size=freq_signal.size)
        freq_signal = freq_signal * phases * mul_noise
        signal = np.fft.irfft(freq_signal)
        data[:, n_ch] = signal * amplitude / np.abs(signal).mean()

    return TimeSeries(data, columns=channels, frequency=frequency,
                      name='SimEEG')


def simulated_eeg_recording(
        duration: float = 60,
        frequency: float = 250,
        amplitude: float = 80,
        theta: float = 2,
        channels: float = 2,
) -> Recording:
    """Create a synthetic EEG-like recording.

    Parameters
    ----------
    duration : float
        Duration of the recording in seconds.
    frequency : float
        Sampling frequency in Hz. Default is 250 Hz.
    amplitude : float
        Mean amplitude value of the signal.
    theta : float
        Power decay with frequency, as 1/f^θ.
    channels : int | Sequence
        Number of channels or a list with channel names.

    Returns
    -------
    recording : Recording
        Simulated EEG recording.
    """
    data = simulated_eeg_signal(
        duration, frequency, amplitude, theta, channels)

    return Recording(data, name='Simulated EEG')
