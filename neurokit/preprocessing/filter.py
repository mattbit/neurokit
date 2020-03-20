import numpy as np
import scipy.signal as ss
from typing import Tuple, Sequence

from ..io import Recording


def bandpass(recording: Recording,
             freqs: Tuple[float, float],
             channels: Sequence[str] = None) -> Recording:
    """Bandpass filter.

    Uses a forward-backward Butterworth of order 2 (effective order 4).

    Parameters
    ----------
    recording : Recording
        The recording to filter.
    freqs : tuple
        The (low, high) frequency for the filter, in Hz.
    channels : Sequence[str], optional
        The channels that must be filtered. If `None`, the filter will be
        applied to all channels.

    Returns
    -------
    recording : Recording
        The recording with bandpass-filtered signals.
    """
    if channels is None:
        channels = recording.channels
    rec = recording.copy()
    Wn = np.asarray(freqs) / (0.5 * rec.frequency)
    sos = ss.butter(2, Wn, btype='bandpass', output='sos')
    values = rec.data.loc[:, channels].values
    rec.data.loc[:, channels] = ss.sosfiltfilt(sos, values, axis=0)
    return rec


def lowpass(recording: Recording,
            freq: float,
            channels: Sequence[str] = None) -> Recording:
    """Lowpass filter.

    Uses a forward-backward Butterworth of order 2 (effective order 4).

    Parameters
    ----------
    recording : Recording
        The recording to filter.
    freq : float
        The critical frequency of the filter in Hz.
    channels : Sequence[str], optional
        The channels that must be filtered. If `None`, the filter will be
        applied to all channels.

    Returns
    -------
    recording : Recording
        The recording with lowpass-filtered signals.
    """
    if channels is None:
        channels = recording.channels
    rec = recording.copy()
    Wn = freq / (0.5 * rec.frequency)
    sos = ss.butter(2, Wn, btype='lowpass', output='sos')
    values = rec.data.loc[: channels].values
    rec.data.loc[:, channels] = ss.sosfiltfilt(sos, values, axis=0)
    return rec


def highpass(recording: Recording,
             freq: float,
             channels: Sequence[str] = None) -> Recording:
    """Highpass filter.

    Uses a forward-backward Butterworth of order 2 (effective order 4).

    Parameters
    ----------
    recording : Recording
        The recording to filter.
    freq : float
        The critical frequency of the filter in Hz.
    channels : Sequence[str], optional
        The channels that must be filtered. If `None`, the filter will be
        applied to all channels.

    Returns
    -------
    recording : Recording
        The recording with highpass-filtered signals.
    """
    if channels is None:
        channels = recording.channels
    rec = recording.copy()
    Wn = freq / (0.5 * rec.frequency)
    sos = ss.butter(2, Wn, btype='highpass', output='sos')
    values = rec.data.loc[: channels].values
    rec.data.loc[:, channels] = ss.sosfiltfilt(sos, values, axis=0)
    return rec


def notch(recording: Recording,
          freq: float,
          channels: Sequence[str] = None,
          qf: float = 30) -> Recording:
    """Notch filter.

    Parameters
    ----------
    recording : Recording
        The recording to filter.
    freq : float
        The critical frequency of the filter in Hz.
    channels : Sequence[str], optional
        The channels that must be filtered. If `None`, the filter will be
        applied to all channels.
    qf : float, optional
        The quality factor of the notch filter.

    Returns
    -------
    recording : Recording
        The recording with highpass-filtered signals.
    """
    if channels is None:
        channels = recording.channels
    rec = recording.copy()
    b, a = ss.iirnotch(freq, qf, rec.frequency)
    values = rec.data.loc[:, channels].values
    rec.data.loc[:, channels] = ss.filtfilt(b, a, values, axis=0)
    return rec
