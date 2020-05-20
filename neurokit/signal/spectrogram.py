import numpy as np
import scipy.signal as ss
from typing import Sequence
import warnings

from ..core import TimeSeries


def spectrogram(timeseries: TimeSeries, window=10., overlap=0.5,
                channels: Sequence = None):
    """Produce the spectrogram for a TimeSeries.

    If multiple channels are selected, it will first calculate the spectrogram
    for each channel individually and then average the result.

    Parameters
    ----------
    timeseries : TimeSeries
        The timeseries object.
    window : float, optional
        Length of the window for the STFT, in seconds.
    overlap : float, optional
        Proportion of overlap between windows (must be less than 1).
    channels : Sequence
        List of channels to use. If `None`, all channels will be used.
    """
    if channels is None:
        channels = timeseries.channels

    n_window = int(timeseries.frequency * window)
    n_overlap = int(timeseries.frequency * window * overlap)

    data = timeseries.loc[:, channels].values
    f, t, S = ss.spectrogram(data, fs=timeseries.frequency,
                             nperseg=n_window, noverlap=n_overlap, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        S = np.nanmean(S, axis=1)

    return f, t, S
