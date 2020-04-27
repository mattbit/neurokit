import numpy as np
import scipy.signal as ss
from typing import Tuple, Sequence

from ..core import TimeSeries


def bandpass(timeseries: TimeSeries,
             freqs: Tuple[float, float],
             channels: Sequence[str] = None,
             order: int = 1) -> TimeSeries:
    """Bandpass filter.


    Parameters
    ----------
    timeseries : TimeSeries
        The timeseries to filter.
    freqs : tuple
        The (low, high) frequency for the filter, in Hz.
    channels : Sequence[str], optional
        The channels that must be filtered. If `None`, the filter will be
        applied to all channels.
    order : int
        Order of the Butterworth filter that will be used. Since the bandpass
        is a combination of two filters and the filtering is done in a
        forward-backward manner, the resulting effective order will be the four
        times this value. Default to 1, corresponding to an effective order 4.

    Returns
    -------
    timeseries : TimeSeries
        The timeseries with bandpass-filtered signals.
    """
    if channels is None:
        channels = timeseries.channels

    Wn = np.asarray(freqs) / (0.5 * timeseries.frequency)
    sos = ss.butter(order, Wn, btype='bandpass', output='sos')
    values = timeseries.loc[:, channels].values

    ts = timeseries.copy()
    ts.loc[:, channels] = ss.sosfiltfilt(sos, values, axis=0)

    return ts


def lowpass(timeseries: TimeSeries,
            freq: float,
            channels: Sequence[str] = None,
            order: int = 2) -> TimeSeries:
    """Lowpass filter.

    Parameters
    ----------
    timeseries : TimeSeries
        The timeseries to filter.
    freq : float
        The critical frequency of the filter in Hz.
    channels : Sequence[str], optional
        The channels that must be filtered. If `None`, the filter will be
        applied to all channels.
    order : int
        Order of the Butterworth filter that will be used. The filtering is
        done in a forward-backward manner, so the resulting effective order
        will be the the double of this value. Default to 2, corresponding to an
        effective order 4.

    Returns
    -------
    timeseries : TimeSeries
        The timeseries with lowpass-filtered signals.
    """
    if channels is None:
        channels = timeseries.channels
    ts = timeseries.copy()
    Wn = freq / (0.5 * ts.frequency)
    sos = ss.butter(order, Wn, btype='lowpass', output='sos')
    values = ts.loc[:, channels].values
    ts.loc[:, channels] = ss.sosfiltfilt(sos, values, axis=0)
    return ts


def highpass(timeseries: TimeSeries,
             freq: float,
             channels: Sequence[str] = None,
             order: int = 2) -> TimeSeries:
    """Highpass filter.

    Parameters
    ----------
    timeseries : TimeSeries
        The timeseries to filter.
    freq : float
        The critical frequency of the filter in Hz.
    channels : Sequence[str], optional
        The channels that must be filtered. If `None`, the filter will be
        applied to all channels.
    order : int
        Order of the Butterworth filter that will be used. The filtering is
        done in a forward-backward manner, so the resulting effective order
        will be the the double of this value. Default to 2, corresponding to an
        effective order 4.

    Returns
    -------
    timeseries : TimeSeries
        The timeseries with highpass-filtered signals.
    """
    if channels is None:
        channels = timeseries.channels
    ts = timeseries.copy()
    Wn = freq / (0.5 * ts.frequency)
    sos = ss.butter(order, Wn, btype='highpass', output='sos')
    values = ts.loc[:, channels].values
    ts.loc[:, channels] = ss.sosfiltfilt(sos, values, axis=0)
    return ts


def notch(timeseries: TimeSeries,
          freq: float,
          channels: Sequence[str] = None,
          qf: float = 30) -> TimeSeries:
    """Notch filter.

    Parameters
    ----------
    timeseries : TimeSeries
        The timeseries to filter.
    freq : float
        The critical frequency of the filter in Hz.
    channels : Sequence[str], optional
        The channels that must be filtered. If `None`, the filter will be
        applied to all channels.
    qf : float, optional
        The quality factor of the notch filter.

    Returns
    -------
    timeseries : TimeSeries
        The timeseries with highpass-filtered signals.
    """
    if channels is None:
        channels = timeseries.channels
    ts = timeseries.copy()
    b, a = ss.iirnotch(freq, qf, ts.frequency)
    values = ts.loc[:, channels].values
    ts.loc[:, channels] = ss.filtfilt(b, a, values, axis=0)
    return ts
