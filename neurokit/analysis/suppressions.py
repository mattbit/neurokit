"""
This module calculates the IES Suppression by taking average of the frontal
electrodes.
"""

import math
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import binary_opening

from ..io import Recording
from ..utils import mask_to_intervals
from ..preprocessing.filter import bandpass


def detect_ies(recording: Recording, channels=None, threshold=8.,
               min_duration=1.):
    """Extract the IES from a Recording.

    @todo See: the paper

    Parameters
    ----------
    recording : neurokit.io.Recording
        The merged recording information in the form of a Recording.
    channels : collection.abc.Sequence
        The channels to consider while calculating IES.
    threshold: float, optional
        The threshold value.
    min_duration: float, optional
        Minimum duration of a suppression in seconds.

    Returns
    -------
    detections : pandas.DataFrame
        The extracted detections of iso-electric suppressions.
    """
    if not channels:
        channels = recording.channels

    avg = recording.data.loc[:, channels].values.mean(axis=1)
    min_length = math.ceil(min_duration * recording.frequency)
    ies_mask = binary_opening(np.abs(avg) < threshold, np.ones(min_length))

    intervals = mask_to_intervals(ies_mask, recording.data.index)
    detections = [{'start': start,
                   'end': end,
                   'channel': None,
                   'description': 'IES'}
                  for start, end in intervals]

    return pd.DataFrame(detections)


def detect_alpha_suppression(recording: Recording, channels=None, frequency_band=None):
    """Extract Alpha Suppression from recording
    Parameters
    ----------
    recording: neurokit.io.Recording
        The merged recording information in the form of a Recording
    channels: collection.abc.Sequence
        The channels to consider while calculating alpha suppressions
    frequency_band:  collection.abc.Sequence
        The frequency band to preserve for filtering in the form of [minFrequency, maxFrequency]

    Returns
    -------
    detections : pandas.DataFrame
        the extracted durations of alpha suppression
    """
    if not channels:
        channels = recording.channels
    if not frequency_band:
        frequency_band = np.asarray([8, 16])
    rec = recording.copy()
    rec.data = recording.data.loc[:, channels]
    rec = bandpass(rec, frequency_band)
    data = rec.data
    detections = []
    for column in data.columns:
        mask = _process(data.loc[:, [column]], rec.start_date, rec.frequency, column)
        intervals = mask_to_intervals(mask, recording.data.index)
        detection = [{'start': start,
                      'end': end,
                      'channel': column,
                      'description': 'alpha suppression'}
                     for start, end in intervals]
        detections.append(detection)
    return detections


def _process(signal, start_date, fs, channel_name, window='1s', r=1.4,
             threshold=0.25, win=1.2):
    """

    Parameters
    ----------
    signal : pandas.DataFrame
        data for the channel
    start_date : pandas.Timestamp
        start date of the recording
    fs : float
        Sampling frequency
    channel_name: str
        name of the channel
    window : str
        duration for calculation of window
    r : float
        threshold for finding maximums and minimums
    threshold : float
        threshold value for thresholding normalized signal
    win :
        window for morphological operation

    Returns
    -------
    detections : numpy.array
        boolean array of alpha suppression detections
    """
    dx = signal.copy()
    dx['positive'] = signal.values > 0
    time = (signal.index - start_date).total_seconds()
    dx['time'] = time
    dx['group_labels'] = dx['positive'].diff().fillna(0).abs().cumsum()
    groups = dx.groupby('group_labels', as_index=False).agg({'positive': 'first',
                                                             channel_name: lambda x: x.abs().max(),
                                                             'time': 'mean'})
    amplitudes = groups.groupby(groups.positive.cumsum(), as_index=False).agg({'time': 'mean',
                                                                               channel_name: 'sum'})
    amplitudes = amplitudes.rename(columns={channel_name: 'amplitude'})
    amplitudes['time'] = pd.to_datetime(amplitudes['time'] * 10 ** 9 + start_date.value)
    amplitudes.set_index('time', inplace=True)
    amplitudes = amplitudes.rolling(window, min_periods=1).mean()
    amplitudes = amplitudes.rolling(2).apply(lambda x: _criteria(x, r)).fillna(0)
    tp = (amplitudes.index - start_date).total_seconds()
    a = np.interp(time, tp, amplitudes.loc[:, 'amplitude'])
    ti = pd.DataFrame({'time': signal.index, 'amplitude': a})
    ti.set_index('time', inplace=True)
    conv_function = _generate_conv_function(fs)
    ti['enhanced'] = np.convolve(ti.loc[:, 'amplitude'], conv_function, 'same') + 1
    dx['enhanced'] = np.multiply(dx[channel_name].values, ti['enhanced'].values)
    maximums = dx.loc[:, 'enhanced'].abs().rolling(window, min_periods=1).max()
    dx['normalized'] = dx['enhanced'].abs() / maximums
    possible_alpha = dx['normalized'] < threshold
    detections = binary_opening(possible_alpha, structure=np.ones(round(win * fs)))
    return detections


def _criteria(pair, threshold):
    """Returns True for threshold*X < Y

    Parameters
    ----------
    pair : list
        pair of amplitude values
    threshold : float
        threshold value

    Returns
    -------
    boolean : bool
    """
    if (pair[0] * threshold) < pair[1]:
        return True
    else:
        return False


def _generate_conv_function(fs, b=0.72, tb=1, eta=1.96):
    """Enhancement signal

    Parameters
    ----------
    fs : float
        sampling frequency
    b : float
        hyper parameter
    tb : float
        hyper parameter
    eta: float
        hyper parameter

    Returns
    -------
    signal : numpy.array
        window of the enhancement signal
    """
    time_range = np.arange(-3, 5, 1 / fs) / tb
    signal = b * np.multiply(np.float_power(time_range, eta), np.exp(-eta * (time_range - 1)))
    signal[time_range < 0] = 0
    return signal
