import numpy as np
import scipy.signal as ss


def filter(recording, freq, order=7):
    if np.isscalar(freq):
        vals = _lowpass(recording.data.values, freq, recording.frequency,
                        order=order)
    else:
        vals = _bandpass(recording.data.values, freq, recording.frequency,
                         order=order)

    recording.data.iloc[:, :] = vals

    return recording


def _lowpass(data, filter_freq, freq, order=7):
    critical_freq = 2 * filter_freq / freq
    b, a = ss.butter(order, critical_freq, btype='low')

    return ss.lfilter(b, a, data, axis=0)


def _bandpass(data, band, freq, order=7):
    '''Butterworth's bandpass filter.'''
    low = 2 * band[0] / freq
    high = 2 * band[1] / freq
    b, a = ss.butter(order, (low, high), btype='band')

    return ss.lfilter(b, a, data, axis=0)
