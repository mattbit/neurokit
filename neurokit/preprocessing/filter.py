import numpy as np
import scipy.signal as ss
from ..io import Recording


def bandpass(recording: Recording, freqs):
    '''Bandpass filter.

    Uses a forward-backward Butterworth of order 2 (effective order 4).'''
    rec = recording.copy()
    Wn = np.asarray(freqs) / (0.5 * rec.frequency)
    sos = ss.butter(2, Wn, btype='bandpass', output='sos')
    rec.data.loc[:] = ss.sosfiltfilt(sos, rec.data.values, axis=0)
    return rec


def lowpass(recording: Recording, freq):
    '''Lowpass filter.

    Uses a forward-backward Butterworth of order 2 (effective order 4).'''
    rec = recording.copy()
    Wn = freq / (0.5 * rec.frequency)
    sos = ss.butter(2, Wn, btype='lowpass', output='sos')
    rec.data.loc[:] = ss.sosfiltfilt(sos, rec.data.values, axis=0)
    return rec


def highpass(recording: Recording, freq):
    '''Highpass filter

    Uses a forward-backward Butterworth of order 2 (effective order 4).'''
    rec = recording.copy()
    Wn = freq / (0.5 * rec.frequency)
    sos = ss.butter(2, Wn, btype='highpass', output='sos')
    rec.data.loc[:] = ss.sosfiltfilt(sos, rec.data.values, axis=0)
    return rec


def notch(recording, freq, qf=30):
    '''Notch filter.'''
    rec = recording.copy()
    b, a = ss.iirnotch(freq, qf, rec.frequency)
    rec.data.loc[:] = ss.filtfilt(b, a, rec.data.values, axis=0)
    return rec
