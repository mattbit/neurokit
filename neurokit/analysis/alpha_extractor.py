import numpy as np
import scipy.signal as ss
from scipy.ndimage import morphology
from scipy import interpolate
from math import floor
def extract(recording,
            channels = None,
            filter_props = {
                'filter_type' : 'butter',
                'freq_band' : [8,16],
                'order' : 4
            }):
    """ Extract alpha suppressions

    Parameters
    ---------
        recording: object
            Recording object from neurokit framework
        channels : tuple
            all the channels used for extracting alpha bursts
        filter_props : dictionary, optional
            Used to filter the signal, must be specified in complete
    Returns
    ---------
        recording : object
            Recording object with added data for alpha bursts masks
    """
    signals = recording.data.loc[:,channels]
    timeseries = recording.data.index.values.astype(np.float64) / 10**9
    timeseries = timeseries - timeseries[0]
    for (chanName, chanData) in signals.iteritems():
        filtered = _do_filter(chanData,recording.frequency, filter_props)
        time_mask = _detect_max_mins(filtered, recording.frequency)
        enhanced_signal = _enhance_signal(timeseries, filtered, time_mask)
        alpha_suppressions = _normalize(enhanced_signal, recording.frequency)
        repaired = _repairs(alpha_suppressions, recording.frequency)

    recording.data['filtered'] = filtered
    recording.data['enhanced'] = enhanced_signal
    recording.data['alphas'] = repaired
    return recording


def _do_filter(signal, fs, filter_props):
    """ Performs filtering according to filter props

    Parameters
    ----------
        signal : array
            Electrode signal
        fs : double
            Sampling frequency of the signal
        filter props : dictionary
            should contain key "filter_key" with possible values {'butter'}
            should contain key "freq_band" with array of two values specifying
            the low pass and high pass frequencies for 'butter' filter
            should contain key "order" with possible integer values for
            'butter' filter

    Returns
    ---------
        filtered : array
            filtered signal
    """
    if filter_props['filter_type'] == 'butter':
        freq_band = filter_props['freq_band']
        order = filter_props['order']
        sos = ss.butter(order, freq_band, 'bp',fs=fs, output='sos')
        filtered = ss.sosfilt(sos, signal)
        return filtered

def _detect_max_mins(filtered_signal,fs, window = 1, threshold=1.4):
    """ calculates the amplitude of oscillations in the singal in the window

    Parameters
    ----------
    filtered_signal : array
        filtered signal from previous step
    fs : double
        sampling frequency
    window : int
        window size in seconds
	threshold : double
		threshold

    Returns
    ----------
    times : array
        mask for the time indexes, set to true when Amp value is present

    """
    window_len = round(window*fs)
    if window_len%2:
        window_len = window_len+1 # making window lengths even
    idx = int(window_len/2)
    amps = np.zeros(len(filtered_signal))
    times = np.zeros(len(filtered_signal), dtype=np.int64)
    rolling_window = filtered_signal[:window_len] 
    for k in range(idx, len(filtered_signal)-idx):
        amps[k] =_find_amps(rolling_window)
        rolling_window = np.roll(rolling_window,-1)
        rolling_window[-1] = filtered_signal[k+idx]
    shifted_amps = np.roll(amps, 1) * threshold
    times = shifted_amps < amps
    return times

def _find_amps(windowed_signal):
    """ find the average amplitude of the windowed signal

    Parameters
    ---------

    windowed_signal : array
        windowed signal

    Returns
    ---------

    amp : double
        average amplitude of this window

    """
    shifted = np.roll(windowed_signal, -1)
    negs = np.multiply(shifted, windowed_signal)
    negs = negs < 0
    difs = np.abs(windowed_signal - shifted)
    amps = difs[negs]
    return amps[:-1].sum()/len(windowed_signal)

def _enhance_signal(timeseries, signal, time_mask,
                    B=0.72, tb=1, eta = 1.96):
    """ Enhance the amplitude of the transient oscillatory EEG regions
    Parameters
    ---------
    timeseries : array
        timeseries
    signal : array
        filtered signal
    amps : array
        calculated Amp(t)
    time_mask : array
        boolean array with mask of times
    B : double
        constant for amplification function
    tb : int
        constant for amplification function
    eta : int
        constant for amplification function

    Returns
    ---------
    enhanced_signal : array
        enhanced signal
    """
    times = timeseries[time_mask]
    ampfun = np.zeros(len(timeseries))
    for ti in times:
        t = timeseries - ti
        ampfun = ampfun+_func(t, B, tb, eta)
    ampfun = ampfun+1
    enhanced_signal = np.multiply(signal, ampfun)
    return enhanced_signal


def _func(t, B, tb, eta):
    """ calculate the value of amplification function at t
    Parameters
    ----------
    t : double
        time in seconds
    B : double
        constant
    tb : double
        constant
    eta : double
        constant

    Returns
    ---------
    value : double
        value of amplification function at t
    """
    value = B*np.multiply(np.float_power(t/tb, eta),np.exp(-eta*(t/tb - 1)))
    value[t < 0] = 0
    return value

def _normalize(signal, fs, threshold = 0.25 ,window = 1):
    """ Normalize  and perform thresholding on the signal
    Step 4 of the procedure
    
    Parameters 
    ----------
    signal : array 
        Signal
    threshold : float, optional
        Signal threshold
    window : int, optional
        Window len for normalization, 1 second default
        
    Returns 
    ----------
    normalized : array
        boolean array of truth values where we detect alpha suppressions
    """
    window_len = round(window*fs)
    idx = 0
    while (idx < (len(signal)-window_len)):
        maxV = np.max(signal[idx:idx+window_len])
        signal[idx:idx+window_len] = np.abs(signal[idx:idx+window_len]/maxV)
        idx=idx+window_len
    normalized = signal > threshold
    return normalized
   
def _repairs(signal, fs, dilate_window = 0.1, erode_window = 1.1):
    """ repairs the noises in the signal
    parameters
    ----------
    signal : array
        boolean array where alpha supressions are detected
    fs : float
        sampling frequency
    dilate_window : float, optional
        value which is used to dilate the signal (in seconds)
    erode_window : float, optional
        value which is used to erode the signal (in seconds)
        
    Returns
    --------
    repaired : array
        boolean array
    """    
    ### TODO fix this section 
    dilate_win = floor(dilate_window*fs)
    erode_win = floor(erode_window*fs)
    dilated = morphology.binary_dilation(signal, structure = np.ones(dilate_win))
    repaired = morphology.binary_erosion(dilated, structure = np.ones(erode_win))
    return repaired
    