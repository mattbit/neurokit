"""
This module calculates the suppressions, in particular iso-electric
suppressions (IES) and α-suppressions (suppressions in the alpha band).
"""

import math
import numpy as np
import pandas as pd
from typing import Sequence, Tuple
from scipy.ndimage.morphology import binary_opening

from ..io import Recording
from ..utils import mask_to_intervals
from ..preprocessing.filters import bandpass


def detect_suppressions(recording: Recording,
                        channels: Sequence = None,
                        threshold: float = None,
                        min_duration: float = 1.):
    """Detect iso-electric suppressions in a Recording.

    The detection procedure is based on the method presented in [1]_.

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

    References
    ----------
    .. [1] Cartailler, Jérôme, et al. "Alpha rhythm collapse predicts
       iso-electric suppressions during anesthesia." Communications Biology
       2.1 (2019).
    """
    if not channels:
        channels = recording.channels

    rec = recording.artifacts_to_nan()
    if threshold is None:
        threshold = _find_threshold(rec.data.loc[:, channels])
    envelope = rec.data.loc[:, channels].abs().values.max(axis=1)
    min_length = math.ceil(min_duration * rec.frequency)
    with np.errstate(invalid='ignore'):
        ies_mask = envelope < threshold
    ies_mask = binary_opening(ies_mask, np.ones(min_length))

    intervals = mask_to_intervals(ies_mask, rec.data.index)
    detections = [{'start': start,
                   'end': end,
                   'channel': None,
                   'description': 'IES'}
                  for start, end in intervals]

    return pd.DataFrame(detections)


def detect_ies(*args, **kwargs):
    """Alias of `detect_suppressions`."""
    return detect_suppressions(*args, **kwargs)


def detect_alpha_suppressions(recording: Recording,
                              channels: Sequence = None,
                              frequency_band: Tuple[float, float] = (8., 16.)):
    """Extract Alpha Suppression from recording.

    Parameters
    ----------
    recording: neurokit.io.Recording
        The Recording object on which detection is performed.
    channels: Sequence
        The channels to consider when calculating α-suppressions.
    frequency_band: tuple[float, float]
        The frequency band used for the detection, in the form
        `(min_freq, max_freq)`. Default is `(8, 16)`.

    Returns
    -------
    detections : pandas.DataFrame
        The detected α-suppressions.
    """
    if not channels:
        channels = recording.channels
    rec = recording.copy()
    rec.data = recording.data.loc[:, channels]
    filtered = bandpass(rec, frequency_band)
    rms_before = np.sqrt(np.mean(rec.data.values**2))
    rms_after = np.sqrt(np.mean(filtered.data.values**2))
    threshold = 8 * rms_after / rms_before

    return detect_suppressions(filtered, threshold=threshold)


def _find_threshold(data: pd.DataFrame, threshold: float = 8.):
    """
    Perform automatic thresholding for the recording signals
    Parameters
    ----------
    data : pandas.DataFrame
        signal data
    threshold : float
        default threshold returned when signal power is normal

    Returns
    -------
    threshold : float
        calculated threshold
    """
    mean_amplitude = data.abs().mean().mean()
    if mean_amplitude < 30:
        threshold = threshold / 1.25
    return threshold
