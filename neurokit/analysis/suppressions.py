"""
This module calculates the IES Suppression by taking average of the frontal
electrodes.
"""

import math
import numpy as np
import pandas as pd
import scipy.signal as ss
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
    return rec
