"""
This module calculates the IES Suppression by taking average of the frontal
electrodes.
"""

import math
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import binary_opening
from scipy.ndimage.morphology import binary_dilation

from ..io import Recording
from ..utils import mask_to_intervals
from ..utils import intervals_to_mask
from ..preprocessing.filter import bandpass
from ..preprocessing.artifact import detect_artifacts


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
    rec = _eliminate_artifacts(recording)
    avg = rec.data.loc[:, channels].values.mean(axis=1)
    min_length = math.ceil(min_duration * rec.frequency)
    ies_mask = binary_opening(np.abs(avg) < threshold, np.ones(min_length))
    intervals = mask_to_intervals(ies_mask, rec.data.index)
    detections = [{'start': start,
                   'end': end,
                   'channel': None,
                   'description': 'IES'}
                  for start, end in intervals]

    return pd.DataFrame(detections)


def detect_alpha_suppressions(recording: Recording, channels=None, frequency_band=(8, 16)):
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
    rec = recording.copy()
    rec.data = recording.data.loc[:, channels]
    filtered = bandpass(rec, frequency_band)
    rms_before = np.sqrt(np.mean(rec.data.loc[:, :].values**2))
    rms_after = np.sqrt(np.mean(filtered.data.loc[:, :].values**2))
    r = rms_after / rms_before
    threshold = 8 * r
    return detect_ies(rec, threshold=threshold)


def _eliminate_artifacts(recording: Recording, min_duration=0.5):
    """Sets detected artifacts to np.nan

    Parameters
    ----------
    recording : neurokit.io.Recording
        Recording of the EEG signal
    min_duration : float, optional
        min duration for dilation (0.5 second)

    Returns
    -------
    rec : neurokit.io.recording
    """
    rec = recording.copy()
    artifacts_intervals = detect_artifacts(rec, detectors={"amplitude"})
    artifacts_mask = intervals_to_mask(artifacts_intervals.loc[:, ['start', 'end']].values, rec.data.index)
    dilated = binary_dilation(artifacts_mask, structure=np.ones(round(rec.frequency * min_duration))).astype(bool)
    di = mask_to_intervals(dilated, rec.data.index)
    print(di)
    rec.data.loc[dilated] = np.nan
    return rec
