"""
This module calculates the IES Suppression by taking average of the frontal
electrodes.
"""

import math
import numpy as np
import pandas as pd
from typing import Sequence, Tuple
from scipy.ndimage.morphology import binary_opening
from scipy.ndimage.morphology import binary_dilation

from ..io import Recording
from ..utils import mask_to_intervals
from ..utils import intervals_to_mask
from ..preprocessing.filter import bandpass
from ..preprocessing.artifact import detect_artifacts


def detect_ies(recording: Recording,
               channels: Sequence = None,
               threshold: float = 8.,
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
    rec = _eliminate_artifacts(recording)
    envelope = recording.data.loc[:, channels].abs().values.max(axis=1)
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
        The deteted α-suppressions.
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
    return detect_ies(filtered, threshold=threshold)


def _eliminate_artifacts(recording: Recording, min_duration: float = 0.5):
    """Sets detected artifacts to np.nan.

    Parameters
    ----------
    recording : neurokit.io.Recording
        Recording of the EEG signal.
    min_duration : float, optional
        Minimum duration for dilation (0.5 second).

    Returns
    -------
    rec : neurokit.io.Recording
    """
    rec = recording.copy()
    artifacts_intervals = detect_artifacts(rec, detectors={"amplitude"})
    artifacts_mask = intervals_to_mask(
        artifacts_intervals.loc[:, ['start', 'end']].values, rec.data.index)
    dilated = binary_dilation(artifacts_mask, structure=np.ones(
        round(rec.frequency * min_duration))).astype(bool)
    rec.data.loc[dilated] = np.nan
    return rec
