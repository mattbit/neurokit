"""
This module calculates the suppressions, in particular iso-electric
suppressions (IES) and α-suppressions (suppressions in the alpha band).
"""

import math
import numpy as np
import pandas as pd
from typing import Sequence, Tuple
from scipy.signal import savgol_filter
from scipy.ndimage.morphology import binary_dilation, binary_erosion

from ..core import Recording
from ..utils import mask_to_intervals


def _detect_suppressions(recording: Recording,
                         channels: Sequence = None,
                         threshold: float = None,
                         min_duration: float = 1.,
                         min_gap: float = 0.):
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
    min_gap: float, optional
        Minimum duration to fill gaps in the suppressions.

    Returns
    -------
    detections : numpy.ndarray
        The boolean mask with detections of iso-electric suppressions.

    References
    ----------
    .. [1] Cartailler, Jérôme, et al. "Alpha rhythm collapse predicts
       iso-electric suppressions during anesthesia." Communications Biology
       2.1 (2019).
    """
    if not channels:
        channels = recording.data.channels

    if min_duration < 0:
        raise ValueError('min_duration should be >= 0')

    if min_gap < 0:
        raise ValueError('min_gap should be >= 0')

    if recording.es.has('artifacts'):
        rec = recording.artifacts_to_nan()
    else:
        rec = recording.copy()

    if threshold is None:
        threshold = _find_threshold(rec.data.loc[:, channels])
    envelope = rec.data.loc[:, channels].abs().values.max(axis=1)
    envelope = savgol_filter(envelope, 3, 1)
    with np.errstate(invalid='ignore'):
        ies_mask = envelope < threshold
    min_length = math.floor(min_duration * rec.frequency)
    dilate_len = math.floor((min_duration + min_gap) * rec.frequency)
    gap_len = math.floor(min_gap * rec.frequency)

    if min_length > 0:
        ies_mask = binary_erosion(ies_mask, np.ones(min_length))

    if dilate_len > 0:
        ies_mask = binary_dilation(ies_mask, np.ones(dilate_len))

    if gap_len > 0:
        ies_mask = binary_erosion(ies_mask, np.ones(gap_len))

    return ies_mask


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
        rms_value = np.sqrt(np.nanmean(data.values ** 2))
        threshold = rms_value * 1.25
    return threshold


def _detect_alpha_suppressions(
    recording: Recording,
    channels: Sequence = None,
    frequency_band: Tuple[float, float] = (8., 16.),
    threshold: float = 0.25
):
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
    detections : numpy.ndarray
        The boolean mask with the detected α-suppressions.
    """
    if not channels:
        channels = recording.data.channels
    rec = recording.copy()
    rec.data = recording.data.loc[:, channels]
    channel_rms = np.sqrt(np.mean(rec.data.values**2, axis=0))
    normalized_channels = rec.data.dot(np.diag(1 / channel_rms))
    rec.data.loc[:, :] = normalized_channels.values
    filtered = rec.filter(*frequency_band)
    return _detect_suppressions(filtered, threshold=threshold)


class SuppressionAnalyzer:
    """Detects isoelectric- and α-suppressions in a Recording."""

    def __init__(self, recording: Recording):
        self.recording = recording
        self.ies_detections_ = None
        self.alpha_detections_ = None
        self.ies_mask_ = None
        self.alpha_mask_ = None

    def detect_ies(self, **kwargs):
        self.ies_mask_ = _detect_suppressions(self.recording, **kwargs)
        intervals = mask_to_intervals(
            self.ies_mask_, self.recording.data.index)
        detections = [{'start': start,
                       'end': end,
                       'channel': None,
                       'description': 'IES'}
                      for start, end in intervals]
        self.ies_detections_ = pd.DataFrame(detections)
        return self.ies_detections_

    def detect_alpha_suppressions(
            self,
            channels: Sequence = None,
            frequency_band: Tuple[float, float] = (8., 16.),
            threshold: float = 0.25
    ):
        """Extract Alpha Suppression from Recording.

        Parameters
        ----------
        channels: Sequence
            The channels to consider when calculating α-suppressions.
        frequency_band: tuple[float, float]
            The frequency band used for the detection, in the form
            `(min_freq, max_freq)`. Default is `(8, 16)`.
        threshold: float
            The normalized threshold for evaluating suppression. Default is `0.25`
        Returns
        -------
        detections : pandas.DataFrame
            The detected α-suppressions.
        """
        if self.ies_mask_ is None:
            self.ies_mask_ = _detect_suppressions(
                self.recording, min_duration=2.5)

        rec = self.recording.copy()
        alpha_mask = _detect_alpha_suppressions(
            rec, channels, frequency_band, threshold)

        alpha_mask[self.ies_mask_] = False

        intervals = mask_to_intervals(alpha_mask, self.recording.data.index)
        detections = [{'start': start,
                       'end': end,
                       'channel': None,
                       'description': 'alpha_suppression'}
                      for start, end in intervals]

        self.alpha_detections_ = pd.DataFrame(detections)

        return self.alpha_detections_
