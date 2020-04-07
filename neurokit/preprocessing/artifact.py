import numpy as np
import pandas as pd
from typing import Tuple
from skimage.filters import apply_hysteresis_threshold
from scipy.ndimage import binary_opening, binary_dilation, gaussian_filter1d

from ..utils import mask_to_intervals


def detect_artifacts(recording, **kwargs):
    artifacts = []
    for ch in recording.channels:
        mask = detect_signal_artifacts(recording.data[ch], **kwargs)

        for start, end in mask_to_intervals(mask, recording.data.index):
            artifacts.append({
                'start': start,
                'end': end,
                'channel': ch,
                'description': 'auto detection'
            })

    return pd.DataFrame(artifacts, columns=['start', 'end', 'channel',
                                            'description'])


def detect_signal_artifacts(signal, detectors=None, pad=0):
    if detectors is None:
        detectors = [HighAmplitudeDetector(),
                     ConstantSignalDetector(),
                     ClippedSignalDetector()]

    mask = np.zeros_like(signal, dtype=bool)

    for detector in detectors:
        mask |= detector.detect(signal)

    if pad > 0:
        return binary_dilation(mask, np.ones(pad))

    return mask


class ArtifactDetector:
    """Detects signal artifacts."""
    multi_channel = False

    def detect(self, signal: np.ndarray) -> np.ndarray:
        """Detect artifacts in the signal.

        Parameters
        ----------
        signal : np.ndarray
            The signal on which detection is performed. If detector is
            `multi_channel`, multidimensional arrays can be used, otherwise a
            1d array is expected.

        Returns
        -------
        detection_mask : np.ndarray
            The boolean mask of the artifact detection.
        """
        raise NotImplementedError("Detection method not implemented")


class ClippedSignalDetector(ArtifactDetector):
    """Detects clipped signal, roughly based on the method described in [1]_.

    Parameters
    ----------
    border_bins : int
        Number of bins to consider to calculate the deviation.
    total_bins : int
        Number of bins used to calculate the histogram.

    References
    ----------
    .. [1] Laguna, Christopher, and Alexander Lerch. "An efficient algorithm
       for clipping detection and declipping audio." Audio Engineering Society
       Convention 141 (2016).
    """

    def detect(self, signal: np.ndarray) -> np.ndarray:
        low_clip, high_clip = self.detect_levels(signal)

        mask = np.zeros(len(signal), dtype=bool)
        if low_clip is not None:
            mask |= signal < low_clip
        if high_clip is not None:
            mask |= signal > high_clip

        return mask

    def detect_levels(self, signal: np.ndarray) -> Tuple[float, float]:
        """Detect clipping levels."""
        # Guess a sensible number of bins
        valid_signal = signal[~np.isnan(signal)]
        values = np.unique(valid_signal)
        if values.size < 100:
            raise ValueError("Not enough unique values to detect clipping.")
        num_bins = min(values.size, 65536)

        # Calculate the histogram
        hist, edges = np.histogram(valid_signal, bins=num_bins)

        # Detect the clipping
        sigma = int(np.round(0.01 * num_bins))
        a, b = 3 * sigma, 10 * sigma
        ref = hist - gaussian_filter1d(hist, sigma, mode='constant')

        _threshold_low = ref[a:b].mean() + 3 * ref[a:b].std()
        _threshold_high = ref[-b:-a].mean() + 3 * ref[-b:-a].std()

        low_level = edges[1] if ref[0] > _threshold_low else None
        high_level = edges[-2] if ref[-1] > _threshold_high else None

        return low_level, high_level


class ConstantSignalDetector(ArtifactDetector):
    """Detects regions with constant signal.

    Non-varying signal may be due to unbranched electrodes or recording
    software glitches.

    Parameters
    ----------
    tol : float
        Tolerance in the difference between subsequent samples, used to detect
        non-varying signals. Default is 0.
    interval : int
        Number of constant valued adjacent samples required to detect an
        artifact. Default is 10, meaning that the signal needs to stay constant
        (variations up to `tol`) for more than 10 consequent samples to be
        considered an artifact.
    """

    def __init__(self, tol=0, interval=10):
        self.tol = tol
        self.interval = interval

    def detect(self, signal: np.ndarray) -> np.ndarray:
        mask = np.abs(np.nan_to_num(np.diff(signal))) <= self.tol
        if self.interval > 0:
            mask = binary_opening(mask, np.ones(self.interval))

        return np.append(mask, mask[-1])


class HighAmplitudeDetector(ArtifactDetector):
    """Detect high amplitude signal with hysteresis thresholding.

    Parameters
    ----------
    low : float
        Lower threshold. If not specified, the median (calculated case by case)
        will be used.
    high : float
        High threshold (the main threshold, amplitude greater than this value
        will be considered artifactual). If not specified, it will be
        determined case by case based on quantile values.
    """

    def __init__(self, low=10, high=200):
        self.low = low
        self.high = high

    def detect(self, signal: np.ndarray) -> np.ndarray:
        abs_signal = np.abs(signal)
        return apply_hysteresis_threshold(abs_signal, self.low, self.high)
