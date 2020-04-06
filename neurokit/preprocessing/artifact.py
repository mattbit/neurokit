import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening, binary_dilation, gaussian_filter1d
from skimage.filters import apply_hysteresis_threshold

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


def detect_signal_artifacts(signal, detectors=None):
    if detectors is None:
        detectors = ['clipped', 'isoelectrical', 'amplitude']

    mask = np.zeros_like(signal, dtype=bool)

    if not isinstance(detectors, dict):
        detectors = {detector: None for detector in detectors}

    for detector, kwargs in detectors.items():
        if kwargs is None:
            kwargs = {}
        if detector == 'clipped':
            mask |= detect_clipped_signal(signal, **kwargs)
        elif detector == 'isoelectrical':
            mask |= detect_isoelectrical_signal(signal, **kwargs)
        elif detector == 'amplitude':
            mask |= detect_high_amplitude_signal(signal, **kwargs)
        else:
            raise Exception(f'Invalid detector "{detector}".')

    return mask


class ArtifactDetector:
    """Detects signal artifacts.
    """
    multi_channel = False

    def detect(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Detection method not implemented")


class ClippingDetector(ArtifactDetector):
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

    def detect(self, signal):
        low_clip, high_clip = self.detect_levels(signal)

        mask = np.zeros(len(signal), dtype=bool)
        if low_clip is not None:
            mask |= signal < low_clip
        if high_clip is not None:
            mask |= signal > high_clip

        return mask

    def detect_levels(self, signal):
        # Guess a sensible number of bins
        valid_signal = signal[~np.isnan(signal)]
        values = np.unique(valid_signal)
        if values.size < 100:
            raise Exception("Not enough unique values to detect clipping.")
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



def detect_isoelectrical_signal(signal, tol=0, opening=10):
    mask = np.abs(np.nan_to_num(np.diff(signal))) <= tol
    if opening > 0:
        mask = binary_opening(mask, iterations=opening)

    return np.append(mask, mask[-1])


def detect_high_amplitude_signal(signal, low=None, high=None):
    abs_signal = np.abs(signal)
    if low is None:
        low = 10
    if high is None:
        high = np.quantile(abs_signal, 0.995)

    return apply_hysteresis_threshold(abs_signal, low, high)
