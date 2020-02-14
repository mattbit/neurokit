import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening
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

    return pd.DataFrame(artifacts, columns=['start', 'end', 'channel', 'description'])


def detect_signal_artifacts(signal, detectors=['clipped', 'isoelectrical', 'amplitude']):
    mask = np.zeros_like(signal, dtype=bool)

    if not isinstance(detectors, dict):
        detectors = {detector: None for detector in detectors}

    for detector, kwargs in detectors.items():
        if kwargs is None:
            kwargs = dict()
        if detector == 'clipped':
            mask |= detect_clipped_signal(signal, **kwargs)
        elif detector == 'isoelectrical':
            mask |= detect_isoelectrical_signal(signal, **kwargs)
        elif detector == 'amplitude':
            mask |= detect_high_amplitude_signal(signal, **kwargs)
        else:
            raise Exception(f'Invalid detector "{detector}".')

    return mask


def detect_clipped_signal(signal):
    hist, edges = np.histogram(signal[~np.isnan(signal)],
                               bins=65536, density=True)

    mask = np.ones(len(signal), dtype=bool)
    if hist[0] > hist[:50].mean():
        mask &= signal > edges[1]

    if hist[-1] > hist[:-50].mean():
        mask &= signal < edges[-2]

    return ~mask


def detect_isoelectrical_signal(signal, tol=0, opening=10):
    mask = np.abs(np.diff(signal)) <= tol
    if opening > 0:
        mask = binary_opening(mask, iterations=opening)

    return np.append(mask, mask[-1])


def detect_high_amplitude_signal(signal, low=None, high=None):
    if low is None:
        low = np.sqrt((signal ** 2).mean())
    if high is None:
        high = 3 * np.sqrt((signal ** 2).mean())

    abs_signal = np.abs(signal)

    return apply_hysteresis_threshold(abs_signal, low, high)
