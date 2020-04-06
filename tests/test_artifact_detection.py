import numpy as np
from pytest import approx
from unittest import mock

from neurokit.preprocessing import ClippingDetector


def test_clipping_levels():
    signal = np.sin(np.arange(1000)) + np.random.normal(0, 0.1, size=1000)
    det = ClippingDetector()

    # No clipping
    low, high = det.detect_levels(signal)
    assert low is None
    assert high is None

    # High clipping
    signal = signal.clip(-np.inf, 0.88)
    low, high = det.detect_levels(signal)

    assert low is None
    assert high == approx(0.88, rel=.005)

    # High and low clipping
    signal = signal.clip(-0.6, 0.88)
    low, high = det.detect_levels(signal)

    assert low == approx(-0.6, rel=.005)
    assert high == approx(0.88, rel=.005)


def test_clipping_detector():
    signal = np.zeros(100)
    clipped_mask = np.random.choice([True, False], size=100)
    signal[clipped_mask] = np.random.choice([10, 5, -10, -2],
                                            size=clipped_mask.sum())

    with mock.patch.object(ClippingDetector, 'detect_levels',
                           return_value=(-1, 1)) as mock_detect_levels:
        det = ClippingDetector()
        detected_mask = det.detect(signal)

    mock_detect_levels.assert_called_once_with(signal)
    assert (detected_mask == clipped_mask).all()
