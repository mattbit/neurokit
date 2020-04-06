import numpy as np
from pytest import approx
from unittest import mock

from neurokit.preprocessing import (ClippedSignalDetector,
                                    ConstantSignalDetector,
                                    HighAmplitudeDetector)


def test_clipping_levels():
    signal = np.sin(np.arange(1000)) + np.random.normal(0, 0.1, size=1000)
    det = ClippedSignalDetector()

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

    with mock.patch.object(ClippedSignalDetector, 'detect_levels',
                           return_value=(-1, 1)) as mock_detect_levels:
        det = ClippedSignalDetector()
        detected_mask = det.detect(signal)

    mock_detect_levels.assert_called_once_with(signal)
    assert (detected_mask == clipped_mask).all()


def test_constant_signal_detector():
    signal = np.cos(np.arange(100))
    signal[10:21] = signal[10]
    signal[30:41] = 1.23
    signal[50:53] = -4
    signal[80:] = 3

    det = ConstantSignalDetector(interval=10)
    detected_mask = det.detect(signal)

    assert detected_mask[10:20].all()
    assert detected_mask[30:40].all()
    assert detected_mask[80:100].all()
    assert detected_mask.sum() == 40


def test_high_amplitude_detector():
    signal = 80 * np.cos(0.25 * np.arange(100))
    signal[30:40] = 240

    det = HighAmplitudeDetector(low=10, high=200)
    detection_mask = det.detect(signal)

    assert detection_mask.sum() == 24
    assert detection_mask[25:40].all()
