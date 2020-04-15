import pytest
import numpy as np
import pandas as pd
from pytest import approx
from unittest.mock import patch
from neurokit.analysis.suppressions import SuppressionAnalyzer
from neurokit.sim import simulated_eeg_recording


@patch('neurokit.analysis.suppressions.SuppressionAnalyzer.detect_ies')
def test_detect_ies(detect_suppressions_mock):
    rec = simulated_eeg_recording()
    detect_suppressions_mock.return_value = 'TEST'
    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_ies(
        channels=['EEG_1'], threshold=3.2, min_duration=0.123)

    detect_suppressions_mock.assert_called_once_with(
        channels=['EEG_1'], threshold=3.2, min_duration=0.123)
    assert detections == 'TEST'


@pytest.mark.repeat(10)
def test_detect_suppressions():
    rec = simulated_eeg_recording(1, duration=10, frequency=100)
    rec = rec.filter(0.5, None)

    # We must start with no suppressions
    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_ies(min_duration=1.)
    print(detections)
    assert len(detections) == 0

    # Add suppressions
    rec.data.iloc[100:250] /= rec.data.iloc[100:250].abs().max()
    rec.data.iloc[300:360] /= rec.data.iloc[300:360].abs().max() / 5
    rec.data.iloc[500:800] /= rec.data.iloc[500:800].abs().max() / 5

    ts = (rec.data.index - rec.start_date).total_seconds()
    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_ies()
    print(detections)
    assert len(detections) == 2

    detections.loc[:, ('start', 'end')] -= rec.start_date
    detections['start'] = detections['start'].dt.total_seconds()
    detections['end'] = detections['end'].dt.total_seconds()

    # Detection may start earlier and finish later than expected because of the
    # randomness of the signal, but they cannot start later or finish earlier.
    assert ts[0] <= detections.loc[0].start <= ts[101]
    assert ts[249] <= detections.loc[0].end <= ts[350]
    assert ts[360] <= detections.loc[1].start <= ts[501]

    detections = analyzer.detect_ies(min_duration=0.5)
    detections.loc[:, ('start', 'end')] -= rec.start_date
    detections['start'] = detections['start'].dt.total_seconds()
    detections['end'] = detections['end'].dt.total_seconds()

    assert len(detections) == 3
    assert ts[200] <= detections.loc[1].start <= ts[301]
    assert ts[359] <= detections.loc[1].end <= ts[460]

    # import matplotlib.pyplot as plt
    # plt.plot(ts, rec.data.EEG_1)
    # plt.hlines([-1.5, 1.5], 0, 10, lw=1)
    # plt.vlines([1, 2.50, 3, 3.6, 5, 8], -50, 50, color='r')
    # plt.ylim(-5, 5)
    detections = analyzer.detect_ies(min_duration=1.49)
    assert len(detections) == 2

    detections = analyzer.detect_ies(threshold=1.5, min_duration=1.)
    detections.loc[:, ('start', 'end')] -= rec.start_date
    detections['start'] = detections['start'].dt.total_seconds()
    detections['end'] = detections['end'].dt.total_seconds()

    assert 1 <= len(detections) <= 2
    assert ts[0] <= detections.loc[0].start <= ts[101]
    assert ts[249] <= detections.loc[0].end <= ts[350]


@pytest.mark.repeat(10)
def test_detect_alpha_suppressions():
    # Generate a weak recording
    rec = simulated_eeg_recording(['EEG_1'], duration=10, frequency=100,
                                  amplitude=8)
    rec = rec.filter(1, 20)

    ts = (rec.data.index - rec.start_date).total_seconds().values

    # Add a strong alpha wave
    alpha_band = 60 * np.cos(2 * np.pi * 9.8 * ts)
    alpha_band[100:250] *= 0.09
    alpha_band[700:730] = 0
    alpha_band[400:600] *= 0.09
    rec.data.loc[:, 'EEG_1'] += alpha_band

    # filtered = rec.filter(9, 11)
    # rms_before = np.sqrt(np.mean(rec.data.loc[:, :].values**2))
    # rms_after = np.sqrt(np.mean(filtered.data.loc[:, :].values**2))
    # r = rms_after / rms_before
    # threshold = 8 * r
    #
    # import matplotlib.pyplot as plt
    # plt.plot(ts, filtered.data.EEG_1)
    # plt.hlines([-threshold, +threshold], 0, 10)
    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_alpha_suppressions()
    assert len(detections) == 2

    det_0_start = (detections.loc[0].start - rec.start_date).total_seconds()
    det_0_end = (detections.loc[0].end - rec.start_date).total_seconds()
    det_1_start = (detections.loc[1].start - rec.start_date).total_seconds()
    det_1_end = (detections.loc[1].end - rec.start_date).total_seconds()

    assert det_0_start == approx(1, abs=0.3)
    assert det_0_end == approx(2.4, abs=0.3)
    assert det_1_start == approx(4, abs=0.3)
    assert det_1_end == approx(6, abs=0.3)


def test_artifacts():
    rec = simulated_eeg_recording(['EEG_1'], duration=10, frequency=100)
    rec = rec.filter(1)

    rec.data.iloc[200:400] = 0
    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_ies()
    assert len(detections) == 1

    # Add artifact
    start = rec.start_date + pd.Timedelta(2.5, unit='s')
    end = rec.start_date + pd.Timedelta(3.99, unit='s')
    rec.artifacts.loc[0, :] = start, end, None, 'test'

    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_ies()
    assert len(detections) == 0
