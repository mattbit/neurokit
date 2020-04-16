import pytest
import numpy as np
import pandas as pd
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

    assert len(detections) == 0

    # Add suppressions
    rec.data.iloc[100:250] /= rec.data.iloc[100:250].abs().max()
    rec.data.iloc[300:360] /= rec.data.iloc[300:360].abs().max() / 5
    rec.data.iloc[500:800] /= rec.data.iloc[500:800].abs().max() / 5

    ts = (rec.data.index - rec.start_date).total_seconds()
    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_ies()

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
    fs = 200
    rec = simulated_eeg_recording(['EEG_1'], duration=10, frequency=fs,
                                  amplitude=80, theta=2.75)
    rec = rec.filter(0.25)

    ts = (rec.data.index - rec.start_date).total_seconds().values

    # Add a strong alpha wave
    alpha_band = 60 * np.cos(2 * np.pi * 9.8 * ts)
    alpha_band[1 * fs:int(2.5 * fs)] = 0.
    alpha_band[7 * fs:int(7.3 * fs)] = 0.
    alpha_band[4 * fs:6 * fs] = 0.
    rec.data.loc[:, 'EEG_1'] += alpha_band

    # import plotly.express as px
    # filtered = rec.filter(8, 12)
    # rms_before = np.sqrt(np.mean(rec.data.loc[:, :].values**2))
    # rms_after = np.sqrt(np.mean(filtered.data.loc[:, :].values**2))
    # r = rms_after / rms_before
    # threshold = 8 * r
    # print(threshold)
    # px.line(x=ts, y=filtered.data['EEG_1'])
    # px.line(x=rec.data.index, y=rec.data['EEG_1'])

    # import matplotlib.pyplot as plt
    # plt.plot(ts, filtered.data.EEG_1)
    # plt.hlines([-threshold, +threshold], 0, 10)

    analyzer = SuppressionAnalyzer(rec)

    # Ensure no iso-electric suppression is detected.
    iso_elec_suppressions = analyzer.detect_ies()
    assert len(iso_elec_suppressions) == 0

    detections = analyzer.detect_alpha_suppressions(frequency_band=(7.5, 12.5))
    assert len(detections) == 2

    det_0_start = (detections.loc[0].start - rec.start_date).total_seconds()
    det_0_end = (detections.loc[0].end - rec.start_date).total_seconds()
    det_1_start = (detections.loc[1].start - rec.start_date).total_seconds()
    det_1_end = (detections.loc[1].end - rec.start_date).total_seconds()
    (detections.loc[1].start - rec.start_date).total_seconds()

    assert 1 <= det_0_start <= 1.75
    assert 2 <= det_0_end <= 2.75
    assert 4 <= det_1_start <= 4.75
    assert 5.5 <= det_1_end <= 6.25


def test_no_ies_as_alpha():
    rec = simulated_eeg_recording(1, duration=10, frequency=100)
    rec = rec.filter(0.5, None)
    ts = (rec.data.index - rec.start_date).total_seconds()
    alpha_band = 60 * np.cos(2 * np.pi * 9.8 * ts)
    rec.data.loc[:, 'EEG_1'] += alpha_band
    rec.data.iloc[200:350] = 0
    analyzer = SuppressionAnalyzer(rec)
    ies_detections = analyzer.detect_ies()

    # should contain 1 ies suppression
    assert len(ies_detections) == 1
    alpha_detections = analyzer.detect_alpha_suppressions()
    # should contain no alpha suppressions
    assert len(alpha_detections) == 0


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
