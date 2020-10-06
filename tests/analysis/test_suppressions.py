"""Test iso-electric suppression detection and alpha suppression detection.



The code used to generate the test recordings is:
```
# Iso-electric suppression tests
rec = simulated_eeg_recording(channels=1, duration=10, frequency=100)
rec = rec.filter(0.5, None)

# α-suppression tests
rec = simulated_eeg_recording(channels=['EEG_1'], duration=10,
                              frequency=fs, amplitude=80, theta=2.75)
rec = rec.filter(0.25)
```

"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from neurokit import EventSeries
from neurokit.io import read_hdf
from neurokit.sim import simulated_eeg_recording
from neurokit.analysis.suppressions import SuppressionAnalyzer


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


@pytest.mark.parametrize('n', range(5))
def test_detect_suppressions(n):
    rec = read_hdf(f'tests/data/suppressions/ies_{n:02}.h5')

    # We must start with no suppressions
    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_ies(min_duration=1., threshold=8.)

    assert len(detections) == 0

    # Add suppressions
    rec.data.iloc[100:250] /= rec.data.iloc[100:250].abs().values.max()
    rec.data.iloc[300:360] /= rec.data.iloc[300:360].abs().values.max() / 5
    rec.data.iloc[500:800] /= rec.data.iloc[500:800].abs().values.max() / 5

    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_ies(threshold=5.)

    assert len(detections) == 2

    # Detection may start earlier and finish later than expected because of the
    # randomness of the signal, but they cannot start later or finish earlier.
    sec = pd.Timedelta('1s')
    assert 0 * sec <= detections.loc[0].start <= 1.01 * sec
    assert 2.48 * sec <= detections.loc[0].end <= 3.5 * sec
    assert 3.6 * sec <= detections.loc[1].start <= 5.01 * sec

    detections = analyzer.detect_ies(threshold=8., min_duration=0.5)
    assert len(detections) == 3
    assert 2 * sec <= detections.loc[1].start <= 3.01 * sec
    assert 3.58 * sec <= detections.loc[1].end <= 4.6 * sec

    detections = analyzer.detect_ies(min_duration=1.48, threshold=8.)
    assert len(detections) == 2

    detections = analyzer.detect_ies(threshold=1.01, min_duration=1.)

    assert len(detections) == 1
    assert 0 * sec <= detections.loc[0].start <= 1.01 * sec
    assert 2.48 * sec <= detections.loc[0].end <= 3.5 * sec


@pytest.mark.parametrize('n', range(5))
def test_detect_alpha_suppressions(n):
    rec = read_hdf(f'tests/data/suppressions/as_{n:02}.h5')

    fs = int(rec.data.frequency)
    ts = rec.data.index.total_seconds().values

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
    iso_elec_suppressions = analyzer.detect_ies(threshold=8.)
    assert len(iso_elec_suppressions) == 0

    detections = analyzer.detect_alpha_suppressions(frequency_band=(7.5, 12.5))

    assert len(detections) == 2

    sec = pd.Timedelta('1s')
    assert 1 * sec <= detections.loc[0].start <= 1.75 * sec
    assert 2 * sec <= detections.loc[0].end <= 2.75 * sec
    assert 4 * sec <= detections.loc[1].start <= 4.75 * sec
    assert 5.5 * sec <= detections.loc[1].end <= 6.25 * sec


@pytest.mark.parametrize('n', range(5))
def test_no_ies_as_alpha(n):
    n = 4
    rec = read_hdf(f'tests/data/suppressions/ies_{n:02}.h5')

    ts = rec.data.index.total_seconds()
    alpha_band = 60 * np.cos(2 * np.pi * 9.8 * ts)
    rec.data.loc[:, 'EEG_1'] += alpha_band
    rec.data.iloc[200:350] /= 100
    analyzer = SuppressionAnalyzer(rec)
    ies_detections = analyzer.detect_ies(threshold=8.)

    # should contain 1 ies suppression
    assert len(ies_detections) == 1
    alpha_detections = analyzer.detect_alpha_suppressions()

    # should contain no alpha suppressions
    assert len(alpha_detections) == 0


def test_artifacts():
    rec = simulated_eeg_recording(channels=['EEG_1'], duration=10,
                                  frequency=100)
    rec = rec.filter(1)

    # Add suppression
    rec.data.iloc[200:400] /= rec.data.iloc[200:400].values.max() / 5

    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_ies(threshold=5.5)
    assert len(detections) == 1

    # Add artifact
    artifacts = EventSeries(name='artifacts')
    artifacts.add(2.5, 3.99, code='test')
    rec.es.add(artifacts)

    analyzer = SuppressionAnalyzer(rec)
    detections = analyzer.detect_ies(threshold=5.5)
    assert len(detections) == 0
