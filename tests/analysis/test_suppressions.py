import numpy as np
import pandas as pd
from pytest import approx
from neurokit.analysis.suppressions import (detect_suppressions,
                                            detect_alpha_suppressions)
from neurokit.sim import simulated_eeg_recording


def test_detect_suppressions():
    rec = simulated_eeg_recording(1, duration=10, frequency=100)
    rec = rec.filter(1, None)

    rec.data.iloc[100:250] /= rec.data.iloc[100:250].max()
    rec.data.iloc[300:350] /= rec.data.iloc[300:350].max() / 5
    rec.data.iloc[500:800] /= rec.data.iloc[500:800].max() / 5

    ts = (rec.data.index - rec.start_date).total_seconds()

    detections = detect_suppressions(rec)
    assert len(detections) == 2

    detections.loc[:, ('start', 'end')] -= rec.start_date
    detections['start'] = detections['start'].dt.total_seconds()
    detections['end'] = detections['end'].dt.total_seconds()

    assert detections.loc[0].start == approx(ts[100], abs=0.2)
    assert detections.loc[0].end == approx(ts[250], abs=0.2)
    assert detections.loc[1].start == approx(ts[500], abs=0.2)

    detections = detect_suppressions(rec, min_duration=0.5)
    detections.loc[:, ('start', 'end')] -= rec.start_date
    detections['start'] = detections['start'].dt.total_seconds()
    detections['end'] = detections['end'].dt.total_seconds()

    assert len(detections) == 3
    assert detections.loc[1].start == approx(ts[300], abs=0.2)
    assert detections.loc[1].end == approx(ts[350], abs=0.2)

    detections = detect_suppressions(rec, min_duration=1.)
    assert len(detections) == 2

    detections = detect_suppressions(rec, threshold=1.5)
    detections.loc[:, ('start', 'end')] -= rec.start_date
    detections['start'] = detections['start'].dt.total_seconds()
    detections['end'] = detections['end'].dt.total_seconds()

    assert len(detections) == 1
    assert detections.loc[0].start == approx(ts[100], abs=0.2)
    assert detections.loc[0].end == approx(ts[250], abs=0.2)


def test_detect_alpha_suppressions():
    # Generate a weak recording
    rec = simulated_eeg_recording(['EEG_1'], duration=10, frequency=100,
                                  amplitude=40)
    rec = rec.filter(1, 20)

    ts = (rec.data.index - rec.start_date).total_seconds().values

    # Add a strong alpha wave
    alpha_band = 60 * np.cos(2 * np.pi * 9.8 * ts)
    alpha_band[100:250] = 0.
    alpha_band[700:730] = 0.
    alpha_band[400:600] = 0.
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

    detections = detect_alpha_suppressions(rec)
    assert len(detections) == 2

    det_0_start = (detections.loc[0].start - rec.start_date).total_seconds()
    det_0_end = (detections.loc[0].end - rec.start_date).total_seconds()
    det_1_start = (detections.loc[1].start - rec.start_date).total_seconds()
    det_1_end = (detections.loc[1].end - rec.start_date).total_seconds()

    assert det_0_start == approx(1, abs=0.2)
    assert det_0_end == approx(2.4, abs=0.2)
    assert det_1_start == approx(4, abs=0.2)
    assert det_1_end == approx(6, abs=0.2)



def test_artifacts():
    rec = simulated_eeg_recording(['EEG_1'], duration=10, frequency=100)
    rec = rec.filter(1)

    rec.data.iloc[200:400] = 0
    detections = detect_suppressions(rec)
    assert len(detections) == 1

    # Add artifact
    start = rec.start_date + pd.Timedelta(2.5, unit='s')
    end = rec.start_date + pd.Timedelta(3.99, unit='s')
    rec.artifacts.loc[0, :] = start, end, None, 'test'
    rec = rec.artifacts_to_nan()

    detections = detect_suppressions(rec)
    assert len(detections) == 0
