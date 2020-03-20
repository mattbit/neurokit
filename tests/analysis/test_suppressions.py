from unittest import TestCase

import pandas as pd
import numpy as np
import datetime
from neurokit.io import Recording
from neurokit.analysis.suppressions import detect_ies
from neurokit.preprocessing.filter import bandpass


class TestSuppressions(TestCase):
    def test_detect_ies(self):
        time = pd.to_datetime(1e9 * np.arange(0, 10, 1e-2))
        data = pd.DataFrame({'CH1': 50 * np.ones(100 * 10),
                             'time': time}).set_index('time')
        data.loc[time[100]:time[200]] = 0
        data.loc[time[300]:time[350]] = 5
        data.loc[time[500]:time[800]] = -5
        rec = Recording(data, frequency=100)

        detections = detect_ies(rec)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections.loc[0].start, time[100])
        self.assertEqual(detections.loc[0].end, time[200])
        self.assertEqual(detections.loc[1].start, time[500])

        detections = detect_ies(rec, min_duration=0.2)
        self.assertEqual(len(detections), 3)
        self.assertEqual(detections.loc[1].start, time[300])
        self.assertEqual(detections.loc[1].end, time[350])

        detections = detect_ies(rec, threshold=1.)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections.loc[0].start, time[100])
        self.assertEqual(detections.loc[0].end, time[200])

    def test_filter(self):
        t = np.arange(0, 10, 1e-2)
        signal = 10*np.sin(2*np.pi*12*t)+10*np.sin(2*np.pi*20*t)
        signal[100:200] = 0.5*np.sin(2*np.pi*12*t[100:200])
        time = pd.to_datetime(1e9 * t)
        data = pd.DataFrame({'CH1': signal,
                             'time': time}).set_index('time')
        rec = Recording(data, frequency=100)
        detections = detect_ies(bandpass(rec, (8, 16)))
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections.loc[0].start, time[100], delta=datetime.timedelta(seconds=0.2))
