from unittest import TestCase

import pandas as pd
import numpy as np

from neurokit.io import Recording
from neurokit.analysis.suppressions import detect_ies


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
