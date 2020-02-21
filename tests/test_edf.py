from unittest import TestCase

from neurokit.io import edf


class TestEDF(TestCase):
    def test_read_edf(self):
        rec = edf.read_edf('tests/data/test.edf')
        self.assertEqual(rec.frequency, 100)
        self.assertEqual(len(rec.channels), 1)
        self.assertEqual(rec.channels[0], 'Calibration')
