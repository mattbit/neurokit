from unittest import TestCase
from pytest import approx
from neurokit.io import edf
import mne


def test_read_edf(self):
    rec = edf.read_edf('tests/data/test.edf')
    assert rec.data.frequency == 100
    assert len(rec.data.channels) == 1
    assert rec.data.channels[0] == 'Calibration'
