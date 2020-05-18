from neurokit.io import edf


def test_read_edf():
    rec = edf.read_edf('tests/data/test.edf')
    assert rec.data.frequency == 100
    assert len(rec.data.channels) == 1
    assert rec.data.channels[0] == 'Calibration'
