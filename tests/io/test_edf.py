import tempfile
from pathlib import Path
from neurokit.io import edf


def test_read_edf():
    rec = edf.read_edf('tests/data/test.edf')
    assert rec.data.frequency == 100
    assert len(rec.data.channels) == 1
    assert rec.data.channels[0] == 'Calibration'


def test_write_edf():
    rec = edf.read_edf('tests/data/test.edf')
    out = tempfile.TemporaryDirectory()

    with tempfile.TemporaryDirectory() as out:
        out_file = Path(out).joinpath('w_test.edf')
        rec.to_edf(out_file)

        new_rec = edf.read_edf(out_file)

    assert rec.data.frequency == new_rec.data.frequency
    assert len(rec.data.channels) == len(new_rec.data.channels)
    assert rec.data.channels[0] == new_rec.data.channels[0]
