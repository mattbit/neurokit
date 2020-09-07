import tempfile
from pathlib import Path
from neurokit.io import hdf


def test_read_hdf():
    rec = hdf.read_hdf('tests/data/test.hdf')

    assert rec.name == 'test'

    assert rec.data.frequency == 100
    assert len(rec.data.channels) == 1
    assert rec.data.channels[0] == 'Calibration'

    assert len(rec.ts) == 2
    assert rec.ts['test_data'].channels[0] == 'CH1'
    assert rec.ts['test_data'].channels[1] == 'CH2'
    assert rec.ts['test_data'].index[0].total_seconds() == 0
    assert rec.ts['test_data'].index[1].total_seconds() == 2
    assert rec.ts['test_data'].index[2].total_seconds() == 3

    assert len(rec.es['test_events']) == 2
    events = [e for e in rec.es['test_events']]
    assert events[0].start.total_seconds() == 0
    assert events[0].end.total_seconds() == 1
    assert events[0].channel == 'CH1'
    assert events[1].start.total_seconds() == 10
    assert events[1].channel == 'CH2'
    assert events[1].code == 'test2'


def test_write_hdf():
    rec = hdf.read_hdf('tests/data/test.hdf')
    out = tempfile.TemporaryDirectory()

    with tempfile.TemporaryDirectory() as out:
        out_file = Path(out).joinpath('w_test.hdf')
        rec.to_hdf(out_file)

        new_rec = hdf.read_hdf(out_file)

    assert rec.data.frequency == new_rec.data.frequency
    assert len(rec.data.channels) == len(new_rec.data.channels)
    assert rec.data.channels[0] == new_rec.data.channels[0]
