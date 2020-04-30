import pandas as pd
from tempfile import TemporaryFile

import neurokit as nk
from neurokit.io import msgpack


def test_write_and_read_msgpack():
    rec = nk.sim.simulated_eeg_recording(pd.to_timedelta('10m').total_seconds(),
                                         channels=['CH1', 'CH2!!!', 'CH$3'],
                                         frequency=150)
    annotations = nk.EventSeries(name='annots')
    annotations.add(0, 3.2, None, 'A1')
    rec.es.add(annotations)

    with TemporaryFile() as tempfile:
        msgpack.write_msgpack(rec, tempfile)
        tempfile.seek(0)
        urec = msgpack.read_msgpack(tempfile)

    assert urec.name == rec.name
    assert (urec.data.values == rec.data.values).all()
    assert (urec.data.channels == rec.data.channels).all()
    assert len(urec.es) == 1
    assert urec.es['annots'].name == 'annots'
    assert len(urec.es['annots']) == 1
    assert urec.es['annots'].data.iloc[0].end.total_seconds() == 3.2
