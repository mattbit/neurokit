from neurokit.sim import simulated_eeg_recording
from neurokit.io.utils import concatenate_recordings
from neurokit.core import EventSeries


def test_concatenation():
    rec1 = simulated_eeg_recording(60, 100)
    rec2 = simulated_eeg_recording(120, 100)

    rec1.data.loc[:] = 1
    es1 = EventSeries(name='events')
    es1.add(5, 10, description='test1')
    rec1.es.add(es1)

    rec2.data.loc[:] = 2
    es2 = EventSeries(name='events')
    es2.add(0, 1, description='test2')
    rec2.es.add(es2)

    rec = concatenate_recordings([rec1, rec2])

    assert rec.frequency == 100
    assert len(rec.data) == 60 * 100 + 120 * 100
    assert (rec.data.iloc[:60 * 100].values == 1).all()
    assert (rec.data.iloc[60 * 100:].values == 2).all()

    assert len(rec.es['events']) == 2
    assert rec.es.events.data.iloc[0].start.total_seconds() == 5
    assert rec.es.events.data.iloc[0].end.total_seconds() == 10
    assert rec.es.events.data.iloc[0].description == 'test1'

    assert rec.es.events.data.iloc[1].start.total_seconds() == 60
    assert rec.es.events.data.iloc[1].end.total_seconds() == 61
    assert rec.es.events.data.iloc[1].description == 'test2'
