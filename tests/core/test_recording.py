import numpy as np
from neurokit import EventSeries
from neurokit.sim import simulated_eeg_recording


def test_shallow_copy():
    original = simulated_eeg_recording(channels=['CH1', 'CH2'], duration=10,
                                       frequency=100)
    original.meta['test'] = True
    original.es.add(EventSeries(name='test_events'))

    copied = original.copy(deep=False)
    assert id(copied) != id(original)
    assert id(copied.data) == id(original.data)
    assert id(copied.patient) == id(original.patient)
    assert id(copied.meta) == id(original.meta)
    assert id(copied.es['test_events']) == id(original.es['test_events'])


def test_deep_copy():
    original = simulated_eeg_recording(channels=['CH1', 'CH2'], duration=10,
                                       frequency=100)
    original.meta['test'] = True
    original.es.add(EventSeries(name='test_events'))
    copied = original.copy(deep=True)
    assert id(copied) != id(original)
    assert id(copied.data) != id(original.data)
    assert id(copied.patient) != id(original.patient)
    assert id(copied.meta) != id(original.meta)
    assert id(copied.es['test_events']) != id(original.es['test_events'])


def test_artifacts_to_nan():
    rec = simulated_eeg_recording(channels=['CH1', 'CH2'], duration=10,
                                  frequency=100)
    events = EventSeries(name='artifacts')

    # Artifact 1: from 1 s to 2 s, all channels
    events.add(1, 2, code='test1')

    # Artifact 2: from 3 s to 5 s, only channel CH1
    events.add(3, 5, channel='CH1', code='test2')

    # Artifact 3: from 4.5 s to 7 s, only channel CH2
    events.add(4.5, 7, channel='CH2', code='test3')

    rec.es.add(events)

    rec_with_nan = rec.artifacts_to_nan()

    rec_with_nan.data.loc['5.1s':, 'CH1'].isna().sum()

    assert np.isnan(rec_with_nan.data['CH1']).sum() == 101 + 201
    assert not rec_with_nan.data.loc['0s':'0.99s', 'CH1'].isna().any()
    assert rec_with_nan.data.loc['1s':'2s', 'CH1'].isna().all()
    assert not rec_with_nan.data.loc['2.01s':'2.99s', 'CH1'].isna().any()
    assert rec_with_nan.data.loc['3s':'5s', 'CH1'].isna().all()
    assert not rec_with_nan.data.loc['5.1s':, 'CH1'].isna().any()

    assert np.isnan(rec_with_nan.data['CH2']).sum() == 101 + 251

    assert rec_with_nan.data.loc['1s':'2s', 'CH2'].isna().all()
    assert not rec_with_nan.data.loc['3s':'4.49s', 'CH2'].isna().any()
    assert rec_with_nan.data.loc['4.5s':'7s', 'CH2'].isna().all()


def test_duration():
    rec = simulated_eeg_recording(channels=['CH1', 'CH2'], duration=10,
                                  frequency=100)
    assert len(rec.data) == 1000
    assert rec.duration.total_seconds() == 10


def test_slice():
    rec = simulated_eeg_recording(channels=['CH1', 'CH2'], duration=10,
                                  frequency=100)
    rec.es.add(EventSeries(name='annotations'))
    rec.es.annotations.add(2, 3, description='test start')
    rec.es.annotations.add(4, 6, description='test middle')
    rec.es.annotations.add(7, 8, description='test end')

    assert len(rec.data) == 1000
    assert rec.duration.total_seconds() == 10

    rec1 = rec.slice(0, 5)
    rec2 = rec.slice(5, 10)

    assert len(rec1.data) == 501  # note that the 5 s sample is included
    assert len(rec2.data) == 500

    assert len(rec1.es.annotations) == 2
    assert rec1.es.annotations.data.iloc[0].description == 'test start'
    assert rec1.es.annotations.data.iloc[1].description == 'test middle'

    assert len(rec2.es.annotations) == 2
    assert rec2.es.annotations.data.iloc[0].description == 'test middle'
    assert rec2.es.annotations.data.iloc[1].description == 'test end'
