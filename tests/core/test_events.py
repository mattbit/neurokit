import pandas as pd

from neurokit.core import EventSeries

test_events = [
    {
        'start': -3,
        'end': 4,
        'code': 'TEST0',
        'channel': 'CH0',
    },
    {
        'start': 1,
        'end': 2,
        'code': 'TEST1',
        'channel': 'CH1',
    },
    {
        'start': 2,
        'end': 4,
        'code': 'TEST2',
        'channel': 'CH0',
    },
    {
        'start': 2,
        'end': 4,
        'code': 'TEST3',
        'channel': 'CH1',
    },
    {
        'start': 5,
        'end': 8,
        'code': 'TEST4',
        'channel': 'CH0',
    },
]


def test_event_series():
    events = EventSeries(test_events, name='test_events')

    assert events.name == 'test_events'
    assert isinstance(events.data, pd.DataFrame)
    assert len(events) == 5


def test_event_series_slicing():
    events = EventSeries(test_events, name='test_events')
    assert len(events['00:00:00':'00:00:03']) == 4
    assert events['00:00:00':'00:00:03'].data.iloc[-1].code == 'TEST3'
    assert len(events[0.:3.]) == 4
    assert events[0:3].data.iloc[-1].code == 'TEST3'


def test_event_add():
    events = EventSeries(test_events, name='test_events')
    events.add(3, 4, channel='CH2', code='ADDED')
    assert len(events) == 6
    assert events.data.iloc[-2].code == 'ADDED'
    assert events.data.iloc[-2].channel == 'CH2'


def test_select_channel():
    events = EventSeries(test_events, name='test_events')
    assert len(events.channel('CH1')) == 2
    assert len(events.channel('CH0')) == 3
    assert list(events.channel('CH1'))[0].code == 'TEST1'

    assert len(events.channel('CH0', 'CH1')) == 5
