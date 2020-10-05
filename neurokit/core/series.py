from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy
from pandas.api.types import is_timedelta64_dtype

from .indexing import FixedTimedeltaIndex, timedelta_range


def _maybe_cast_timedelta(obj, unit='s'):
    if isinstance(obj, pd.Timedelta):
        return obj
    if isinstance(obj, str):
        return pd.to_timedelta(obj)

    return pd.to_timedelta(obj, unit=unit)


class EventSeries:
    """A frame of events."""
    _cols = ['start', 'end', 'channel', 'code', 'description']
    _index = ['start', 'end']

    def __init__(self, data=None, name=None):
        self.name = name
        self.data = pd.DataFrame([] if data is None else data,
                                 columns=self._cols)

        if not is_timedelta64_dtype(self.data['start']):
            self.data['start'] = pd.to_timedelta(self.data['start'], unit='s')
        if not is_timedelta64_dtype(self.data['end']):
            self.data['end'] = pd.to_timedelta(self.data['end'], unit='s')

        self.data.index = pd.MultiIndex.from_frame(
            self.data.loc[:, self._index])
        self.data.sort_index(inplace=True)

    def add(self, start, end=None, channel=None, code=None, description=None):
        start = _maybe_cast_timedelta(start)
        end = _maybe_cast_timedelta(end)

        event = pd.Series(data=[start, end, channel, code, description],
                          index=self._cols,
                          name=(start, end))
        self.data = self.data.append(event, ignore_index=False).sort_index()

    def channel(self, *channels):
        data = self.data[self.data.channel.isin(channels)].copy()
        return EventSeries(data, name=self.name)

    def copy(self, deep=True):
        if not deep:
            return self.__copy__()

        return self.__deepcopy__()

    def to_dict(self):
        events = [{'start': e.start,
                   'end': e.end,
                   'channel': e.channel,
                   'code': e.code,
                   'description': e.description}
                  for e in self.data.itertuples()]

        return {'name': self.name, 'events': events}

    @classmethod
    def from_dict(cls, data):
        events = data.get('events', [])
        for e in events:
            if not isinstance(e['start'], pd.Timedelta):
                e['start'] = pd.to_timedelta(e['start'], unit='ns')
            if not isinstance(e['end'], pd.Timedelta):
                e['end'] = pd.to_timedelta(e['end'], unit='ns')
        df = pd.DataFrame(events, columns=cls._cols)
        return cls(data=df, name=data.get('name'))

    def __copy__(self):
        return EventSeries(self.data, self.name)

    def __deepcopy__(self, memo=None):
        return EventSeries(deepcopy(self.data), deepcopy(self.name))

    def __iter__(self):
        return self.data.itertuples()

    def __repr__(self):
        return f"<EventSeries '{self.name}' ({len(self.data)} events)>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = _maybe_cast_timedelta(key.start)
            stop = _maybe_cast_timedelta(key.stop)

            idx = np.ones(len(self.data), dtype=bool)

            if start is not None:
                end_idx = self.data.index.get_level_values('end')
                idx &= end_idx > start

            if stop is not None:
                start_idx = self.data.index.get_level_values('start')
                idx &= start_idx < stop

            data = self.data.loc[idx].loc[::key.step]

            return EventSeries(data, name=self.name)

        return self.data.loc[key]


class BaseTimeSeries(pd.DataFrame):
    """Base TimeSeries class"""
    _metadata = ['name', 'filters']

    @property
    def _constructor(self):
        return BaseTimeSeries

    @property
    def _constructor_expanddim(self):
        return NotImplementedError("Not supported for TimeSeries!")


class TimeSeries(BaseTimeSeries):
    """"""
    _metadata = ['name', 'filters']

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name', None)
        self.filters = kwargs.pop('filters', {})
        frequency = kwargs.pop('frequency', None)
        offset = kwargs.pop('offset', 0.)
        super().__init__(*args, **kwargs)

        if frequency:
            period_ns = f'{round(1e9 / frequency)}N'
            self.index = timedelta_range(
                start=0, freq=period_ns, periods=self.shape[0])
        elif not isinstance(self.index, pd.TimedeltaIndex):
            raise ValueError('A TimedeltaIndex must be defined '
                             + 'or a frequency must be specified')
        elif not isinstance(self.index, FixedTimedeltaIndex):
            self.index = FixedTimedeltaIndex(self.index)

        if not isinstance(offset, pd.Timedelta):
            offset = pd.Timedelta(offset, unit='s')

        self.index += offset

    @property
    def _constructor(self):
        return TimeSeries

    @property
    def _constructor_expanddim(self):
        return NotImplementedError("Not supported for TimeSeries!")

    @property
    def frequency(self):
        return round(len(self) / self.duration.total_seconds(), 9)

    @property
    def offset(self):
        return self.index.min()

    @property
    def channels(self):
        return self.columns

    @property
    def duration(self):
        return self.index[-1] - 2 * self.index[0] + self.index[1]

    def filter(  # skipcq: PYL-W0221
            self,
            low: float,
            high: float = None,
            **kwargs
    ) -> TimeSeries:
        from ..signal import filters
        if not low:
            return filters.lowpass(self, high, **kwargs)

        if not high:
            return filters.highpass(self, low, **kwargs)

        return filters.bandpass(self, (low, high), **kwargs)

    def to_dict(self, **kwargs):  # skipcq: PYL-W0221
        return {
            'name': self.name,
            'offset': self.offset,
            'frequency': self.frequency,
            'channels': [{'name': ch, 'data': self[ch].values}
                         for ch in self.channels],
        }

    @classmethod
    def from_dict(cls, data, **kwargs):  # skipcq: PYL-W0221
        chs = data.get('channels', [])
        ch_names = [ch.get('name') for ch in chs]
        ch_data = np.array([ch.get('data') for ch in chs], dtype=np.float).T
        offset = pd.to_timedelta(data.get('offset', 0), unit='ns')

        return cls(
            ch_data,
            columns=ch_names,
            name=data.get('name'),
            frequency=data.get('frequency'),
            offset=offset
        )


class UnevenTimeSeries(BaseTimeSeries):
    """Unevenly spaced time series."""

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name', None)
        self.filters = kwargs.pop('filters', {})
        super().__init__(*args, **kwargs)

        if not isinstance(self.index, pd.TimedeltaIndex):
            raise ValueError('A TimedeltaIndex must be defined.')

        if not isinstance(self.index, FixedTimedeltaIndex):
            self.index = FixedTimedeltaIndex(self.index)

    @property
    def _constructor(self):
        return UnevenTimeSeries

    @property
    def _constructor_expanddim(self):
        return NotImplementedError("Not supported for TimeSeries!")

    @property
    def channels(self):
        return self.columns

    @property
    def duration(self):
        return self.index[-1] - self.index[0]

    def to_dict(self, **kwargs):  # skipcq: PYL-W0221
        return {
            'name': self.name,
            'index': self.index.total_seconds().values,
            'channels': [{'name': ch, 'data': self[ch].values}
                         for ch in self.channels],
        }

    @classmethod
    def from_dict(cls, data, **kwargs):  # skipcq: PYL-W0221
        chs = data.get('channels', [])
        ch_names = [ch.get('name') for ch in chs]
        ch_data = np.array([ch.get('data') for ch in chs], dtype=np.float).T

        return cls(
            ch_data,
            columns=ch_names,
            index=pd.to_timedelta(data.get('index'), unit='s'),
            name=data.get('name'),
        )


def timeseries_from_dict(ts_data):
    """Create a timeseries from a dict structure.

    The correct type (evenly or unevenly spaced) is inferred from the data.
    """
    cls = TimeSeries if 'frequency' in ts_data else UnevenTimeSeries

    return cls.from_dict(ts_data)
