from __future__ import annotations

import pandas as pd
from pandas.api.types import is_timedelta64_dtype


class EventSeries:
    """A frame of events."""
    __cols = ['start', 'end', 'channel', 'code', 'description']
    __index = ['start', 'end']

    def __init__(self, data, name=None):
        self.name = name
        self.data = pd.DataFrame(data, columns=EventSeries.__cols)

        if not is_timedelta64_dtype(self.data['start']):
            self.data['start'] = pd.to_timedelta(self.data['start'], unit='s')
        if not is_timedelta64_dtype(self.data['end']):
            self.data['end'] = pd.to_timedelta(self.data['end'], unit='s')

        self.data = self.data.set_index(
            EventSeries.__index, drop=False).sort_index()

    def add(self, start, end=None, channel=None, code=None, description=None):
        if not isinstance(start, pd.Timedelta):
            start = pd.to_timedelta(start, unit='s')
        if not isinstance(end, pd.Timedelta):
            end = pd.to_timedelta(end, unit='s')

        event = pd.Series(data=[start, end, channel, code, description],
                          index=EventSeries.__cols,
                          name=(start, end))
        self.data = self.data.append(event, ignore_index=False).sort_index()

    def channel(self, *channels):
        data = self.data[self.data.channel.isin(channels)].copy()
        return EventSeries(data, name=self.name)

    def __iter__(self):
        return self.data.itertuples()

    def __repr__(self):
        return f"<Events '{self.name}' ({len(self.data)} events)>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if not isinstance(start, pd.Timedelta):
                start = pd.to_timedelta(start, unit='s')
            if not isinstance(stop, pd.Timedelta):
                stop = pd.to_timedelta(stop, unit='s')

            data = self.data.loc[(None, start):(stop, None):key.step]

            return EventSeries(data, name=self.name)

        return self.data.loc[key]


class TimeSeries(pd.DataFrame):
    """"""
    _metadata = ['name']

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name', None)
        frequency = kwargs.pop('frequency', None)
        offset = kwargs.pop('offset', 0.)
        super().__init__(*args, **kwargs)

        if frequency:
            period_ns = f'{round(1e9 / frequency)}N'
            self.index = pd.timedelta_range(
                start=0, freq=period_ns, periods=self.shape[0])
        elif not isinstance(self.index, pd.TimedeltaIndex):
            raise ValueError('A TimedeltaIndex must be defined '
                             + 'or a frequency must be specified')

        if not isinstance(offset, pd.Timedelta):
            offset = pd.Timedelta(offset, unit='s')

        self.index += offset

    @property
    def _constructor(self):
        return TimeSeries

    @property
    def frequency(self):
        return (len(self) - 1) / self.duration.total_seconds()

    @property
    def channels(self):
        return self.columns

    @property
    def duration(self):
        return self.index[-1] - self.index[0]

    def filter(self, low: float, high: float = None, **kwargs) -> TimeSeries:
        from ..preprocessing import filters
        if not low:
            return filters.lowpass(self, high, **kwargs)

        if not high:
            return filters.highpass(self, low, **kwargs)

        return filters.bandpass(self, (low, high), **kwargs)
