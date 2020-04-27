from __future__ import annotations

import pandas as pd


class EventSeries:
    """A frame of events."""

    def __init__(self, name=None):
        self.name = name
        self.data = pd.DataFrame(
            columns=['id', 'start', 'end', 'channel', 'description'])
        self.data = self.data.set_index('start', drop=False)

    def add(self, start, end=None, channel=None, description=None):
        self.data.append({
            'start': start,
            'end': end,
            'channels': channel,
            'description': description,
        })

    def __repr__(self):
        return f"<Events '{self.name}' ({len(self.data)} events)>"


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
