from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Sequence, Union

from .common import NamedItemsBag
from .series import EventSeries, TimeSeries


class Patient:
    def __init__(
        self,
        id_: str = None,
        description: str = None,
        name: str = None,
        sex: str = None,
        age: float = None,
        height: float = None,
        weight: float = None
    ):
        self.id_ = id_
        self.description = description
        self.name = name
        self.sex = sex
        self.age = age
        self.height = height
        self.weight = weight


class Recording:
    """Representation of a neurophysiological recording.

    A recording object is a group of timeseries and associated event series.

    Parameters
    ----------
    data
    name
    series : Sequence[TimeSeries]
    events : Sequence[EventSeries]
    patient : Patient
    meta : dict


    """

    def __init__(
        self,
        data=None,
        name: str = None,
        timeseries: Sequence[TimeSeries] = None,
        events: Sequence[EventSeries] = None,
        patient: Patient = None,
        meta: dict = None,
    ):
        self.name = name
        self.meta = meta or {}
        self.patient = patient or Patient()

        if data is not None:
            if not isinstance(data, TimeSeries):
                data = TimeSeries(data, name='data')
            timeseries = [data] + list(timeseries)

        if not timeseries:
            raise ValueError('At least a timeseries is required.')

        self.ts = NamedItemsBag(timeseries, dtype=TimeSeries)
        self._main_ts = timeseries[0].name

        self.es = NamedItemsBag(events, dtype=EventSeries)

    @property
    def data(self):
        return self.ts[self._main_ts]

    @data.setter
    def data(self, value: TimeSeries):
        self.ts[self._main_ts] = value

    @property
    def timeseries(self):
        return self.ts

    @property
    def events(self):
        return self.es

    @property
    def frequency(self):
        return self.data.frequency

    @property
    def duration(self):
        return self.data.duration

    def copy(self, deep=True):
        if not deep:
            return self.__copy__()
        return self.__deepcopy__()

    def artifacts_to_nan(
        self,
        series: Union[str, EventSeries] = 'artifacts',
        pad: float = 0
    ) -> Recording:
        """Convert artifacts to NaN.

        Parameters
        ----------
        series : str
            Name of the series containing the artifacts
        pad : float
            Padding (in seconds) that should be added around artifacts. For
            example, if `pad = 1` all values from 1 second before the beginning
            of the artifact to 1 second after its end will be set to `np.nan`.

        Returns
        -------
        recording : neurokit.Recording
            A copy of the original recording with `numpy.nan` instead of
            artifacted signal for the main series.
        """
        if isinstance(series, str):
            series = self.es[str]

        rec = self.copy()
        dt = pd.Timedelta(pad, unit='s')
        for artifact in series:
            start = artifact.start - dt
            end = artifact.end + dt
            chs = slice(None) if artifact.channel is None else artifact.channel
            rec.data.loc[start:end, chs] = np.nan

        return rec

    def filter(self, *args, **kwargs) -> Recording:
        rec = self.copy()
        rec.data = rec.data.filter(*args, **kwargs)
        return rec

    def to_edf(self, filename, **kwargs):
        from .edf import write_edf

        return write_edf(self, filename, **kwargs)

    def to_nkr(self, filename, **kwargs):
        from neurokit.io.nk import write_nkr

        return write_nkr(self, filename, **kwargs)

    def __copy__(self):
        return Recording(id_=self.name,
                         series=self.series,
                         events=self.events,
                         patient=self.patient,
                         meta=self.meta)

    def __deepcopy__(self, memo=None):
        return Recording(id_=deepcopy(self.name, memo),
                         series=deepcopy(self.series, memo),
                         events=deepcopy(self.events, memo),
                         patient=deepcopy(self.patient, memo),
                         meta=deepcopy(self.meta, memo))

    def __repr__(self):
        return f"<Recording '{self.name}' ({len(self.ts)} series)>"

    def __getattr__(self, name):
        return self.ts[name]

    def __getitem__(self, name):
        return self.ts[name]
