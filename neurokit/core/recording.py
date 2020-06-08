from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Sequence, Union
from .common import NamedItemsBag
from .series import (EventSeries, BaseTimeSeries,
                     TimeSeries, timeseries_from_dict)


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
        self.id = id_
        self.description = description
        self.name = name
        self.sex = sex
        self.age = age
        self.height = height
        self.weight = weight

    def to_dict(self):
        return {
            'id': self.id,
            'description': self.description,
            'name': self.name,
            'sex': self.sex,
            'age': self.age,
            'height': self.height,
            'weight': self.weight,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id_=data.get('id'),
            description=data.get('description'),
            name=data.get('name'),
            sex=data.get('sex'),
            age=data.get('age'),
            height=data.get('height'),
            weight=data.get('weight'),
        )


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
        timeseries = list(timeseries) if timeseries is not None else []

        if data is not None:
            if not isinstance(data, TimeSeries):
                data = TimeSeries(data, name='data')
            timeseries = [data] + timeseries

        if not timeseries:
            raise ValueError('At least a timeseries is required.')

        self.ts = NamedItemsBag(timeseries, dtype=BaseTimeSeries)
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
            series = self.es[series]
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

    def to_dict(self):
        return {
            'name': self.name,
            'meta': self.meta,
            'patient': self.patient.to_dict(),
            'timeseries': [ts.to_dict() for ts in self.ts],
            'events': [e.to_dict() for e in self.es],
        }

    @classmethod
    def from_dict(cls, data):
        name = data.get('name')
        meta = data.get('meta', {})
        patient = Patient.from_dict(data.get('patient', {}))
        ts = [timeseries_from_dict(ts_data)
              for ts_data in data.get('timeseries', [])]
        es = [EventSeries.from_dict(es_data)
              for es_data in data.get('events', [])]

        return cls(
            name=name,
            timeseries=ts,
            events=es,
            patient=patient,
            meta=meta,
        )

    @classmethod
    def from_mne_raw(cls, raw):
        """Create a Recording from an `mne.io.Raw` object.

        Parameters
        ----------
        raw : mne.io.Raw
            The Raw object.

        Returns
        -------
        recording : neurokit.Recording
            The neurokit recording object.
        """
        from ..io._mne import _recording_from_mne_raw

        return _recording_from_mne_raw(raw)

    def to_edf(self, filename, **kwargs):
        from ..io.edf import write_edf

        return write_edf(self, filename, **kwargs)

    def to_nkr(self, filename, **kwargs):
        from ..io.nk import write_nkr

        return write_nkr(self, filename, **kwargs)

    def to_msgpack(self, filename, **kwargs):
        from ..io.msgpack import write_msgpack

        return write_msgpack(self, filename, **kwargs)

    def slice(self, start, end=None):
        if start is not None and not isinstance(start, pd.Timedelta):
            start = pd.to_timedelta(start, unit='s')
        if end is not None and not isinstance(end, pd.Timedelta):
            end = pd.to_timedelta(end, unit='s')

        timeseries = NamedItemsBag([ts.loc[start:end] for ts in self.ts])
        events = NamedItemsBag([es[start:end] for es in self.es])

        return Recording(name=self.name, meta=self.meta, patient=self.patient,
                         timeseries=timeseries, events=events)

    def __copy__(self):
        return Recording(name=self.name,
                         timeseries=self.ts,
                         events=self.es,
                         patient=self.patient,
                         meta=self.meta)

    def __deepcopy__(self, memo=None):
        return Recording(name=deepcopy(self.name, memo),
                         timeseries=deepcopy(self.ts, memo),
                         events=deepcopy(self.es, memo),
                         patient=deepcopy(self.patient, memo),
                         meta=deepcopy(self.meta, memo))

    def __repr__(self):
        return f"<Recording '{self.name}' ({len(self.ts)} series)>"

    def __getattr__(self, name):
        try:
            return self.ts[name]
        except KeyError:
            raise AttributeError()

    def __getitem__(self, name):
        return self.ts[name]
