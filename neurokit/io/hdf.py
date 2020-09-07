import h5py
import datetime
import numpy as np
import pandas as pd

from ..core import Recording, Patient
from ..core import TimeSeries, UnevenTimeSeries, EventSeries


_hdf_string = h5py.string_dtype(encoding='utf-8')


def _parse_ts_group(group, name):
    data = group.get('data')[()]
    chs = group.get('channel')[()]
    time = pd.to_timedelta(group.get('time')[()], unit='s')
    frequency = group.attrs.get('frequency')

    if frequency is None:
        return UnevenTimeSeries(data, columns=chs, index=time, name=name)

    return TimeSeries(data, columns=chs, frequency=frequency, offset=time[0],
                      name=name)


def _parse_es_group(group, name):
    data = group.get('data')[()]
    return EventSeries(data, name=name)


def _encode_attr_val(value):
    if isinstance(value, datetime.datetime):
        return value.utcnow().isoformat()
    if isinstance(value, datetime.timedelta):
        return pd.Timedelta(value).total_seconds()

    return value


def read_hdf(filename):
    """Read a HDF5 encoded Recording."""
    with h5py.File(filename, 'r') as f:
        version = f.attrs.get('_nkversion')

        if version != 0:
            raise ValueError(f'Invalid neurokit HDF version ({version}).')

        name = f.attrs.get('name')
        meta = dict(f['meta'].attrs)
        patient = Patient.from_dict(f['patient'].attrs)
        ts = [_parse_ts_group(ts_group, ts_name)
              for ts_name, ts_group in f['timeseries'].items()]
        es = [_parse_es_group(es_group, es_name)
              for es_name, es_group in f['eventseries'].items()]

    return Recording(name=name, timeseries=ts, events=es,
                     patient=patient, meta=meta)


def write_hdf(recording: Recording, filename):
    """Write a Recording object in the HDF5 format."""
    with h5py.File(filename, 'w') as f:
        # General metadata
        f.attrs['_nkversion'] = 0
        f.attrs['name'] = str(recording.name)
        meta_group = f.create_group('meta')
        meta_group.attrs.update((name, _encode_attr_val(val)) for (name, val) in recording.meta.items() if val is not None)

        # Patient data
        patient_meta = recording.patient.to_dict()
        patient_group = f.create_group('patient')
        patient_group.attrs.update((name, _encode_attr_val(val)) for (name, val) in patient_meta.items() if val is not None)

        # Timeseries
        ts_group = f.create_group('timeseries')
        for ts in recording.ts:
            g = ts_group.create_group(ts.name)
            g['data'] = ts.values
            g['time'] = ts.index.total_seconds()
            g['time'].make_scale('time')
            g['channel'] = np.array([ch for ch in ts.channels], dtype=_hdf_string)
            g['channel'].make_scale('channel')
            g['data'].dims[0].attach_scale(g['time'])
            g['data'].dims[1].attach_scale(g['channel'])
            g['data'].dims[0].label = 'time'
            g['data'].dims[1].label = 'channel'

            if hasattr(ts, 'frequency'):
                g.attrs['frequency'] = ts.frequency

        # Events
        es_group = f.create_group('eventseries')
        es_dtype = np.dtype([
            ('start', np.float64),
            ('end', np.float64),
            ('channel', _hdf_string),
            ('code', _hdf_string),
            ('description', _hdf_string),
        ])
        for es in recording.es:
            dx = es.data.copy()
            dx['start'] = dx['start'].dt.total_seconds()
            dx['end'] = dx['end'].dt.total_seconds()
            dx.reset_index(drop=True)

            g = es_group.create_group(es.name)
            g['data'] = np.array(dx.to_records(index=False), dtype=es_dtype)
