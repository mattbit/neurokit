import h5py
import numpy as np

from ..core import Recording


_hdf_string = h5py.string_dtype(encoding='utf-8')

def read_hdf(filename):
    """Read a HDF5 encoded Recording."""
    raise NotImplementedError()

def write_hdf(recording: Recording, filename):
    """Write a Recording object in the HDF5 format."""
    with h5py.File(filename, 'w') as f:
        # General metadata
        f['_appversion'] = 0
        f['name'] = recording.name
        meta_group = f.create_group('meta')
        meta_group.attrs.update((name, val) for (name, val) in recording.meta.items() if val is not None)

        # Patient data
        patient_meta = recording.patient.to_dict()
        patient_group = f.create_group('patient')
        patient_group.attrs.update((name, val) for (name, val) in patient_meta.items() if val is not None)

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
