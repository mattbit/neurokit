import mne
import pandas as pd
from ..core import Recording, TimeSeries, EventSeries, Patient


def _recording_from_mne_raw(raw: mne.io.Raw):
    meta = {
        'date': raw.info.get('meas_date', None),
        'resolution': [ch['range'] / ch['cal']
                       for ch in raw.info.get('chs', [])],
    }

    # Timeseries
    filters = {
        'lowpass': raw.info.get('lowpass', None),
        'highpass': raw.info.get('highpass', None),
        'notch': raw.info.get('notch', None),
    }
    data = raw.to_data_frame(time_format='datetime').set_index('time')
    data.index -= meta['date'] if meta['date'] else data.index.min()
    data = TimeSeries(data, filters=filters, name='data')

    # Annotations
    annotations = []
    for annotation in raw.annotations:
        onset = pd.Timedelta(seconds=annotation['onset'])
        duration = pd.Timedelta(seconds=annotation['duration'])
        annotations.append({
            'start': data.index.min() + onset,
            'end': data.index.min() + onset + duration,
            'code': None,
            'channel': None,
            'description': annotation['description']
        })
    events = [EventSeries(annotations, name='annotations')]

    # Patient
    patient = Patient()
    if raw.info['subject_info']:
        patient.id = raw.info['subject_info'].get('his_id')

    return Recording(timeseries=[data], events=events, meta=meta,
                     patient=patient)
