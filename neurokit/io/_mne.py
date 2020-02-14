import mne
import logging
import pandas as pd
from .model import Recording


def _recording_from_mne_raw(raw: mne.io.Raw):
    mne.utils.logger.setLevel(logging.WARNING)

    timestamp, microseconds = raw.info['meas_date']
    initial_time = timestamp * 10**9 + microseconds * 10**3

    data = raw.to_data_frame(scaling_time=10**9)

    data.index = pd.to_datetime(initial_time + data.index, unit='ns')

    meta = {
        'frequency': raw.info['sfreq'],
        'resolution': [ch['range'] / ch['cal'] for ch in raw.info['chs']],
    }

    # Annotations
    annot_list = []
    for annotation in raw.annotations:
        onset = pd.Timedelta(seconds=annotation['onset'])
        duration = pd.Timedelta(seconds=annotation['duration'])
        annot_list.append({
            'start': data.index.min() + onset,
            'end': data.index.min() + onset + duration,
            'channel': None,
            'description': annotation['description']
        })

    annotations = pd.DataFrame(
        annot_list, columns=['start', 'end', 'channel', 'description'])

    return Recording(data, meta=meta, annotations=annotations)
