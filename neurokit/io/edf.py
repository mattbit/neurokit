import mne
import logging
import numpy as np
import pandas as pd
from fractions import Fraction
from pyedflib import EdfWriter

from .model import Recording
from ..internal import LoggingContext


def read_edf(path):
    """Read an EDF/EDF+ file"""
    mne.utils.logger.setLevel(logging.WARNING)

    raw = mne.io.read_raw_edf(path, stim_channel=None,
                              preload=True, verbose=False)
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


def write_edf(recording, path):
    writer = EdfWriter(str(path), len(recording.channels))
    try:
        duration, samples_per_record = _calc_datarecord_params(
            recording.frequency)
        if recording.id is not None:
            writer.setAdmincode(recording.id)
            writer.setPatientCode(recording.id)
            writer.setPatientName(recording.id)

        # patient_info = ' '.join(
        #     f'{key}={value}' for key, value in recording.patient.items())
        # writer.setPatientAdditional(patient_info)

        writer.setStartdatetime(recording.start_date)

        phys_max = recording.data.values.max()
        phys_min = recording.data.values.min()

        for n, channel in enumerate(recording.channels):
            writer.setLabel(n, channel)
            writer.setPhysicalDimension(n, 'uV')
            writer.setSamplefrequency(n, samples_per_record)
            writer.setLabel(n, channel)
            writer.setDigitalMaximum(n, 32767)
            writer.setDigitalMinimum(n, -32768)
            writer.setPhysicalMaximum(n, phys_max)
            writer.setPhysicalMinimum(n, phys_min)

        writer.setDatarecordDuration(duration * 100000)

        data = recording.data.to_numpy()
        if data.shape[0] % samples_per_record != 0:
            pad = samples_per_record - data.shape[0] % samples_per_record
            data = np.pad(data, ((0, pad), (0, 0)))

        num_records = data.shape[0] // samples_per_record
        num_channels = len(recording.channels)
        raw = data.reshape((num_records, samples_per_record, num_channels))
        for block in raw:
            writer.blockWritePhysicalSamples(block.ravel('F'))

        # Write annotations
        for item in recording.annotations.itertuples():
            onset = (item.start - recording.start_date).total_seconds()
            duration = (item.end - item.start).total_seconds()
            writer.writeAnnotation(onset, duration, item.description)

        # Write artifacts as annotations
        # for item in recording.artifacts.itertuples():
        #     onset = (item.start - recording.start_date).total_seconds()
        #     duration = (item.end - item.start).total_seconds()
        #     description = f'ARTIFACT | {item.channel} | {item.description}'
        #     writer.writeAnnotation(onset, duration, description)

    finally:
        writer.close()


def _calc_datarecord_params(frequency):
    f = Fraction(frequency).limit_denominator(60)

    return f.denominator, f.numerator
