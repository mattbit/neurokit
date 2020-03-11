import dateparser
import unidecode
import chardet
import shutil
import re
import mne
import logging
import numpy as np
import pandas as pd
from fractions import Fraction
from pyedflib import EdfWriter
from pathlib import Path

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


def read_edf(path):
    """Read an EDF/EDF+ file"""

    raw = mne.io.read_raw_edf(path, stim_channel=None,
                              preload=True, verbose=False)
    return _recording_from_mne_raw(raw)


def write_edf(recording, path, artifacts=False):
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

        writer.set_number_of_annotation_signals(1 + artifacts)

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
        if artifacts:
            for item in recording.artifacts.itertuples():
                onset = (item.start - recording.start_date).total_seconds()
                duration = (item.end - item.start).total_seconds()
                description = f'ARTIFACT | {item.channel} | {item.description}'
                writer.writeAnnotation(onset, duration, description)

    finally:
        writer.close()


def _calc_datarecord_params(frequency):
    f = Fraction(frequency).limit_denominator(60)

    return f.denominator, f.numerator


class PatientInfo:
    _re = re.compile(
        r'^(?P<code>[^\s]+)\s+(?P<sex>[MFX])\s+(?P<date>(?:\d{2}-\w{3,4}\.?-\d{4}|X))\s+(?P<name>[^\s]+)(?P<fields>(?:\s+[^\s]+)*)', re.UNICODE | re.IGNORECASE)

    def __init__(self, code=None, sex=None, date=None, name=None, *extras):
        self.code = code
        self.sex = sex
        self.date = date
        self.name = name
        self.extras = extras

    @classmethod
    def parse(cls, raw):
        raw = raw.strip()
        match = cls._re.match(raw)
        if not match:
            return cls(None, None, None, None, *raw.split(' '))

        code = match['code']
        sex = match['sex'].upper()
        date = dateparser.parse(match['date'])
        name = match['name']
        extras = match['fields'].strip().split()

        return cls(code, sex, date, name, *extras)

    def anonymize(self):
        self.code = None
        self.sex = None
        self.date = None
        self.name = None
        self.extras = []
        return self

    def format(self):
        fmt_code = self.code or 'X'
        fmt_sex = self.sex or 'X'
        fmt_date = self.date.strftime('%d-%b-%Y').upper() if self.date else 'X'
        fmt_name = self.name or 'X'
        fmt_extras = ' '.join(self.extras)
        fmt_info = f'{fmt_code} {fmt_sex} {fmt_date} {fmt_name} {fmt_extras}'
        return unidecode.unidecode(fmt_info).strip()


class RecordingInfo:
    _re = re.compile(
        r'^Startdate\s+(?P<date>\d{2}-\w{3,4}\.?-\d{4})(?P<fields>(?:\s+[^\s]+)*)', re.UNICODE | re.IGNORECASE)

    def __init__(self, date, code=None, technician=None, equipment=None, *extras):
        self.date = date
        self.code = code
        self.technician = technician
        self.equipment = equipment
        self.extras = extras

    @classmethod
    def parse(cls, raw):
        raw = raw.strip()
        match = cls._re.match(raw)
        if not match:
            # @todo: ParseException
            raise Exception('Invalid recording information')

        date = dateparser.parse(match['date'])
        fields = match['fields'].strip().split(' ')
        missing = 3 - len(fields)
        if missing > 0:
            fields += [None] * missing

        code = fields[0]
        technician = fields[1]
        equipment = fields[2]
        extras = fields[3:]

        return cls(date, code, technician, equipment, *extras)

    def anonymize(self):
        self.code = None
        self.technician = None
        self.equipment = None
        self.extras = []
        return self

    def format(self):
        fmt_date = self.date.strftime('%d-%b-%Y').upper()
        fmt_code = self.code or 'X'
        fmt_technician = self.technician or 'X'
        fmt_equipment = self.equipment or 'X'
        fmt_extras = ' '.join(self.extras)
        fmt_info = f'Startdate {fmt_date} {fmt_code} {fmt_technician} {fmt_equipment} {fmt_extras}'
        return unidecode.unidecode(fmt_info).strip()


def fix_edf(file, dest, anonymize=False):
    file = Path(file)
    dest = Path(dest)
    with file.open('rb') as input_file:
        raw_header = input_file.read(256)

    # Detect the encoding
    charset = chardet.detect(raw_header)
    encoding = charset['encoding']
    if encoding != 'ascii':
        logging.warning(f'Detected non-standard encoding: {encoding}.')

    # header_size = int(raw_header[184:192].decode(encoding))
    pat_info = PatientInfo.parse(raw_header[8:88].decode(encoding))
    rec_info = RecordingInfo.parse(raw_header[88:168].decode(encoding))

    if anonymize:
        pat_info.anonymize()
        rec_info.anonymize()

    shutil.copy(file, dest)
    output = dest.open('rb+')

    output.seek(8)
    output.write(pat_info.format().encode('ascii').ljust(
        80)[:80] + rec_info.format().encode('ascii').ljust(80)[:80])
    output.close()
