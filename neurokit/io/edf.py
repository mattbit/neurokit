import re
import mne
import math
import shutil
import logging
import datetime
import unidecode
import dateparser
import numpy as np
from pathlib import Path
from fractions import Fraction
from pyedflib import EdfWriter
from ._mne import _recording_from_mne_raw
from ..internals import import_optional_dependency


def read_edf(path):
    """Read an EDF/EDF+ file"""

    raw = mne.io.read_raw_edf(path, stim_channel=None,
                              preload=True, verbose=False)
    return _recording_from_mne_raw(raw)


def write_edf(recording, path):
    writer = EdfWriter(str(path), len(recording.data.channels))
    try:
        duration, samples_per_record = _calc_datarecord_params(
            recording.frequency)
        if recording.name is not None:
            id_string = str(recording.name)
            writer.setAdmincode(id_string)
            writer.setPatientCode(id_string)

        # patient_info = ' '.join(
        #     f'{key}={value}' for key, value in recording.patient.items())
        # writer.setPatientAdditional(patient_info)

        if 'date' in recording.meta:
            start_date = recording.meta['date'] + recording.data.index.min()
        else:
            start_date = datetime.datetime.fromtimestamp(0)
        writer.setStartdatetime(start_date)

        phys_max = np.nanmax(recording.data.values)
        phys_min = np.nanmin(recording.data.values)

        for n, channel in enumerate(recording.data.channels):
            writer.setLabel(n, channel)
            writer.setPhysicalDimension(n, 'uV')
            writer.setSamplefrequency(n, samples_per_record)
            writer.setLabel(n, channel)
            writer.setDigitalMaximum(n, 32767)
            writer.setDigitalMinimum(n, -32768)
            writer.setPhysicalMaximum(n, phys_max)
            writer.setPhysicalMinimum(n, phys_min)

        writer.setDatarecordDuration(duration * 100000)

        n_annotation = 1
        if recording.es.has('annotations'):
            duration = recording.duration.total_seconds()
            n_annotation = math.ceil(
                20 * len(recording.es.annotations) / duration)
        writer.set_number_of_annotation_signals(min(n_annotation, 64))

        data = recording.data.to_numpy()
        if data.shape[0] % samples_per_record != 0:
            pad = samples_per_record - data.shape[0] % samples_per_record
            data = np.pad(data, ((0, pad), (0, 0)))

        num_records = data.shape[0] // samples_per_record
        num_channels = len(recording.data.channels)
        raw = data.reshape((num_records, samples_per_record, num_channels))
        for block in raw:
            writer.blockWritePhysicalSamples(block.ravel('F'))

        # Write annotations
        if recording.es.has('annotations'):
            start_interval = recording.data.index.min()
            for item in recording.es['annotations']:
                onset = (item.start - start_interval).total_seconds()
                duration = (item.end - item.start).total_seconds()
                writer.writeAnnotation(onset, duration, item.description)

    finally:
        writer.close()


def _calc_datarecord_params(frequency):
    f = Fraction(frequency).limit_denominator(60)

    return f.denominator, f.numerator


class PatientInfo:
    _re = re.compile(
        r'^(?P<code>[^\s]+)\s+(?P<sex>[MFX])\s+(?P<date>(?:\d{2}-\w{3,4}\.?-\d{4}|X))\s+(?P<name>[^\s]+)(?P<fields>(?:\s+[^\s]+)*)', re.UNICODE | re.IGNORECASE)

    def __init__(self, code=None, sex=None, date=None, name=None, extras=None):
        self.code = code
        self.sex = sex
        self.date = date
        self.name = name
        self.extras = extras or []

    @classmethod
    def parse(cls, raw):
        raw = raw.strip()
        match = cls._re.match(raw)
        if not match:
            return cls(None, None, None, None, raw.split(' '))

        code = match['code']
        sex = match['sex'].upper()
        date = dateparser.parse(match['date'])
        name = match['name']
        extras = match['fields'].strip().split()

        return cls(code, sex, date, name, extras)

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

    def __init__(self, date, code=None, technician=None, equipment=None, extras=None):
        self.date = date
        self.code = code
        self.technician = technician
        self.equipment = equipment
        self.extras = extras or []

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

        return cls(date, code, technician, equipment, extras)

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
    chardet = import_optional_dependency('chardet')

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
