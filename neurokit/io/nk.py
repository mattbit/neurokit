import os
import json
import shutil
import tarfile
import tempfile
import pandas as pd

from ..core.series import EventSeries
from ..core.recording import Recording
from ..internals import import_optional_dependency


def read_nkr(filename):
    """Read a Recording from a parquet file."""
    fastparquet = import_optional_dependency('fastparquet')

    tf = tarfile.open(filename, mode='r')
    tmp = tempfile.mkdtemp(prefix='nk_tmp')

    members = [tf.getmember('meta.json'),
               tf.getmember('data.parq')]
    try:
        members.append(tf.getmember('artifacts.parq'))
    except KeyError:
        pass

    try:
        members.append(tf.getmember('annotations.parq'))
    except KeyError:
        pass

    tf.extractall(tmp, members=members)
    tf.close()

    # Meta
    with open(os.path.join(tmp, 'meta.json'), 'r') as f:
        meta = json.load(f)

    # Data
    data_path = os.path.join(tmp, 'data.parq')
    data = fastparquet.ParquetFile(data_path).to_pandas()
    data.index = pd.to_datetime(data.index)
    meta['date'] = data.index.min()

    eventseries = []

    # Artifacts
    art_path = os.path.join(tmp, 'artifacts.parq')
    if os.path.exists(art_path):
        artifacts = fastparquet.ParquetFile(art_path).to_pandas()
        artifacts.start = pd.to_datetime(artifacts.start)
        artifacts.end = pd.to_datetime(artifacts.end)
        eventseries.append(EventSeries(artifacts, name='artifacts'))

    # Annotations
    annots_path = os.path.join(tmp, 'annotations.parq')
    if os.path.exists(annots_path):
        annotations = fastparquet.ParquetFile(annots_path).to_pandas()
        annotations['start'] = pd.to_datetime(annotations['start'])
        annotations['end'] = pd.to_datetime(annotations['end'])
        eventseries.append(EventSeries(annotations, name='annotations'))

    shutil.rmtree(tmp)

    return Recording(data, name=meta.get('id'), meta=meta,
                     events=eventseries)


def write_nkr(recording, filename):
    """Write a Recording to a parquet file."""
    raise NotImplementedError('The nkr format has been deprecated.')
