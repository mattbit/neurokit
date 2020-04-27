import os
import json
import shutil
import tarfile
import tempfile
import fastparquet
import pandas as pd

from ..core import Recording


def read_nkr(filename):
    """Read a Recording from a parquet file."""
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

    # Artifacts
    art_path = os.path.join(tmp, 'artifacts.parq')
    artifacts = None
    if os.path.exists(art_path):
        artifacts = fastparquet.ParquetFile(art_path).to_pandas()
        artifacts.start = pd.to_datetime(artifacts.start)
        artifacts.end = pd.to_datetime(artifacts.end)

    # Annotations
    annots_path = os.path.join(tmp, 'annotations.parq')
    annotations = None
    if os.path.exists(annots_path):
        annotations = fastparquet.ParquetFile(annots_path).to_pandas()
        annotations['start'] = pd.to_datetime(annotations['start'])
        annotations['end'] = pd.to_datetime(annotations['end'])

    shutil.rmtree(tmp)

    return Recording(data, annotations=annotations, artifacts=artifacts,
                     meta=meta)


def write_nkr(recording, filename):
    """Write a Recording to a parquet file."""
    tf = tarfile.open(filename, mode='w')
    tmp = tempfile.mkdtemp(prefix='nk_tmp')

    # Metadata
    meta_path = os.path.join(tmp, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(recording.meta, f)
    tf.add(meta_path, 'meta.json')

    # Data
    data = recording.data.copy()
    data.index = data.index.astype('int64')
    data_path = os.path.join(tmp, 'data.parq')
    fastparquet.write(data_path, data, compression='gzip')
    del data
    tf.add(data_path, 'data.parq')

    # Artifacts
    artifacts = recording.artifacts.copy()
    artifacts.start = artifacts.start.astype('int64')
    artifacts.end = artifacts.end.astype('int64')
    art_path = os.path.join(tmp, 'artifacts.parq')
    fastparquet.write(art_path, artifacts, compression='gzip')
    tf.add(art_path, 'artifacts.parq')

    # Annotations
    annots = recording.annotations.copy()
    annots['start'] = annots['start'].astype('int64')
    annots['end'] = annots['end'].astype('int64')
    annots_path = os.path.join(tmp, 'annotations.parq')
    fastparquet.write(annots_path, annots, compression='gzip')
    tf.add(annots_path, 'annotations.parq')

    tf.close()

    shutil.rmtree(tmp)
