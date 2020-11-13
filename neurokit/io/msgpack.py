import io
import msgpack
import datetime
import numpy as np
import pandas as pd

from ..core import Recording
from ..internals import import_optional_dependency


def read_msgpack(filename):
    """Read a msgpack encoded Recording."""
    msgpack = import_optional_dependency('msgpack')

    if isinstance(filename, io.IOBase):
        rec_data = msgpack.unpack(filename, object_pairs_hook=_msgpack_decoder)
    else:
        with open(filename, 'rb') as f:
            rec_data = msgpack.unpack(f, object_pairs_hook=_msgpack_decoder)

    return Recording.from_dict(rec_data)


def write_msgpack(recording: Recording, filename):
    """Write a Recording object in the msgpack format."""
    msgpack = import_optional_dependency('msgpack')

    if isinstance(filename, io.IOBase):
        msgpack.pack(recording.to_dict(),
                     filename,
                     use_bin_type=False,
                     strict_types=False,
                     default=_msgpack_encoder)
    else:
        with open(filename, 'wb') as output_file:
            msgpack.pack(recording.to_dict(),
                         output_file,
                         use_bin_type=False,
                         strict_types=False,
                         default=_msgpack_encoder)


def _msgpack_encoder(obj):
    if isinstance(obj, datetime.datetime):
        obj = pd.to_datetime(obj)
        seconds = obj.value // 1000000000
        nanoseconds = obj.microsecond * 1000 + obj.nanosecond
        return msgpack.ext.Timestamp(seconds, nanoseconds)
    if isinstance(obj, datetime.timedelta):
        return int(pd.to_timedelta(obj).asm8.astype(int))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def _maybe_cast(obj):
    if isinstance(obj, msgpack.ext.Timestamp):
        return pd.to_datetime(obj.seconds * 1000000000 + obj.nanoseconds,
                              unit='ns')
    if isinstance(obj, list):
        return np.array(obj)

    return obj


def _msgpack_decoder(items):
    return {key: _maybe_cast(value) for key, value in items}
