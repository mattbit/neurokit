import numpy as np
import pandas as pd
from typing import Sequence
from collections import defaultdict

from ..core.recording import Recording, BaseTimeSeries, TimeSeries, EventSeries
from ..core.common import NamedItemsBag
from ..utils import mask_to_intervals


def concatenate_recordings(recordings: Sequence[Recording]):
    if len(recordings) == 0:
        raise ValueError('No recordings to concatenate')

    if len(recordings) == 1:
        return recordings[0]

    # Calculate offsets with respect to the main timeseries.
    offsets = np.cumsum([pd.Timedelta(0)]
                        + [rec.duration for rec in recordings[:-1]])

    _ts = defaultdict(lambda: [])
    _es = defaultdict(lambda: [])
    for rec, offset in zip(recordings, offsets):
        # Timeseries
        for series in rec.ts:
            s = series.copy()
            s.index += offset
            _ts[s.name].append(s)

        # Events
        for events in rec.es:
            e = events.data.copy()
            e.start += offset
            e.end += offset
            _es[events.name].append(e)

    _finalized_ts = []
    for name, ss in _ts.items():
        series = pd.concat(ss)
        series.name = name
        _finalized_ts.append(series)

    ts = NamedItemsBag(_finalized_ts, dtype=BaseTimeSeries)
    es = NamedItemsBag([EventSeries(data=pd.concat(ee), name=name)
                        for name, ee in _es.items()], dtype=EventSeries)

    # Prepare recording with meta
    recording = recordings[0].copy()
    for rec in reversed(recordings):
        recording.meta.update(rec.meta)

    # ensure correct start date
    if 'date' in recordings[0].meta:
        recording.meta['date'] = recordings[0].meta['date']

    # use max resolution
    resolutions = [rec.meta.get('resolution', []) for rec in recordings]
    recording.meta['resolution'] = np.max(resolutions, axis=0)

    recording.ts = ts
    recording.es = es

    return recording


def merge_sequential_recordings(recordings: Sequence[Recording], **kwargs):
    """Merge sequential recordings.

    It uses the `recording.meta['date']` value to define if the objects
    are sequential.
    """
    for r in recordings:
        if 'date' not in r.meta:
            raise ValueError('Recordings must have a `meta[\'date\']`'
                             'attribute to allow sequential merging.')

    recs = sorted(recordings, key=lambda r: r.meta['date'])
    groups = [[recs[0]]]

    for rec in recs[1:]:
        last_group = groups[-1]
        if is_sequential_recording(last_group[-1], rec, **kwargs):
            # append to existing group
            last_group.append(rec)
        else:
            # new group
            groups.append([rec])

    # merge groups
    merged = []

    for group in groups:
        merged.append(concatenate_recordings(group))

    return merged


def is_sequential_recording(recording: Recording,
                            other: Recording,
                            tol='auto'):
    """Check if recordings are compatible and sequential."""
    shared_series = {s.name for s in recording.ts} & {s.name for s in other.ts}

    for name in shared_series:
        s1, s2 = recording.ts[name], other.ts[name]

        # No incompatible types (UnevenTimeSeries, TimeSeries)
        if type(s1) != type(s2):
            return False

        # If not Uneven, frequency must match
        if (isinstance(s1, TimeSeries) and s1.frequency != s2.frequency):
            return False

    recording_end_date = (recording.meta['date'] + recording.data.duration)
    time_diff = (other.meta['date'] - recording_end_date).total_seconds()
    if tol == 'auto':
        tol = 1 / recording.frequency

    if time_diff > tol or time_diff < 0:
        return False

    return True


def split_recording(recording, times):
    """Split recording at given time."""
    if not times:
        return [recording.copy()]

    if not isinstance(times, Sequence):
        times = [times]

    times = [None] + list(times) + [None]

    return [recording.slice(start, end)
            for start, end in zip(times[:-1], times[1:])]


def detect_empty_signal(recording):
    """Detect all zeros, electrodes unbranched segments."""
    power = recording.data.values ** 2
    eps = (np.array(recording.meta['resolution']) / 2)[np.newaxis]

    return (power < eps ** 2).sum(axis=1) == recording.data.shape[1]


def trim_empty(recording):
    empty_signal = detect_empty_signal(recording)
    intervals = mask_to_intervals(empty_signal)

    if len(intervals) == 0:
        return recording.copy()

    start = None
    end = None

    if intervals[0][0] == 0:
        start = recording.data.index[intervals[0][1] - 1]

    if intervals[-1][1] == len(recording.data):
        end = recording.data.index[intervals[-1][0]]

    return recording.slice(start, end)


def split_in_segments(recording, min_break=300, trim=True):
    # Detect breaks (all zero, electrodes unbranched)
    no_signal = detect_empty_signal(recording)

    breakpoints = []
    for start, end in mask_to_intervals(no_signal, index=recording.data.index):
        if (end - start).total_seconds() >= min_break:
            breakpoints.append(start + (end - start) / 2)

    segments = split_recording(recording, breakpoints)
    if trim:
        return [trim_empty(s) for s in segments]

    return segments


def reset_offset(recording):
    rec = recording.copy()
    offset = rec.data.index.min()

    for ts in rec.ts:
        ts.index -= offset

    for es in rec.es:
        es.data.start -= offset
        es.data.end -= offset
        es.index = pd.MultiIndex.from_frame(es.data.loc[:, ('start', 'end')])

    if 'date' in rec.meta:
        rec.meta['date'] += offset

    return rec
