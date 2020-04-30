from collections.abc import Sequence

from ..core.recording import Recording


def merge_recordings(recordings: Sequence[Recording], **kwargs):
    """Merge sequential recordings.

    It uses the `recording.meta['date']` value to define if the objects
    are sequential.
    """
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
    raise NotImplementedError('This method is not implemented yet!')
    # for group in groups:
    #     if len(group) < 2:
    #         merged.append(group[0])
    #         continue
    #
    #     data = pd.concat([r.data for r in group])
    #     artifacts = pd.concat([r.artifacts for r in group])
    #     annotations = pd.concat([r.annotations for r in group])
    #
    #     meta = {}
    #     for r in group:
    #         meta.update(r.meta)
    #     ids = [r.id for r in group if r.id is not None]
    #     res = np.array([r.meta['resolution'] for r in group]).max(axis=0)
    #
    #     meta.update({
    #         'id': '+'.join(ids) if ids else None,
    #         'frequency': group[0].frequency,
    #         'resolution': res.tolist()
    #     })
    #
    #     merged.append(Recording(data, annotations=annotations,
    #                             artifacts=artifacts, meta=meta))
    #
    # return merged


def is_sequential_recording(recording: Recording, other: Recording, tol='auto'):
    """Check if recordings are compatible and sequential."""
    if recording.data.frequency != other.data.frequency:
        return False

    if recording.data.channels != other.data.channels:
        return False

    recording_end_date = (recording.meta['date'] + recording.data.duration)
    time_diff = (other.meta['date'] - recording_end_date).total_seconds()
    if tol == 'auto':
        tol = 1 / recording.frequency

    if time_diff > tol or time_diff < 0:
        return False

    return True


def split_recording(recording, times):
    """Split recording at given time or timedelta."""
    raise NotImplementedError('The method is not implemented yet')
    # if not times:
    #     return [recording]
    #
    # times = [None] + list(_ensure_sequence(times)) + [None]
    #
    # return [recording[start:end] for start, end in zip(times[:-1], times[1:])]


def detect_empty_signal(recording):
    """Detect all zeros, electrodes unbranched segments."""
    raise NotImplementedError('The method is not implemented yet')
    # power = recording.data.values ** 2
    # eps = (np.array(recording.meta['resolution']) / 2)[np.newaxis]
    #
    # return (power < eps ** 2).sum(axis=1) == recording.data.shape[1]


def trim_empty(recording):
    raise NotImplementedError('The method is not implemented yet')
    #
    # empty_signal = detect_empty_signal(recording)
    # intervals = mask_to_intervals(empty_signal)
    #
    # if len(intervals) == 0:
    #     return recording.copy()
    #
    # start = recording.data.index.min()
    # end = recording.data.index.max()
    #
    # if intervals[0][0] == 0:
    #     start = recording.data.index[intervals[0][1] - 1]
    #
    # if intervals[-1][1] == len(recording.data):
    #     end = recording.data.index[intervals[-1][0]]
    #
    # return recording[start:end].copy()


def split_in_segments(recording, min_break=300, trim=True):
    raise NotImplementedError('The method is not implemented yet')
    # Detect breaks (all zero, electrodes unbranched)
    # empty_signal = detect_empty_signal(recording)
    #
    # breakpoints = []
    # for start, end in mask_to_intervals(empty_signal, index=recording.data.index):
    #     if (end - start).total_seconds() >= min_break:
    #         breakpoints.append(start + (end - start) / 2)
    #
    # segments = split_recording(recording, breakpoints)
    # if trim:
    #     return [trim_empty(s) for s in segments]
    #
    # return segments


def _ensure_sequence(obj):
    if isinstance(obj, Sequence):
        return obj

    return [obj]
