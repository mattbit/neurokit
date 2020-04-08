import numpy as np
import pandas as pd


class Recording:
    """Neurophysiological Recording"""

    def __init__(self, data, meta=None, annotations=None, artifacts=None, frequency=None, id=None):
        self.data = data

        if annotations is None:
            annotations = pd.DataFrame(
                columns=['start', 'end', 'channel', 'description'])
        self.annotations = annotations

        self.meta = meta or {}
        if 'id' not in self.meta:
            self.meta['id'] = id
        if frequency is not None:
            self.meta['frequency'] = frequency

        if 'frequency' not in self.meta:
            self.meta['frequency'] = (
                len(data) - 1) / self.duration.total_seconds()

        if artifacts is None:
            artifacts = pd.DataFrame(
                columns=['start', 'end', 'channel', 'description'])
        self.artifacts = artifacts

    @property
    def id(self):
        return self.meta['id']

    @property
    def frequency(self):
        return self.meta['frequency']

    @property
    def channels(self):
        return list(self.data.columns)

    @property
    def duration(self):
        return self.end_date - self.start_date

    @property
    def start_date(self):
        return self.data.index.min()

    @property
    def end_date(self):
        return self.data.index.max()

    def __repr__(self):
        duration_sec = self.duration.total_seconds()
        hh = int(duration_sec // 3600)
        mm = int(duration_sec % 3600 / 60)
        ss = round(duration_sec % 60)
        duration = f'{hh:02}:{mm:02}:{ss:02}'

        return f"<Recording '{self.id}' {self.start_date} ({duration})>"

    def copy(self, empty=False):
        data = pd.DataFrame(
            columns=self.channels) if empty else self.data.copy()
        artifacts = None if empty else self.artifacts.copy()
        annotations = None if empty else self.annotations.copy()

        return Recording(data, self.meta.copy(),
                         annotations=annotations, artifacts=artifacts)

    def filter(self, low, high=None):
        from ..preprocessing import filter as _filter
        if not low:
            return _filter.lowpass(self, high)

        if not high:
            return _filter.highpass(self, low)

        return _filter.bandpass(self, (low, high))

    def artifacts_to_nan(self, pad=0):
        """Convert artifacts to NaN.

        Parameters
        ----------
        pad : float
            Padding (in seconds) that should be added around artifacts. For
            example, if `pad = 1` all values from 1 second before the beginning
            of the artifact to 1 second after its end will be set to `np.nan`.

        Returns
        -------
        recording : neurokit.io.Recording
            A copy of the orignal recording with `numpy.nan` instead of
            artifacted signal.
        """
        rec = self.copy()
        dt = pd.Timedelta(pad, unit='s')
        for artifact in self.artifacts.itertuples():
            start = artifact.start - dt
            end = artifact.end + dt
            chs = slice(None) if artifact.channel is None else artifact.channel
            rec.data.loc[start:end, chs] = np.nan

        return rec

    def to_edf(self, filename, **kwargs):
        from .edf import write_edf

        return write_edf(self, filename, **kwargs)

    def to_nkr(self, filename, **kwargs):
        from neurokit.io.nk import write_nkr

        return write_nkr(self, filename, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise Exception('Recording can only be sliced.')

        try:
            start = self.start_date + \
                pd.to_timedelta(key.start) if key.start is not None else None
            end = self.start_date + \
                pd.to_timedelta(key.stop) if key.stop is not None else None
        except ValueError:
            start, end = pd.to_datetime(key.start), pd.to_datetime(key.stop)

        data = self.data.loc[start:end]
        artifacts = _slice_intervals(self.artifacts, start, end)
        annotations = _slice_intervals(self.annotations, start, end)

        return Recording(data, self.meta,
                         annotations=annotations, artifacts=artifacts)


def _slice_intervals(df, start, end):
    mask = np.ones(len(df), dtype=bool)

    if end is not None:
        mask &= (df['start'] < end)

    if start is not None:
        mask &= (df['end'] > start)

    return df[mask]
