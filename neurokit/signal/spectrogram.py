import numpy as np
import scipy.signal as ss
from types import Sequence

from ..io.model import Recording


def spectrogram(recording: Recording, window=10., overlap=0.5,
                channels: Sequence = None, remove_artifacts=True):
    """Produce the spectrogram for a Recording.

    If multiple channels are selected, it will first calculate the spectrogram
    for each channel individually and then average the result.

    Parameters
    ----------
    recording : Recording
        The recording object.
    window : float, optional
        Length of the window for the STFT, in seconds.
    overlap : float, optional
        Proportion of overlap between windows (must be less than 1).
    channels : Sequence
        List of channels to use. If `None`, all channels will be used.
    remove_artifacts : bool
        Whether to remove the artifacts associated to the recording before
        calculating the spectrogram.
    """
    if channels is None:
        channels = recording.channels

    n_window = int(recording.frequency * window)
    n_overlap = int(recording.frequency * window * overlap)

    data_na = recording.data.copy()
    if remove_artifacts:
        for artifact in recording.artifacts.itertuples():
            data_na.loc[artifact.start:artifact.end, artifact.channel] = np.nan

    data = data_na.loc[:, channels].values
    f, t, S = ss.spectrogram(data, fs=recording.frequency,
                             nperseg=n_window, noverlap=n_overlap, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.nansum(S, axis=1) / (len(channels) - np.isnan(S).sum(axis=1))

    return f, t, S
