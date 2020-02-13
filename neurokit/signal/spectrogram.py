import numpy as np
import scipy.signal as ss


def spectrogram(recording, window=10, overlap=0.5, channels=None, remove_artifacts=True):
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
