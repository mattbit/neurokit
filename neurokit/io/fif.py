import mne
from ._mne import _recording_from_mne_raw


def read_fif(path):
    '''Read a MNE FIF file.'''
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    return _recording_from_mne_raw(raw)
