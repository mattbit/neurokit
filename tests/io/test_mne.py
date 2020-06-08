import mne
from neurokit import Recording


def test_read_edf():
    raw = mne.io.read_raw_edf('tests/data/test.edf', verbose=False)

    rec = Recording.from_mne_raw(raw)

    assert (rec.data.values.T == raw.get_data() * 1e6).all()
    assert raw.info['ch_names'][0] == rec.data.channels[0]
    assert rec.data.frequency == raw.info['sfreq']
