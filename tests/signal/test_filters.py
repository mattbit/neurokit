import pytest
import numpy as np

from neurokit.sim import simulated_eeg_signal
from neurokit.signal import filters


def test_filters_show_nan_warning():
    ts = simulated_eeg_signal(duration=1)

    # No warnings if everything is fine
    with pytest.warns(None) as record:
        filters.lowpass(ts, 10.)
        filters.highpass(ts, 10.)
        filters.bandpass(ts, (10., 20.))
        filters.notch(ts, 10.)

    assert len(record) == 0

    # Warning if contains np.nan.
    ts.iloc[100] = np.nan

    with pytest.warns(RuntimeWarning) as record:
        filters.lowpass(ts, 10.)
    assert len(record) == 1

    with pytest.warns(RuntimeWarning) as record:
        filters.highpass(ts, 10.)
    assert len(record) == 1

    with pytest.warns(RuntimeWarning) as record:
        filters.bandpass(ts, (10., 20.))
    assert len(record) == 1

    with pytest.warns(RuntimeWarning) as record:
        filters.notch(ts, 10.)
    assert len(record) == 1
