import numpy as np
import pandas as pd
from neurokit.sim import simulated_eeg_recording


def test_artifacts_to_nan():
    rec = simulated_eeg_recording(channels=['CH1', 'CH2'], duration=10,
                                  frequency=100)

    # Artifact 1: from 1 s to 2 s, all channels
    start_1 = rec.start_date + pd.Timedelta(1, unit='s')
    end_1 = rec.start_date + pd.Timedelta(2, unit='s')
    rec.artifacts.loc[0, :] = start_1, end_1, None, 'test 1'

    # Artifact 2: from 3 s to 5 s, only channel CH1
    start_2 = rec.start_date + pd.Timedelta(3, unit='s')
    end_2 = rec.start_date + pd.Timedelta(5, unit='s')
    rec.artifacts.loc[1, :] = start_2, end_2, 'CH1', 'test 2'

    # Artifact 3: from 4.5 s to 7 s, only channel CH2
    start_3 = rec.start_date + pd.Timedelta(4.5, unit='s')
    end_3 = rec.start_date + pd.Timedelta(7, unit='s')
    rec.artifacts.loc[2, :] = start_3, end_3, 'CH2', 'test 3'

    rec_with_nan = rec.artifacts_to_nan()

    assert np.isnan(rec_with_nan.data['CH1']).sum() == 101 + 201
    assert np.isnan(rec_with_nan.data['CH2']).sum() == 101 + 251
    assert rec_with_nan.data.loc[start_1:end_1, 'CH1'].isna().all()
    assert rec_with_nan.data.loc[start_1:end_1, 'CH2'].isna().all()

    assert rec_with_nan.data.loc[start_2:end_2, 'CH1'].isna().all()
    _start_3_pre = start_3 - pd.Timedelta(0.01, unit='s')
    assert not rec_with_nan.data.loc[start_2:_start_3_pre, 'CH2'].isna().any()

    assert not rec_with_nan.data.loc[end_2:start_3, 'CH1'].isna().any()
    assert rec_with_nan.data.loc[start_3:end_3, 'CH2'].isna().all()
