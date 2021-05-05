import pandas as pd
from neurokit import TimeSeries


def test_time_series_reduce():
    ts = TimeSeries({'CH1': [1, 2, 3, 4, 5],
                     'CH2': [2, 2, 2, 2, 2]},
                    frequency=128)

    assert isinstance(ts.mean(), pd.Series)
    assert ts.mean()['CH1'] == 3
    assert ts.mean()['CH2'] == 2
