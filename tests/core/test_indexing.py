import pandas as pd
from neurokit.core.indexing import FixedTimedeltaIndex, timedelta_range


def test_timedelta_range():
    idx = timedelta_range(start=0, end='10s', freq='100ms')
    assert len(idx) == 101
    assert isinstance(idx, FixedTimedeltaIndex)


def test_fixed_timedelta_index():
    idx = timedelta_range(0, '10s', freq='100ms')
    assert idx.slice_indexer('1s', '2s') == slice(10, 21)
    assert idx.slice_indexer(None, '2s') == slice(0, 21)
    assert idx.slice_indexer(None, '2.0001s') == slice(0, 21)

    assert isinstance(idx + pd.Timedelta(0), FixedTimedeltaIndex)
