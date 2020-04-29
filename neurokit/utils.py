import numpy as np
import pandas as pd

from typing import Sequence, Tuple, Any


def mask_to_intervals(mask: np.ndarray,
                      index: Sequence = None) -> Sequence[Tuple[Any, Any]]:
    """Convert a boolean mask to a sequence of intervals.

    Caveat: when no index is given, the returned values correspond to the
    Python pure integer indexing (starting element included, ending element
    excluded). When an index is passed, pandas label indexing convention
    with strict inclusion is used.
    For example `mask_to_intervals([0, 1, 1, 0])` will return `[(1, 3)]`,
    but `mask_to_intervals([0, 1, 1, 0], ["a", "b", "c", "d"])` will return
    the value `[("b", "c")]`.

    Parameters
    ----------
    mask : numpy.ndarray
        A boolean array.
    index : Sequence, optional
        Elements to use as indices for determining interval start and end. If
        no index is given, integer array indices are used.

    Returns
    -------
    intervals : Sequence[Tuple[Any, Any]]
        A sequence of (start_index, end_index) tuples. Mindful of the caveat
        described above concerning the indexing convention.
    """
    if not np.any(mask):
        return []

    edges = np.flatnonzero(np.diff(np.pad(mask, 1)))
    intervals = edges.reshape((len(edges) // 2, 2))

    if index is not None:
        return [(index[i], index[j - 1]) for i, j in intervals]

    return [(i, j) for i, j in intervals]


def intervals_to_mask(intervals: Sequence[Tuple[Any, Any]],
                      index: Sequence) -> np.ndarray:
    """Convert a sequence of intervals to a boolean mask.

    Caveat: for intervals, pandas label based indexing is used (star and end
    are strictly included).

    Parameters
    ----------
    intervals : Sequence
        A sequence of tuples of (start, end) indices.
    index : Sequence
        The index referenced by the intervals. Passing an ordered
        `pandas.Index` object can drastically improve performances.

    Returns
    -------
    mask : numpy.ndarray
        A booelan array with same length as the index, with `True` elements
        inside the intervals (boundaries stricty included).
    """
    mask = np.zeros(len(index), dtype=bool)

    if not isinstance(index, pd.Index):
        index = pd.Index(index)

    for start, end in intervals:
        mask[index.get_loc(start, method='ffill'):index.get_loc(end, method='bfill') + 1] = True

    return mask
