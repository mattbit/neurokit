import numpy as np
import pandas as pd


def mask_to_intervals(mask, index=None):
    if not np.any(mask):
        return []

    edges = np.flatnonzero(np.diff(np.pad(mask, 1)))
    intervals = edges.reshape((len(edges) // 2, 2))

    if index is not None:
        return [(index[i], index[j - 1]) for i, j in intervals]

    return intervals


def intervals_to_mask(intervals, index):
    mask = np.zeros(len(index), dtype=bool)

    for start, end in intervals:
        mask[index.get_loc(start):index.get_loc(end)] = True

    return mask
