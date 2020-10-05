"""Restoration functions that are specific for Masimo Sedline."""

import pywt
import numpy as np
import pandas as pd
import networkx as nx
import scipy.ndimage as ndi

from ..utils import mask_to_intervals
from ..io.utils import detect_empty_signal
from ..preprocessing import detect_signal_artifacts
from ..preprocessing import ConstantSignalDetector, HighAmplitudeDetector


def detect_scale_changes(recording, channels=None, merge_interval=1):
    if not channels:
        channels = recording.data.columns

    data = recording.data
    iso = ndi.binary_opening(detect_empty_signal(recording), iterations=2)

    idx = data.index
    intervals = [(idx[i], idx[j - 1]) for i, j in mask_to_intervals(iso)
                 if 0.49 < (idx[j - 1] - idx[i]).total_seconds()]

    if not intervals:
        return []

    if merge_interval and merge_interval > 0:
        merged = [intervals[0]]
        for interval in intervals[1:]:
            if (interval[0] - merged[-1][1]).total_seconds() < merge_interval:
                merged[-1] = (merged[-1][0], interval[1])
            else:
                merged.append(interval)
        intervals = merged

    intervals = [(None, idx.min())] + intervals + [(idx.max(), None)]

    detections = []
    for n in range(1, len(intervals) - 1):
        start, end = intervals[n]
        prev_end = intervals[n - 1][1]
        next_start = intervals[n + 1][0]
        mid_time = start + (end - start) / 2
        window = pd.Timedelta(seconds=5)

        mad_ratio = None
        wav_ratio = None

        if window.total_seconds() > 0:
            pre_vals = data.loc[max(
                prev_end, mid_time - window):start, channels].values
            post_vals = data.loc[end:min(
                next_start, mid_time + window), channels].values

            if min(len(pre_vals), len(post_vals)) >= 30:
                pre_mask = np.zeros(len(pre_vals), dtype=bool)
                for ch in range(pre_vals.shape[1]):
                    _raw_rms = np.sqrt((pre_vals[:, ch] ** 2).mean())
                    _detectors = [
                        ConstantSignalDetector(),
                        HighAmplitudeDetector(low=_raw_rms, high=3 * _raw_rms)
                    ]
                    pre_mask |= detect_signal_artifacts(
                        pre_vals[:, ch], _detectors)
                _pre_vals = pre_vals[~pre_mask]

                post_mask = np.zeros(len(post_vals), dtype=bool)
                for ch in range(post_vals.shape[1]):
                    _raw_rms = np.sqrt((post_vals[:, ch] ** 2).mean())
                    _detectors = [
                        ConstantSignalDetector(),
                        HighAmplitudeDetector(low=_raw_rms, high=3 * _raw_rms)
                    ]
                    post_mask |= detect_signal_artifacts(
                        post_vals[:, ch], _detectors)
                _post_vals = post_vals[~post_mask]

                # Robust scale estimator: MAD
                if pre_mask.mean() < 0.5 and post_mask.mean() < 0.5:
                    mad_pre = np.median(np.abs(pre_vals), axis=0)
                    mad_post = np.median(np.abs(post_vals), axis=0)
                    mad_ratio = (mad_pre / mad_post).mean()

                if min(len(_pre_vals), len(_post_vals)) > 128:
                    _wavelet = 'db4'
                    _, pre_cD, _ = pywt.wavedec(pre_vals, _wavelet, level=2,
                                                axis=0)
                    pre_D = pywt.waverec([None, pre_cD, None], _wavelet,
                                         axis=0)
                    _, post_cD, _ = pywt.wavedec(post_vals, _wavelet, level=2,
                                                 axis=0)
                    post_D = pywt.waverec(
                        [None, post_cD, None], _wavelet, axis=0)

                    pre_D = pre_D[:len(pre_vals)]
                    post_D = post_D[:len(post_vals)]

                    wav_ratio = np.sqrt(np.mean((pre_D[~pre_mask]**2).mean(
                        axis=0) / (post_D[~post_mask]**2).mean(axis=0)))

        detections.append({
            'start': start,
            'end': end,
            'wav_ratio': wav_ratio,
            'mad_ratio': mad_ratio,
            'pre_vals': pre_vals,
            'post_vals': post_vals,
        })

    return detections


def find_best_scale_sequence(detections, scales=None):
    if scales is None:
        scales = [5, 10, 25, 50]

    if not detections:
        raise Exception('No scale change detections specified!')

    num_det = len(detections)
    T = nx.DiGraph()
    T.add_node('source')
    T.add_node('target')

    T.add_nodes_from([(n, s) for s in scales for n in range(num_det + 1)])
    T.add_edges_from([('source', (0, s)) for s in scales], weight=0)

    for n, detection in enumerate(detections):
        # Detection of change of scale (n) → (n + 1)
        for s1 in scales:
            for s2 in scales:
                ratio = s2 / s1
                loss = 0
                if detection['wav_ratio'] is not None:
                    loss += (detection['wav_ratio'] - ratio)**2
                elif detection['mad_ratio'] is not None:
                    loss += (detection['mad_ratio'] - ratio)**2
                else:
                    loss = 0.1 * int(s1 != s2)

                T.add_edge((n, s1), (n + 1, s2), loss=loss)

    T.add_edges_from([((num_det, s), 'target') for s in scales], weight=0)

    shortest_path = nx.shortest_path(T, 'source', 'target', weight='loss')

    return [scale for _, scale in shortest_path[1:-1]]


def scale_changes_correction(recording, base_scale=5, logfile=None, **kwargs):
    fixed = recording.copy()
    detections = detect_scale_changes(recording, **kwargs)
    if not detections:
        return fixed

    scales = find_best_scale_sequence(detections)

    starts = [recording.data.index.min()] + [d['start'] for d in detections]
    ends = [d['end'] for d in detections] + [recording.data.index.max()]

    for start, end, scale in zip(starts, ends, scales):
        if scale != base_scale:
            fixed.data.loc[start:end] *= scale / base_scale

    for det in detections:
        fixed.data.loc[det['start']:det['end']] = 0

    if logfile is not None:
        with open(logfile, 'w+') as f:
            f.write(f'SEDLINE SCALE CHANGE {recording.meta["date"]}\n')

            for det, s1, s2 in zip(detections, scales[:-1], scales[1:]):
                if s1 != s2:
                    det_time = recording.meta['date'] + det['start']
                    f.write(f'{det_time}: {s1} µV/mm → {s2} µV/mm\n')

    return fixed
