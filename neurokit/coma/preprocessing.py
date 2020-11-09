"""Preprocessing pipeline.
[1]: Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., & Robbins, K. A.
     "The PREP pipeline: standardized preprocessing for large-scale EEG
     analysis". Frontiers in neuroinformatics, 9, 16 (2015).
"""
import mne
import logging
import numpy as np
import pandas as pd
from scipy import stats
import scipy.signal as ss
import matplotlib.pyplot as plt
from autoreject import AutoReject, Ransac
from scipy.signal.windows import hann

logging.getLogger().setLevel(logging.INFO)

#

CHANNELS = ['ECG1+', 'Fp1', 'Fp2', 'F7', 'F8', 'T3', 'C3',
            'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
WINDOW_DURATION = 4.  # in seconds
MAX_FREQUENCY = 30.  # Hz

_std_montage = mne.channels.make_standard_montage('standard_1020')


def preprocess(raw: mne.io.Raw):
    raw.pick_channels(CHANNELS)
    raw.set_channel_types({'ECG1+': 'ecg'})
    raw.set_montage(_std_montage)

    raw = filter_raw(raw)
    raw = select_optimal_crop(raw)

    bad_channels = find_bad_channels(raw)
    raw.info['bads'].extend(bad_channels)

    raw = fix_artifacts(raw)
    return raw


def filter_raw(raw):
    return raw.filter(.5, 45, phase='zero-double').notch_filter(50)


def calculate_sef(raw):
    nperseg = int(WINDOW_DURATION * raw.info['sfreq'])
    noverlap = nperseg // 8

    f, t, S = ss.spectrogram(raw.get_data(['eeg']), fs=raw.info['sfreq'],
                             window='hann', nperseg=nperseg, noverlap=noverlap,
                             mode='psd')

    S_avg = np.mean(S[:, f <= MAX_FREQUENCY], axis=0)
    f = f[f <= MAX_FREQUENCY]

    # TODO: could be better
    cum_power = S_avg.cumsum(axis=0)
    sef = f[np.argmax((cum_power / cum_power[-1]) >= 0.95, axis=0)]

    if np.isnan(S_avg).any():
        sef[np.isnan(S_avg).any(axis=0)] = np.nan

    return t, sef


def select_optimal_crop(raw):
    if raw.n_times / raw.info['sfreq'] <= 2400:  # 40 minutes
        return raw.copy()

    times, sef = calculate_sef(raw)
    med_sef = np.median(sef)

    sef_series = pd.Series(sef, index=pd.to_timedelta(times, unit='s'))

    start = (sef_series.rolling('30min', min_periods=None).mean()
             - med_sef).abs().idxmin()
    end = start + pd.to_timedelta(30, unit='min')
    logging.info(f'Recording crop: ({start}, {end})')

    return raw.copy().crop(start.total_seconds(), end.total_seconds())


def find_bad_channels(raw):
    epochs = make_fixed_length_epochs(raw).pick('eeg')

    chs = mne.pick_types(raw.info, eeg=True)
    interpolates = np.arange(1, 5)
    consensus = np.arange(0, 1, 1 / len(chs))
    ar = AutoReject(interpolates, consensus,
                    random_state=42,
                    verbose=False,
                    thresh_method='bayesian_optimization')
    ar.fit(epochs)

    log = ar.get_reject_log(epochs)
    epochs_data = epochs.get_data()
    ii, jj = np.nonzero(log.labels > 0)
    for i, j in zip(ii, jj):
        epochs_data[i, j] = np.nan

    data = np.concatenate(epochs_data, axis=-1)

    # don't do this
    clean_raw = mne.io.RawArray(data, raw.copy().pick('eeg').info,
                                first_samp=raw.first_samp)

    bad_channels = prep_by_sef(clean_raw)
    bad_channels |= prep_by_correlation(clean_raw)
    bad_channels |= prep_by_epoch_variance(epochs_data)
    bad_channels |= prep_by_epoch_flatness(epochs_data)

    # Before running RANSAC, we remove the bad channels we have detected until
    # now. For more information, see [1].
    bad_ch_names = np.array(clean_raw.info['ch_names'])[bad_channels]
    clean_raw.info['bads'].extend(bad_ch_names)

    epochs = make_fixed_length_epochs(clean_raw)

    picks = mne.pick_types(clean_raw.info, eeg=True, exclude='bads')
    ransac = Ransac(verbose=False, picks=picks)
    ransac.fit(epochs)

    all_bad_channels = set(clean_raw.info['bads']).union(ransac.bad_chs_)

    return all_bad_channels


def prep_by_sef(raw):
    sefs = [calculate_sef(raw.copy().pick(ch))[1]
            for ch in range(raw.info['nchan'])]
    avg_sefs = np.mean(sefs, axis=1)
    return np.abs(stats.zscore(avg_sefs)) > 3


def prep_by_correlation(raw):
    corr = np.ma.corrcoef(np.ma.masked_invalid(raw.get_data())).data
    corr += np.diag(np.full(corr.shape[0], np.nan))
    return np.abs(stats.zscore(np.nanmean(corr, axis=0))) > 3


def prep_by_epoch_variance(epochs_data):
    variances = np.nanvar(epochs_data, axis=-1)
    log_var = np.log(np.nanmean(variances, axis=0))
    return np.abs(stats.zscore(log_var)) > 3


def prep_by_epoch_flatness(epochs_data):
    diff = np.nanmax(epochs_data, axis=-1) - np.nanmin(epochs_data, axis=-1)
    diff_mean = np.log(np.nanmean(diff, axis=0))
    return np.abs(stats.zscore(diff_mean)) > 3


def fix_artifacts(raw):
    epochs = make_fixed_length_epochs(raw).pick(['eeg', 'ecg'])

    picks = mne.pick_types(epochs.info, eeg=True, exclude='bads')
    interpolates = np.arange(1, np.ceil(len(picks) / 4) + 1, dtype=int)
    consensus = np.linspace(0, 1, len(picks) + 1)
    ar = AutoReject(interpolates, consensus,
                    picks=picks,
                    thresh_method='bayesian_optimization',
                    random_state=42, verbose=False)
    epochs_clean, log = ar.fit_transform(epochs, return_log=True)

    epochs_data = epochs.get_data()
    epochs_data[~log.bad_epochs] = epochs_clean.get_data()
    data = np.concatenate(epochs_data, axis=-1)

    # Create annotations
    onsets = epochs.events[log.bad_epochs, 0] / raw.info['sfreq']
    annotations = mne.Annotations(onsets, WINDOW_DURATION, 'bad_segment')

    fixed_raw = mne.io.RawArray(data, info=epochs.info,
                                first_samp=raw.first_samp)
    fixed_raw.set_annotations(annotations)

    return fixed_raw


def make_fixed_length_epochs(raw):
    events = mne.make_fixed_length_events(raw, 1, duration=WINDOW_DURATION)
    delta = 1. / raw.info['sfreq']
    epochs = mne.Epochs(raw, events, event_id=[1], tmin=0.,
                        tmax=WINDOW_DURATION - delta,
                        reject_by_annotation=True,
                        preload=True, baseline=None)
    return epochs


def remove_ecg(raw, tmin=-0.1, tmax=0.2):
    sf = raw.info['sfreq']
    events, _, _ = mne.preprocessing.find_ecg_events(raw, ch_name='ECG1+', reject_by_annotation=True)
    if len(events) < 1:
        return raw

    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, picks=['eeg'],
                        baseline=(None, None), detrend=0, event_repeated='error',
                        preload=True)
    data = epochs.get_data()
    data_qrs = np.zeros_like(data)
    for offset in range(30, data.shape[0] - 30):
        data_qrs[offset] = data[offset - 30:offset + 30].mean(axis=0)

    window = hann(data.shape[-1])
    projection = np.expand_dims(np.sum(data_qrs * data, axis=2), axis=2)
    normalization = np.expand_dims(np.sum(data_qrs ** 2, axis=2), axis=2)
    data_qrs *= projection / normalization * window.reshape(1,1,-1)
    data_qrs = np.nan_to_num(data_qrs)
    data_update = np.copy(raw.get_data())
    lower_time = int(round(tmin * sf))
    upper_time = int(round(tmax * sf) + 1)
    for idx, t in enumerate(events[:, 0]):
        if (t < lower_time) or (raw.times.size - t < lower_time):
            continue
        if idx < 30:
            data_update[:, t + lower_time:t + upper_time] -= data_qrs[30]
        if idx >= data_qrs.shape[0] - 30:
            data_update[:, t + lower_time:t + upper_time] -= data_qrs[data_qrs.shape[0] - 30]
        else:
            data_update[:, t + lower_time:t + upper_time] -= data_qrs[idx]

    return mne.io.RawArray(data_update, raw.info, first_samp=raw.first_samp)
