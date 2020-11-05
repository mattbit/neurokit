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
    ecg_events, ch, heartbeat = mne.preprocessing.find_ecg_events(raw,ch_name='ECG1+', reject_by_annotation=True)
    ecg_epochs = mne.Epochs(raw, ecg_events, 999, tmin, tmax, picks='ECG1+', baseline=(None,None),
                            reject_by_annotation=True, verbose=False)
    ## moving the center of event to the minimum of the amplitude
    ecg_epochs.event[:,0] += int(round(tmin * sf)) + np.argmin(np.squeeze(ecg_epochs.get_data()), axis=1)

    eeg_chs = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    data = raw.get_data()
    data_updated = np.copy(data)
    dt = (60*60) * sf / heartbeat
    dt_eps = int(round(tmax * sf) + 1)
    time_segmentation = np.arange(raw.first_samp + dt_eps, raw.last_samp + dt - dt_eps, dt )
    for ch in eeg_chs:

        for start , stop in zip(time_segmentation, time_segmentation[1:]):
            mask = np.ma.masked_outside(ecg_epochs.events[:,0], start, stop).mask
            try:
                epochs_ecg_centered = mne.Epochs(raw, ecg_epochs.events[~mask,:], 999, tmin, tmax, picks=ch,
                                    baseline=(None, None), detrend=1, verbose=False, event_repeated='drop',
                                    preload=True)
            except ValueError:
                continue
            if len(epochs_ecg_centered) < 2:
                continue

            event_eeg = ecg_epochs.events[~mask,:][epochs_ecg_centered.selection,:]

            eeg_centering = np.argmax(np.abs(np.diff(np.squeeze(
                epochs_ecg_centered.get_data())[:,:,(int(round((-tmin) * sf)) - 3):(int(round((-tmin) * sf)) + 4)],
                axis=1)), axis=1) - 1

            event_eeg[:,0] += eeg_centering - 3

            epochs_eeg_centered = mne.Epochs(raw, event_eeg, 999, tmin, tmax, picks=ch,
                                    baseline=(None, None), detrend=1, verbose=False, event_repeated='drop',
                                    preload=True)
            event_eeg = event_eeg[epochs_eeg_centered.selection, :]
            mean_qrs = np.mean(epochs_eeg_centered.get_data().T,axis=2)

            for i, event in enumerate(event_eeg[:-1,0]):
                lower_bound = event + int(round(tmin * sf))
                upper_bound = event + int(round(tmax * sf) + 1)
                signal = data_updated[ch,lower_bound - raw.first_samp:upper_bound - raw.first_samp]
                signal_baseline_corr = np.squeeze(data)[i, :]
                data_updated[ch,lower_bound - raw.first_samp:upper_bound - raw.first_samp] = \
                    signal - ((np.dot(signal_baseline_corr, mean_qrs) / np.dot(mean_qrs, mean_qrs)) * mean_qrs)

    return mne.io.RawArray(data_updated, raw.info, first_samp=raw.first_samp)





        





