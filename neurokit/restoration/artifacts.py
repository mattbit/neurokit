import pywt
import logging
import numpy as np
import pandas as pd


class ArtifactRestorator:
    """Restores artifacts."""

    def __init__(self):
        self.window = pd.Timedelta(seconds=1)

    def restore(self, recording):
        """Restore all signals in a recording.

        Parameters
        ----------
        recording : neurokit.model.Recording

        Returns
        -------
        fixed : neurokit.model.Recording
            The fixed recording.
        """
        fixed = recording.copy()
        fixed_artifacts_idx = []
        for af in recording.artifacts.itertuples():
            signal = fixed.data.loc[af.start:af.end, af.channel]
            duration = af.end - af.start
            reference = fixed.data.loc[af.start - duration:af.end, af.channel]

            if np.isnan(reference).any():
                logging.warning(f'Cannot fix artifact at {af.start}')
                continue

            restored = restore_wavelet_renormalization(signal, reference)
            fixed.data.loc[af.start:af.end, af.channel] = restored
            fixed_artifacts_idx.append(af.Index)

        fixed.artifacts.drop(index=fixed_artifacts_idx, inplace=True)

        return fixed


def restore_wavelet_renormalization(signal, reference, wavelet='db2'):
    """Fix a signal using a wavelet renormalization.

    Parameters
    ----------
    signal : numpy.ndarray
        1D signal containing artifacts.
    reference : numpy.ndarray
        1D clean signal which will be used as a reference for the correction.

    Returns
    -------
    reconstruction : numpy.ndarray
        Restored signal, same shape as `signal`.
    """
    ref_coeffs = pywt.wavedec(reference, wavelet, mode='reflect')
    coeffs = pywt.wavedec(signal, wavelet, mode='reflect')

    rms = np.array([np.sqrt(np.mean(c**2)) for c in coeffs])
    ref_rms = np.array([np.sqrt(np.mean(c**2)) for c in ref_coeffs])

    for n in np.arange(len(coeffs)):
        coeffs[n] *= ref_rms[n] / rms[n]

    reconstruction = pywt.waverec(coeffs, wavelet, mode='reflect')

    return reconstruction[:len(signal)]
