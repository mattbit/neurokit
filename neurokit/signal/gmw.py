"""Generalized Morse Wavelet"""

import numpy as np
from scipy import fft
from scipy.special import loggamma, eval_genlaguerre


class MorseWavelet:
    def __init__(self, β=8, γ=3):
        self.β = β
        self.γ = γ

    def psi_f(self, freqs: np.ndarray, scale: float = 1, k: int = 0):
        """Wavelet function in the frequency domain.

        By default, they are normalized to have unitary norm.

        Parameters
        ----------
        freqs : np.ndarray
            Array of frequency values.
        scale : float
            Scale factor.
        k : int
            Order of the wavelet.
        """
        r = (2 * self.β + 1) / self.γ

        # I calculate the log to avoid explosing functions and improve the
        # numerical precision, particularly for the Γ(k + 1)/Γ(k + r).
        logAk = 0.5 * (np.log(np.pi * self.γ) + r * np.log(2) +
                       loggamma(k + 1) - loggamma(k + r))

        # The effective frequency in radians
        aω = scale * 2 * np.pi * freqs

        H = np.heaviside(aω, 0.5)

        # As before, I use the log instead of doing a ω^β.
        with np.errstate(divide='ignore', invalid='ignore'):
            wav0 = scale * H * \
                np.sqrt(2) * np.exp(logAk - aω**γ + self.β * np.log(aω))

        # Remove the term which explosed. Not very elegant, but it works.
        if np.isscalar(wav0):
            wav0 = 0 if np.isnan(wav0) else wav0
        else:
            wav0[np.isnan(wav0)] = 0

        # Finally, I add the generalized Laguerre polynomial term.
        return wav0 * eval_genlaguerre(k, r - 1, 2 * aω**self.γ)

    def psi_t(self, length, scale: float = 1, k: int = 0):
        """THIS IS EXPERIMENTAL!!!

        What should we do? Find a frequency such that the wavelet psi_f goes
        to zero, then do an ifft based on that.
        """
        raise NotImplementedError()

    def cwt(self, signal: np.ndarray, scales: np.ndarray, num_k=0, norm=1):
        """Continuous wavelet transform"""
        if norm not in ('energy', 'analytic', None):
            raise ValueError(f'Invalid norm `{norm}`.')

        # Find the fast length for the FFT
        n = len(signal)
        fast_len = fft.next_fast_len(n)

        # Signal in frequency domain
        fs = fft.fftfreq(fast_len)
        f_sig = fft.fft(sig, n=fast_len)

        # Compute the wavelet transform
        W = np.zeros((scales.size, n), dtype=complex)
        for i, scale in enumerate(scales):
            # Select the correct normalization value.
            if norm == 'energy':
                norm = np.sqrt(scale)
            elif norm == 'analytic':
                norm = scale
            else:
                norm = 1

            # Average over orthogonal wavelets, k = 0, …, num_k - 1
            for k in range(num_k + 1):
                # Do the convolution for each scale in the frequency domain, then
                # invert the FFT to get the wavelet transform.
                W[i] += fft.ifft(f_sig * self.psi_f(fs) / Δt, n=n)
            W[i] /= (norm * num_k)

        return W
