import warnings
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d

try:
    from numba import jit
except ImportError:
    warnings.warn(
        'We recommend installing `numba` to speed up Gaussian decomposition.'
    )

    def jit(fun, *args, **kwargs):
        return fun


@jit
def _sum_of_gaussians(x, params):
    return (params[0::3] * np.exp(- 0.5 * (x.reshape(x.size, 1) - params[1::3])**2 / params[2::3]**2)).sum(axis=1)


@jit
def _ffit(params, x, y):
    return (params[0::3] * np.exp(- 0.5 * (x.reshape(x.size, 1) - params[1::3])**2 / params[2::3]**2)).sum(axis=1) - y


def sum_of_gaussians(x, components):
    """Sum of Gaussian components.

    Parameters
    ----------
    x : numpy.ndarray
        1D array with x values of the curve.
    components : list
        List of components as (amplitude, location, scale).
    Returns
    -------
    """
    return _sum_of_gaussians(x, np.array(components).ravel())


class DecompositionError(RuntimeError):
    """Error in the decomposition."""


class GaussianDecomposition:
    """Decompose a signal in a sum of Gaussian curves.

    Each Gaussian has the form:
        f(x) = amplitude * exp(- (x - loc)**2 / (2 * scale**2))

    Parameters
    ----------
    alpha : float
        Smoothing coefficient (standard deviantion of Gaussian derivatives).
    min_sigma : float
        Minimum standard deviation of each component. Defaults to `alpha / 2`
        if not explicitly set.
    min_distance : int
        Minimum distance between components location.
    min_amplitude : float
        Minimum amplitude of a peak. If `normalize` is set to true, this
        will indicate a relative amplitude with respect to the maximum.
    max_ls_iter : int
        Maximum number of least-squares iterations. If a fit is not found
        before this limit is reached, the guessed parameters are used.
    normalize : bool
        Whether to normalize the input for numerical stability.

    Attributes
    ----------
    components_ : numpy.ndarray, shape=(n_components, 3)
        2D array where each row represents (amplitude, location, scale) of a
        Gaussian component, with largest amplitude components first.
    n_components_ : int
        Number of components.

    Raises
    ------
    DecompositionError
        If it cannot find a valid decomposition (e.g. if least squares fit did
        not converge before `max_ls_iter` iterations).

    References
    ----------
    .. [1] Lindner, Robert R., et al. "Autonomous gaussian decomposition."
       The Astronomical Journal 149.4 (2015).
    """

    def __init__(self, alpha=3, min_sigma=None, min_distance=10,
                 min_amplitude=0, max_ls_iter=20, normalize=True):
        self.alpha = alpha
        self.min_sigma = alpha / 2 if min_sigma is None else min_sigma
        self.min_distance = min_distance
        self.min_amplitude = min_amplitude
        self.components_ = None
        self.n_components_ = None
        self.max_ls_iter = max_ls_iter
        self.normalize = normalize

    def fit(self, x):
        """Fit Gaussian components to a signal.

        Parameters
        ----------
        x : numpy.ndarray
            1D array representing the signal to decompose.

        Returns
        -------
        self : GaussianDecomposition
            Returns the instance itself.
        """
        # Normalization between 0 and 1 for numerical stability
        if self.normalize:
            _offset = x.min()
            _scale = x.max() - x.min()
            y = (x - _offset) / _scale
        else:
            y = x

        # Initial guess
        params_guess = self._guess_initial(y)
        params_guess[:, 2] = np.maximum(params_guess[:, 2], self.min_sigma)

        if len(params_guess) == 0:
            raise DecompositionError("Cannot guess initial parameters.")

        # Setup optimization bounds
        bounds_l = np.full_like(params_guess, -np.inf)
        bounds_h = np.full_like(params_guess, +np.inf)
        bounds_l[:, 0] = 0.
        bounds_h[:, 0] = 1.01 * y.max()
        bounds_l[:, 1] = params_guess[:, 1] - 10
        bounds_h[:, 1] = params_guess[:, 1] + 10
        bounds_l[:, 2] = self.min_sigma
        bounds = (bounds_l.ravel(), bounds_h.ravel())

        base_x = np.arange(y.size, dtype=float)

        r = least_squares(_ffit, params_guess.ravel(), args=(
            base_x, y), bounds=bounds, max_nfev=self.max_ls_iter)

        if not r.success:
            warnings.warn("Least squares fit did not converge. "
                          + "Try increasing `max_ls_iter`.")

        params = sorted(r.x.reshape(len(r.x) // 3, 3),
                        key=lambda x: x[0], reverse=True)
        params = np.array(params)
        if self.normalize:
            params[:, 0] = params[:, 0] * _scale + _offset

        self.components_ = params
        self.n_components_ = len(self.components_)

        return self

    def _guess_initial(self, x):
        x_clip = x.clip(0)
        u0 = gaussian_filter1d(x_clip, self.alpha, order=0, mode="wrap")
        u2 = gaussian_filter1d(x_clip, self.alpha, order=2, mode="wrap")
        u3 = gaussian_filter1d(x_clip, self.alpha, order=3, mode="wrap")
        u4 = gaussian_filter1d(x_clip, self.alpha, order=4, mode="wrap")

        idx, = np.nonzero(np.diff(np.sign(u3)))
        idx = idx[(x[idx] > 0) & (u2[idx] < 0) & (u4[idx] > 0)]

        # Remove small amplitude components
        idx = idx[x[idx] >= self.min_amplitude]

        # Select location by distance
        idx = list(idx)
        distances = np.diff(idx)
        while (distances < self.min_distance).any():
            d_min = np.argmin(distances)
            i, j = idx[d_min], idx[d_min + 1]
            if u0[i] < u0[j]:
                del idx[d_min]
            else:
                del idx[d_min + 1]

            distances = np.diff(idx)

        locs = np.array(idx) + 0.5
        amps = u0[idx]
        scales = np.sqrt(- u0[idx] / u2[idx])

        return np.vstack([amps, locs, scales]).T
