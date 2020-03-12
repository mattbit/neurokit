import numpy as np
from numba import jit
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


@jit
def _sum_of_gaussians(x, *params):
    params = np.array(params)
    return (params[0::3] * np.exp(- 0.5 * (x.reshape(x.size, 1) - params[1::3])**2 / params[2::3]**2)).sum(axis=1)


def sum_of_gaussians(x, components):
    '''Sum of Gaussian components.

    Parameters
    ----------
    x : numpy.ndarray
        1D array with x values of the curve.
    components : list
        List of components as (amplitude, location, scale).
    Returns
    -------
    '''
    return _sum_of_gaussians(x, *np.array(components).ravel())


class GaussianDecomposition:
    '''Decompose a signal in a sum of Gaussian curves.

    Each Gaussian has the form:
        f(x) = amplitude * exp(- (x - loc)**2 / (2 * scale**@))

    Parameters
    ----------
    alpha : float
        Smoothing coefficient (standard deviantion of Gaussian derivatives).
    min_distance : int
        Minimum distance between components location.
    max_ls_iter : int
        Maximum number of least-squares iterations. If a fit is not found
        before this limit is reached, the guessed parameters are used.

    Attributes
    ----------
    components_ : numpy.ndarray, shape=(n_components, 3)
        2D array where each row represents (amplitude, location, scale) of a
        Gaussian component, with largest amplitude components first.
    n_components_ : int
        Number of components.
    '''

    def __init__(self, alpha=3, min_distance=10, max_ls_iter=20):
        self.alpha = alpha
        self.min_distance = min_distance
        self.components_ = None
        self.n_components_ = None
        self.max_ls_iter = max_ls_iter

    def fit(self, x):
        '''Fit Gaussian components to a signal.

        Parameters
        ----------
        x : numpy.ndarray
            1D array representing the signal to decompose.

        Returns
        -------
        self : GaussianDecomposition
            Returns the instance itself.
        '''
        # Initial guess
        params_guess = self._guess_initial(x)

        # Setup optimization bounds
        bounds_l = np.full_like(params_guess, -np.inf)
        bounds_h = np.full_like(params_guess, +np.inf)
        bounds_l[:, 0] = 0.
        bounds_l[:, 1] = params_guess[:, 1] - 10
        bounds_h[:, 1] = params_guess[:, 1] + 10
        bounds_l[:, 2] = 1e-9
        bounds = (bounds_l.ravel(), bounds_h.ravel())

        try:
            _fit_params, _ = curve_fit(_sum_of_gaussians, np.arange(x.size),
                                       x, p0=params_guess.ravel(),
                                       bounds=bounds,
                                       max_nfev=self.max_ls_iter)
        except RuntimeError:
            _fit_params = params_guess.ravel()

        params = sorted(_fit_params.reshape(len(_fit_params) // 3, 3),
                        key=lambda x: x[0], reverse=True)

        self.components_ = np.array(params)
        self.n_components_ = len(self.components_)

        return self

    def _guess_initial(self, x):
        x_clip = x.clip(0)
        u0 = gaussian_filter1d(x_clip, self.alpha, order=0, mode='wrap')
        u2 = gaussian_filter1d(x_clip, self.alpha, order=2, mode='wrap')
        u3 = gaussian_filter1d(x_clip, self.alpha, order=3, mode='wrap')
        u4 = gaussian_filter1d(x_clip, self.alpha, order=4, mode='wrap')

        idx, = np.nonzero(np.diff(np.sign(u3)))
        idx = list(idx[(x[idx] > 0) & (u2[idx] < 0) & (u4[idx] > 0)])

        # Select location by distance
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
