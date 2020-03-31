import pytest
import numpy as np
from pytest import approx

from neurokit.analysis.gaussian_decomposition import (GaussianDecomposition,
                                                      sum_of_gaussians,
                                                      DecompositionError)


def test_sum_of_gaussians():
    xs = np.arange(100)

    ys = sum_of_gaussians(xs, [(10, 30, 5)])
    assert ys[30] == 10
    assert ys[70] == approx(0, abs=1e-6)

    ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])
    assert ys[30] == approx(10, abs=1e-6)
    assert ys[70] == approx(20, abs=1e-6)


def test_initial_guess():
    xs = np.arange(100)
    ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])

    dec = GaussianDecomposition(alpha=1, min_distance=5)
    guessed_components = dec._guess_initial(ys)

    assert len(guessed_components) == 2

    a1, μ1, σ1 = guessed_components[0]
    assert a1 == approx(10., abs=1.5)
    assert μ1 == approx(30., abs=1)
    assert σ1 == approx(5., abs=1)

    a0, μ0, σ0 = guessed_components[1]
    assert a0 == approx(20., abs=1.5)
    assert μ0 == approx(70., abs=1)
    assert σ0 == approx(3., abs=1)


def test_fit():
    xs = np.arange(100)
    ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])

    dec = GaussianDecomposition(alpha=1, min_distance=5)
    dec.fit(ys)

    assert dec.n_components_ == 2

    a0, μ0, σ0 = dec.components_[0]
    assert a0 == approx(20., abs=1e-3)
    assert μ0 == approx(70., abs=1e-3)
    assert σ0 == approx(3., abs=1e-3)

    a1, μ1, σ1 = dec.components_[1]
    assert a1 == approx(10., abs=1e-3)
    assert μ1 == approx(30., abs=1e-3)
    assert σ1 == approx(5., abs=1e-3)


def test_fit_with_distance():
    xs = np.arange(100)
    ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])

    dec = GaussianDecomposition(alpha=1, min_distance=100)
    dec.fit(ys)

    assert dec.n_components_ == 1

    a0, μ0, σ0 = dec.components_[0]
    assert a0 == approx(20., abs=1e-3)
    assert μ0 == approx(70., abs=1e-3)
    assert σ0 == approx(3., abs=1e-3)


def test_fit_with_noise():
    xs = np.arange(100)
    ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])
    ys += np.random.normal(0, 0.1, size=ys.size)

    dec = GaussianDecomposition(alpha=4, min_distance=10, max_ls_iter=100)
    dec.fit(ys)

    a0, μ0, σ0 = dec.components_[0]
    assert a0 == approx(20., abs=0.5)
    assert μ0 == approx(70., abs=0.5)
    assert σ0 == approx(3., abs=0.5)


def test_fit_small_values():
    xs = np.arange(100)
    ys = sum_of_gaussians(xs, [(10e-12, 30, 5), (20e-12, 70, 3)])

    dec = GaussianDecomposition(alpha=4, min_distance=10, max_ls_iter=100)
    dec.fit(ys)

    a0, μ0, σ0 = dec.components_[0]
    assert a0 == approx(20e-12, abs=0.5e-12)
    assert μ0 == approx(70, abs=0.5)
    assert σ0 == approx(3., abs=0.5)


def test_raises_exception_if_least_squares_does_not_converge():
    xs = np.arange(100)
    ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])

    dec = GaussianDecomposition(alpha=1, min_sigma=10, min_distance=5,
                                max_ls_iter=10)

    with pytest.raises(DecompositionError):
        dec.fit(ys)


def test_raises_exception_if_cannot_guess_initial_params():
    xs = np.arange(100)
    ys = np.ones_like(xs)
    dec = GaussianDecomposition(alpha=1, min_distance=5, normalize=False)

    with pytest.raises(DecompositionError):
        dec.fit(ys)
