import numpy as np
from unittest import TestCase

from neurokit.analysis.gaussian_decomposition import (GaussianDecomposition,
                                                      sum_of_gaussians)


class TestGaussianDecomposition(TestCase):
    def test_sum_of_gaussians(self):
        xs = np.arange(100)

        ys = sum_of_gaussians(xs, [(10, 30, 5)])
        self.assertEqual(ys[30], 10)
        self.assertAlmostEqual(ys[70], 0, places=6)

        ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])
        self.assertAlmostEqual(ys[30], 10, places=6)
        self.assertAlmostEqual(ys[70], 20, places=6)

    def test_initial_guess(self):
        xs = np.arange(100)
        ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])

        dec = GaussianDecomposition(alpha=1, min_distance=5)
        guessed_components = dec._guess_initial(ys)

        self.assertEqual(len(guessed_components), 2)

        a1, μ1, σ1 = guessed_components[0]
        self.assertAlmostEqual(a1, 10., delta=1.5)
        self.assertAlmostEqual(μ1, 30., delta=1)
        self.assertAlmostEqual(σ1, 5., delta=1)

        a0, μ0, σ0 = guessed_components[1]
        self.assertAlmostEqual(a0, 20., delta=1.5)
        self.assertAlmostEqual(μ0, 70., delta=1)
        self.assertAlmostEqual(σ0, 3., delta=1)

    def test_fit(self):
        xs = np.arange(100)
        ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])

        dec = GaussianDecomposition(alpha=1, min_distance=5)
        dec.fit(ys)

        self.assertEqual(dec.n_components_, 2)

        a0, μ0, σ0 = dec.components_[0]
        self.assertAlmostEqual(a0, 20., places=3)
        self.assertAlmostEqual(μ0, 70., places=3)
        self.assertAlmostEqual(σ0, 3., places=3)

        a1, μ1, σ1 = dec.components_[1]
        self.assertAlmostEqual(a1, 10., places=3)
        self.assertAlmostEqual(μ1, 30., places=3)
        self.assertAlmostEqual(σ1, 5., places=3)

    def test_fit_with_distance(self):
        xs = np.arange(100)
        ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])

        dec = GaussianDecomposition(alpha=1, min_distance=100)
        dec.fit(ys)

        self.assertEqual(dec.n_components_, 1)

        a0, μ0, σ0 = dec.components_[0]
        self.assertAlmostEqual(a0, 20., places=3)
        self.assertAlmostEqual(μ0, 70., places=3)
        self.assertAlmostEqual(σ0, 3., places=3)

    def test_fit_with_noise(self):
        xs = np.arange(100)
        ys = sum_of_gaussians(xs, [(10, 30, 5), (20, 70, 3)])
        ys += np.random.normal(0, 0.1, size=ys.size)

        dec = GaussianDecomposition(alpha=4, min_distance=10, max_ls_iter=100)
        dec.fit(ys)

        a0, μ0, σ0 = dec.components_[0]
        self.assertAlmostEqual(a0, 20., delta=0.5)
        self.assertAlmostEqual(μ0, 70., delta=0.5)
        self.assertAlmostEqual(σ0, 3., delta=0.5)

    def test_fit_small_values(self):
        xs = np.arange(100)
        ys = sum_of_gaussians(xs, [(10e-12, 30, 5), (20e-12, 70, 3)])

        dec = GaussianDecomposition(alpha=4, min_distance=10, max_ls_iter=100)
        dec.fit(ys)

        a0, μ0, σ0 = dec.components_[0]
        self.assertAlmostEqual(a0, 20e-12, delta=0.5e-12)
        self.assertAlmostEqual(μ0, 70, delta=0.5)
        self.assertAlmostEqual(σ0, 3., delta=0.5)
