"""Tests for :mod:`pyts.transformation` module."""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
from itertools import product
import numpy as np
from ..transformation import BOSS, WEASEL


standard_library.install_aliases()


def test_BOSS():
    """Testing 'BOSS'."""
    # Parameters
    X = np.arange(40).reshape(4, 10).astype('float64')

    # Test 1
    boss = BOSS(n_coefs=4, window_size=5, norm_mean=False, norm_std=False)
    boss.fit_transform(X)

    # Test 2
    X = (np.arange(1, 5).reshape(4, 1) * np.ones((4, 10))).astype('float64')
    boss = BOSS(n_coefs=2, window_size=10, norm_mean=True, norm_std=False)
    arr_actual = boss.fit_transform(X).toarray().ravel()
    arr_desired = np.ones(4)  # Expected words: ["dd", "dd", "dd", "dd"]
    np.testing.assert_allclose(arr_actual, arr_desired)

    # Test 2
    X = (np.arange(1, 5).reshape(4, 1) * np.ones((4, 10))).astype('float64')
    boss = BOSS(n_coefs=2, window_size=10, norm_mean=False, norm_std=False)
    arr_actual = boss.fit_transform(X).toarray()
    arr_desired = np.eye(4)  # Expected words: ["ad", "bd", "cd", "dd"]
    np.testing.assert_allclose(arr_actual, arr_desired)

    # Test: loop
    rng = np.random.RandomState(41)
    X_noise = rng.randn(20, 10)

    n_coefs_list = [None, 4, 6]
    window_size_list = [6, 8]
    anova_list = [True, False]
    norm_mean_list = [True, False]
    norm_std_list = [True, False]
    n_bins_list = [2, 4]
    quantiles_list = ['gaussian', 'empirical']
    variance_selection_list = [True, False]
    variance_threshold_list = [0, 0.001]
    numerosity_reduction_list = [True, False]
    for (n_coefs, window_size, anova, norm_mean, norm_std, n_bins,
         quantiles, variance_selection, variance_threshold,
         numerosity_reduction) in product(*[n_coefs_list,
                                            window_size_list,
                                            anova_list,
                                            norm_mean_list,
                                            norm_std_list,
                                            n_bins_list,
                                            quantiles_list,
                                            variance_selection_list,
                                            variance_threshold_list,
                                            numerosity_reduction_list]):
        boss = BOSS(n_coefs, window_size, anova, norm_mean, norm_std,
                    n_bins, quantiles, variance_selection, variance_threshold,
                    numerosity_reduction)
        boss.fit(X_noise, overlapping=True).transform(X_noise)
        boss.fit(X_noise, overlapping=False).transform(X_noise)
        boss.fit_transform(X_noise, overlapping=True)
        boss.fit_transform(X_noise, overlapping=False)


def test_WEASEL():
    """Testing 'WEASEL'."""
    # Parameters
    rng = np.random.RandomState(123)
    X = rng.randn(4, 100)
    y = np.array([0, 0, 1, 1])

    # Test 1
    weasel = WEASEL(n_coefs=4, window_sizes=[10, 20])
    weasel.fit_transform(X, y)

    # Test 2
    weasel = WEASEL(n_coefs=4, window_sizes=(6, 15))
    weasel.fit_transform(X, y)

    # Test 3
    weasel = WEASEL(n_coefs=3, window_sizes=(6, 15), norm_mean=False,
                    norm_std=False, n_bins=3, pvalue_threshold=0.5)
    weasel.fit(X, y)
    weasel.transform(X[2:])

    # Test: loop
    rng = np.random.RandomState(41)
    X_noise = rng.randn(20, 10)
    y = rng.randint(3, size=20)

    n_coefs_list = [4, 6]
    window_sizes_list = [[6, 7], [6, 8]]
    norm_mean_list = [True, False]
    norm_std_list = [True, False]
    n_bins_list = [2, 4]
    variance_selection_list = [True, False]
    variance_threshold_list = [0, 0.001]
    pvalue_threshold_list = [0.2, 0.9]
    for (n_coefs, window_sizes, norm_mean, norm_std, n_bins,
         variance_selection, variance_threshold,
         pvalue_threshold) in product(*[n_coefs_list,
                                        window_sizes_list,
                                        norm_mean_list,
                                        norm_std_list,
                                        n_bins_list,
                                        variance_selection_list,
                                        variance_threshold_list,
                                        pvalue_threshold_list]):
        weasel = WEASEL(n_coefs, window_sizes, norm_mean, norm_std,
                        n_bins, variance_selection, variance_threshold,
                        pvalue_threshold)
        weasel.fit(X_noise, y, overlapping=True).transform(X_noise)
        weasel.fit_transform(X_noise, y)
