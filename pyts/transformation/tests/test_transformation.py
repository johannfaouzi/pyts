"""Tests for :mod:`pyts.transformation` module."""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
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
