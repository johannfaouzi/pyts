"""Tests for :mod:`pyts.quantization` module."""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
from itertools import product
import numpy as np
import scipy.stats
from ..quantization import SAX, MCB


standard_library.install_aliases()


def test_SAX():
    """Testing 'SAX'."""
    # Test 1
    X = np.tile(np.arange(3), 5)
    sax = SAX(n_bins=3, quantiles='empirical')
    str_actual = sax.fit_transform(X[np.newaxis, :])[0]
    str_desired = np.array(["a", "b", "c"] * 5)
    np.testing.assert_array_equal(str_actual, str_desired)

    # Test 2
    X = np.tile(np.arange(-0.75, 1, 0.5), 3)
    sax = SAX(n_bins=4, quantiles='gaussian')
    str_actual = sax.fit_transform(X[np.newaxis, :])[0]
    str_desired = np.array(["a", "b", "c", "d"] * 3)
    np.testing.assert_array_equal(str_actual, str_desired)


def test_MCB():
    """Testing 'MCB'."""
    # Test 1
    X = np.arange(40).reshape(4, 10)
    mcb = MCB(n_bins=4, quantiles='empirical')
    arr_actual = mcb.fit_transform(X)
    arr_desired = np.repeat(np.array(["a", "b", "c", "d"]), 10).reshape(4, 10)
    np.testing.assert_array_equal(arr_desired, arr_actual)

    # Test 2
    X = scipy.stats.norm.ppf(np.linspace(0, 1, 42)[1:-1]).reshape(4, 10)
    mcb = MCB(n_bins=4, quantiles='empirical')
    arr_actual = mcb.fit_transform(X)
    arr_desired = np.repeat(np.array(["a", "b", "c", "d"]), 10).reshape(4, 10)
    np.testing.assert_array_equal(arr_desired, arr_actual)

    # Test 3
    mcb = MCB(n_bins=4, quantiles='gaussian')
    arr_actual = mcb.fit_transform(X)
    arr_desired = np.repeat(np.array(["a", "b", "c", "d"]), 10).reshape(4, 10)
    np.testing.assert_array_equal(arr_desired, arr_actual)

    # Test 4
    mcb = MCB(n_bins=2, quantiles='entropy')
    y = np.array([0, 0, 1, 2])
    arr_actual = mcb.fit_transform(X, y)
    arr_desired = np.repeat(np.array(["a", "a", "b", "b"]), 10).reshape(4, 10)
    np.testing.assert_array_equal(arr_desired, arr_actual)

    # Test: loop
    n_bins_list = [2, 3, 4]
    quantiles_list = ['empirical', 'gaussian', 'entropy']
    for n_bins, quantiles in product(*[n_bins_list, quantiles_list]):
        mcb = MCB(n_bins, quantiles)
        mcb.fit(X, y).transform(X)
