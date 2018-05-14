"""Tests for :mod:`pyts.decomposition` module."""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
from itertools import product
import numpy as np
from ..decomposition import SSA


standard_library.install_aliases()


def test_SSA():
    """Testing 'SSA'."""
    # Parameter
    size = 30
    X = np.arange(size)
    X_tiled = np.tile(X, 2).reshape(2, -1)
    window_size = 18

    # Test 1
    ssa = SSA(window_size=2)
    arr_actual = ssa.fit_transform(X[np.newaxis, :])[0].sum(axis=0)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 2
    ssa = SSA(window_size=2)
    arr_actual = ssa.fit_transform(X_tiled).sum(axis=1)
    np.testing.assert_allclose(arr_actual, X_tiled, atol=1e-5, rtol=0.)

    # Test 3
    ssa = SSA(window_size=8)
    arr_actual = ssa.fit_transform(X[np.newaxis, :])[0].sum(axis=0)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 4
    ssa = SSA(window_size=8)
    arr_actual = ssa.fit_transform(X_tiled).sum(axis=1)
    np.testing.assert_allclose(arr_actual, X_tiled, atol=1e-5, rtol=0.)

    # Test 5: window_size
    for new_window_size in range(1, window_size + 1):
        arr_actual = ssa._ssa(X, size, new_window_size, None).sum(axis=0)
        np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 6: grouping (None)
    grouping = None
    arr_actual = ssa._ssa(X, size, window_size, grouping).sum(axis=0)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 7: grouping (integer)
    for grouping in range(1, window_size + 1):
        arr_actual = ssa._ssa(X, size, window_size, grouping).sum(axis=0)
        np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 8: grouping (array-like)
    grouping = [[0, 1, 2], [3, 4]]
    arr_actual = ssa._ssa(X, size, window_size, grouping).sum(axis=0)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test: loop
    window_size_list = [10, 12, 17]
    grouping_list = [None, 2, 3, [[0, 1, 2], [3, 4, 5]]]
    for window_size, grouping in product(*[window_size_list,
                                           grouping_list]):
        ssa = SSA(window_size=window_size, grouping=grouping)
        ssa.fit(X_tiled).transform(X_tiled)
