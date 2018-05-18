"""Tests for :mod:`pyts.bop` module."""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
import numpy as np
from ..bow import BOW


standard_library.install_aliases()


def test_BOW():
    """Testing 'BOW'."""
    # Parameter
    X = np.repeat(["a", "b", "c", "d"], 3)[np.newaxis, :]

    # Test 1
    window_size = 4
    vsm = BOW(window_size=window_size, numerosity_reduction=False)
    arr_actual = vsm.fit_transform(X)
    arr_desired = np.array([''.join(X[0, i: i + window_size])
                            for i in range(len(X[0]) - window_size + 1)])
    arr_desired = np.array([' '.join(arr_desired)])
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    window_size = 2
    vsm = BOW(window_size=window_size, numerosity_reduction=True)
    arr_actual = vsm.fit_transform(X)
    arr_desired = []
    for i in range(len(X[0]) - window_size + 1):
        substring = ''.join(X[0, i: i + window_size])
        if i == 0:
            arr_desired.append(substring)
        else:
            substring = ''.join(X[0, i: i + window_size])
            if substring != arr_desired[-1]:
                arr_desired.append(substring)
    arr_desired = np.array([' '.join(arr_desired)])
    np.testing.assert_array_equal(arr_actual, arr_desired)
