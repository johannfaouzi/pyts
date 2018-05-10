"""Tests for :mod:`pyts.preprocessing` module."""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
import numpy as np
from ..preprocessing import StandardScaler


standard_library.install_aliases()


def test_StandardScaler():
    """Testing 'StandardScaler'."""
    # Parameter
    X = np.arange(30).reshape(3, 10).astype('float64')

    # Test 1
    standardscaler = StandardScaler(norm_mean=False, norm_std=False)
    arr_actual = standardscaler.fit_transform(X)
    arr_desired = X
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    standardscaler = StandardScaler(norm_mean=True, norm_std=False)
    arr_actual = standardscaler.fit_transform(X)
    X_mean = X - X.mean(axis=1).reshape(3, 1)
    arr_desired = X_mean
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 3
    standardscaler = StandardScaler(norm_mean=False, norm_std=True)
    arr_actual = standardscaler.fit_transform(X)
    arr_desired = X / X.std(axis=1).reshape(3, 1)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 4
    standardscaler = StandardScaler(norm_mean=True, norm_std=True)
    arr_actual = standardscaler.fit_transform(X)
    arr_desired = X_mean / X.std(axis=1).reshape(3, 1)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
