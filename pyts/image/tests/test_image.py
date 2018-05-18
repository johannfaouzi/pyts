"""Tests for :mod:`pyts.image` module."""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
from itertools import product
import numpy as np
from ..image import GASF, GADF, MTF, RecurrencePlots


standard_library.install_aliases()


def test_GASF():
    """Testing 'GASF'."""
    # Parameter
    size = 9
    X = np.linspace(-1, 1, size)

    # Test 1
    ones = np.ones(size)
    gasf = GASF(image_size=size)
    arr_actual = gasf.transform(X[np.newaxis, :])[0]
    arr_desired = np.outer(X, X) - np.outer(np.sqrt(ones - X ** 2),
                                            np.sqrt(ones - X ** 2))
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    size_new = 3
    ones_new = np.ones(size_new)
    gasf = GASF(image_size=size_new)
    arr_actual = gasf.transform(X[np.newaxis, :])[0]
    X_new = np.linspace(-1, 1, size_new)
    arr_sqrt = np.sqrt(ones_new - X_new**2)
    arr_desired = np.outer(X_new, X_new) - np.outer(arr_sqrt, arr_sqrt)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test: loop
    image_size_list = [3, 5]
    overlapping_list = [True, False]
    scale_list = [-1, 0]
    for image_size, overlapping, scale in product(*[image_size_list,
                                                    overlapping_list,
                                                    scale_list]):
        gasf = GASF(image_size, overlapping, scale)
        gasf.fit_transform(X[np.newaxis, :])


def test_GADF():
    """Testing 'GADF'."""
    # Parameter
    size = 9
    X = np.linspace(-1, 1, size)

    # Test 1
    ones = np.ones(size)
    gadf = GADF(image_size=size)
    arr_actual = gadf.transform(X[np.newaxis, :])[0]
    arr_sqrt = np.sqrt(ones - X**2)
    arr_desired = np.outer(arr_sqrt, X) - np.outer(X, arr_sqrt)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    size_new = 3
    ones_new = np.ones(size_new)
    gadf = GADF(image_size=size_new)
    arr_actual = gadf.transform(X[np.newaxis, :])[0]
    X_new = np.linspace(-1, 1, size_new)
    arr_new_sqrt = np.sqrt(ones_new - X_new**2)
    arr_desired = np.outer(arr_new_sqrt, X_new) - np.outer(X_new, arr_new_sqrt)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test: loop
    image_size_list = [3, 5]
    overlapping_list = [True, False]
    scale_list = [-1, 0]
    for image_size, overlapping, scale in product(*[image_size_list,
                                                    overlapping_list,
                                                    scale_list]):
        gadf = GADF(image_size, overlapping, scale)
        gadf.fit_transform(X[np.newaxis, :])


def test_MTF():
    """Testing 'MTF'."""
    # Parameter
    size = 9
    X = np.linspace(-1, 1, size)

    # Test 1
    mtf = MTF(image_size=size, n_bins=3)
    arr_actual = mtf.transform(X[np.newaxis, :])[0]
    MTF_arr = np.array([[2., 1., 0.], [0., 2., 1.], [0., 0., 2.]])
    MTF_arr = np.multiply(MTF_arr.T, (MTF_arr.sum(axis=1) ** (-1))).T
    arr_desired = np.repeat(np.tile(MTF_arr, 3), 3).reshape(9, 9)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    size_new = 3
    mtf = MTF(image_size=size_new, n_bins=3)
    arr_actual = mtf.transform(X[np.newaxis, :])[0]
    MTF_arr = np.array([[2., 1., 0.], [0., 2., 1.], [0., 0., 2.]])
    arr_desired = np.multiply(MTF_arr.T, (MTF_arr.sum(axis=1) ** (-1))).T
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test: loop
    image_size_list = [3, 5]
    n_bins_list = [2, 4]
    quantiles_list = ['empirical', 'gaussian']
    overlapping_list = [True, False]
    for (image_size, n_bins,
         quantiles, overlapping) in product(*[image_size_list,
                                              n_bins_list,
                                              quantiles_list,
                                              overlapping_list]):
        mtf = MTF(image_size, n_bins, quantiles, overlapping)
        mtf.fit_transform(X[np.newaxis, :])


def test_RecurrencePlots():
    """Testing 'RecurrencePlots'."""
    # Parameter
    size = 9
    X = np.linspace(-1, 1, size)

    # Test 1
    rp = RecurrencePlots(dimension=1)
    arr_actual = rp.transform(X[np.newaxis, :])[0]
    arr_desired = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            arr_desired[i, j] = abs(X[i] - X[j])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    percentage = 50
    rp = RecurrencePlots(dimension=1,
                         epsilon='percentage_distance',
                         percentage=percentage)
    arr_actual = rp.transform(X[np.newaxis, :])[0]
    arr_desired = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            threshold = percentage * (X.max() - X.min()) / 100
            arr_desired[i, j] = abs(X[i] - X[j]) < threshold
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 3
    percentage = 50
    rp = RecurrencePlots(dimension=1,
                         epsilon='percentage_points',
                         percentage=percentage)
    arr_actual = rp.transform(X[np.newaxis, :])[0]
    arr_desired = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            arr_desired[i, j] = abs(X[i] - X[j])
    arr_desired = arr_desired < np.percentile(arr_desired, q=50)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test: loop
    dimension_list = [1, 2, 3]
    epsilon_list = [None, 'percentage_points', 'percentage_distance', 3.]
    percentage_list = [10, 40]
    for (dimension, epsilon,
         percentage) in product(*[dimension_list,
                                  epsilon_list,
                                  percentage_list]):
        rp = RecurrencePlots(dimension, epsilon, percentage)
        rp.fit_transform(X[np.newaxis, :])
