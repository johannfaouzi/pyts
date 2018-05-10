"""Tests for :mod:`pyts.utils` module."""

import numpy as np
from ..utils import segmentation, numerosity_reduction, dtw, fast_dtw


def test_segmentation():
    """Testing 'segmentation'."""
    # Test 1
    bounds = np.array([0, 4, 8, 12, 16, 20])
    window_size = 4
    overlapping = False
    res_actual = segmentation(20, window_size, overlapping)
    res_start = bounds[:-1]
    res_end = bounds[1:]
    res_size = 5
    np.testing.assert_array_equal(res_actual[0], res_start)
    np.testing.assert_array_equal(res_actual[1], res_end)
    np.testing.assert_equal(res_actual[2], res_size)


def test_numerosity_reduction():
    """Testing 'numerosity_reduction'."""
    # Test 1
    array = np.array(["aaa", "aaa", "aaa", "bbb", "bbb", "ccc", "aaa"])
    arr_actual = numerosity_reduction(array)
    arr_desired = ' '.join(["aaa", "bbb", "ccc", "aaa"])
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_dtw():
    """Testing 'dtw'."""
    # Parameter 1
    x1 = np.array([1, -1, -1, 1, 1])
    x2 = np.array([1, 1, -1, -1, 1])

    # Test 1
    cost_actual = dtw(x1, x2)
    cost_desired = 0
    np.testing.assert_equal(cost_actual, cost_desired)

    # Parameter 2
    x1 = np.array([1, -1, 1, 1, 1, -1])
    x2 = np.ones(6)
    x3 = - np.ones(6)

    # Test 1
    cost_actual = dtw(x1, x2)
    cost_desired = 4
    np.testing.assert_equal(cost_actual, cost_desired)

    # Test 2
    cost_actual = dtw(x1, x3)
    cost_desired = 8
    np.testing.assert_equal(cost_actual, cost_desired)

    # Test 3
    cost_actual = dtw(x2, x3)
    cost_desired = 12
    np.testing.assert_equal(cost_actual, cost_desired)


def test_fast_dtw():
    """Testing 'fast_dtw'."""
    # Parameter
    x1 = np.array([1, -1, 1, 1, 1, -1])
    x2 = np.ones(6)
    x3 = - np.ones(6)

    # Test 1
    cost_actual = fast_dtw(x1, x2, window_size=2)
    cost_desired = 2
    np.testing.assert_equal(cost_actual, cost_desired)

    # Test 2
    cost_actual = fast_dtw(x1, x3, window_size=2)
    cost_desired = 4
    np.testing.assert_equal(cost_actual, cost_desired)

    # Test 3
    cost_actual = fast_dtw(x2, x3, window_size=2)
    cost_desired = 6
    np.testing.assert_equal(cost_actual, cost_desired)
