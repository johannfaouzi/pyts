"""Testing for Lower Bounds of Dynamic Time Warping."""

import numpy as np
import pytest
import re
from math import sqrt

from pyts.metrics.lower_bounds import (
    _lower_bound_yi_x_y, _lower_bound_yi_X_Y, _warping_envelope, _clip
)
from pyts.metrics import (lower_bound_improved, lower_bound_keogh,
                          lower_bound_kim, lower_bound_yi)
from pyts.metrics import dtw, sakoe_chiba_band
from pyts.metrics.dtw import _dtw_sakoechiba
from sklearn.metrics import pairwise_distances


@pytest.mark.parametrize(
    'X, Y, err_msg',
    [([[1, 1]], [[1]], "Found input variables with inconsistent numbers of "
                       "timestamps: [2, 1]"),
     ([[1]], [[1, 2]], "Found input variables with inconsistent numbers of "
                       "timestamps: [1, 2]"),
     ([[3, 1, 1]], [[1]], "Found input variables with inconsistent numbers of "
                          "timestamps: [3, 1]")]
)
def test_check_consistent_lengths(X, Y, err_msg):
    """Test 'lower_bound_yi' parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        lower_bound_yi(X, Y)


@pytest.mark.parametrize(
    'x, y, float_desired',
    [([1, 5, 3, 2, 8], [3, 2, 5, 4, 5], sqrt(10)),
     ([4, 5, 3, 6, 8], [1, 2, 0, 1, 2], sqrt(66)),
     ([4, 5, 3, 6, 8], [1, 5, 3, 2, 1], sqrt(19)),
     ([3, 2, 5, 4, 5], [1, 5, 3, 2, 8], sqrt(10)),
     ([1, 2, 0, 1, 2], [4, 5, 3, 6, 8], sqrt(66)),
     ([1, 5, 3, 2, 1], [4, 5, 3, 6, 8], sqrt(19))]
)
def test_lower_bound_yi_x_y(x, y, float_desired):
    """Test '_lower_bound_yi_x_y' function."""
    x, y = np.asarray(x), np.asarray(y)
    float_actual = _lower_bound_yi_x_y(x, min(x), max(x), y, min(y), max(y))
    np.testing.assert_allclose(float_actual, float_desired, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    'X, Y',
    [([[3, 5, 1, 4, 6], [3, 8, 2, 4, 2]],
      [[4, 5, 9, 2, 3], [4, 3, 5, 2, 3], [5, 9, 3, 3, 4]])]
)
def test_lower_bound_yi_X_Y(X, Y):
    """Test '_lower_bound_yi_X_Y' function."""
    X, Y = np.asarray(X), np.asarray(Y)
    arr_actual = _lower_bound_yi_X_Y(X, X.min(axis=1), X.max(axis=1),
                                     Y, Y.min(axis=1), Y.max(axis=1))
    arr_desired = np.empty((2, 3))
    for i in range(2):
        for j in range(3):
            arr_desired[i, j] = _lower_bound_yi_x_y(
                X[i], X[i].min(), X[i].max(),
                Y[j], Y[j].min(), Y[j].max()
            )
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    'X_train, X_test, arr_desired',
    [([[5, 4, 3, 2, 1], [1, 8, 4, 3, 2], [6, 3, 5, 4, 7]],
      [[2, 1, 8, 4, 5]],
      np.sqrt([[9, 0, 6]]))]
)
def test_actual_results_lower_bound_yi(X_train, X_test, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = lower_bound_yi(X_train, X_test)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    'X_train, X_test, arr_desired',
    [([[2, 1, 8, 4, 5], [1, 2, 3, 4, 5]],
      [[5, 4, 3, 2, 1], [1, 8, 4, 3, 2], [6, 3, 5, 4, 7]],
      [[4, 4], [3, 3], [4, 5]])]
)
def test_actual_results_lower_bound_kim(X_train, X_test, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = lower_bound_kim(X_train, X_test)
    np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'X, region, err_msg',
    [([0, 1], [[0, 1], [0, 1]], "X must be a two- or three-dimensional."),
     ([[[[0, 1, 2]]]], [[0, 1], [0, 1]],
      "X must be a two- or three-dimensional.")]
)
def test_parameter_check_warping_envelope(X, region, err_msg):
    """Test '_warping_envelope' parameter validation.."""
    with pytest.raises(ValueError, match=err_msg):
        _warping_envelope(X, region)


@pytest.mark.parametrize(
    'X, region, lower_desired, upper_desired',
    [([[0, 1, 2, 3], [1, 5, 3, -1]], [[0, 0, 1, 2], [2, 3, 4, 4]],
      [[0, 0, 1, 2], [1, 1, -1, -1]], [[1, 2, 3, 3], [5, 5, 5, 3]]),
     ([[[0, 1, 2, 3], [1, 5, 3, -1]]], [[0, 0, 1, 2], [2, 3, 4, 4]],
      [[[0, 0, 1, 2], [1, 1, -1, -1]]], [[[1, 2, 3, 3], [5, 5, 5, 3]]])]
)
def test_actual_results_warping_envelope(X, region,
                                         lower_desired, upper_desired):
    """Test that the actual results are the expected ones."""
    lower_actual, upper_actual = _warping_envelope(X, region)
    np.testing.assert_array_equal(lower_actual, lower_desired)
    np.testing.assert_array_equal(upper_actual, upper_desired)


@pytest.mark.parametrize(
    'X, lower, upper, err_msg',
    [([[0], [1]], [6], [[1], [1]],
      "'lower' must be two- or three-dimensional."),
     ([[0], [1]], [[[[6]]]], [[1], [1]],
      "'lower' must be two- or three-dimensional."),
     ([[0], [1]], [[[6]]], [[1], [1]],
      "'lower' and 'upper' must have the same shape ((1, 1, 1) != (2, 1))")]
)
def test_parameter_check_clip(X, lower, upper, err_msg):
    """Test '_clip' parameter validation.."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        _clip(X, lower, upper)


@pytest.mark.parametrize(
    'X, lower, upper, arr_desired',
    [([[0, 1, 2, 3], [1, 5, 3, -1]],
      [[0, 3, 3, 3], [-1, 2, 4, 6]],
      [[1, 5, 4, 6], [1, 3, 6, 8]],
      [[[0, 3, 3, 3], [0, 2, 4, 6]], [[1, 5, 3, 3], [1, 3, 4, 6]]]),

     ([[0, 1, 2, 3], [1, 5, 3, -1]],
      [[[0, 3, 3, 3], [-1, 2, 4, 6]]],
      [[[1, 5, 4, 6], [1, 3, 6, 8]]],
      [[[0, 3, 3, 3]], [[1, 3, 4, 6]]])]
)
def test_actual_results_clip(X, lower, upper, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = _clip(X, lower, upper)
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_actual_results_lower_bound_keogh():
    """Test that the actual results are the expected ones."""
    # Toy dataset
    X_train = np.asarray([[0, 1, 2, 3],
                          [1, 2, 3, 4]])
    X_test = np.asarray([[0, 2.5, 3.5, 6]])

    # Region = Sakoe-Chiba band (w=0)
    region = [[0, 1, 2, 3],
              [1, 2, 3, 4]]
    arr_actual = lower_bound_keogh(X_train, X_test, region)
    arr_desired = np.sqrt(np.sum(
        (X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=-1
    ))
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)

    # Region = Sakoe-Chiba band (w=1)
    region_window = [[0, 0, 1, 2],
                     [2, 3, 4, 4]]
    arr_actual_window = lower_bound_keogh(X_train, X_test, region_window)
    # lower = [[0, 0, 1, 2], [1, 1, 2, 3]]
    # upper = [[1, 2, 3, 3], [2, 3, 4, 4]]
    # X_proj = [[0, 2, 3, 3], [1, 2.5, 3.5, 4]]
    # LB_Keogh = [[sqrt(0.25 + 0.25 + 9), sqrt(1 + 4)]]
    arr_desired_window = np.sqrt([[9.5, 5]])
    np.testing.assert_allclose(arr_actual_window, arr_desired_window,
                               atol=1e-5, rtol=0)


def test_actual_results_lower_bound_improved():
    """Test that the actual results are the expected ones."""
    # Toy dataset
    X_train = np.asarray([[0, 1, 2, 3],
                          [1, 2, 3, 4]])
    X_test = np.asarray([[0, 2.5, 3.5, 3.3]])

    # Region = Sakoe-Chiba band (w=0)
    region = [[0, 1, 2, 3],
              [1, 2, 3, 4]]
    arr_actual = lower_bound_improved(X_train, X_test, region)
    arr_desired = np.sqrt(np.sum(
        (X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=-1
    ))
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)

    # Region = Sakoe-Chiba band (w=1)
    region_window = [[0, 0, 1, 2],
                     [2, 3, 4, 4]]
    arr_actual_window = lower_bound_improved(X_train, X_test, region_window)
    # lower_train = [[0, 0, 1, 2], [1, 1, 2, 3]
    # upper_train = [[1, 2, 3, 3], [2, 3, 4, 4]]
    # X_test_proj = [[0, 2, 3, 3], [1, 2.5, 3.5, 3.3]
    # LB_Keogh^2 = [[0.25 + 0.25 + 0.09, 1]] = [[0.59, 1]]
    # lower_test = [[0, 0, 2, 3], [1, 1, 2.5, 3.3]]
    # upper_test = [[2, 3, 3, 3], [2.5, 3.5, 3.5, 3.5]]
    # X_train_proj = [[0, 1, 2, 3], [1, 2, 3, 3.5]]
    # LB_Improved^2 = [[0, 0.25]]
    arr_desired_window = np.sqrt([[0.59 + 0, 1 + 0.25]])
    np.testing.assert_allclose(arr_actual_window, arr_desired_window,
                               atol=1e-5, rtol=0)


def test_lower_bounds_inequalities():
    """Test that the expected inequalities are verified."""
    # Toy dataset
    rng = np.random.RandomState(42)
    n_samples_train, n_samples_test, n_timestamps = 20, 30, 60
    window_size = 0.1
    X_train = rng.randn(n_samples_train, n_timestamps)
    X_test = rng.randn(n_samples_test, n_timestamps)

    # DTW
    X_dtw = pairwise_distances(X_test, X_train, dtw)
    region = sakoe_chiba_band(n_timestamps, window_size=window_size)
    X_dtw_window = pairwise_distances(X_test, X_train, _dtw_sakoechiba,
                                      window_size=window_size)

    # Lower bounds
    lb_yi = lower_bound_yi(X_train, X_test)
    lb_kim = lower_bound_kim(X_train, X_test)
    lb_keogh = lower_bound_keogh(X_train, X_test, region)
    lb_improved = lower_bound_improved(X_train, X_test, region)

    # Sanity check
    EPS = 1e-8
    np.testing.assert_array_less(lb_yi, X_dtw + EPS)
    np.testing.assert_array_less(lb_kim, X_dtw + EPS)
    np.testing.assert_array_less(lb_keogh, X_dtw_window + EPS)
    np.testing.assert_array_less(lb_improved, X_dtw_window + EPS)
    np.testing.assert_array_less(lb_keogh, lb_improved + EPS)
