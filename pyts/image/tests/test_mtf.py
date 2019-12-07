"""Testing for Markov Transition Field."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.image.mtf import (_markov_transition_matrix,
                            _markov_transition_field,
                            _aggregated_markov_transition_field)
from pyts.image import MarkovTransitionField


X = [[0, 1, 2, 3], [1, 0, 1, 1]]


@pytest.mark.parametrize(
    'X_binned, n_bins, arr_desired',
    [([[0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]], 2, [[[3, 3], [3, 1]]]),

     ([[0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]], 3,
      [[[3, 3, 0], [3, 1, 0], [0, 0, 0]]]),

     ([[0, 1, 2, 3], [0, 2, 1, 3]], 4,
      [[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
       [[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]]])]
)
def test_markov_transition_matrix(X_binned, n_bins, arr_desired):
    """Test that the actual results are the expected ones."""
    X_binned = np.asarray(X_binned)
    n_samples, n_timestamps = X_binned.shape
    arr_actual = _markov_transition_matrix(
        X_binned, n_samples, n_timestamps, n_bins)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'X_binned, X_mtm, n_bins, arr_desired',
    [([[1, 0, 1, 0, 0, 1, 0]], [[[1, 2], [3, 0]]], 2,
      [[[0, 3, 0, 3, 3, 0, 3],
        [2, 1, 2, 1, 1, 2, 1],
        [0, 3, 0, 3, 3, 0, 3],
        [2, 1, 2, 1, 1, 2, 1],
        [2, 1, 2, 1, 1, 2, 1],
        [0, 3, 0, 3, 3, 0, 3],
        [2, 1, 2, 1, 1, 2, 1]]]),

     ([[0, 1, 2, 0, 1], [0, 2, 0, 1, 1]],
      [[[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, .5, .5], [0, 1, 0], [1, 0, 0]]],
      3,
      [[[0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0]],
       [[0.0, 0.5, 0.0, 0.5, 0.5],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0]]])]
)
def test_markov_transition_field(X_binned, X_mtm, n_bins, arr_desired):
    """Test that the actual results are the expected ones."""
    X_binned = np.asarray(X_binned)
    X_mtm = np.asarray(X_mtm)
    n_samples, n_timestamps = X_binned.shape
    arr_actual = _markov_transition_field(
        X_binned, X_mtm, n_samples, n_timestamps, n_bins
    )
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'image_size, start, end, X_mtf, arr_desired',
    [(2, [0, 2], [2, 4],
      [[[0, 1, 2, 0], [1, 0, 2, 0], [1, 1, 0, 0], [0, 1, 2, 2]],
       [[2, 1, 2, 0], [0, 1, 3, 0], [0, 1, 2, 0], [0, 0, 0, 0]]],
      [[[0.5, 1.0], [0.75, 1.0]], [[1.0, 1.25], [0.25, 0.5]]]),

     (3, [0, 1, 2], [2, 3, 4],
      [[[0, 1, 2, 0], [1, 0, 2, 0], [1, 1, 0, 0], [0, 1, 2, 2]],
       [[2, 1, 2, 0], [0, 1, 3, 0], [0, 1, 2, 0], [0, 0, 0, 0]]],
      [[[0.5, 1.25, 1.0], [0.75, 0.75, 0.5], [0.75, 1, 1.0]],
       [[1.0, 1.75, 1.25], [0.5, 1.75, 1.25], [0.25, 0.75, 0.5]]])]
)
def test_aggregated_markov_transition_field(
    image_size, start, end, X_mtf, arr_desired
):
    """Test that the actual results are the expected ones."""
    X_mtf = np.asarray(X_mtf)
    start = np.asarray(start)
    end = np.asarray(end)
    n_samples, _ = X_mtf.shape[:2]
    arr_actual = _aggregated_markov_transition_field(
        X_mtf, n_samples, image_size, start, end)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'image_size': '4'}, TypeError,
      "'image_size' must be an integer or a float."),

     ({'n_bins': [0, 1]}, TypeError,
      "'n_bins' must be an integer."),

     ({'image_size': 0}, ValueError,
      "If 'image_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps (got 0)."),

     ({'image_size': 2.}, ValueError,
      "If 'image_size' is a float, it must be greater than 0 and lower than "
      "or equal to 1 (got {0}).".format(2.)),

     ({'n_bins': 1}, ValueError,
      "'n_bins' must be greater than or equal to 2."),

     ({'strategy': 'whoops'}, ValueError,
      "'strategy' must be 'uniform', 'quantile' or 'normal'.")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    mtf = MarkovTransitionField(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        mtf.transform(X)


@pytest.mark.parametrize(
    'params, X, arr_desired',
    [({'image_size': 2, 'n_bins': 4},
      [np.arange(8)], [[[0.375, 0.125], [0.000, 0.500]]]),

     ({'image_size': 2, 'n_bins': 4, 'strategy': 'uniform'},
      [np.arange(8)], [[[0.375, 0.125], [0.000, 0.500]]]),

     ({'image_size': 0.25, 'n_bins': 4},
      [np.arange(8)], [[[0.375, 0.125], [0.000, 0.500]]]),

     ({'image_size': 7, 'n_bins': 2, 'overlapping': True},
      [np.arange(8)],
      [[[0.75, 0.75, 0.75, 0.50, 0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75, 0.50, 0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75, 0.50, 0.25, 0.25, 0.25],
        [0.375, 0.375, 0.375, 0.5, 0.625, 0.625, 0.625],
        [0, 0, 0, 0.5, 1, 1, 1],
        [0, 0, 0, 0.5, 1, 1, 1],
        [0, 0, 0, 0.5, 1, 1, 1]]]),

     ({'image_size': 7, 'n_bins': 2, 'overlapping': True,
       'strategy': 'uniform'}, [np.arange(8)],
      [[[0.75, 0.75, 0.75, 0.50, 0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75, 0.50, 0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75, 0.50, 0.25, 0.25, 0.25],
        [0.375, 0.375, 0.375, 0.5, 0.625, 0.625, 0.625],
        [0, 0, 0, 0.5, 1, 1, 1],
        [0, 0, 0, 0.5, 1, 1, 1],
        [0, 0, 0, 0.5, 1, 1, 1]]]),

     ({'image_size': 2, 'n_bins': 2, 'strategy': 'uniform'}, X,
      [[[0.5, 0.5], [0, 1]], [[0.5, 0.75], [0.5, 0.5]]])]
)
def test_actual_results(params, X, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = MarkovTransitionField(**params).fit_transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_flatten():
    """Test the 'flatten' parameter."""
    arr_false = MarkovTransitionField(n_bins=2).transform(X).reshape(2, -1)
    arr_true = MarkovTransitionField(n_bins=2, flatten=True).transform(X)
    np.testing.assert_allclose(arr_false, arr_true, atol=1e-5, rtol=0.)
