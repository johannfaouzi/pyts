"""Testing for Piecewise Aggregate Approximation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.approximation.paa import _paa
from pyts.approximation import PiecewiseAggregateApproximation


X = [np.arange(6), [0, 5, 1, 2, 4, 3]]


@pytest.mark.parametrize(
    'X, start, end, arr_desired',
    [([[0, 1, 2, 3], [1, 3, 0, 2], [4, 8, 2, 6]], [0, 2], [2, 4],
      [[0.5, 2.5], [2.0, 1.0], [6.0, 4.0]]),

     ([[0, 1, 2, 3], [1, 3, 0, 2], [4, 8, 2, 6]], [0, 1, 2], [2, 3, 4],
      [[0.5, 1.5, 2.5], [2.0, 1.5, 1.0], [6.0, 5.0, 4.0]]),

     ([[0, 1, 2, 3], [1, 3, 0, 2], [4, 8, 2, 6]], [0, 1, 2, 3], [1, 2, 3, 4],
      [[0, 1, 2, 3], [1, 3, 0, 2], [4, 8, 2, 6]])]
)
def test_actual_results_paa(X, start, end, arr_desired):
    """Test that the actual results are the expected ones."""
    X = np.asarray(X)
    start = np.asarray(start)
    end = np.asarray(end)
    n_timestamps_new = start.size
    n_samples, n_timestamps = X.shape
    arr_actual = _paa(X, n_samples, n_timestamps, start, end, n_timestamps_new)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'window_size': None, 'output_size': None}, TypeError,
      "'window_size' and 'output_size' cannot be both None."),

     ({'window_size': '3'}, TypeError,
      "If specified, 'window_size' must be an integer or a float."),

     ({'window_size': None, 'output_size': '3'}, TypeError,
      "If specified, 'output_size' must be an integer or a float."),

     ({'window_size': 0}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps (got 0)."),

     ({'window_size': 2.}, ValueError,
      "If 'window_size' is a float, it must be greater than 0 and lower than "
      "or equal to 1 (got {0}).".format(2.)),

     ({'window_size': None, 'output_size': 0}, ValueError,
      "If 'output_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps (got 0)."),

     ({'window_size': None, 'output_size': 2.}, ValueError,
      "If 'output_size' is a float, it must be greater than 0 and lower than "
      "or equal to 1 (got {0}).".format(2.))]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    paa = PiecewiseAggregateApproximation(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        paa.transform(X)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({}, X),
     ({'window_size': 2}, [[0.5, 2.5, 4.5], [2.5, 1.5, 3.5]]),

     ({'window_size': None, 'output_size': 3},
      [[0.5, 2.5, 4.5], [2.5, 1.5, 3.5]]),

     ({'window_size': None, 'output_size': 0.5},
      [[0.5, 2.5, 4.5], [2.5, 1.5, 3.5]]),

     ({'window_size': 0.5}, [[1, 4], [2, 3]])]
)
def test_actual_results(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = PiecewiseAggregateApproximation(**params).fit_transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
