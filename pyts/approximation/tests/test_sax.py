"""Testing for Symbolic Aggregate Approximation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.approximation import SymbolicAggregateApproximation


X = np.arange(30).reshape(3, 10)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_bins': '3'}, TypeError, "'n_bins' must be an integer."),

     ({'alphabet': 'whoops'}, TypeError,
      "'alphabet' must be None, 'ordinal' or array-like with shape (n_bins,) "
      "(got {0})".format('whoops')),

     ({'n_bins': 1}, ValueError,
      "'n_bins' must be greater than or equal to 2 and lower than "
      "or equal to min(n_timestamps, 26) (got 1)."),

     ({'n_bins': 15}, ValueError,
      "'n_bins' must be greater than or equal to 2 and lower than "
      "or equal to min(n_timestamps, 26) (got 15)."),

     ({'strategy': 'whoops'}, ValueError,
      "'strategy' must be either 'uniform', 'quantile' or 'normal' "
      "(got {0})".format('whoops')),

     ({'alphabet': ['a', 'b', 'c']}, ValueError,
      "If 'alphabet' is array-like, its shape must be equal to (n_bins,).")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    sax = SymbolicAggregateApproximation(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        sax.transform(X)


@pytest.mark.parametrize(
    'params, X, arr_desired',
    [({}, [[0, 1, 2, 3]], [['a', 'b', 'c', 'd']]),

     ({'strategy': 'uniform'}, [[0, 1, 2, 3]], [['a', 'b', 'c', 'd']]),

     ({}, [[0, 4, 2, 6]], [['a', 'c', 'b', 'd']]),

     ({}, [[-5, -8, -7, -6]], [['d', 'a', 'b', 'c']]),

     ({'alphabet': 'ordinal'}, [[0, 1, 2, 3]], [[0, 1, 2, 3]]),

     ({'alphabet': ['d', 'c', 'b', 'a']}, [[0, 1, 2, 3]],
      [['d', 'c', 'b', 'a']]),

     ({'alphabet': ['0', '1', '2', '3']}, [[0, 3, 2, 1]],
      [['0', '3', '2', '1']])]
)
def test_actual_results(params, X, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = SymbolicAggregateApproximation(**params).fit_transform(X)
    np.testing.assert_array_equal(arr_actual, arr_desired)
