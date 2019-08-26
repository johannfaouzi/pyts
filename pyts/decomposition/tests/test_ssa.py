"""Testing for Singular Spectrum Analysis."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.decomposition.ssa import _outer_dot, _diagonal_averaging
from pyts.decomposition import SingularSpectrumAnalysis


rng = np.random.RandomState(42)
X = rng.randn(4, 30)


@pytest.mark.parametrize(
    'v, X, arr_desired',
    [(np.zeros((2, 2, 2)), np.ones((2, 2, 2)), np.zeros((2, 2, 2, 2))),

     (np.ones((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2, 2, 2))),

     ([[[1, 2], [3, 4]]],
      np.ones((1, 2, 2)),
      [[[[4, 4], [12, 12]], [[12, 12], [24, 24]]]]),

     ([[[1, 2], [3, 4]]],
      [[[2, 1], [-2, -3]]],
      [[[[-4, -8], [-12, -24]], [[-8, -20], [-16, -40]]]])]
)
def test_outer_dot(v, X, arr_desired):
    """Test that the actual results are the expected ones."""
    v = np.asarray(v).astype('float64')
    X = np.asarray(X).astype('float64')
    n_samples, n_windows, window_size = X.shape
    arr_actual = _outer_dot(v, X, n_samples, window_size, n_windows)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'X, arr_desired',
    [([[[[0, 1, 2], [3, 4, 5]]]], [[[0, (1 + 3) / 2, (2 + 4) / 2, 5]]]),

     ([[[[0, 1, 2, 0, -1], [3, -2, 4, 5, -3], [1, 3, 2, 4, 5]]]],
      [[[0, (1 + 3) / 2, (2 - 2 + 1) / 3, (0 + 4 + 3) / 3,
         (-1 + 5 + 2) / 3, (-3 + 4) / 2, 5]]])]
)
def test_diagonal_averaging(X, arr_desired):
    """Test that the actual results are the expected ones."""
    X = np.asarray(X, dtype='float64')
    n_samples, grouping_size, window_size, n_windows = X.shape
    n_timestamps = window_size + n_windows - 1
    gap = max(n_windows, window_size)

    arr_actual = _diagonal_averaging(
        X, n_samples, n_timestamps, window_size,
        n_windows, grouping_size, gap
    )
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'window_size': '4'}, TypeError,
      "'window_size' must be an integer or a float."),

     ({'groups': '3'}, TypeError,
      "'groups' must be either None, an integer or array-like."),

     ({'window_size': 1}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 2 "
      "and lower than or equal to n_timestamps (got 1)."),

     ({'window_size': 0.}, ValueError,
      "If 'window_size' is a float, it must be greater than 0 and lower than "
      "or equal to 1 (got {0}).".format(0.)),

     ({'groups': 6}, ValueError,
      "If 'groups' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to 'window_size'."),

     ({'groups': [[0, 2, 5]]}, ValueError,
      "If 'groups' is array-like, all the values in 'groups' must be integers "
      "between 0 and ('window_size' - 1).")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    ssa = SingularSpectrumAnalysis(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        ssa.transform(X)


@pytest.mark.parametrize(
    'params',
    [({}),
     ({'window_size': 2}),
     ({'window_size': 25}),
     ({'window_size': 0.2}),
     ({'groups': 3}),
     ({'window_size': 10, 'groups': 10}),
     ({'groups': [[0, 1], [2, 3]]}),
     ({'groups': [[0, 2], [1, 3]]}),
     ({'window_size': 8, 'groups': [[0, 2, 4, 6], [1, 3, 5, 7]]})]
)
def test_actual_results(params):
    """Test that the actual results are the expected ones."""
    ssa = SingularSpectrumAnalysis(**params)
    arr_actual = ssa.fit_transform(X).sum(axis=1)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)
