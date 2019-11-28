"""Testing for discretizers."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.preprocessing.discretizer import _uniform_bins, _digitize
from pyts.preprocessing import KBinsDiscretizer


n_samples, n_timestamps = 3, 10
X = np.arange(n_samples * n_timestamps).reshape(n_samples, n_timestamps)


@pytest.mark.parametrize(
    'sample_min, sample_max, n_bins, arr_desired',
    [(np.arange(50), np.arange(20, 70), 2, [np.arange(10, 60)]),
     (np.arange(30, 80), np.arange(-10, 40), 2, [np.arange(10, 60)]),
     (np.arange(30), np.arange(30, 60), 3,
      [np.arange(10, 40), np.arange(20, 50)])]
)
def test_uniform_bins(sample_min, sample_max, n_bins, arr_desired):
    """Test '_uniform_bins' function."""
    arr_actual = _uniform_bins(sample_min, sample_max,
                               n_samples=sample_min.size, n_bins=n_bins)
    np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'X, bins, arr_desired',
    [(X, np.array([4.5, 14.5, 24.5]),
      [[i] * 5 + [i + 1] * 5 for i in range(3)]),
     (X, np.array([[-0.5, 1.5, 8.5, 11.5],
                   [9.5, 11.5, 18.5, 21.5],
                   [19.5, 21.5, 28.5, 31.5]]),
      [[1] * 2 + [2] * 7 + [3] for _ in range(3)])]
)
def test_digitize(X, bins, arr_desired):
    """Test '_digitize' function."""
    X_float = np.asarray(X, dtype='float64')
    arr_actual = _digitize(X_float, bins)
    np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_bins': None}, TypeError, "'n_bins' must be an integer."),

     ({'n_bins': 1}, ValueError,
      "'n_bins' must be greater than or equal to 2 and lower than "
      "or equal to n_timestamps (got 1)."),

     ({'n_bins': n_timestamps + 1}, ValueError,
      "'n_bins' must be greater than or equal to 2 and lower than "
      "or equal to n_timestamps (got {0}).".format(n_timestamps + 1)),

     ({'strategy': 'whoops'}, ValueError,
      "'strategy' must be either 'uniform', 'quantile' or 'normal' (got {0})."
      .format('whoops'))]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    discretizer = KBinsDiscretizer(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        discretizer.transform(X)


def test_constant_sample():
    """Test that a ValueError is raised with a constant sample."""
    discretizer = KBinsDiscretizer()
    with pytest.raises(ValueError, match="At least one sample is constant."):
        discretizer.fit_transform(np.ones((10, 15)))


def test_warning_smaller_n_bins():
    """Test that a warning is raised when n_bins will be smaller."""
    X_new = np.r_[np.zeros((1, n_timestamps)), X]
    discretizer = KBinsDiscretizer()
    warning_msg = (
        "Some quantiles are equal. The number of bins will be "
        "smaller for sample {0}. Consider decreasing the number "
        "of bins or removing these samples.".format([0])
    )
    with pytest.warns(None, match=re.escape(warning_msg)):
        discretizer._compute_bins(
            X_new, n_samples + 1, n_bins=5, strategy='quantile'
        )


@pytest.mark.parametrize(
    'params, X, arr_desired',
    [({'n_bins': 10, 'strategy': 'quantile'}, X,
      [list(range(10))] * n_samples),

     ({'n_bins': 10, 'strategy': 'uniform'}, X,
      [list(range(10))] * n_samples),

     ({'n_bins': 5, 'strategy': 'quantile'}, X,
      [np.repeat(np.arange(5), 2)] * n_samples),

     ({'n_bins': 5, 'strategy': 'uniform'}, X,
      [np.repeat(np.arange(5), 2)] * n_samples),

     ({'n_bins': 3, 'strategy': 'normal'}, [[-0.5, 0, 0.5]], [[0, 1, 2]])]
)
def test_actual_results(params, X, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = KBinsDiscretizer(**params).fit_transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
