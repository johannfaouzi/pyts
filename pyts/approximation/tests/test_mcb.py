"""Testing for Multiple Coefficient Binning."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.approximation.mcb import _uniform_bins, _digitize
from pyts.approximation import MultipleCoefficientBinning


X = np.arange(30).reshape(10, 3)
y = [0, 0, 0, 1, 1, 2, 1, 0, 0, 2]


@pytest.mark.parametrize(
    'timestamp_min, timestamp_max, n_bins, arr_desired',
    [(np.arange(5), np.arange(2, 7), 2, np.arange(1, 6).reshape(5, 1)),

     ([4, 8, -2, 2], [8, 20, 2, 10], 4,
      [[5, 6, 7], [11, 14, 17], [-1, 0, 1], [4, 6, 8]]),

     ([0, 10, 10], [12, 28, 70], 6,
      [[2, 4, 6, 8, 10], [13, 16, 19, 22, 25], [20, 30, 40, 50, 60]])]
)
def test_uniform_bins(timestamp_min, timestamp_max, n_bins, arr_desired):
    """Test that the actual results are the expected ones."""
    timestamp_min = np.asarray(timestamp_min)
    timestamp_max = np.asarray(timestamp_max)
    n_timestamps = timestamp_min.size
    arr_actual = _uniform_bins(timestamp_min, timestamp_max,
                               n_timestamps, n_bins)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'X, bins, arr_desired',
    [([[0, 1], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [1, 0]],
      [0.2, 0.4, 0.6, 0.8],
      [[0, 4], [0, 3], [1, 2], [2, 1], [3, 0], [4, 0]]),

     ([[0, 1], [0.4, 1.8], [0.4, 1.6], [0.6, 1.4], [0.2, 1.2], [1, 2]],
      [0.2, 0.4, 0.6, 0.8],
      [[0, 4], [1, 4], [1, 4], [2, 4], [0, 4], [4, 4]]),

     ([[0, 1], [0.4, 1.8], [0.4, 1.6], [0.6, 1.4], [0.2, 1.2], [1, 2]],
      [[0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]],
      [[0, 4], [1, 4], [1, 4], [2, 4], [0, 4], [4, 4]]),

     ([[0, 1], [0.4, 1.8], [0.4, 1.6], [0.6, 1.4], [0.2, 1.2], [1, 2]],
      [[0.2, 0.4, 0.6, 0.8], [0.2, 1.4, 1.6, 2.8]],
      [[0, 1], [1, 3], [1, 2], [2, 1], [0, 1], [4, 3]])]
)
def test_digitize(X, bins, arr_desired):
    """Test that the actual results are the expected ones."""
    X = np.asarray(X)
    bins = np.asarray(bins)
    arr_actual = _digitize(X, bins)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_bins': '3'}, TypeError, "'n_bins' must be an integer."),

     ({'alphabet': 'whoops'}, TypeError,
      "'alphabet' must be None, 'ordinal' or array-like with shape (n_bins,) "
      "(got {0})".format('whoops')),

     ({'n_bins': 1}, ValueError,
      "'n_bins' must be greater than or equal to 2 and lower than "
      "or equal to min(n_samples, 26) (got 1)."),

     ({'n_bins': 15}, ValueError,
      "'n_bins' must be greater than or equal to 2 and lower than "
      "or equal to min(n_samples, 26) (got 15)."),

     ({'strategy': 'whoops'}, ValueError,
      "'strategy' must be either 'uniform', 'quantile', 'normal' or 'entropy' "
      "(got {0})".format('whoops')),

     ({'alphabet': ['a', 'b', 'c']}, ValueError,
      "If 'alphabet' is array-like, its shape must be equal to (n_bins,).")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    mcb = MultipleCoefficientBinning(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        mcb.fit(X, y)


def test_constant_sample():
    """Test that a ValueError is raised with a constant sample."""
    discretizer = MultipleCoefficientBinning()
    msg_error = "At least one timestamp is constant."
    with pytest.raises(ValueError, match=msg_error):
        discretizer.fit_transform(np.ones((10, 15)))


def test_consistent_length():
    """Test that a ValueError is raised with a constant sample."""
    msg_error = (
        "The number of timestamps in X must be the same as "
        "the number of timestamps when `fit` was called "
        "({0} != {1}).".format(3, 1)
    )
    mcb = MultipleCoefficientBinning()
    mcb.fit(X, y)
    with pytest.raises(ValueError, match=re.escape(msg_error)):
        mcb.transform(X[:, :1])


def test_identical_bin_edges():
    """Test that an error is raised when two consecutive bins are equal."""
    msg_error = ("At least two consecutive quantiles are equal. "
                 "Consider trying with a smaller number of bins or "
                 "removing timestamps with low variation.")
    mcb = MultipleCoefficientBinning(n_bins=6, strategy='quantile')
    with pytest.raises(ValueError, match=msg_error):
        mcb.fit(np.r_[np.zeros((4, 2)), np.ones((4, 2))])


def test_y_none_entropy():
    """Test that an error is raised when y is None and 'entropy' is used."""
    msg_error = "y cannot be None if strategy='entropy'."
    mcb = MultipleCoefficientBinning(n_bins=2, strategy='entropy')
    with pytest.raises(ValueError, match=msg_error):
        mcb.fit(X, None)


def test_high_n_bins_entropy():
    """Test that an error is raised when 'n_bins' is too large."""
    msg_error = ("The number of bins is too high for timestamp {0}. "
                 "Consider trying with a smaller number of bins or "
                 "removing this timestamp.".format(0))
    mcb = MultipleCoefficientBinning(n_bins=9, strategy='entropy')
    with pytest.raises(ValueError, match=msg_error):
        mcb.fit(X, y)


@pytest.mark.parametrize(
    'params, X, y, arr_desired',
    [({}, X, None,
      np.asarray([['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'd']] * 3).T),

     ({'strategy': 'uniform'}, X, None,
      np.asarray([['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'd']] * 3).T),

     ({'alphabet': 'ordinal'}, X, None,
      np.asarray([[0, 0, 0, 1, 1, 2, 2, 3, 3, 3]] * 3).T),

     ({'n_bins': 10, 'strategy': 'quantile', 'alphabet': 'ordinal'}, X, None,
      np.asarray([np.arange(10)] * 3).T),

     ({'n_bins': 10, 'strategy': 'uniform', 'alphabet': 'ordinal'}, X, None,
      np.asarray([np.arange(10)] * 3).T),

     ({'n_bins': 5, 'strategy': 'quantile', 'alphabet': 'ordinal'}, X, None,
      np.asarray([np.repeat(np.arange(5), 2)] * 3).T),

     ({'n_bins': 5, 'strategy': 'uniform', 'alphabet': 'ordinal'}, X, None,
      np.asarray([np.repeat(np.arange(5), 2)] * 3).T),

     ({'n_bins': 3, 'strategy': 'normal'}, [[-0.5], [0], [0.5]], None,
      [['a'], ['b'], ['c']]),

     ({'n_bins': 6, 'strategy': 'entropy'}, X, y,
      np.asarray([['a', 'a', 'a', 'b', 'b', 'c', 'd', 'e', 'e', 'f']] * 3).T),

     ({'n_bins': 6, 'strategy': 'entropy', 'alphabet': 'ordinal'}, X, y,
      np.asarray([[0, 0, 0, 1, 1, 2, 3, 4, 4, 5]] * 3).T)]
)
def test_actual_results(params, X, y, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = MultipleCoefficientBinning(**params).fit_transform(X, y)
    np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'params',
    [({}),
     ({'strategy': 'uniform'}),
     ({'alphabet': 'ordinal'}),
     ({'n_bins': 10, 'strategy': 'quantile', 'alphabet': 'ordinal'}),
     ({'n_bins': 5, 'strategy': 'quantile', 'alphabet': 'ordinal'}),
     ({'n_bins': 5, 'strategy': 'normal'}),
     ({'n_bins': 6, 'strategy': 'entropy'}),
     ({'n_bins': 6, 'strategy': 'entropy', 'alphabet': 'ordinal'})]
)
def test_fit_transform(params):
    """Test that fit and transform yield the same results as fit_transform."""
    arr_1 = MultipleCoefficientBinning(**params).fit(X, y).transform(X)
    arr_2 = MultipleCoefficientBinning(**params).fit_transform(X, y)
    np.testing.assert_array_equal(arr_1, arr_2)
