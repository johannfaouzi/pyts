"""Testing for Multiple Coefficient Binning."""

import numpy as np
import pytest
import re
from ..mcb import _uniform_bins, _digitize, MultipleCoefficientBinning


def test_uniform_bins():
    """Test '_uniform_bins' function."""
    timestamp_min = np.arange(5)
    timestamp_max = np.arange(2, 7)
    arr_actual = _uniform_bins(timestamp_min, timestamp_max,
                               n_timestamps=5, n_bins=2)
    arr_desired = ((timestamp_min + timestamp_max) / 2).reshape(5, 1)
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_digitize():
    """Test '_digitize' function."""
    n_samples = 10
    n_bins = 5
    X = np.asarray([np.linspace(0, 1, n_samples),
                    np.linspace(0, 1, n_samples)[::-1]]).T
    bins = np.percentile(X, np.linspace(0, 100, n_bins + 1)[1:-1], axis=0).T

    # Accuracte result (1d) check
    arr_actual = _digitize(X, bins[0])
    arr_desired = np.asarray([np.repeat(np.arange(n_bins), 2),
                              np.repeat(np.arange(n_bins), 2)[::-1]]).T
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Accuracte result (2d) check
    arr_actual = _digitize(X, bins)
    arr_desired = np.asarray([np.repeat(np.arange(n_bins), 2),
                              np.repeat(np.arange(n_bins), 2)[::-1]]).T
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_MultipleCoefficientBinning():
    """Test 'MultipleCoefficientBinning' class."""
    X = np.arange(14).reshape(7, 2)
    y = [0, 0, 0, 1, 1, 2, 1]

    # Parameter check
    msg_error = "'n_bins' must be an integer."
    with pytest.raises(TypeError, match=msg_error):
        mcb = MultipleCoefficientBinning(n_bins=None, strategy='quantile')
        mcb.fit_transform(X, y)

    msg_error = re.escape(
        "'n_bins' must be greater than or equal to 2 and lower than "
        "or equal to n_samples (got {0}).".format(1)
    )
    with pytest.raises(ValueError, match=msg_error):
        mcb = MultipleCoefficientBinning(n_bins=1, strategy='quantile')
        mcb.fit_transform(X, y)

    msg_error = re.escape(
        "'strategy' must be either 'uniform', 'quantile', "
        "'normal' or 'entropy' (got {0}).".format('whoops')
    )
    with pytest.raises(ValueError, match=msg_error):
        mcb = MultipleCoefficientBinning(n_bins=2, strategy='whoops')
        mcb.fit_transform(X, y)

    # Consistent lengths check
    msg_error = re.escape(
        "The number of timestamps in X must be the same as "
        "the number of timestamps when `fit` was called "
        "({0} != {1}).".format(2, 1)
    )
    with pytest.raises(ValueError, match=msg_error):
        mcb = MultipleCoefficientBinning(n_bins=3, strategy='quantile')
        mcb.fit(X, y).transform(X[:, :1])

    # Constant feature check
    msg_error = ("At least one timestamp is constant.")
    with pytest.raises(ValueError, match=msg_error):
        mcb = MultipleCoefficientBinning(n_bins=3, strategy='quantile')
        mcb.fit(np.ones((10, 2)))

    # 'quantile' bins check
    msg_error = ("At least two consecutive quantiles are equal. "
                 "You should try with a smaller number of bins or "
                 "remove features with low variation.")
    with pytest.raises(ValueError, match=msg_error):
        mcb = MultipleCoefficientBinning(n_bins=6, strategy='quantile')
        mcb.fit(np.r_[np.zeros((4, 2)), np.ones((4, 2))])

    # 'entropy' bins check
    msg_error = ("The number of bins is too high for feature {0}. "
                 "Try with a smaller number of bins or remove "
                 "this feature.".format(0))
    with pytest.raises(ValueError, match=msg_error):
        mcb = MultipleCoefficientBinning(n_bins=6, strategy='entropy')
        mcb.fit(X, y)

    # Test 1
    mcb = MultipleCoefficientBinning(
        n_bins=3, strategy='uniform', alphabet='ordinal'
    )
    arr_actual = mcb.fit_transform(X)
    arr_desired = [[0, 0],
                   [0, 0],
                   [0, 0],
                   [1, 1],
                   [1, 1],
                   [2, 2],
                   [2, 2]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    mcb = MultipleCoefficientBinning(
        n_bins=3, strategy='quantile', alphabet='ordinal'
    )
    arr_actual = mcb.fit_transform(X)
    arr_desired = [[0, 0],
                   [0, 0],
                   [0, 0],
                   [1, 1],
                   [1, 1],
                   [2, 2],
                   [2, 2]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 3
    mcb = MultipleCoefficientBinning(
        n_bins=3, strategy='normal', alphabet='ordinal'
    )
    arr_actual = mcb.fit_transform(X)
    arr_desired = [[1, 2],
                   [2, 2],
                   [2, 2],
                   [2, 2],
                   [2, 2],
                   [2, 2],
                   [2, 2]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 4
    mcb = MultipleCoefficientBinning(
        n_bins=4, strategy='entropy', alphabet='ordinal'
    )
    arr_actual = mcb.fit_transform(X, y)
    arr_desired = [[0, 0],
                   [0, 0],
                   [0, 0],
                   [1, 1],
                   [1, 1],
                   [2, 2],
                   [3, 3]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 5
    mcb = MultipleCoefficientBinning(
        n_bins=2, strategy='entropy', alphabet='ordinal'
    )
    arr_actual = mcb.fit_transform(X, y)
    arr_desired = [[0, 0],
                   [0, 0],
                   [0, 0],
                   [1, 1],
                   [1, 1],
                   [1, 1],
                   [1, 1]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 5
    mcb = MultipleCoefficientBinning(
        n_bins=2, strategy='entropy', alphabet=None
    )
    arr_actual = mcb.fit_transform(X, y)
    arr_desired = [['a', 'a'],
                   ['a', 'a'],
                   ['a', 'a'],
                   ['b', 'b'],
                   ['b', 'b'],
                   ['b', 'b'],
                   ['b', 'b']]
    np.testing.assert_array_equal(arr_actual, arr_desired)
