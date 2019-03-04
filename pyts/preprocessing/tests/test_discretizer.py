"""Testing for discretizers."""

import numpy as np
from itertools import product
from warnings import catch_warnings
from ..discretizer import _uniform_bins, _digitize, KBinsDiscretizer


def test_uniform_bins():
    """Test '_uniform_bins' function."""
    sample_min = np.arange(5)
    sample_max = np.arange(2, 7)
    arr_actual = _uniform_bins(sample_min, sample_max, n_samples=5, n_bins=2)
    arr_desired = ((sample_min + sample_max) / 2).reshape(1, 5)
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_digitize():
    """Test '_digitize' function."""
    n_timestamps = 10
    n_bins = 5
    X = np.asarray([np.linspace(0, 1, n_timestamps),
                    np.linspace(0, 1, n_timestamps)[::-1]])
    bins = np.percentile(X, np.linspace(0, 100, n_bins + 1)[1:-1], axis=1).T

    # Accuracte result (1d) check
    arr_actual = _digitize(X, bins[0])
    arr_desired = np.asarray([np.repeat(np.arange(n_bins), 2),
                              np.repeat(np.arange(n_bins), 2)[::-1]])
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Accuracte result (2d) check
    arr_actual = _digitize(X, bins)
    arr_desired = np.asarray([np.repeat(np.arange(n_bins), 2),
                              np.repeat(np.arange(n_bins), 2)[::-1]])
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_KBinsDiscretizer():
    """Test 'KBinsDiscretizer' class."""
    # Parameter check
    def value_error_list(n_bins, strategy):
        value_error_list_ = [
            "'n_bins' must be greater than or equal to 2 and lower than "
            "or equal to n_timestamps (got {0}).".format(n_bins),
            "'strategy' must be either 'uniform', 'quantile' "
            "or 'normal' (got {0}).".format(strategy)
        ]
        return value_error_list_

    def type_error_list():
        type_error_list_ = ["'n_bins' must be an integer."]
        return type_error_list_

    X = np.arange(15).reshape(3, 5)
    n_bins_list = ['0', 0, 2]
    strategy_list = [0, 'uniform']
    for (n_bins, strategy) in product(n_bins_list, strategy_list):
        discretizer = KBinsDiscretizer(n_bins=n_bins, strategy=strategy)
        try:
            discretizer.fit_transform(X)
        except ValueError as e:
            if str(e) not in value_error_list(n_bins, strategy):
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) not in type_error_list():
                raise TypeError("Unexpected TypeError: {}".format(e))

    # Constant check
    X = np.ones((10, 15))
    discretizer = KBinsDiscretizer(n_bins=3, strategy='uniform')
    try:
        discretizer._check_constant(X)
    except ValueError as e:
        if str(e) != "At least one sample is constant.":
            raise ValueError("Unexpected ValueError: {}".format(e))

    # Compute bins check
    X = [0] * 5 + 1 * [2] + [2] * 5
    X = np.asarray(X).reshape(1, -1)
    discretizer = KBinsDiscretizer()
    with catch_warnings(record=True) as w:
        discretizer._compute_bins(X, n_samples=1,
                                  n_bins=5, strategy='quantile')
        warning = (
            "Some quantiles are equal. The number of bins will be "
            "smaller for sample {0}. Consider decreasing the number "
            "of bins or removing these samples.".format([0])
        )
        assert str(w[-1].message) == warning

    # Test 1
    X = np.arange(15).reshape(3, 5)
    arr_actual = KBinsDiscretizer(
        n_bins=5, strategy='quantile').fit_transform(X)
    arr_desired = X - X.min(axis=1)[:, None]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    X = np.arange(15).reshape(3, 5)
    arr_actual = KBinsDiscretizer(
        n_bins=5, strategy='uniform').fit_transform(X)
    arr_desired = X - X.min(axis=1)[:, None]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 3
    X = [[-0.5, 0, 0.5]]
    arr_actual = KBinsDiscretizer(
        n_bins=3, strategy='normal').fit_transform(X)
    arr_desired = [[0, 1, 2]]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
