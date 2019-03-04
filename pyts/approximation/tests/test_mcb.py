"""Testing for Multiple Coefficient Binning."""

import numpy as np
from itertools import product
from ..mcb import MultipleCoefficientBinning


def test_MultipleCoefficientBinning():
    """Test 'MultipleCoefficientBinning' class."""
    X = np.arange(14).reshape(7, 2)
    y = [0, 0, 0, 1, 1, 2, 1]

    # Parameter check
    def type_error_list():
        type_error_list_ = ["'n_bins' must be an integer."]
        return type_error_list_

    def value_error_list(n_bins, strategy):
        value_error_list_ = [
            "'n_bins' must be greater than or equal to 2 and lower than "
            "or equal to n_samples (got {0}).".format(n_bins),
            "'strategy' must be either 'uniform', 'quantile', "
            "'normal' or 'entropy' (got {0}).".format(strategy),
        ]
        return value_error_list_

    n_bins_list = [None, 1, 2]
    strategy_list = [None, 'quantile']

    for (n_bins, strategy) in product(n_bins_list, strategy_list):
        mcb = MultipleCoefficientBinning(n_bins, strategy)
        try:
            mcb.fit_transform(X, y)
        except ValueError as e:
            if str(e) in value_error_list(n_bins, strategy):
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) in type_error_list():
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))

    # Consistent lengths check
    mcb = MultipleCoefficientBinning(n_bins=3, strategy='quantile')
    try:
        mcb.fit(X, y).transform(X[:, :5])
    except ValueError as e:
        value_error = (
            "The number of timestamps in X must be the same as "
            "the number of timestamps when `fit` was called "
            "({0} != {1})".format(7, 5)
        )
        if str(e) == value_error:
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    # Constant feature check
    mcb = MultipleCoefficientBinning(n_bins=3, strategy='quantile')
    try:
        mcb.fit(np.ones((10, 2)))
    except ValueError as e:
        if str(e) == "At least one timestamp is constant.":
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    # 'quantile' bins check
    mcb = MultipleCoefficientBinning(n_bins=6, strategy='quantile')
    try:
        mcb.fit(np.r_[np.zeros((4, 2)), np.ones((4, 2))])
    except ValueError as e:
        value_error = (
            "At least two consecutive quantiles are equal. "
            "You should try with a smaller number of bins or "
            "remove features with low variation."
        )
        if str(e) == value_error:
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    # 'entropy' bins check
    mcb = MultipleCoefficientBinning(n_bins=6, strategy='entropy')
    try:
        mcb.fit(X, y)
    except ValueError as e:
        value_error = (
            "The number of bins is too high for feature {0}. "
            "Try with a smaller number of bins or remove "
            "this feature.".format(0)
        )
        if str(e) == value_error:
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    # Test 1
    mcb = MultipleCoefficientBinning(
        n_bins=3, strategy='uniform', alphabet='ordinal'
    )
    arr_actual = mcb.fit_transform(X)
    arr_desired = [[0, 0],
                   [0, 0],
                   [1, 1],
                   [1, 1],
                   [2, 2],
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
                   [1, 1],
                   [1, 1],
                   [2, 2],
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
