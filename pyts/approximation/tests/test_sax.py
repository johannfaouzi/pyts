"""Testing for Symbolic Aggregate Approximation."""

from itertools import product
import numpy as np
from ..sax import SymbolicAggregateApproximation


def test_SymbolicAggregateApproximation():
    """Test 'SymbolicAggregateApproximation' class."""
    # Parameter check
    def value_error_list(n_bins, strategy):
        value_error_list_ = [
            "'n_bins' must be greater than or equal to 2 and lower than "
            "or equal to n_timestamps (got {0}).".format(n_bins),
            "'n_bins' is unexpectedly high. You should try with a smaller "
            "value.",
            "'strategy' must be either 'uniform', 'quantile' "
            "or 'normal' (got {0})".format(strategy),
            "'alphabet' must be None or array-like with shape (n_bins, ).",
            "If 'alphabet' is array-like, its shape must be equal to "
            "(n_bins, )."
        ]
        return value_error_list_

    def type_error_list(alphabet):
        type_error_list_ = [
            "'n_bins' must be an integer.",
            "'alphabet' must be None, 'ordinal' or array-like "
            "with shape (n_bins,) (got {0})".format(alphabet)
        ]
        return type_error_list_

    X = np.arange(15).reshape(3, 5)
    n_bins_list = ['0', 0, 2]
    strategy_list = [0, 'uniform']
    alphabet_list = [None, ['a'], ['a', 'b'], 'unexpected']
    for (n_bins, strategy, alphabet) in product(
        n_bins_list, strategy_list, alphabet_list
    ):
        sax = SymbolicAggregateApproximation(
            n_bins=n_bins, strategy=strategy, alphabet=alphabet)
        try:
            sax.fit_transform(X)
        except ValueError as e:
            if str(e) not in value_error_list(n_bins, strategy):
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) not in type_error_list(alphabet):
                raise TypeError("Unexpected TypeError: {}".format(e))

    # Test 1
    X = [[0, 1, 2, 3, 4]]
    arr_actual = SymbolicAggregateApproximation(
        n_bins=5, strategy='quantile').fit_transform(X)
    arr_desired = [['a', 'b', 'c', 'd', 'e']]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    X = [[0, 1, 2, 3, 4]]
    arr_actual = SymbolicAggregateApproximation(
        n_bins=5, strategy='quantile', alphabet=['e', 'd', 'c', 'b', 'a']
    ).fit_transform(X)
    arr_desired = [['e', 'd', 'c', 'b', 'a']]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 3
    X = [[0, 1, 2, 3, 4, 5]]
    arr_actual = SymbolicAggregateApproximation(
        n_bins=3, strategy='quantile').fit_transform(X)
    arr_desired = [['a', 'a', 'b', 'b', 'c', 'c']]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 4
    X = [[0, 1, 2, 3, 4, 5]]
    arr_actual = SymbolicAggregateApproximation(
        n_bins=3, strategy='quantile', alphabet=('a', 'b', 'c')
    ).fit_transform(X)
    arr_desired = [['a', 'a', 'b', 'b', 'c', 'c']]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 5
    X = [[0, 1, 2, 3, 4, 5]]
    arr_actual = SymbolicAggregateApproximation(
        n_bins=3, strategy='quantile', alphabet=['0', '1', '2']
    ).fit_transform(X)
    arr_desired = [['0', '0', '1', '1', '2', '2']]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 6
    X = [[0, 1, 2, 3, 4, 5]]
    arr_actual = SymbolicAggregateApproximation(
        n_bins=3, strategy='quantile', alphabet='ordinal'
    ).fit_transform(X)
    arr_desired = [[0, 0, 1, 1, 2, 2]]
    np.testing.assert_array_equal(arr_actual, arr_desired)
