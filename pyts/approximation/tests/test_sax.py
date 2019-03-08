"""Testing for Symbolic Aggregate Approximation."""

import numpy as np
import pytest
import re
from ..sax import SymbolicAggregateApproximation


def test_SymbolicAggregateApproximation():
    """Test 'SymbolicAggregateApproximation' class."""
    # Parameter check
    X = np.arange(15).reshape(3, 5)

    msg_error = "'n_bins' must be an integer."
    with pytest.raises(TypeError, match=msg_error):
        sax = SymbolicAggregateApproximation(
            n_bins=None, strategy='quantile', alphabet=None
        )
        sax.fit_transform(X)

    msg_error = re.escape("'alphabet' must be None, 'ordinal' or array-like "
                          "with shape (n_bins,) (got {0})".format('whoops'))
    with pytest.raises(TypeError, match=msg_error):
        sax = SymbolicAggregateApproximation(
            n_bins=2, strategy='quantile', alphabet='whoops'
        )
        sax.fit_transform(X)

    msg_error = re.escape(
        "'n_bins' must be greater than or equal to 2 and lower than "
        "or equal to n_timestamps (got {0}).".format(15)
    )
    with pytest.raises(ValueError, match=msg_error):
        sax = SymbolicAggregateApproximation(
            n_bins=15, strategy='quantile', alphabet=None
        )
        sax.fit_transform(X)

    msg_error = ("'n_bins' is unexpectedly high. You should try with a "
                 "smaller value.")
    with pytest.raises(ValueError, match=msg_error):
        sax = SymbolicAggregateApproximation(
            n_bins=10000000, strategy='quantile', alphabet=None
        )
        sax.fit_transform(np.c_[np.zeros((2, 10000000)),
                                np.ones((2, 10000000))])

    msg_error = re.escape("'strategy' must be either 'uniform', 'quantile' "
                          "or 'normal' (got {0})".format('whoops'))
    with pytest.raises(ValueError, match=msg_error):
        sax = SymbolicAggregateApproximation(
            n_bins=3, strategy='whoops', alphabet=None
        )
        sax.fit_transform(X)

    msg_error = re.escape(
        "If 'alphabet' is array-like, its shape must be equal to (n_bins,)."
    )
    with pytest.raises(ValueError, match=msg_error):
        sax = SymbolicAggregateApproximation(
            n_bins=3, strategy='quantile', alphabet=[0, 1]
        )
        sax.fit_transform(X)

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
