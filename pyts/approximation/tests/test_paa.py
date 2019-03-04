"""Testing for Piecewise Aggregate Approximation."""

from itertools import product
import numpy as np
from ..paa import _paa, PiecewiseAggregateApproximation


def test_paa():
    """Test '_paa' function."""
    # Test 1
    n_samples, n_timestamps, n_timestamps_new = 3, 4, 2
    start, end = [0, 2], [2, 4]
    X = np.asarray([[0, 1, 2, 3],
                    [1, 3, 0, 2],
                    [4, 8, 2, 6]])
    arr_actual = _paa(X, n_samples, n_timestamps, start, end, n_timestamps_new)
    arr_desired = np.asarray([[0.5, 2.5],
                              [2.0, 1.0],
                              [6.0, 4.0]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    n_samples, n_timestamps, n_timestamps_new = 3, 4, 3
    start, end = [0, 1, 2], [2, 3, 4]
    X = np.asarray([[0, 1, 2, 3],
                    [1, 3, 0, 2],
                    [4, 8, 2, 6]])
    arr_actual = _paa(X, n_samples, n_timestamps, start, end, n_timestamps_new)
    arr_desired = np.asarray([[0.5, 1.5, 2.5],
                              [2.0, 1.5, 1.0],
                              [6.0, 5.0, 4.0]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_PiecewiseAggregateApproximation():
    """Test 'PiecewiseAggregateApproximation' class."""
    X = np.arange(9).reshape(1, 9)

    # Parameter check
    def type_error_list():
        type_error_list_ = [
            "'window_size' and 'output_size' cannot be both None.",
            "If specified, 'window_size' must be an integer "
            "or a float.",
            "If specified, 'output_size' must be an integer "
            "or a float."
        ]
        return type_error_list_

    def value_error_list(window_size, output_size):
        value_error_list_ = [
            "If 'window_size' is an integer, it must be greater "
            "than or equal to 1 and lower than or equal to the size "
            "of each time series (i.e. the size of the last dimension "
            "of X) (got {0}).".format(window_size),
            "If 'window_size' is a float, it must be greater "
            "than 0 and lower than or equal to 1 "
            "(got {0}).".format(window_size),
            "If 'output_size' is an integer, it must be greater "
            "than or equal to 1 and lower than or equal to the size "
            "of each time series (i.e. the size of the last dimension "
            "of X) (got {0}).".format(output_size),
            "If 'output_size' is a float, it must be greater "
            "than 0 and lower than or equal to 1 "
            "(got {0}).".format(output_size)
        ]
        return value_error_list_

    window_size_list = [1., 2., -1, 2, 3, None]
    output_size_list = [1., 2., -1, 2, None]
    overlapping_list = [True, False]

    for (window_size, output_size, overlapping) in product(
        window_size_list, output_size_list, overlapping_list
    ):
        paa = PiecewiseAggregateApproximation(
            window_size, output_size, overlapping)
        try:
            paa.fit_transform(X)
        except ValueError as e:
            if str(e) in value_error_list(window_size, output_size):
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) in type_error_list():
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))

    # Test 1 (window_size = 1 check)
    paa = PiecewiseAggregateApproximation(
        window_size=1, output_size=None, overlapping=False)
    arr_actual = paa.fit_transform(X)
    arr_desired = X
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2 (window_size > 1 check)
    paa = PiecewiseAggregateApproximation(
        window_size=3, output_size=None, overlapping=False)
    arr_actual = paa.fit_transform(X)
    arr_desired = np.array([1, 4, 7]).reshape(1, 3)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 3 (output_size specified check)
    X = np.arange(9).reshape(1, 9)
    paa = PiecewiseAggregateApproximation(
        window_size=None, output_size=3, overlapping=False)
    arr_actual = paa.fit_transform(X)
    print(arr_actual)
    arr_desired = np.array([1, 4, 7]).reshape(1, 3)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
