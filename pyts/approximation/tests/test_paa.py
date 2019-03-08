"""Testing for Piecewise Aggregate Approximation."""

import numpy as np
import pytest
import re
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
    msg_error = "'window_size' and 'output_size' cannot be both None."
    with pytest.raises(TypeError, match=msg_error):
        paa = PiecewiseAggregateApproximation(
            window_size=None, output_size=None, overlapping=False
        )
        paa.fit_transform(X)

    msg_error = "If specified, 'window_size' must be an integer or a float."
    with pytest.raises(TypeError, match=msg_error):
        paa = PiecewiseAggregateApproximation(
            window_size="3", output_size=None, overlapping=False
        )
        paa.fit_transform(X)

    msg_error = "If specified, 'output_size' must be an integer or a float."
    with pytest.raises(TypeError, match=msg_error):
        paa = PiecewiseAggregateApproximation(
            window_size=None, output_size="3", overlapping=False
        )
        paa.fit_transform(X)

    msg_error = re.escape(
        "If 'window_size' is an integer, it must be greater "
        "than or equal to 1 and lower than or equal to the size "
        "of each time series (i.e. the size of the last dimension "
        "of X) (got {0}).".format(0)
    )
    with pytest.raises(ValueError, match=msg_error):
        paa = PiecewiseAggregateApproximation(
            window_size=0, output_size=None, overlapping=False
        )
        paa.fit_transform(X)

    msg_error = re.escape("If 'window_size' is a float, it must be greater "
                          "than 0 and lower than or equal to 1 "
                          "(got {0}).".format(2.))
    with pytest.raises(ValueError, match=msg_error):
        paa = PiecewiseAggregateApproximation(
            window_size=2., output_size=None, overlapping=False
        )
        paa.fit_transform(X)

    msg_error = re.escape(
        "If 'output_size' is an integer, it must be greater "
        "than or equal to 1 and lower than or equal to the size "
        "of each time series (i.e. the size of the last dimension "
        "of X) (got {0}).".format(0)
    )
    with pytest.raises(ValueError, match=msg_error):
        paa = PiecewiseAggregateApproximation(
            window_size=None, output_size=0, overlapping=False
        )
        paa.fit_transform(X)

    msg_error = re.escape("If 'output_size' is a float, it must be greater "
                          "than 0 and lower than or equal to 1 "
                          "(got {0}).".format(2.))
    with pytest.raises(ValueError, match=msg_error):
        paa = PiecewiseAggregateApproximation(
            window_size=None, output_size=2., overlapping=False
        )
        paa.fit_transform(X)

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
