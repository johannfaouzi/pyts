"""Testing for utility tools."""

import numpy as np
from itertools import product
from ..utils import segmentation, windowed_view


def test_segmentation():
    """Test 'segmentation' function."""
    # Parameters check
    def type_error_list():
        type_error_list_ = ["'ts_size' must be an integer.",
                            "'window_size' must be an integer.",
                            "'n_segments' must be None or an integer."]
        return type_error_list_

    def value_error_list(ts_size, window_size, n_segments):
        value_error_list_ = [
            "'ts_size' must be an integer greater than or equal "
            "to 2 (got {0}).".format(ts_size),
            "'window_size' must be an integer greater than or "
            "equal to 1 (got {0}).".format(window_size),
            "'window_size' must be lower than or equal to "
            "'ts_size' ({0} > {1}).".format(window_size, ts_size),
            "If 'n_segments' is an integer, it must be lower than or "
            "equal to 'ts_size' ({0} > {1}).".format(n_segments, ts_size)
        ]
        return value_error_list_

    ts_size_list = [-1, 2, 3, None]
    window_size_list = [-1, 1, 2, 4, None]
    overlapping_list = [True, False]
    n_segments_list = [-1, 1, 2, 4, None]

    for (ts_size, window_size, overlapping, n_segments) in product(
        ts_size_list, window_size_list, overlapping_list, n_segments_list
    ):
        try:
            segmentation(ts_size, window_size, overlapping, n_segments)
        except ValueError as e:
            if str(e) in value_error_list(ts_size, window_size, n_segments):
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) in type_error_list():
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))

    # Test 1
    bounds = np.array([0, 4, 8, 12, 16, 20])
    window_size = 4
    overlapping = False
    res_actual = segmentation(20, window_size, overlapping)
    res_start = bounds[:-1]
    res_end = bounds[1:]
    res_size = 5
    np.testing.assert_array_equal(res_actual[0], res_start)
    np.testing.assert_array_equal(res_actual[1], res_end)
    np.testing.assert_equal(res_actual[2], res_size)


def test_windowed_view():
    """Test 'windowed_view' function."""
    X = np.arange(10).reshape(2, 5)

    # Parameters check
    def type_error_list():
        type_error_list_ = ["'window_size' must be an integer.",
                            "'window_step' must be an integer."]
        return type_error_list_

    def value_error_list():
        value_error_list_ = [
            "'window_size' must be an integer between 1 and n_timestamps.",
            "'window_step' must be an integer between 1 and n_timestamps."
        ]
        return value_error_list_

    window_size_list = [-1, 1, 2, 4, None]
    window_step_list = [-1, 1, 2, 4, None]

    for (window_size, window_step) in product(
        window_size_list, window_step_list
    ):
        try:
            windowed_view(X, window_size, window_step)
        except ValueError as e:
            if str(e) in value_error_list():
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) in type_error_list():
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))

    # Test 1
    arr_actual = windowed_view(X, window_size=3, window_step=1)
    arr_desired = [[[0, 1, 2],
                    [1, 2, 3],
                    [2, 3, 4]],
                   [[5, 6, 7],
                    [6, 7, 8],
                    [7, 8, 9]]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    arr_actual = windowed_view(X, window_size=3, window_step=2)
    arr_desired = [[[0, 1, 2],
                    [2, 3, 4]],
                   [[5, 6, 7],
                    [7, 8, 9]]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 3
    arr_actual = windowed_view(X, window_size=2, window_step=3)
    arr_desired = [[[0, 1],
                    [3, 4]],
                   [[5, 6],
                    [8, 9]]]
    np.testing.assert_array_equal(arr_actual, arr_desired)
