"""Testing for utility tools."""

import numpy as np
import pytest
import re
from ..utils import segmentation, windowed_view


def test_segmentation():
    """Test 'segmentation' function."""
    # Parameters check
    msg_error = "'ts_size' must be an integer."
    with pytest.raises(TypeError, match=msg_error):
        segmentation(ts_size=None, window_size=2,
                     overlapping=False, n_segments=None)

    msg_error = "'window_size' must be an integer."
    with pytest.raises(TypeError, match=msg_error):
        segmentation(ts_size=10, window_size={},
                     overlapping=False, n_segments=None)

    msg_error = "'n_segments' must be None or an integer."
    with pytest.raises(TypeError, match=msg_error):
        segmentation(ts_size=10, window_size=2,
                     overlapping=False, n_segments="3")

    msg_error = re.escape(
        "'ts_size' must be an integer greater than or equal "
        "to 2 (got {0}).".format(1)
    )
    with pytest.raises(ValueError, match=msg_error):
        segmentation(ts_size=1, window_size=2,
                     overlapping=False, n_segments=None)

    msg_error = re.escape(
        "'window_size' must be an integer greater than or "
        "equal to 1 (got {0}).".format(0)
    )
    with pytest.raises(ValueError, match=msg_error):
        segmentation(ts_size=10, window_size=0,
                     overlapping=False, n_segments=None)

    msg_error = re.escape(
        "'window_size' must be lower than or equal to "
        "'ts_size' ({0} > {1}).".format(15, 10)
    )
    with pytest.raises(ValueError, match=msg_error):
        segmentation(ts_size=10, window_size=15,
                     overlapping=False, n_segments=None)

    msg_error = re.escape(
        "If 'n_segments' is an integer, it must be greater than or "
        "equal to 2 and lower than or equal to 'ts_size' "
        "({0} > {1}).".format(12, 10)
    )
    with pytest.raises(ValueError, match=msg_error):
        segmentation(ts_size=10, window_size=3,
                     overlapping=False, n_segments=12)

    # Test 1
    bounds = np.array([0, 4, 8, 12, 16, 20])
    window_size = 4
    overlapping = False
    res_actual = segmentation(20, window_size, overlapping)
    start_desired = bounds[:-1]
    end_desired = bounds[1:]
    size_desired = 5
    np.testing.assert_array_equal(res_actual[0], start_desired)
    np.testing.assert_array_equal(res_actual[1], end_desired)
    np.testing.assert_equal(res_actual[2], size_desired)

    # Test 2
    window_size = 8
    overlapping = True
    res_actual = segmentation(20, window_size, overlapping)
    start_desired = [0, 6, 12]
    end_desired = [8, 14, 20]
    size_desired = 3
    np.testing.assert_array_equal(res_actual[0], start_desired)
    np.testing.assert_array_equal(res_actual[1], end_desired)
    np.testing.assert_equal(res_actual[2], size_desired)


def test_windowed_view():
    """Test 'windowed_view' function."""
    X = np.arange(10).reshape(2, 5)

    # Parameters check
    msg_error = "'window_size' must be an integer."
    with pytest.raises(TypeError, match=msg_error):
        windowed_view(X, window_size="3", window_step=1)

    msg_error = "'window_step' must be an integer."
    with pytest.raises(TypeError, match=msg_error):
        windowed_view(X, window_size=2, window_step=None)

    msg_error = "'window_size' must be an integer between 1 and n_timestamps."
    with pytest.raises(ValueError, match=msg_error):
        windowed_view(X, window_size=7, window_step=1)

    msg_error = "'window_step' must be an integer between 1 and n_timestamps."
    with pytest.raises(ValueError, match=msg_error):
        windowed_view(X, window_size=2, window_step=0)

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
