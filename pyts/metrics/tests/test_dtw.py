"""Testing for Dynamic Time Warping and its variants."""

import numpy as np
from itertools import product
from numba import njit
from ..dtw import (
    _square, _absolute, cost_matrix, accumulated_cost_matrix, _check_input_dtw,
    _return_path, dtw_classic, dtw_region, sakoe_chiba_band, dtw_sakoechiba,
    itakura_parallelogram, dtw_itakura, _multiscale_region, dtw_multiscale,
    dtw_fast, dtw, show_options
)


def test_square():
    """Test '_square' function."""
    assert _square(1, 2) == 1
    assert _square(1, 3) == 4
    assert _square(5, 2) == 9


def test_absolute():
    """Test '_absolute' function."""
    assert _absolute(1, 2) == 1
    assert _absolute(1, 3) == 2
    assert _absolute(5, 2) == 3


def test_cost_matrix():
    """Test 'cost_matrix' function."""
    x = np.arange(3)
    y = np.arange(3)[::-1]

    # Parameter check
    try:
        cost_matrix(x[:, None], y, dist='square')
    except ValueError as e:
        if str(e) == "'x' must a one-dimensional array.":
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    try:
        cost_matrix(x, y[:, None], dist='square')
    except ValueError as e:
        if str(e) == "'y' must a one-dimensional array.":
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    try:
        cost_matrix(x, y[:2], dist='square')
    except ValueError as e:
        if str(e) == "'x' and 'y' must have the same shape.":
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    try:
        cost_matrix(x, y, dist='sqaure')
    except ValueError as e:
        if str(e) == ("'dist' must be either 'square', 'absolute' or "
                      "callable (got {0}).".format('sqaure')):
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    try:
        @njit()
        def _dist(x):
            return x
        cost_matrix(x, y, dist=_dist)
    except ValueError as e:
        if str(e) == "Calling dist(1, 1) did not work.":
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    try:
        @njit()
        def _dist(x, y):
            return "abc"
        cost_matrix(x, y, dist=_dist)
    except ValueError as e:
        if str(e) == ("Calling dist(1, 1) did not return a float or an "
                      "integer."):
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    try:
        region = np.asarray([[1, 2]])
        cost_matrix(x, y, dist='square', region=region)
    except ValueError as e:
        if str(e) == ("The shape of 'region' must be equal to "
                      "(2, n_timestamps) (got {0})".format((region.shape))):
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    # Test 1
    arr_actual = cost_matrix(x, y, dist='square')
    arr_desired = [[4, 1, 0],
                   [1, 0, 1],
                   [0, 1, 4]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    arr_actual = cost_matrix(x, y, dist='absolute')
    arr_desired = [[2, 1, 0],
                   [1, 0, 1],
                   [0, 1, 2]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 3
    @njit()
    def dist(x, y):
        return (x - y) ** 4
    arr_actual = cost_matrix(x, y, dist=dist)
    arr_desired = [[16, 1, 0],
                   [1, 0, 1],
                   [0, 1, 16]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 4
    arr_actual = cost_matrix(x, y, dist='square',
                             region=[[0, 0, 1], [2, 3, 3]])
    arr_desired = [[4, 1, np.inf],
                   [1, 0, 1],
                   [np.inf, 1, 4]]
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_accumulated_cost_matrix():
    """Test 'accumulated_cost_matrix' function."""
    # Parameter check
    try:
        accumulated_cost_matrix(np.ones((4, 3)))
    except ValueError as e:
        if str(e) == "'cost_mat' must be a square matrix.":
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    try:
        region = np.asarray([[1, 2]])
        accumulated_cost_matrix(np.ones((4, 4)), region=region)
    except ValueError as e:
        if str(e) == ("The shape of 'region' must be equal to "
                      "(2, n_timestamps) (got {0})".format(region.shape)):
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    cost_mat = [[0, 2, 5], [2, 1, 4], [5, 4, 5]]

    # Test 1
    arr_actual = accumulated_cost_matrix(cost_mat, region=None)
    arr_desired = [[0, 2, 7], [2, 1, 5], [7, 5, 6]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    arr_actual = accumulated_cost_matrix(
        cost_mat, region=[[0, 0, 1], [2, 3, 3]]
    )
    arr_desired = [[0, 2, np.inf], [2, 1, 5], [np.inf, 5, 6]]
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_return_path():
    """Test '_return_path' function."""
    # Test 1
    acc_cost_mat = np.asarray([[0, 2, 7], [2, 1, 5], [7, 5, 6]])
    arr_actual = _return_path(acc_cost_mat)
    arr_desired = [[0, 1, 2], [0, 1, 2]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    acc_cost_mat = np.asarray([[0, 0, 0], [2, 3, 1], [7, 5, 2]])
    arr_actual = _return_path(acc_cost_mat)
    arr_desired = [[0, 0, 1, 2], [0, 1, 2, 2]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 3
    acc_cost_mat = np.asarray([[0, 0, np.inf], [2, 3, 1], [np.inf, 5, 2]])
    arr_actual = _return_path(acc_cost_mat)
    arr_desired = [[0, 0, 1, 2], [0, 1, 2, 2]]
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_check_input_dtw():
    """Test '_check_input_dtw' function."""
    x = np.arange(3)
    y = np.arange(3)[::-1]

    try:
        _check_input_dtw(x[:, None], y)
    except ValueError as e:
        if str(e) == "'x' must a one-dimensional array.":
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    try:
        _check_input_dtw(x, y[:, None])
    except ValueError as e:
        if str(e) == "'y' must a one-dimensional array.":
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    try:
        _check_input_dtw(x, y[:2])
    except ValueError as e:
        if str(e) == "'x' and 'y' must have the same shape.":
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))


def test_dtw_classic():
    """Test 'dtw_classic' function."""
    x = np.arange(3)
    y = np.arange(1, 4)

    res = dtw_classic(x, y, dist='square', return_cost=True,
                      return_accumulated=True, return_path=True)
    cost_matrix = [[1, 4, 9],
                   [0, 1, 4],
                   [1, 0, 1]]
    acculumated_cost_matrix = [[1, 5, 14],
                               [1, 2, 6],
                               [2, 1, 2]]
    dtw_score = 2
    path = [[0, 1, 2, 2], [0, 0, 1, 2]]

    assert res[0] == dtw_score
    np.testing.assert_array_equal(res[1], cost_matrix)
    np.testing.assert_array_equal(res[2], acculumated_cost_matrix)
    np.testing.assert_array_equal(res[3], path)


def test_dtw_region():
    """Test 'dtw_region' function."""
    x = np.arange(3)
    y = np.arange(1, 4)

    # Region check
    region = [[0, 1], [0, 1], [0, 1]]
    try:
        dtw_region(x, y, dist='square', region=region)
    except ValueError as e:
        if str(e) == ("If 'region' is not None, it must be array-like "
                      "with shape (2, n_timestamps)."):
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))

    # Test 1
    res = dtw_region(x, y, dist='square', region=None, return_cost=True,
                     return_accumulated=True, return_path=True)
    cost_matrix = [[1, 4, 9],
                   [0, 1, 4],
                   [1, 0, 1]]
    acculumated_cost_matrix = [[1, 5, 14],
                               [1, 2, 6],
                               [2, 1, 2]]
    dtw_score = 2
    path = [[0, 1, 2, 2], [0, 0, 1, 2]]

    assert res[0] == dtw_score
    np.testing.assert_array_equal(res[1], cost_matrix)
    np.testing.assert_array_equal(res[2], acculumated_cost_matrix)
    np.testing.assert_array_equal(res[3], path)

    # Test 2
    region = [[0, 0, 1], [2, 3, 3]]
    res = dtw_region(x, y, dist='square', region=region, return_cost=True,
                     return_accumulated=True, return_path=True)
    cost_matrix = [[1, 4, np.inf],
                   [0, 1, 4],
                   [np.inf, 0, 1]]
    acculumated_cost_matrix = [[1, 5, np.inf],
                               [1, 2, 6],
                               [np.inf, 1, 2]]
    dtw_score = 2
    path = [[0, 1, 2, 2], [0, 0, 1, 2]]

    assert res[0] == dtw_score
    np.testing.assert_array_equal(res[1], cost_matrix)
    np.testing.assert_array_equal(res[2], acculumated_cost_matrix)
    np.testing.assert_array_equal(res[3], path)


def test_sakoe_chiba_band():
    """Test 'sakoe_chiba_band' function."""
    # Parameter check
    n_timestamps_list = [None, 6]
    window_size_list = [None, -1, 2.]
    for (n_timestamps, window_size) in product(
        n_timestamps_list, window_size_list
    ):
        try:
            sakoe_chiba_band(n_timestamps, window_size)
        except ValueError as e:
            if str(e) in [
                "'n_timestamps' must be an integer greater than or "
                "equal to 2.",
                "If 'window_size' is an integer, it must be greater "
                "than or equal to 0 and lower than 'n_timestamps'.",
                "If 'window_size' is a float, it must be between 0 and 1."
            ]:
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) in ["'n_timestamps' must be an intger.",
                          "'window_size' must be an integer or a float."]:
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))

    # Test 1
    n_timestamps, window_size = 4, 2
    arr_actual = sakoe_chiba_band(n_timestamps, window_size)
    arr_desired = [[0, 0, 0, 1], [3, 4, 4, 4]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    n_timestamps, window_size = 4, 0.5
    arr_actual = sakoe_chiba_band(n_timestamps, window_size)
    arr_desired = [[0, 0, 0, 1], [3, 4, 4, 4]]
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_dtw_sakoechiba():
    """Test 'dtw_sakoechiba' function."""
    x = np.arange(4)
    y = np.arange(1, 5)

    res = dtw_sakoechiba(
        x, y, dist='square', window_size=2, return_cost=True,
        return_accumulated=True, return_path=True
    )
    cost_matrix = [[1, 4, 9, np.inf],
                   [0, 1, 4, 9],
                   [1, 0, 1, 4],
                   [np.inf, 1, 0, 1]]
    acculumated_cost_matrix = [[1, 5, 14, np.inf],
                               [1, 2, 6, 15],
                               [2, 1, 2, 6],
                               [np.inf, 2, 1, 2]]
    dtw_score = 2
    path = [[0, 1, 2, 3, 3], [0, 0, 1, 2, 3]]

    assert res[0] == dtw_score
    np.testing.assert_array_equal(res[1], cost_matrix)
    np.testing.assert_array_equal(res[2], acculumated_cost_matrix)
    np.testing.assert_array_equal(res[3], path)


def test_itakura_parallelogram():
    """Test 'itakura_parallelogram' function."""
    # Parameter check
    n_timestamps_list = [None, -1, 4]
    max_slope_list = [None, 0, 2]
    for (n_timestamps, max_slope) in product(
        n_timestamps_list, max_slope_list
    ):
        try:
            itakura_parallelogram(n_timestamps, max_slope)
        except TypeError as e:
            if str(e) in ["'n_timestamps' must be an intger.",
                          "'max_slope' must be an integer or a float."]:
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))
        except ValueError as e:
            if str(e) in ["'n_timestamps' must be an integer greater than "
                          "or equal to 2.",
                          "'max_slope' must be a number greater "
                          "than or equal to 1."]:
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
    # Test
    n_timestamps, max_slope = 4, 2
    arr_actual = itakura_parallelogram(n_timestamps, max_slope)
    arr_desired = [[0, 1, 1, 3], [1, 3, 3, 4]]
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_dtw_itakura():
    """Test 'dtw_itakura' function."""
    x = np.arange(4)
    y = np.arange(1, 5)

    res = dtw_itakura(x, y, dist='square', max_slope=2., return_cost=True,
                      return_accumulated=True, return_path=True)

    cost_matrix = [[1, np.inf, np.inf, np.inf],
                   [np.inf, 1, 4, np.inf],
                   [np.inf, 0, 1, np.inf],
                   [np.inf, np.inf, np.inf, 1]]
    acculumated_cost_matrix = [[1, np.inf, np.inf, np.inf],
                               [np.inf, 2, 6, np.inf],
                               [np.inf, 2, 3, np.inf],
                               [np.inf, np.inf, np.inf, 4]]
    dtw_score = 4
    path = [[0, 1, 2, 3], [0, 1, 2, 3]]

    assert res[0] == dtw_score
    np.testing.assert_array_equal(res[1], cost_matrix)
    np.testing.assert_array_equal(res[2], acculumated_cost_matrix)
    np.testing.assert_array_equal(res[3], path)


def test_multiscale_region():
    """Test '_multiscale_region' function."""
    # Test 1
    path = np.asarray([[0, 0, 1, 2], [0, 1, 2, 2]])
    arr_actual = _multiscale_region(
        n_timestamps=6, resolution_level=2, n_timestamps_reduced=3,
        path=path, radius=0)
    arr_desired = [[0, 0, 4, 4, 4, 4], [4, 4, 6, 6, 6, 6]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    path = np.asarray([[0, 0, 1, 2], [0, 1, 2, 2]])
    arr_actual = _multiscale_region(
        n_timestamps=6, resolution_level=2, n_timestamps_reduced=3,
        path=path, radius=1)
    arr_desired = [[0, 0, 0, 0, 2, 2], [6, 6, 6, 6, 6, 6]]
    np.testing.assert_array_equal(arr_actual, arr_desired)


def test_dtw_multiscale():
    """Test 'dtw_multiscale' function."""
    x = np.arange(4)
    y = np.arange(1, 5)

    # Parameter check
    resolution_list = [None, -1, 2]
    radius_list = ["0", -1, 1]
    for (resolution, radius) in product(resolution_list, radius_list):
        try:
            dtw_multiscale(x, y, resolution=resolution, radius=radius)
        except TypeError as e:
            if str(e) in ["'resolution' must be an integer.",
                          "'radius' must be an integer."]:
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))
        except ValueError as e:
            if str(e) in ["'resolution' must be a positive integer.",
                          "'radius' must be a non-negative integer."]:
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))

    # Test 1
    res = dtw_multiscale(x, y, resolution=2, radius=0, return_cost=True,
                         return_accumulated=True, return_path=True)
    cost_matrix = [[1, 4, np.inf, np.inf],
                   [0, 1, np.inf, np.inf],
                   [np.inf, np.inf, 1, 4],
                   [np.inf, np.inf, 0, 1]]
    acculumated_cost_matrix = [[1, 5, np.inf, np.inf],
                               [1, 2, np.inf, np.inf],
                               [np.inf, np.inf, 3, 7],
                               [np.inf, np.inf, 3, 4]]
    dtw_score = 4
    path = [[0, 1, 2, 3], [0, 1, 2, 3]]

    assert res[0] == dtw_score
    np.testing.assert_array_equal(res[1], cost_matrix)
    np.testing.assert_array_equal(res[2], acculumated_cost_matrix)
    np.testing.assert_array_equal(res[3], path)


def test_dtw_fast():
    """Test 'dtw_fast' function."""
    x = np.arange(4)
    y = np.arange(1, 5)

    res = dtw_fast(x, y, dist='square', radius=0, return_cost=True,
                   return_accumulated=True, return_path=True)
    cost_matrix = [[1, 4, np.inf, np.inf],
                   [0, 1, np.inf, np.inf],
                   [np.inf, np.inf, 1, 4],
                   [np.inf, np.inf, 0, 1]]
    acculumated_cost_matrix = [[1, 5, np.inf, np.inf],
                               [1, 2, np.inf, np.inf],
                               [np.inf, np.inf, 3, 7],
                               [np.inf, np.inf, 3, 4]]
    dtw_score = 4
    path = [[0, 1, 2, 3], [0, 1, 2, 3]]

    assert res[0] == dtw_score
    np.testing.assert_array_equal(res[1], cost_matrix)
    np.testing.assert_array_equal(res[2], acculumated_cost_matrix)
    np.testing.assert_array_equal(res[3], path)


def test_dtw():
    """Test 'dtw' function."""
    x = np.arange(4)
    y = np.arange(1, 5)

    # method='classic' check
    scalar_actual = dtw(x, y, dist='square', method='classic', options=None)
    scalar_desired = dtw_classic(x, y)
    assert scalar_actual == scalar_desired

    # method='sakoechiba' check
    scalar_actual = dtw(x, y, dist='square', method='sakoechiba',
                        options={'window_size': 2})
    scalar_desired = dtw_sakoechiba(x, y, window_size=2)
    assert scalar_actual == scalar_desired

    # method='itakura' check
    scalar_actual = dtw(x, y, dist='square', method='itakura',
                        options={'max_slope': 3.})
    scalar_desired = dtw_itakura(x, y, max_slope=3.)
    assert scalar_actual == scalar_desired

    # method='multiscale' check
    scalar_actual = dtw(x, y, dist='square', method='multiscale',
                        options={'resolution': 2, 'radius': 1})
    scalar_desired = dtw_multiscale(x, y, resolution=2, radius=1)
    assert scalar_actual == scalar_desired

    # method='fast' check
    scalar_actual = dtw(x, y, dist='square', method='fast',
                        options={'radius': 1})
    scalar_desired = dtw_fast(x, y, radius=1)
    assert scalar_actual == scalar_desired

    # Other
    try:
        dtw(x, y, dist='square', method='unexpected', options=None)
    except ValueError as e:
        if str(e) == ("'method' must be either 'classic', 'sakoechiba', "
                      "'itakura', 'multiscale' or 'fast'."):
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))


def test_show_options():
    """Test 'show_options' function."""
    show_options(method=None, disp=True)
    show_options(method=None, disp=False)
    show_options(method='classic', disp=True)
    show_options(method='sakoechiba', disp=True)
    show_options(method='itakura', disp=True)
    show_options(method='multiscale', disp=True)
    show_options(method='fast', disp=True)
    try:
        show_options(method='unexpected', disp=True)
    except ValueError as e:
        if str(e) == ("'method' must be either None, 'classic', "
                      "'sakoechiba', 'itakura', 'multiscale' or 'fast'."):
            pass
        else:
            raise ValueError("Unexpected ValueError: {}".format(e))
