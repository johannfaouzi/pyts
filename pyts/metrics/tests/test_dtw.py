"""Testing for Dynamic Time Warping and its variants."""

import numpy as np
import pytest
import re
from numba import njit
from ..dtw import (
    _square, _absolute, _check_input_dtw, _multiscale_region, _return_path,
    _return_results, cost_matrix, accumulated_cost_matrix, dtw_classic,
    dtw_region, sakoe_chiba_band, dtw_sakoechiba, itakura_parallelogram,
    dtw_itakura, dtw_multiscale, dtw_fast, dtw, show_options
)


x = np.array([0, 1, 2])
y = np.array([2, 0, 1])

params_return = {'return_cost': True,
                 'return_accumulated': True,
                 'return_path': True}


@njit
def power_four(x, y):
    """Power four function with a njit decorator."""
    return (x - y) ** 4


@pytest.mark.parametrize(
    'x, y, scalar_desired',
    [(1, 2, 1),
     (1, 3, 4),
     (3, 1, 4),
     (5, 2, 9)]
)
def test_square(x, y, scalar_desired):
    """Test that the actual results are the expected ones."""
    assert _square(x, y) == scalar_desired


@pytest.mark.parametrize(
    'x, y, scalar_desired',
    [(1, 2, 1),
     (1, 3, 2),
     (3, 1, 2),
     (5, 2, 3)]
)
def test_absolute(x, y, scalar_desired):
    """Test that the actual results are the expected ones."""
    assert _absolute(x, y) == scalar_desired


@pytest.mark.parametrize(
    'params, err_msg',
    [({'x': x[:, None], 'y': y}, "'x' must be a one-dimensional array."),
     ({'x': x, 'y': y[:, None]}, "'y' must be a one-dimensional array."),
     ({'x': x, 'y': y[:2]}, "'x' and 'y' must have the same shape.")]
)
def test_check_input_dtw(params, err_msg):
    """Test parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        _check_input_dtw(**params)


@pytest.mark.parametrize(
    'params, err_msg',
    [({'dist': 'who'},
      "'dist' must be either 'square', 'absolute' or callable (got who)."),

     ({'dist': lambda x: x}, "Calling dist(1, 1) did not work."),

     ({'dist': lambda x, y: str(x) + str(y)},
      "Calling dist(1, 1) did not return a float or an integer."),

     ({'region': [[1, 1]]},
      "The shape of 'region' must be equal to (2, n_timestamps) "
      "(got (1, 2)).")]
)
def test_parameter_check_cost_matrix(params, err_msg):
    """Test parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        cost_matrix(x, y, **params)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({}, [[4, 0, 1], [1, 1, 0], [0, 4, 1]]),

     ({'dist': 'absolute'}, [[2, 0, 1], [1, 1, 0], [0, 2, 1]]),

     ({'dist': power_four}, [[16, 0, 1], [1, 1, 0], [0, 16, 1]]),

     ({'region': [[0, 0, 1], [2, 3, 3]]},
      [[4, 0, np.inf], [1, 1, 0], [np.inf, 4, 1]])]
)
def test_actual_results_cost_matrix(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = cost_matrix(x, y, **params)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, err_msg',
    [({'cost_mat': np.ones((4, 3))}, "'cost_mat' must be a square matrix."),

     ({'cost_mat': np.ones((4, 4)), 'region': [[1, 1]]},
      "The shape of 'region' must be equal to (2, n_timestamps) "
      "(got {0})".format((1, 2)))]
)
def test_parameter_check_accumulated_cost_matrix(params, err_msg):
    """Test parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        accumulated_cost_matrix(**params)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'cost_mat': [[0, 2, 5], [2, 1, 4], [5, 4, 5]]},
      [[0, 2, 7], [2, 1, 5], [7, 5, 6]]),

     ({'cost_mat': [[0, 2, 5], [2, 1, 4], [5, 4, 5]],
       'region': [[0, 0, 1], [2, 3, 3]]},
      [[0, 2, np.inf], [2, 1, 5], [np.inf, 5, 6]])]
)
def test_actual_results_accumulated_cost_matrix(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = accumulated_cost_matrix(**params)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'acc_cost_mat, arr_desired',
    [(np.asarray([[0, 2, 7], [2, 1, 5], [7, 5, 6]]), [[0, 1, 2], [0, 1, 2]]),

     (np.asarray([[0, 0, 0], [2, 3, 1], [7, 5, 2]]),
      [[0, 0, 1, 2], [0, 1, 2, 2]]),

     (np.asarray([[0, 0, np.inf], [2, 3, 1], [np.inf, 5, 2]]),
      [[0, 0, 1, 2], [0, 1, 2, 2]])]
)
def test_actual_results_return_path(acc_cost_mat, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = _return_path(acc_cost_mat)
    np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'params, res_desired',
    [({}, 2),

     ({'return_cost': True}, (2, [[2]])),

     ({'return_accumulated': True}, (2, [[3]])),

     ({'return_path': True}, (2, [[0], [0]])),

     ({'return_cost': True, 'return_accumulated': True},
      (2, [[2]], np.asarray([[3]]))),

     ({'return_cost': True, 'return_path': True}, (2, [[2]], [[0], [0]])),

     ({'return_accumulated': True, 'return_path': True},
      (2, np.asarray([[3]]), [[0], [0]])),

     ({'return_cost': True, 'return_accumulated': True,
       'return_path': True}, (2, [[2]], np.asarray([[3]]), [[0], [0]]))]
)
def test_actual_results_return_results(params, res_desired):
    """Test that the actual results are the expected ones."""
    res_actual = _return_results(dtw_dist=2, cost_mat=[[2]],
                                 acc_cost_mat=np.asarray([[3]]), **params)
    if isinstance(res_desired, tuple):
        for actual, desired in zip(res_actual, res_desired):
            np.testing.assert_allclose(actual, desired)
    else:
        np.testing.assert_allclose(res_actual, res_desired)


@pytest.mark.parametrize(
    'params, res_desired',
    [({},
      {'cost_mat': [[4, 0, 1], [1, 1, 0], [0, 4, 1]],
       'acc_cost_mat': [[4, 4, 5], [5, 5, 4], [5, 9, 5]],
       'path': [[0, 0, 1, 2], [0, 1, 2, 2]],
       'dtw': 5}),

     ({'dist': 'absolute'},
      {'cost_mat': [[2, 0, 1], [1, 1, 0], [0, 2, 1]],
       'acc_cost_mat': [[2, 2, 3], [3, 3, 2], [3, 5, 3]],
       'path': [[0, 0, 1, 2], [0, 1, 2, 2]],
       'dtw': 3})]
)
def test_actual_results_dtw_classic(params, res_desired):
    """Test that the actual results are the expected ones."""
    (dtw_actual, cost_mat_actual,
     acc_cost_mat_actual, path_actual) = dtw_classic(
        x, y, **params_return, **params
    )
    np.testing.assert_allclose(cost_mat_actual, res_desired['cost_mat'])
    np.testing.assert_allclose(dtw_actual, res_desired['dtw'])
    np.testing.assert_allclose(path_actual, res_desired['path'])
    np.testing.assert_allclose(acc_cost_mat_actual,
                               res_desired['acc_cost_mat'])


@pytest.mark.parametrize(
    'params, err_msg',
    [({'region': [[1, 1]]},
      "If 'region' is not None, it must be array-like with shape "
      "(2, n_timestamps).")]
)
def test_parameter_check_dtw_region(params, err_msg):
    """Test parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        dtw_region(x, y, **params)


@pytest.mark.parametrize(
    'params, res_desired',
    [({},
      {'cost_mat': [[4, 0, 1], [1, 1, 0], [0, 4, 1]],
       'acc_cost_mat': [[4, 4, 5], [5, 5, 4], [5, 9, 5]],
       'path': [[0, 0, 1, 2], [0, 1, 2, 2]],
       'dtw': 5}),

     ({'dist': 'absolute'},
      {'cost_mat': [[2, 0, 1], [1, 1, 0], [0, 2, 1]],
       'acc_cost_mat': [[2, 2, 3], [3, 3, 2], [3, 5, 3]],
       'path': [[0, 0, 1, 2], [0, 1, 2, 2]],
       'dtw': 3}),

     ({'region': [[0, 0, 1], [2, 3, 3]]},
      {'cost_mat': [[4, 0, np.inf], [1, 1, 0], [np.inf, 4, 1]],
       'acc_cost_mat': [[4, 4, np.inf], [5, 5, 4], [np.inf, 9, 5]],
       'path': [[0, 0, 1, 2], [0, 1, 2, 2]],
       'dtw': 5}),

     ({'dist': 'absolute', 'region': [[0, 0, 1], [2, 3, 3]]},
      {'cost_mat': [[2, 0, np.inf], [1, 1, 0], [np.inf, 2, 1]],
       'acc_cost_mat': [[2, 2, np.inf], [3, 3, 2], [np.inf, 5, 3]],
       'path': [[0, 0, 1, 2], [0, 1, 2, 2]],
       'dtw': 3})]
)
def test_actual_results_dtw_region(params, res_desired):
    """Test that the actual results are the expected ones."""
    (dtw_actual, cost_mat_actual,
     acc_cost_mat_actual, path_actual) = dtw_region(
        x, y, **params_return, **params
    )
    np.testing.assert_allclose(cost_mat_actual, res_desired['cost_mat'])
    np.testing.assert_allclose(dtw_actual, res_desired['dtw'])
    np.testing.assert_allclose(path_actual, res_desired['path'])
    np.testing.assert_allclose(acc_cost_mat_actual,
                               res_desired['acc_cost_mat'])


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_timestamps': '3'}, TypeError,
      "'n_timestamps' must be an intger."),

     ({'n_timestamps': 3, 'window_size': '0.5'}, TypeError,
      "'window_size' must be an integer or a float."),

     ({'n_timestamps': 1}, ValueError,
      "'n_timestamps' must be an integer greater than or equal to 2."),

     ({'n_timestamps': 10, 'window_size': 20}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 0 "
      "and lower than 'n_timestamps'."),

     ({'n_timestamps': 10, 'window_size': 2.}, ValueError,
      "If 'window_size' is a float, it must be between 0 and 1.")]
)
def test_parameter_check_sakoe_chiba_band(params, error, err_msg):
    """Test parameter validation."""
    with pytest.raises(error, match=re.escape(err_msg)):
        sakoe_chiba_band(**params)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'n_timestamps': 4, 'window_size': 2}, [[0, 0, 0, 1], [3, 4, 4, 4]]),
     ({'n_timestamps': 4, 'window_size': 0.5}, [[0, 0, 0, 1], [3, 4, 4, 4]]),
     ({'n_timestamps': 4, 'window_size': 3}, [[0, 0, 0, 0], [4, 4, 4, 4]]),
     ({'n_timestamps': 4, 'window_size': 1}, [[0, 0, 1, 2], [2, 3, 4, 4]]),
     ({'n_timestamps': 4, 'window_size': 0}, [[0, 1, 2, 3], [1, 2, 3, 4]])]
)
def test_actual_results_sakoe_chiba_band(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = sakoe_chiba_band(**params)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, res_desired',
    [({'window_size': 2},
      {'cost_mat': [[4, 0, 1], [1, 1, 0], [0, 4, 1]],
       'acc_cost_mat': [[4, 4, 5], [5, 5, 4], [5, 9, 5]],
       'path': [[0, 0, 1, 2], [0, 1, 2, 2]],
       'dtw': 5}),

     ({'window_size': 2, 'dist': 'absolute'},
      {'cost_mat': [[2, 0, 1], [1, 1, 0], [0, 2, 1]],
       'acc_cost_mat': [[2, 2, 3], [3, 3, 2], [3, 5, 3]],
       'path': [[0, 0, 1, 2], [0, 1, 2, 2]],
       'dtw': 3}),

     ({'window_size': 1},
      {'cost_mat': [[4, 0, np.inf], [1, 1, 0], [np.inf, 4, 1]],
       'acc_cost_mat': [[4, 4, np.inf], [5, 5, 4], [np.inf, 9, 5]],
       'path': [[0, 0, 1, 2], [0, 1, 2, 2]],
       'dtw': 5}),

     ({'window_size': 1, 'dist': 'absolute'},
      {'cost_mat': [[2, 0, np.inf], [1, 1, 0], [np.inf, 2, 1]],
       'acc_cost_mat': [[2, 2, np.inf], [3, 3, 2], [np.inf, 5, 3]],
       'path': [[0, 0, 1, 2], [0, 1, 2, 2]],
       'dtw': 3}),

     ({'window_size': 0},
      {'cost_mat':
          [[4, np.inf, np.inf], [np.inf, 1, np.inf], [np.inf, np.inf, 1]],
       'acc_cost_mat':
           [[4, np.inf, np.inf], [np.inf, 5, np.inf], [np.inf, np.inf, 6]],
       'path': [[0, 1, 2], [0, 1, 2]],
       'dtw': 6})]
)
def test_actual_results_dtw_sakoechiba(params, res_desired):
    """Test that the actual results are the expected ones."""
    (dtw_actual, cost_mat_actual,
     acc_cost_mat_actual, path_actual) = dtw_sakoechiba(
        x, y, **params_return, **params
    )
    np.testing.assert_allclose(cost_mat_actual, res_desired['cost_mat'])
    np.testing.assert_allclose(dtw_actual, res_desired['dtw'])
    np.testing.assert_allclose(path_actual, res_desired['path'])
    np.testing.assert_allclose(acc_cost_mat_actual,
                               res_desired['acc_cost_mat'])


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_timestamps': '3'}, TypeError,
      "'n_timestamps' must be an intger."),

     ({'n_timestamps': 10, 'max_slope': '3'}, TypeError,
      "'max_slope' must be an integer or a float."),

     ({'n_timestamps': 1}, ValueError,
      "'n_timestamps' must be an integer greater than or equal to 2."),

     ({'n_timestamps': 10, 'max_slope': 0.5}, ValueError,
      "'max_slope' must be a number greater than or equal to 1.")]
)
def test_parameter_check_itakura_parallelogram(params, error, err_msg):
    """Test parameter validation."""
    with pytest.raises(error, match=re.escape(err_msg)):
        itakura_parallelogram(**params)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'n_timestamps': 4, 'max_slope': 2}, [[0, 1, 1, 3], [1, 3, 3, 4]]),
     ({'n_timestamps': 4, 'max_slope': 8}, [[0, 1, 1, 3], [1, 3, 3, 4]]),
     ({'n_timestamps': 4, 'max_slope': 1}, [[0, 1, 2, 3], [1, 2, 3, 4]])]
)
def test_actual_results_itakura_parallelogram(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = itakura_parallelogram(**params)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, res_desired',
    [({},
      {'cost_mat':
          [[4, np.inf, np.inf], [np.inf, 1, np.inf], [np.inf, np.inf, 1]],
       'acc_cost_mat':
           [[4, np.inf, np.inf], [np.inf, 5, np.inf], [np.inf, np.inf, 6]],
       'path': [[0, 1, 2], [0, 1, 2]],
       'dtw': 6}),

     ({'dist': 'absolute'},
      {'cost_mat':
          [[2, np.inf, np.inf], [np.inf, 1, np.inf], [np.inf, np.inf, 1]],
       'acc_cost_mat':
          [[2, np.inf, np.inf], [np.inf, 3, np.inf], [np.inf, np.inf, 4]],
       'path': [[0, 1, 2], [0, 1, 2]],
       'dtw': 4}),

     ({'max_slope': 8},
      {'cost_mat':
          [[4, np.inf, np.inf], [np.inf, 1, np.inf], [np.inf, np.inf, 1]],
       'acc_cost_mat':
          [[4, np.inf, np.inf], [np.inf, 5, np.inf], [np.inf, np.inf, 6]],
       'path': [[0, 1, 2], [0, 1, 2]],
       'dtw': 6}),

     ({'max_slope': 8, 'dist': 'absolute'},
      {'cost_mat':
          [[2, np.inf, np.inf], [np.inf, 1, np.inf], [np.inf, np.inf, 1]],
       'acc_cost_mat':
          [[2, np.inf, np.inf], [np.inf, 3, np.inf], [np.inf, np.inf, 4]],
       'path': [[0, 1, 2], [0, 1, 2]],
       'dtw': 4})]
)
def test_actual_results_dtw_itakura(params, res_desired):
    """Test that the actual results are the expected ones."""
    (dtw_actual, cost_mat_actual,
     acc_cost_mat_actual, path_actual) = dtw_itakura(
        x, y, **params_return, **params
    )
    np.testing.assert_allclose(cost_mat_actual, res_desired['cost_mat'])
    np.testing.assert_allclose(dtw_actual, res_desired['dtw'])
    np.testing.assert_allclose(path_actual, res_desired['path'])
    np.testing.assert_allclose(acc_cost_mat_actual,
                               res_desired['acc_cost_mat'])


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'n_timestamps': 6, 'resolution_level': 2, 'n_timestamps_reduced': 3,
       'path': np.asarray([[0, 0, 1, 2], [0, 1, 2, 2]]), 'radius': 0},
      [[0, 0, 4, 4, 4, 4], [4, 4, 6, 6, 6, 6]]),

     ({'n_timestamps': 6, 'resolution_level': 2, 'n_timestamps_reduced': 3,
       'path': np.asarray([[0, 0, 1, 2], [0, 1, 2, 2]]), 'radius': 1},
      [[0, 0, 0, 0, 2, 2], [6, 6, 6, 6, 6, 6]])]
)
def test_actual_results_multiscale_region(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = _multiscale_region(**params)
    np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'resolution': '3'}, TypeError, "'resolution' must be an integer."),
     ({'radius': '3'}, TypeError, "'radius' must be an integer."),
     ({'resolution': 0}, ValueError,
      "'resolution' must be a positive integer."),
     ({'radius': -3}, ValueError, "'radius' must be a non-negative integer.")]
)
def test_parameter_check_dtw_multiscale(params, error, err_msg):
    """Test parameter validation."""
    with pytest.raises(error, match=re.escape(err_msg)):
        dtw_multiscale(x, y, **params)


@pytest.mark.parametrize(
    'params, res_desired',
    [({},
      {'cost_mat':
          [[1, 4, np.inf, np.inf], [0, 1, np.inf, np.inf],
           [np.inf, np.inf, 1, 4], [np.inf, np.inf, 0, 1]],
       'acc_cost_mat':
           [[1, 5, np.inf, np.inf], [1, 2, np.inf, np.inf],
            [np.inf, np.inf, 3, 7], [np.inf, np.inf, 3, 4]],
       'path': [[0, 1, 2, 3], [0, 1, 2, 3]],
       'dtw': 4}),

     ({'radius': 1},
      {'cost_mat':
          [[1, 4, 9, 16], [0, 1, 4, 9],
           [1, 0, 1, 4], [4, 1, 0, 1]],
       'acc_cost_mat':
           [[1, 5, 14, 30], [1, 2, 6, 15],
            [2, 1, 2, 6], [6, 2, 1, 2]],
       'path': [[0, 1, 2, 3, 3], [0, 0, 1, 2, 3]],
       'dtw': 2}),

     ({'radius': 2},
      {'cost_mat':
          [[1, 4, 9, 16], [0, 1, 4, 9],
           [1, 0, 1, 4], [4, 1, 0, 1]],
       'acc_cost_mat':
           [[1, 5, 14, 30], [1, 2, 6, 15],
            [2, 1, 2, 6], [6, 2, 1, 2]],
       'path': [[0, 1, 2, 3, 3], [0, 0, 1, 2, 3]],
       'dtw': 2})]
)
def test_actual_results_dtw_multiscale(params, res_desired):
    """Test that the actual results are the expected ones."""
    x = np.arange(4)
    y = np.arange(1, 5)

    (dtw_actual, cost_mat_actual,
     acc_cost_mat_actual, path_actual) = dtw_multiscale(
        x, y, **params_return, **params
    )
    np.testing.assert_allclose(cost_mat_actual, res_desired['cost_mat'])
    np.testing.assert_allclose(dtw_actual, res_desired['dtw'])
    np.testing.assert_allclose(path_actual, res_desired['path'])
    np.testing.assert_allclose(acc_cost_mat_actual,
                               res_desired['acc_cost_mat'])


@pytest.mark.parametrize(
    'params, err_msg',
    [({'method': 'loop'},
      "'method' must be either 'classic', 'sakoechiba', 'itakura', "
      "'multiscale' or 'fast'.")]
)
def test_parameter_check_dtw(params, err_msg):
    """Test parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        dtw(x, y, **params)


@pytest.mark.parametrize(
    'params, res_desired',
    [({}, dtw_classic(x, y, **params_return)),

     ({'method': 'sakoechiba'}, dtw_sakoechiba(x, y, **params_return)),

     ({'method': 'sakoechiba', 'options': {'window_size': 2}},
      dtw_sakoechiba(x, y, window_size=2, **params_return)),

     ({'method': 'itakura'}, dtw_itakura(x, y, **params_return)),

     ({'method': 'itakura', 'options': {'max_slope': 8}},
      dtw_itakura(x, y, max_slope=8, **params_return)),

     ({'method': 'multiscale'}, dtw_multiscale(x, y, **params_return)),

     ({'method': 'multiscale', 'options': {'resolution': 1}},
      dtw_multiscale(x, y, resolution=1, **params_return)),

     ({'method': 'multiscale', 'options': {'radius': 2}},
      dtw_multiscale(x, y, radius=2, **params_return)),

     ({'method': 'fast'}, dtw_fast(x, y, **params_return)),

     ({'method': 'fast', 'options': {'radius': 1}},
      dtw_fast(x, y, radius=1, **params_return))]
)
def test_actual_results_dtw(params, res_desired):
    """Test that the actual results are the expected ones."""
    (dtw_actual, cost_mat_actual, acc_cost_mat_actual, path_actual) = dtw(
        x, y, **params_return, **params
    )
    np.testing.assert_allclose(dtw_actual, res_desired[0])
    np.testing.assert_allclose(cost_mat_actual, res_desired[1])
    np.testing.assert_allclose(acc_cost_mat_actual, res_desired[2])
    np.testing.assert_allclose(path_actual, res_desired[3])


@pytest.mark.parametrize(
    'params, err_msg',
    [({'method': 'whoops'},
      "'method' must be either None, 'classic', 'sakoechiba', 'itakura', "
      "'multiscale' or 'fast'.")]
)
def test_parameter_check_show_options(params, err_msg):
    """Test parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        show_options(**params)


@pytest.mark.parametrize(
    'params, res_desired',
    [({}, None),
     ({'method': 'classic'}, None),
     ({'method': 'sakoechiba'}, None),
     ({'method': 'itakura'}, None),
     ({'method': 'multiscale'}, None),
     ({'method': 'fast'}, None)]
)
def test_actual_results_show_options(params, res_desired):
    """Test that the actual results are the expected ones."""
    res_actual = show_options(disp=True, **params)
    np.testing.assert_equal(res_actual, res_desired)
