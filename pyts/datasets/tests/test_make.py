"""Testing for datatset generation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.datasets import make_cylinder_bell_funnel


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_samples': '3'}, TypeError, "'n_samples' must be an integer."),

     ({'n_samples': -1}, ValueError,
      "'n_samples' must be a positive integer."),

     ({'weights': [1]}, ValueError,
      "'weights' must be None or a list with 2 or 3 elements (got 1)."),

     ({'weights': [1, 2, 3, 4]}, ValueError,
      "'weights' must be None or a list with 2 or 3 elements (got 4)."),

     ({'weights': [0.5, 1.5]}, ValueError,
      "'sum(weights)' cannot be larger than 1 if len(weights) == 2 (got 2.0)")]
)
def test_parameter_check_make_cylinder_bell_funnel(params, error, err_msg):
    """Test parameter validation."""
    with pytest.raises(error, match=re.escape(err_msg)):
        make_cylinder_bell_funnel(**params)


@pytest.mark.parametrize(
    'params, class_balance_desired',
    [({'n_samples': 9}, [3, 3, 3]),
     ({'n_samples': 90}, [30, 30, 30]),
     ({'n_samples': 10, 'weights': [0, 0.5, 0.5]}, [0, 5, 5]),
     ({'n_samples': 10, 'weights': [0, 1., 1.]}, [0, 10, 10]),
     ({'n_samples': 10, 'weights': [0, 1., 0]}, [0, 10]),
     ({'n_samples': 10, 'weights': np.array([0, 0.5, 0.5])}, [0, 5, 5]),
     ({'n_samples': 10, 'weights': (0, 0.5, 0.5)}, [0, 5, 5])]
)
def test_class_balance_make_cylinder_bell_funnel(params,
                                                 class_balance_desired):
    """Test that the class balance is the expected one."""
    X, y = make_cylinder_bell_funnel(**params)
    class_balance_actual = np.bincount(y)
    np.testing.assert_array_equal(class_balance_actual, class_balance_desired)


@pytest.mark.parametrize(
    'params',
    [{},
     {'return_params': True},
     {'n_samples': 100, 'return_params': True}])
def test_return_params_make_cylinder_bell_funnel(params):
    """Test the return objects."""
    res = make_cylinder_bell_funnel(**params)
    assert isinstance(res[0], np.ndarray)
    assert isinstance(res[1], np.ndarray)
    if 'return_params' in params.keys() and params['return_params']:
        parameters = res[2]
        assert isinstance(parameters, dict)
        assert isinstance(parameters['a'], (int, np.integer))
        assert isinstance(parameters['b'], (int, np.integer))
        assert isinstance(parameters['eta'], (float, np.floating))
        assert isinstance(parameters['epsilon'], np.ndarray)
        if 'n_samples' in params:
            assert parameters['epsilon'].shape == (params['n_samples'], 128)
        else:
            assert parameters['epsilon'].shape == (30, 128)


@pytest.mark.parametrize('params', [{}, {'shuffle': False}])
def test_shuffle_make_cylinder_bell_funnel(params):
    """Test that shuffling works as expected."""
    _, y = make_cylinder_bell_funnel(**params)
    arr_desired_no_shuffle = np.repeat(np.arange(3), 10)
    if 'shuffle' in params.keys() and (not params['shuffle']):
        np.testing.assert_array_equal(y, arr_desired_no_shuffle)
    else:
        assert not np.array_equal(y, arr_desired_no_shuffle)
