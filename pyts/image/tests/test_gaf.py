"""Testing for Gramian Angular Field."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.image.gaf import _gasf, _gadf
from pyts.image import GramianAngularField


X = np.arange(9).reshape(1, 9)
pi_6 = np.cos(np.pi / 6).item()


@pytest.mark.parametrize(
    'X_cos, X_sin, arr_desired',
    [([[-1, 0, 1]], [[-1, 0, 1]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),

     ([[-1, 0, 1]], [[1, 0, 1]], [[[0, 0, -2], [0, 0, 0], [-2, 0, 0]]]),

     ([[-1, 0, 1], [-1, 0, 1]], [[1, 0, 1], [1, 0, 1]],
      [[[0, 0, -2], [0, 0, 0], [-2, 0, 0]],
       [[0, 0, -2], [0, 0, 0], [-2, 0, 0]]])]
)
def test_actual_results_gasf(X_cos, X_sin, arr_desired):
    """Test that the actual results are the expected ones."""
    X_cos = np.asarray(X_cos)
    X_sin = np.asarray(X_sin)
    arr_actual = _gasf(X_cos, X_sin,
                       n_samples=X_cos.shape[0], image_size=X_cos.shape[1])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'X_cos, X_sin, arr_desired',
    [([[-1, 0, 1]], [[-1, 0, 1]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),

     ([[-1, 0, 1]], [[1, 0, 1]], [[[0, 0, 2], [0, 0, 0], [-2, 0, 0]]]),

     ([[-1, 0, 1], [-1, 0, 1]], [[1, 0, 1], [1, 0, 1]],
      [[[0, 0, 2], [0, 0, 0], [-2, 0, 0]],
       [[0, 0, 2], [0, 0, 0], [-2, 0, 0]]])]
)
def test_actual_results_gadf(X_cos, X_sin, arr_desired):
    """Test that the actual results are the expected ones."""
    X_cos = np.asarray(X_cos)
    X_sin = np.asarray(X_sin)
    arr_actual = _gadf(X_cos, X_sin,
                       n_samples=X_cos.shape[0], image_size=X_cos.shape[1])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'image_size': '4'}, TypeError,
      "'image_size' must be an integer or a float."),

     ({'sample_range': [0, 1]}, TypeError,
      "'sample_range' must be None or a tuple."),

     ({'image_size': 0}, ValueError,
      "If 'image_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps (got 0)."),

     ({'image_size': 2.}, ValueError,
      "If 'image_size' is a float, it must be greater than 0 and lower than "
      "or equal to 1 (got {0}).".format(2.)),

     ({'sample_range': (-1, 0, 1)}, ValueError,
      "If 'sample_range' is a tuple, its length must be equal to 2."),

     ({'sample_range': (-2, 2)}, ValueError,
      "If 'sample_range' is a tuple, it must satisfy "
      "-1 <= sample_range[0] < sample_range[1] <= 1."),

     ({'method': 'a'}, ValueError,
      "'method' must be either 'summation', 's', 'difference' or 'd'."),

     ({'sample_range': None}, ValueError,
      "If 'sample_range' is None, all the values of X must be between "
      "-1 and 1.")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    gaf = GramianAngularField(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        gaf.transform(X)


@pytest.mark.parametrize(
    'params, X, arr_desired',
    [({}, [[-1, 0, 1]], [[[1, 0, -1], [0, -1, 0], [-1, 0, 1]]]),

     ({'sample_range': None}, [[-1, 0, 1]],
      [[[1, 0, -1], [0, -1, 0], [-1, 0, 1]]]),

     ({'image_size': 3}, [[-1, 0, 1]],
      [[[1, 0, -1], [0, -1, 0], [-1, 0, 1]]]),

     ({'image_size': 3}, [np.arange(9)],
      [[[1, 0, -1], [0, -1, 0], [-1, 0, 1]]]),

     ({'image_size': 3, 'overlapping': True}, [np.arange(9)],
      [[[1, 0, -1], [0, -1, 0], [-1, 0, 1]]]),

     ({'method': 'd'}, [[-1, 0, 1]],
      [[[0, 1, 0], [-1, 0, 1], [0, -1, 0]]]),

     ({'image_size': 3, 'method': 'd'}, [np.arange(9)],
      [[[0, 1, 0], [-1, 0, 1], [0, -1, 0]]]),

     ({'sample_range': (0, 1)}, [[-1, 0, 1]],
      [[[-1, -pi_6, 0], [-pi_6, -0.5, 0.5], [0, 0.5, 1]]]),

     ({'sample_range': (0, 1), 'method': 'd'}, [[-1, 0, 1]],
      [[[0, 0.5, 1], [-0.5, 0, pi_6], [-1, -pi_6, 0]]])]
)
def test_actual_results(params, X, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = GramianAngularField(**params).fit_transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_flatten():
    """Test the 'flatten' parameter."""
    arr_false = GramianAngularField().transform(X).reshape(1, -1)
    arr_true = GramianAngularField(flatten=True).transform(X)
    np.testing.assert_allclose(arr_false, arr_true, atol=1e-5, rtol=0.)
