import numpy as np
import pytest
import re

from pyts.transformation.rocket import (
    generate_kernels, apply_one_kernel_one_sample, apply_all_kernels)
from pyts.transformation import ROCKET


X = np.arange(14, dtype='float64').reshape(2, 7)
weights = np.array([[1, -1, 0, 0],
                    [-1, 2, -1, 0],
                    [-0.5, 0.5, 0, 0]])
lengths = np.array([2, 3, 4])
biases = np.array([-1., 0., 1.])
dilations = np.array([1, 2, 2])
paddings = np.array([0, 0, 3])


@pytest.mark.parametrize(
    'n_kernels, n_timestamps, kernel_sizes, seed',
    [(10, 100, (4, 6, 8), 42),
     (10, 100, (4, 6, 8), 43),
     (50, 100, (4, 6, 8), 42),
     (10, 64, (4, 6, 8), 42),
     (10, 600, (4, 6, 8), 42)]
)
def test_generate_kernels(n_kernels, n_timestamps, kernel_sizes, seed):
    """Test the generated kernels."""
    kernel_sizes = np.asarray(kernel_sizes)
    weights, lengths, biases, dilations, paddings = generate_kernels(
        n_kernels, n_timestamps, kernel_sizes, seed
    )
    # Check shapes
    assert weights.shape == (n_kernels, max(kernel_sizes))
    for arr in (lengths, biases, dilations, paddings):
        assert arr.shape == (n_kernels,)

    # Check range of values
    assert np.all(np.isin(lengths, kernel_sizes))
    assert np.all(np.logical_and(-1 <= biases, biases <= 1))
    (n_timestamps - 1) // (min(kernel_sizes) - 1)
    upper_bound_dilation = (n_timestamps - 1) // (min(kernel_sizes) - 1)
    assert np.all(
        np.logical_and(1 <= dilations, dilations <= upper_bound_dilation))
    upper_bound_padding = (
        (lengths - 1) * (n_timestamps - 1) // (min(kernel_sizes) - 1)) // 2
    assert np.all(
        np.logical_and(0 <= paddings, paddings <= upper_bound_padding))

    # Check zero mean of weights
    np.testing.assert_allclose(weights.mean(axis=1), 0, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'weight': np.zeros(2), 'length': 2, 'bias': 0,
       'dilation': 1, 'padding': 0}, [0, 0]),
     ({'weight': np.zeros(2), 'length': 2, 'bias': -1,
       'dilation': 1, 'padding': 0}, [-1, 0]),
     ({'weight': np.zeros(2), 'length': 2, 'bias': 1,
       'dilation': 1, 'padding': 0}, [1, 1]),
     ({'weight': np.zeros(2), 'length': 2, 'bias': 1,
       'dilation': 2, 'padding': 0}, [1, 1]),
     ({'weight': np.zeros(3), 'length': 2, 'bias': 1,
       'dilation': 1, 'padding': 0}, [1, 1]),
     ({'weight': np.zeros(3), 'length': 2, 'bias': 1,
       'dilation': 1, 'padding': 1}, [1, 1]),
     ({'weight': np.zeros(3), 'length': 2, 'bias': 1,
       'dilation': 2, 'padding': 2}, [1, 1]),
     ({'weight': weights[0], 'length': lengths[0],
       'bias': biases[0], 'dilation': dilations[0],
       'padding': paddings[0]}, [-2, 0]),
     ({'weight': weights[1], 'length': lengths[1],
       'bias': biases[1], 'dilation': dilations[1],
       'padding': paddings[1]}, [0, 0]),
     ({'weight': weights[2], 'length': lengths[2],
       'bias': biases[2], 'dilation': dilations[2],
       'padding': paddings[2]}, [2, 1])]
)
def test_apply_one_kernel_one_sample(params, arr_desired):
    """Test the convolutions applied to one time series."""
    arr_actual = apply_one_kernel_one_sample(
        x=X[0], n_timestamps=7, **params)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'X': X, 'weights': np.zeros((3, 4)), 'lengths': np.array([2, 3, 4]),
       'biases': np.zeros(3), 'dilations': np.ones(3, dtype='int64'),
       'paddings': np.zeros(3, dtype='int64')}, np.zeros((2, 6))),

     ({'X': np.zeros((2, 7)), 'weights': np.ones((3, 4)),
       'lengths': np.array([2, 3, 4]), 'biases': np.zeros(3),
       'dilations': np.ones(3, dtype='int64'),
       'paddings': np.zeros(3, dtype='int64')}, np.zeros((2, 6))),

     ({'X': np.zeros((2, 7)), 'weights': np.ones((3, 4)),
       'lengths': np.array([2, 3, 4]), 'biases': np.ones(3),
       'dilations': np.ones(3, dtype='int64'),
       'paddings': np.zeros(3, dtype='int64')}, np.ones((2, 6))),

     ({'X': X, 'weights': weights, 'lengths': lengths,
       'biases': biases, 'dilations': dilations, 'paddings': paddings},
      [[-2, 0, 0, 0, 2, 1], [-2, 0, 0, 0, 5, 1]]),

     ({'X': X[:1], 'weights': weights, 'lengths': lengths,
       'biases': biases, 'dilations': dilations, 'paddings': paddings},
      [[-2, 0, 0, 0, 2, 1]]),

     ({'X': X[1:], 'weights': weights, 'lengths': lengths,
       'biases': biases, 'dilations': dilations, 'paddings': paddings},
      [[-2, 0, 0, 0, 5, 1]])]
)
def test_apply_all_kernels(params, arr_desired):
    """Test the application of all kernels to all samples."""
    arr_actual = apply_all_kernels(**params)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_kernels': 'yolo'}, TypeError,
      "'n_kernels' must be an integer (got yolo)."),

     ({'kernel_sizes': {0: 1}}, TypeError,
      "'kernel_sizes' must be a list, a tuple or an array (got {0: 1})."),

     ({'kernel_sizes': np.arange(1)}, ValueError,
      "All the values in 'kernel_sizes' must be greater than or equal to 1 "
      "(0 < 1)."),

     ({'kernel_sizes': np.arange(3)}, ValueError,
      "All the values in 'kernel_sizes' must be greater than or equal to 1 "
      "(0 < 1)."),

     ({'kernel_sizes': np.arange(1, 9)}, ValueError,
      "All the values in 'kernel_sizes' must be lower than or equal to "
      "'n_timestamps' (8 > 7)."),

     ({'kernel_sizes': np.arange(1, 14)}, ValueError,
      "All the values in 'kernel_sizes' must be lower than or equal to "
      "'n_timestamps' (13 > 7).")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    clf = ROCKET(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        clf.fit(X)
