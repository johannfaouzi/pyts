"""Testing for 'windowed_view' function."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.utils import windowed_view


X = np.arange(10).reshape(2, 5)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'X': X, 'window_size': '3', 'window_step': 3}, TypeError,
      "'window_size' must be an integer."),
     ({'X': X, 'window_size': 3, 'window_step': '3'}, TypeError,
      "'window_step' must be an integer."),
     ({'X': X, 'window_size': 7, 'window_step': 3}, ValueError,
      "'window_size' must be an integer between 1 and n_timestamps."),
     ({'X': X, 'window_size': 3, 'window_step': 7}, ValueError,
      "'window_step' must be an integer between 1 and n_timestamps.")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation in segmentation."""
    with pytest.raises(error, match=re.escape(err_msg)):
        windowed_view(**params)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'X': X, 'window_size': 3, 'window_step': 1},
      [[[0, 1, 2], [1, 2, 3], [2, 3, 4]], [[5, 6, 7], [6, 7, 8], [7, 8, 9]]]),
     ({'X': X, 'window_size': 3, 'window_step': 2},
      [[[0, 1, 2], [2, 3, 4]], [[5, 6, 7], [7, 8, 9]]]),
     ({'X': X, 'window_size': 2, 'window_step': 3},
      [[[0, 1], [3, 4]], [[5, 6], [8, 9]]])]
)
def test_accurate_results(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = windowed_view(**params)
    np.testing.assert_array_equal(arr_actual, arr_desired)
