"""Testing for 'segmentation' function."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.utils import segmentation


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'ts_size': None, 'window_size': 3}, TypeError,
      "'ts_size' must be an integer."),
     ({'ts_size': 4, 'window_size': None}, TypeError,
      "'window_size' must be an integer."),
     ({'ts_size': 4, 'window_size': 3, 'n_segments': "3"}, TypeError,
      "'n_segments' must be None or an integer."),
     ({'ts_size': 1, 'window_size': 2}, ValueError,
      "'ts_size' must be an integer greater than or equal to 2 (got 1)."),
     ({'ts_size': 10, 'window_size': 0}, ValueError,
      "'window_size' must be an integer greater than or equal to 1 (got 0)."),
     ({'ts_size': 10, 'window_size': 15}, ValueError,
      "'window_size' must be lower than or equal to 'ts_size' (15 > 10)."),
     ({'ts_size': 10, 'window_size': 3, 'n_segments': 1}, ValueError,
      "If 'n_segments' is an integer, it must be greater than or equal to 2 "
      "(got 1)."),
     ({'ts_size': 10, 'window_size': 3, 'n_segments': 12}, ValueError,
      "If 'n_segments' is an integer, it must be lower than or equal to "
      "'ts_size' (12 > 10).")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    with pytest.raises(error, match=re.escape(err_msg)):
        segmentation(**params)


@pytest.mark.parametrize(
    ('params, start_desired, end_desired, size_desired'),
    [({'ts_size': 20, 'window_size': 4, 'overlapping': False},
      [0, 4, 8, 12, 16], [4, 8, 12, 16, 20], 5),
     ({'ts_size': 20, 'window_size': 8, 'overlapping': True},
      [0, 6, 12], [8, 14, 20], 3)]
)
def test_accurate_results(params, start_desired, end_desired, size_desired):
    """Test that the actual results are the expected ones."""
    start_actual, end_actual, size_actual = segmentation(**params)
    np.testing.assert_array_equal(start_actual, start_desired)
    np.testing.assert_array_equal(end_actual, end_desired)
    np.testing.assert_equal(size_actual, size_desired)
