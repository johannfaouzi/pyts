"""Testing for BOSS metric."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from math import sqrt
from pyts.metrics import boss


x = np.arange(1, 6)
y = np.arange(1, 6)[::-1]
z = [0, 0, 0, 10, 0]


@pytest.mark.parametrize(
    'x, y, err_msg',
    [(x.reshape(1, -1), y, "'x' must a one-dimensional array."),
     (x, y.reshape(1, -1), "'y' must a one-dimensional array."),
     (x[:2], y, "'x' and 'y' must have the same shape.")]
)
def test_parameter_check(x, y, err_msg):
    """Test parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        boss(x, y)


@pytest.mark.parametrize(
    'x, y, arr_desired',
    [(x, y, sqrt(np.sum((x - y) ** 2))),
     (y, x, sqrt(np.sum((x - y) ** 2))),
     (x, z, sqrt(np.sum((x - z) ** 2))),
     (z, x, 6),
     (y, z, sqrt(np.sum((y - z) ** 2))),
     (z, y, 8)]
)
def test_actual_results(x, y, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = boss(x, y)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
