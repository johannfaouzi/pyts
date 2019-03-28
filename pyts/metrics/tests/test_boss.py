"""Testing for BOSS metric."""

import numpy as np
import pytest
from math import sqrt
from ..boss import boss


def test_boss():
    """Test 'boss' function."""
    x = np.arange(1, 6)
    y = np.arange(1, 6)[::-1]
    z = [0, 0, 0, 10, 0]

    # Parameter check
    msg_error = "'x' must a one-dimensional array."
    with pytest.raises(ValueError, match=msg_error):
        boss(x.reshape(1, 5), y)

    msg_error = "'y' must a one-dimensional array."
    with pytest.raises(ValueError, match=msg_error):
        boss(x, y.reshape(1, 5))

    msg_error = "'x' and 'y' must have the same shape."
    with pytest.raises(ValueError, match=msg_error):
        boss(x[:2], y)

    # Test 1
    scalar_actual = boss(x, y)
    scalar_desired = sqrt(np.sum((x - y) ** 2))
    np.testing.assert_allclose([scalar_actual], [scalar_desired])

    # Test 2
    scalar_actual = boss(y, x)
    scalar_desired = sqrt(np.sum((x - y) ** 2))
    np.testing.assert_allclose([scalar_actual], [scalar_desired])

    # Test 3
    scalar_actual = boss(x, z)
    scalar_desired = sqrt(np.sum((x - z) ** 2))
    np.testing.assert_allclose([scalar_actual], [scalar_desired])

    # Test 4
    scalar_actual = boss(z, x)
    scalar_desired = sqrt((10 - 4) ** 2)
    np.testing.assert_allclose([scalar_actual], [scalar_desired])

    # Test 5
    scalar_actual = boss(y, z)
    scalar_desired = sqrt(np.sum((y - z) ** 2))
    np.testing.assert_allclose([scalar_actual], [scalar_desired])

    # Test 6
    scalar_actual = boss(z, y)
    scalar_desired = sqrt((10 - 2) ** 2)
    np.testing.assert_allclose([scalar_actual], [scalar_desired])
