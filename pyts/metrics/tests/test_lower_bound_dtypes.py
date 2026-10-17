"""Numeric input conversion must precede lower-bound arithmetic."""

import numpy as np
import pytest

from pyts.metrics import (
    lower_bound_improved,
    lower_bound_keogh,
    lower_bound_kim,
    lower_bound_yi,
)


@pytest.mark.parametrize(
    'bound',
    [lower_bound_yi, lower_bound_kim, lower_bound_keogh, lower_bound_improved],
)
@pytest.mark.parametrize('train_dtype', [np.uint8, np.int16, np.float32, np.float64])
@pytest.mark.parametrize('test_dtype', [np.uint8, np.int16, np.float32, np.float64])
def test_lower_bounds_numeric_dtypes(bound, train_dtype, test_dtype):
    """Constant-series offsets have exact lower bounds for every window."""
    train = np.array([[1, 1, 1, 1], [5, 5, 5, 5]], dtype=train_dtype)
    test = np.array([[3, 3, 3, 3]], dtype=test_dtype)
    if bound in (lower_bound_keogh, lower_bound_improved):
        actual = bound(train, test, [[0, 0, 1, 2], [2, 3, 4, 4]])
    else:
        actual = bound(train, test)
    expected = [[2, 2]] if bound is lower_bound_kim else [[4, 4]]
    np.testing.assert_allclose(actual, expected)


def test_yi_promotes_before_integer_square():
    """Integer storage must not overflow in the distance calculation."""
    train = np.zeros((1, 4), dtype=np.int64)
    test = np.full((1, 4), 2_000_000_000, dtype=np.int64)
    np.testing.assert_allclose(lower_bound_yi(train, test), [[4_000_000_000]])
