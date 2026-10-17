"""Testing for Piecewise Aggregate Approximation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import re

import numpy as np
import pytest

from pyts.approximation import PiecewiseAggregateApproximation
from pyts.approximation._paa import _paa

X = [np.arange(6), [0, 5, 1, 2, 4, 3]]


@pytest.mark.parametrize(
    'dtype', ['float16', 'float32', 'float64', 'int16', 'int32', 'int64', 'uint16']
)
@pytest.mark.parametrize('overlapping', [False, True])
def test_numeric_dtypes(dtype, overlapping):
    """Numeric inputs reach the compiled averaging kernel after conversion."""
    X = np.array([[0, 3, 6, 9, 12], [15, 12, 9, 6, 3]], dtype=dtype)
    transformer = PiecewiseAggregateApproximation(
        window_size=None, output_size=2, overlapping=overlapping
    )
    expected = [[3, 9], [12, 6]] if overlapping else [[1.5, 9], [13.5, 6]]
    np.testing.assert_allclose(transformer.transform(X), expected)


@pytest.mark.parametrize('dtype', ['float32', 'int16', 'int64'])
def test_identity_dtype(dtype):
    """The window-size-one fast path keeps the original data and dtype."""
    X = np.arange(6, dtype=dtype).reshape(2, 3)
    result = PiecewiseAggregateApproximation(window_size=1).transform(X)
    assert result is X
    assert result.dtype == X.dtype


@pytest.mark.parametrize(
    'X, start, end, arr_desired',
    [
        (
            [[0, 1, 2, 3], [1, 3, 0, 2], [4, 8, 2, 6]],
            [0, 2],
            [2, 4],
            [[0.5, 2.5], [2.0, 1.0], [6.0, 4.0]],
        ),
        (
            [[0, 1, 2, 3], [1, 3, 0, 2], [4, 8, 2, 6]],
            [0, 1, 2],
            [2, 3, 4],
            [[0.5, 1.5, 2.5], [2.0, 1.5, 1.0], [6.0, 5.0, 4.0]],
        ),
        (
            [[0, 1, 2, 3], [1, 3, 0, 2], [4, 8, 2, 6]],
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [[0, 1, 2, 3], [1, 3, 0, 2], [4, 8, 2, 6]],
        ),
    ],
)
def test_actual_results_paa(X, start, end, arr_desired):
    """Test that the actual results are the expected ones."""
    X = np.asarray(X)
    start = np.asarray(start)
    end = np.asarray(end)
    n_timestamps_new = start.size
    n_samples, n_timestamps = X.shape
    arr_actual = _paa(X, n_samples, n_timestamps, start, end, n_timestamps_new)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [
        (
            {'window_size': None, 'output_size': None},
            TypeError,
            "'window_size' and 'output_size' cannot be both None.",
        ),
        (
            {'window_size': '3'},
            TypeError,
            "If specified, 'window_size' must be an integer or a float.",
        ),
        (
            {'window_size': None, 'output_size': '3'},
            TypeError,
            "If specified, 'output_size' must be an integer or a float.",
        ),
        (
            {'window_size': 0},
            ValueError,
            "If 'window_size' is an integer, it must be greater than or equal to 1 "
            "and lower than or equal to n_timestamps (got 0).",
        ),
        (
            {'window_size': 2.0},
            ValueError,
            "If 'window_size' is a float, it must be greater than 0 and lower than "
            f"or equal to 1 (got {2.0}).",
        ),
        (
            {'window_size': None, 'output_size': 0},
            ValueError,
            "If 'output_size' is an integer, it must be greater than or equal to 1 "
            "and lower than or equal to n_timestamps (got 0).",
        ),
        (
            {'window_size': None, 'output_size': 2.0},
            ValueError,
            "If 'output_size' is a float, it must be greater than 0 and lower than "
            f"or equal to 1 (got {2.0}).",
        ),
    ],
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    paa = PiecewiseAggregateApproximation(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        paa.transform(X)


@pytest.mark.parametrize(
    'params, arr_desired',
    [
        ({}, X),
        ({'window_size': 2}, [[0.5, 2.5, 4.5], [2.5, 1.5, 3.5]]),
        (
            {'window_size': None, 'output_size': 3},
            [[0.5, 2.5, 4.5], [2.5, 1.5, 3.5]],
        ),
        (
            {'window_size': None, 'output_size': 0.5},
            [[0.5, 2.5, 4.5], [2.5, 1.5, 3.5]],
        ),
        ({'window_size': 0.5}, [[1, 4], [2, 3]]),
    ],
)
def test_actual_results(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = PiecewiseAggregateApproximation(**params).fit_transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.0)
