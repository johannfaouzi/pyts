"""Testing for Recurrence Plot."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import re

import numpy as np
import pytest

from pyts.image import RecurrencePlot
from pyts.image._recurrence import _trajectories

X = np.asarray([[0, 1, 1, 0, 2, 3], [1, 0, 3, 2, 1, 0]])


@pytest.mark.parametrize(
    'params, arr_desired',
    [
        (
            {'dimension': 3, 'time_delay': 1},
            np.asarray(
                [
                    [[0, 1, 1], [1, 1, 0], [1, 0, 2], [0, 2, 3]],
                    [[1, 0, 3], [0, 3, 2], [3, 2, 1], [2, 1, 0]],
                ]
            ),
        ),
        (
            {'dimension': 2, 'time_delay': 2},
            np.asarray(
                [
                    [[0, 1], [1, 0], [1, 2], [0, 3]],
                    [[1, 3], [0, 2], [3, 1], [2, 0]],
                ]
            ),
        ),
        (
            {'dimension': 3, 'time_delay': 2},
            np.asarray([[[0, 1, 2], [1, 0, 3]], [[1, 3, 1], [0, 2, 0]]]),
        ),
        (
            {'dimension': 1, 'time_delay': 1},
            np.asarray(
                [
                    [[0], [1], [1], [0], [2], [3]],
                    [[1], [0], [3], [2], [1], [0]],
                ]
            ),
        ),
        (
            {'dimension': 1, 'time_delay': 5},
            np.asarray(
                [
                    [[0], [1], [1], [0], [2], [3]],
                    [[1], [0], [3], [2], [1], [0]],
                ]
            ),
        ),
    ],
)
def test_actual_results_trajectories(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = _trajectories(X, **params)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [
        (
            {'dimension': '3'},
            TypeError,
            "'dimension' must be an integer or a float.",
        ),
        (
            {'time_delay': '3'},
            TypeError,
            "'time_delay' must be an integer or a float.",
        ),
        (
            {'threshold': '3'},
            TypeError,
            "'threshold' must be either None, 'point', 'distance', a float or an "
            "integer.",
        ),
        (
            {'percentage': '3'},
            TypeError,
            "'percentage' must be a float or an integer.",
        ),
        (
            {'dimension': 10},
            ValueError,
            "If 'dimension' is an integer, it must be greater than or equal to 1 "
            "and lower than or equal to n_timestamps (got 10).",
        ),
        (
            {'dimension': 2.0},
            ValueError,
            "If 'dimension' is a float, it must be greater than 0 and lower than or "
            f"equal to 1 (got {2.0}).",
        ),
        (
            {'time_delay': 12},
            ValueError,
            "If 'time_delay' is an integer, it must be greater than or equal to 1 "
            "and lower than or equal to n_timestamps (got 12).",
        ),
        (
            {'time_delay': 2.0},
            ValueError,
            "If 'time_delay' is a float, it must be greater than 0 and lower than "
            f"or equal to 1 (got {2.0}).",
        ),
        (
            {'time_delay': 3, 'dimension': 3},
            ValueError,
            "The number of trajectories must be positive. Consider trying with "
            "smaller values for 'dimension' and 'time_delay'.",
        ),
        (
            {'threshold': -1},
            ValueError,
            "If 'threshold' is a float or an integer, it must be greater than or "
            "equal to 0.",
        ),
        (
            {'percentage': 200, 'threshold': 'point'},
            ValueError,
            "'percentage' must be between 0 and 100.",
        ),
    ],
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    recurrence = RecurrencePlot(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        recurrence.transform(X)


@pytest.mark.parametrize(
    'params, arr_desired',
    [
        (
            {'dimension': 3, 'time_delay': 1},
            [
                [[0, 2, 3, 5], [2, 0, 5, 11], [3, 5, 0, 6], [5, 11, 6, 0]],
                [
                    [0, 11, 12, 11],
                    [11, 0, 11, 12],
                    [12, 11, 0, 3],
                    [11, 12, 3, 0],
                ],
            ],
        ),
        (
            {'dimension': 2, 'time_delay': 2},
            [
                [[0, 2, 2, 4], [2, 0, 4, 10], [2, 4, 0, 2], [4, 10, 2, 0]],
                [[0, 2, 8, 10], [2, 0, 10, 8], [8, 10, 0, 2], [10, 8, 2, 0]],
            ],
        ),
        (
            {'dimension': 3, 'time_delay': 2},
            [[[0, 3], [3, 0]], [[0, 3], [3, 0]]],
        ),
        (
            {'dimension': 0.5, 'time_delay': 0.3},
            [[[0, 3], [3, 0]], [[0, 3], [3, 0]]],
        ),
        (
            {
                'dimension': 0.5,
                'time_delay': 0.3,
                'threshold': 'point',
                'percentage': 50,
            },
            [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
        ),
        (
            {
                'dimension': 0.5,
                'time_delay': 0.3,
                'threshold': 'distance',
                'percentage': 50,
            },
            [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
        ),
        (
            {'dimension': 1, 'time_delay': 1},
            [
                [
                    [0, 1, 1, 0, 4, 9],
                    [1, 0, 0, 1, 1, 4],
                    [1, 0, 0, 1, 1, 4],
                    [0, 1, 1, 0, 4, 9],
                    [4, 1, 1, 4, 0, 1],
                    [9, 4, 4, 9, 1, 0],
                ],
                [
                    [0, 1, 4, 1, 0, 1],
                    [1, 0, 9, 4, 1, 0],
                    [4, 9, 0, 1, 4, 9],
                    [1, 4, 1, 0, 1, 4],
                    [0, 1, 4, 1, 0, 1],
                    [1, 0, 9, 4, 1, 0],
                ],
            ],
        ),
        (
            {'dimension': 1, 'time_delay': 5},
            [
                [
                    [0, 1, 1, 0, 4, 9],
                    [1, 0, 0, 1, 1, 4],
                    [1, 0, 0, 1, 1, 4],
                    [0, 1, 1, 0, 4, 9],
                    [4, 1, 1, 4, 0, 1],
                    [9, 4, 4, 9, 1, 0],
                ],
                [
                    [0, 1, 4, 1, 0, 1],
                    [1, 0, 9, 4, 1, 0],
                    [4, 9, 0, 1, 4, 9],
                    [1, 4, 1, 0, 1, 4],
                    [0, 1, 4, 1, 0, 1],
                    [1, 0, 9, 4, 1, 0],
                ],
            ],
        ),
    ],
)
def test_actual_results_recurrence_plot(params, arr_desired):
    """Test that the actual results are the expected ones."""
    recurrence = RecurrencePlot(**params)
    arr_actual = recurrence.fit_transform(X) ** 2
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.0)


def test_flatten():
    """Test the 'flatten' parameter."""
    arr_false = RecurrencePlot().transform(X).reshape(2, -1)
    arr_true = RecurrencePlot(flatten=True).transform(X)
    np.testing.assert_allclose(arr_false, arr_true, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    'dtype',
    [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.bool_,
    ],
)
@pytest.mark.parametrize('dimension', [1, 3])
def test_numeric_dtypes(dtype, dimension):
    """Safely convertible numeric inputs produce Euclidean distances."""
    X_typed = X.astype(dtype)
    trajectories = np.asarray(
        [
            [row[i : i + dimension] for i in range(X.shape[1] - dimension + 1)]
            for row in X_typed.astype(np.float64)
        ]
    )
    expected = np.linalg.norm(
        trajectories[:, :, None] - trajectories[:, None, :], axis=-1
    )

    actual = RecurrencePlot(dimension=dimension).transform(X_typed)

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize('dimension', [1, 2])
def test_large_int64_precision(dimension):
    """Keep small differences between integers above float64 precision."""
    X_large = np.asarray([[2**53, 2**53 + 1, 2**53 + 3]], dtype=np.int64)
    if dimension == 1:
        expected = [[[0, 1, 3], [1, 0, 2], [3, 2, 0]]]
    else:
        expected = [[[0, np.sqrt(5)], [np.sqrt(5), 0]]]

    actual = RecurrencePlot(dimension=dimension).transform(X_large)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize('dtype', [np.int32, np.uint32])
@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_integer_dtype_full_range(dtype, dimension):
    """Embedded distances do not overflow for newly accepted integer dtypes."""
    bounds = np.iinfo(dtype)
    X_range = np.asarray(
        [[bounds.min, bounds.max, bounds.min, bounds.max]], dtype=dtype
    )
    # Alternating trajectories are either identical or differ by the full
    # integer span in every component; Python integer subtraction is exact.
    span = int(bounds.max) - int(bounds.min)
    phases = np.arange(5 - dimension) % 2
    expected = (phases[:, None] != phases[None, :]) * (span * np.sqrt(dimension))

    actual = RecurrencePlot(dimension=dimension).transform(X_range)

    np.testing.assert_allclose(actual[0], expected, rtol=1e-15, atol=0)
