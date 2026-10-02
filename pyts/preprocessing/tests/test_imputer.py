"""Testing for imputers."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import re

import numpy as np
import pytest

from pyts.preprocessing import InterpolationImputer

X = [[np.nan, 1, 2, 3, np.nan, 5, 6, np.nan]]


@pytest.mark.parametrize(
    'params, error, err_msg',
    [
        (
            {'missing_values': np.inf},
            ValueError,
            "'missing_values' cannot be infinity.",
        ),
        (
            {'missing_values': "3"},
            ValueError,
            "'missing_values' must be an integer, a float, None or np.nan "
            "(got {!s})".format("3"),
        ),
        (
            {'strategy': 'whoops'},
            ValueError,
            "'strategy' must be an integer or one of 'linear', 'nearest', "
            "'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next' "
            "(got {})".format('whoops'),
        ),
    ],
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    imputer = InterpolationImputer(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        imputer.transform(X)


@pytest.mark.parametrize(
    'params, X, arr_desired',
    [
        (
            {'missing_values': None},
            [[None, 10, 8, None, 4, 2, None]],
            [[12, 10, 8, 6, 4, 2, 0]],
        ),
        (
            {'missing_values': np.nan},
            [[np.nan, 10, 8, np.nan, 4, 2, np.nan]],
            [[12, 10, 8, 6, 4, 2, 0]],
        ),
        (
            {'missing_values': 45.0},
            [[45.0, 10, 8, 45.0, 4, 2, 45.0]],
            [[12, 10, 8, 6, 4, 2, 0]],
        ),
        (
            {'missing_values': 78},
            [[78, 10, 8, 78, 4, 2, 78]],
            [[12, 10, 8, 6, 4, 2, 0]],
        ),
        (
            {'missing_values': None, 'strategy': 'quadratic'},
            [[None, 9, 4, None, 0, 1, None]],
            [[16, 9, 4, 1, 0, 1, 4]],
        ),
        (
            {'missing_values': None, 'strategy': 'previous'},
            [[5, 9, 4, None, 0, 1, None]],
            [[5, 9, 4, 4, 0, 1, 1]],
        ),
        (
            {'missing_values': None, 'strategy': 'next'},
            [[None, 9, 4, None, 0, 1, 8]],
            [[9, 9, 4, 0, 0, 1, 8]],
        ),
        (
            {'missing_values': None, 'strategy': 'nearest'},
            [[None, 9, 4, None, 0, 1, 8]],
            [[9, 9, 4, 4, 0, 1, 8]],
        ),
        (
            {'missing_values': None, 'strategy': 'zero'},
            [[5, 9, 4, None, 0, 1, None]],
            [[5, 9, 4, 4, 0, 1, 1]],
        ),
        (
            {'missing_values': None, 'strategy': 'slinear'},
            [[None, 10, 8, None, 4, 2, None]],
            [[12, 10, 8, 6, 4, 2, 0]],
        ),
        (
            {'missing_values': None, 'strategy': 'cubic'},
            [[None, 0, 1, 8, 27, 64, None]],
            [[-1, 0, 1, 8, 27, 64, 125]],
        ),
        (
            {'missing_values': None, 'strategy': 2},
            [[None, 9, 4, None, 0, 1, None]],
            [[16, 9, 4, 1, 0, 1, 4]],
        ),
    ],
)
def test_actual_results(params, X, arr_desired):
    """Test that the actual results are the expected ones."""
    imputer = InterpolationImputer(**params)
    arr_actual = imputer.fit_transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    'strategy, min_points',
    [
        ('zero', 1),
        ('linear', 2),
        ('slinear', 2),
        ('quadratic', 3),
        ('cubic', 4),
        ('previous', 1),
        ('next', 1),
        ('nearest', 1),
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ],
)
def test_min_points_required(strategy, min_points):
    """Each strategy succeeds with exactly its minimum, raises below it."""
    n_timestamps = min_points + 3
    rng = np.random.default_rng(0)
    values = rng.normal(size=n_timestamps)

    def make_row(n_known):
        row = values.copy()
        keep = rng.choice(n_timestamps, size=n_known, replace=False)
        mask = np.ones(n_timestamps, dtype=bool)
        mask[keep] = False
        row[mask] = np.nan
        return row

    imputer = InterpolationImputer(strategy=strategy)
    # Exactly the minimum number of known points: should succeed.
    imputer.transform([make_row(min_points)])

    # One fewer than the minimum: should raise a clear error.
    n_known = min_points - 1
    err_msg = (
        f"Sample 0 has {n_known} non-missing value(s), but "
        f"strategy={strategy!r} requires at least {min_points}."
    )
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        InterpolationImputer(strategy=strategy).transform([make_row(n_known)])


def test_min_points_required_names_correct_sample():
    """The error message points at the offending row, not just any row."""
    X = [
        [1.0, 2.0, 3.0, 4.0],
        [np.nan, 5.0, np.nan, np.nan],
    ]
    imputer = InterpolationImputer(strategy='linear')
    err_msg = (
        "Sample 1 has 1 non-missing value(s), but strategy='linear' "
        "requires at least 2."
    )
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer.transform(X)
