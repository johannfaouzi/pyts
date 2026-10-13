"""Testing for Singular Spectrum Analysis."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import re
from math import ceil

import numpy as np
import pytest

from pyts.decomposition import SingularSpectrumAnalysis

rng = np.random.RandomState(42)
X = rng.randn(4, 30)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [
        (
            {'window_size': '4'},
            TypeError,
            "'window_size' must be an integer or a float.",
        ),
        (
            {'groups': '3'},
            TypeError,
            "'groups' must be either None, an integer, 'auto' or array-like.",
        ),
        (
            {'lower_frequency_bound': 1},
            TypeError,
            "'lower_frequency_bound' must be a float.",
        ),
        (
            {'lower_frequency_contribution': 1},
            TypeError,
            "'lower_frequency_contribution' must be a float.",
        ),
        (
            {'chunksize': 1.0},
            TypeError,
            "'chunksize' must be None or an integer.",
        ),
        (
            {'window_size': 1},
            ValueError,
            "If 'window_size' is an integer, it must be greater than or equal to 2 "
            "and lower than or equal to n_timestamps (got 1).",
        ),
        (
            {'window_size': 0.0},
            ValueError,
            "If 'window_size' is a float, it must be greater than 0 and lower than "
            f"or equal to 1 (got {0.0}).",
        ),
        (
            {'groups': 6},
            ValueError,
            "If 'groups' is an integer, it must be greater than or equal to 1 "
            "and lower than or equal to 'window_size'.",
        ),
        (
            {'groups': [[0, 2, 5]]},
            ValueError,
            "If 'groups' is array-like, all the values in 'groups' must be integers "
            "between 0 and ('window_size' - 1).",
        ),
        (
            {'lower_frequency_bound': 0.6},
            ValueError,
            "'lower_frequency_bound' must be greater than 0 and lower than 0.5.",
        ),
        (
            {'lower_frequency_contribution': 1.2},
            ValueError,
            "'lower_frequency_contribution' must be greater than 0 and lower than 1.",
        ),
        (
            {'chunksize': 0},
            ValueError,
            "If 'chunksize' is an integer, it must be positive (got 0)",
        ),
        (
            {'n_jobs': 0},
            ValueError,
            "'n_jobs' must be None or an integer not equal to zero (got 0).",
        ),
        (
            {'n_jobs': [1, 2]},
            ValueError,
            "'n_jobs' must be None or an integer not equal to zero (got [1, 2]).",
        ),
    ],
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    ssa = SingularSpectrumAnalysis(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        ssa.transform(X)


@pytest.mark.parametrize(
    'params',
    [
        ({}),
        ({'window_size': 2}),
        ({'window_size': 25}),
        ({'window_size': 0.2}),
        ({'groups': 3}),
        ({'window_size': 10, 'groups': 10}),
        ({'groups': [[0, 1], [2, 3]]}),
        ({'groups': [[0, 2], [1, 3]]}),
        ({'groups': np.array([[0, 1], [2, 3]])}),
        ({'groups': np.array([[0, 2], [1, 3]])}),
        ({'groups': np.array([[0], [1, 2, 3]], dtype=object)}),
        ({'groups': np.array([[0, 2], [1, 3]]), 'chunksize': 1}),
        ({'window_size': 8, 'groups': [[0, 2, 4, 6], [1, 3, 5, 7]]}),
        ({'groups': 'auto'}),
        ({'window_size': 2, 'groups': 'auto'}),
        ({'window_size': 5, 'groups': 'auto'}),
        ({'groups': 'auto', 'lower_frequency_contribution': 0.99}),
        ({'groups': 'auto', 'lower_frequency_bound': 0.01}),
        ({'chunksize': 1}),
        ({'chunksize': 2}),
        ({'chunksize': 2, 'n_jobs': 1}),
        ({'chunksize': 2, 'n_jobs': 2}),
    ],
)
def test_actual_results(params):
    """Test that the actual results are the expected ones."""
    ssa = SingularSpectrumAnalysis(**params)
    arr_actual = ssa.fit_transform(X).sum(axis=1)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize('fraction', [0.001, 0.21, 0.5, 1.0])
@pytest.mark.parametrize('group_kind', ['single', 'count', 'indices'])
@pytest.mark.parametrize('chunksize', [None, 2])
def test_fractional_window_groups(fraction, group_kind, chunksize):
    X = np.random.RandomState(42).normal(size=(5, 30))
    window_size = max(2, ceil(fraction * X.shape[1]))
    groups = {
        'single': 1,
        'count': 2,
        'indices': [[0], list(range(1, window_size))],
    }[group_kind]
    actual = SingularSpectrumAnalysis(
        window_size=fraction, groups=groups, chunksize=chunksize
    ).fit_transform(X)
    expected = SingularSpectrumAnalysis(
        window_size=window_size, groups=groups, chunksize=chunksize
    ).fit_transform(X)
    np.testing.assert_array_equal(actual, expected)
    reconstructed = actual if group_kind == 'single' else actual.sum(axis=1)
    np.testing.assert_allclose(reconstructed, X, atol=1e-12)


@pytest.mark.parametrize('fraction', [0.001, 0.21, 1.0])
@pytest.mark.parametrize('group_kind', ['count', 'indices'])
def test_fractional_window_rejects_out_of_range_groups(fraction, group_kind):
    X = np.ones((2, 30))
    window_size = max(2, ceil(fraction * X.shape[1]))
    groups = window_size + 1 if group_kind == 'count' else [[window_size]]
    with pytest.raises(ValueError, match="If 'groups'"):
        SingularSpectrumAnalysis(window_size=fraction, groups=groups).transform(X)
