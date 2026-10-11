"""Grouping uses the resolved SSA window, including fractional windows."""

from math import ceil

import numpy as np
import pytest

from pyts.decomposition import SingularSpectrumAnalysis


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
