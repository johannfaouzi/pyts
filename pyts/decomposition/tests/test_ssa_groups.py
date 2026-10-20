"""Testing the grouping of the Singular Spectrum Analysis components."""

# License: BSD-3-Clause

import numpy as np
import pytest

from pyts.decomposition import SingularSpectrumAnalysis

rng = np.random.RandomState(0)
t = np.arange(60)
X = np.r_[
    [np.sin(t / 3.0) * 2 + t / 20.0 + rng.randn(60) * 0.3],
    [np.cos(t / 7.0) + rng.randn(60) * 0.5],
    rng.randn(3, 60),
]


@pytest.mark.parametrize('lower_frequency_bound', [0.05, 0.2, 0.3, 0.35, 0.45, 0.49])
@pytest.mark.parametrize('lower_frequency_contribution', [0.5, 0.85])
@pytest.mark.parametrize('window_size', [10, 11, 20])
def test_auto_groups_partition_the_components(
    lower_frequency_bound, lower_frequency_contribution, window_size
):
    """Each component is in exactly one group, so the groups sum back to X.

    With a ``lower_frequency_bound`` above the fixed residual frequency, a
    component used to be put in both the trend and the residual groups.
    """
    ssa = SingularSpectrumAnalysis(
        window_size=window_size,
        groups='auto',
        lower_frequency_bound=lower_frequency_bound,
        lower_frequency_contribution=lower_frequency_contribution,
    )
    X_new = ssa.fit_transform(X)
    assert X_new.shape == (X.shape[0], 3, X.shape[1])
    np.testing.assert_allclose(X_new.sum(axis=1), X, atol=1e-10)


@pytest.mark.parametrize('groups', [np.int64(2), np.int32(3), np.uint8(1)])
def test_numpy_integer_groups(groups):
    """A NumPy integer number of groups behaves like the Python integer."""
    X_new = SingularSpectrumAnalysis(window_size=6, groups=groups).fit_transform(X)
    expected = SingularSpectrumAnalysis(window_size=6, groups=int(groups))
    np.testing.assert_array_equal(X_new, expected.fit_transform(X))
