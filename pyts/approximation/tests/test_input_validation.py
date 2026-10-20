"""Validation of alphabets, DFT inputs and single-segment PAA."""

# License: BSD-3-Clause

import re

import numpy as np
import pytest

from pyts.approximation import (
    DiscreteFourierTransform,
    MultipleCoefficientBinning,
    PiecewiseAggregateApproximation,
    SymbolicAggregateApproximation,
    SymbolicFourierApproximation,
)
from pyts.datasets import load_gunpoint

X_gp, _, y_gp, _ = load_gunpoint(return_X_y=True)
rng = np.random.RandomState(0)
X_rand = rng.randn(12, 10)


@pytest.mark.parametrize(
    'estimator',
    [
        MultipleCoefficientBinning(n_bins=3),
        SymbolicAggregateApproximation(n_bins=3),
        SymbolicFourierApproximation(n_coefs=4, n_bins=3),
    ],
    ids=lambda estimator: type(estimator).__name__,
)
def test_alphabet_as_numpy_array(estimator):
    """A NumPy array alphabet behaves like the equivalent list."""
    expected = estimator.set_params(alphabet=['x', 'y', 'z']).fit_transform(X_rand)
    actual = estimator.set_params(alphabet=np.array(['x', 'y', 'z'])).fit_transform(
        X_rand
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('scale', [1e-12, 1e-5, 1e8])
def test_dft_anova_is_scale_invariant(scale):
    """ANOVA selects the same coefficients whatever the scale of the data."""
    reference = DiscreteFourierTransform(n_coefs=4, anova=True).fit(X_gp, y_gp)
    scaled = DiscreteFourierTransform(n_coefs=4, anova=True).fit(X_gp * scale, y_gp)
    np.testing.assert_array_equal(scaled.support_, reference.support_)


def test_dft_anova_excludes_rounding_noise():
    """Once the series are centered, their sum is rounding noise only."""
    n_timestamps = X_gp.shape[1]
    dft = DiscreteFourierTransform(
        n_coefs=n_timestamps - 1, anova=True, norm_mean=True
    ).fit(X_gp, y_gp)
    assert 0 not in dft.support_
    assert dft.support_.size == n_timestamps - 1


@pytest.mark.parametrize('anova', [False, True])
def test_dft_transform_checks_n_timestamps(anova):
    """Series of another length than in ``fit`` are rejected."""
    dft = DiscreteFourierTransform(n_coefs=4, anova=anova).fit(X_gp, y_gp)
    msg = (
        "The number of timestamps in X must be the same as the number of "
        "timestamps when `fit` was called (150 != 100)."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        dft.transform(X_gp[:, :100])


@pytest.mark.parametrize('output_size', [1, 0.1])
@pytest.mark.parametrize('overlapping', [False, True])
def test_paa_single_segment(output_size, overlapping):
    """A single segment is the mean of the whole series."""
    paa = PiecewiseAggregateApproximation(
        window_size=None, output_size=output_size, overlapping=overlapping
    )
    np.testing.assert_allclose(
        paa.transform(X_rand), X_rand.mean(axis=1, keepdims=True)
    )
