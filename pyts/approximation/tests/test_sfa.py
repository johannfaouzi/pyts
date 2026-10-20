"""Testing for Symbolic Fourier Approximation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
from sklearn.feature_selection import f_classif

from pyts.approximation import (
    DiscreteFourierTransform,
    MultipleCoefficientBinning,
    SymbolicFourierApproximation,
)

rng = np.random.RandomState(42)
n_samples, n_timestamps = 5, 8
X = rng.randn(n_samples, n_timestamps)
y = rng.randint(2, size=n_samples)


def _compute_expected_results(
    X,
    y=None,
    n_coefs=None,
    n_bins=4,
    strategy='quantile',
    drop_sum=False,
    anova=False,
    norm_mean=False,
    norm_std=False,
    alphabet=None,
):
    """Compute the expected results."""
    X = np.array(X, copy=True)
    if norm_mean:
        X -= X.mean(axis=1)[:, None]
    if norm_std:
        X /= X.std(axis=1)[:, None]
    X_fft = np.fft.rfft(X)
    X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
    X_fft = X_fft.reshape(n_samples, -1, order='F')
    if drop_sum:
        X_fft = X_fft[:, 2:-1]
    else:
        X_fft = np.hstack([X_fft[:, :1], X_fft[:, 2:-1]])
    if n_coefs is not None:
        if anova:
            _, p = f_classif(X_fft, y)
            support = np.argsort(p)[:n_coefs]
            X_fft = X_fft[:, support]
        else:
            X_fft = X_fft[:, :n_coefs]

    mcb = MultipleCoefficientBinning(
        n_bins=n_bins, strategy=strategy, alphabet=alphabet
    )
    arr_desired = mcb.fit_transform(X_fft)
    return arr_desired


@pytest.mark.parametrize(
    'params',
    [
        ({}),
        ({'n_coefs': 3}),
        ({'n_bins': 2}),
        ({'strategy': 'uniform'}),
        ({'drop_sum': True}),
        ({'anova': True}),
        ({'norm_mean': True, 'drop_sum': True}),
        ({'norm_std': True}),
        ({'norm_mean': True, 'norm_std': True, 'drop_sum': True}),
        ({'n_coefs': 2, 'drop_sum': True, 'anova': True}),
    ],
)
def test_actual_results(params):
    """Test that the actual results are the expected ones."""
    arr_actual = SymbolicFourierApproximation(**params).fit_transform(X, y)
    arr_desired = _compute_expected_results(X, y, **params)
    np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'params',
    [
        ({}),
        ({'n_coefs': 3}),
        ({'n_bins': 2}),
        ({'strategy': 'uniform'}),
        ({'drop_sum': True}),
        ({'anova': True}),
        ({'norm_mean': True, 'drop_sum': True}),
        ({'norm_std': True}),
        ({'norm_mean': True, 'norm_std': True, 'drop_sum': True}),
        ({'n_coefs': 2, 'drop_sum': True, 'anova': True}),
    ],
)
def test_fit_transform(params):
    """Test that fit and transform yield the same results as fit_transform."""
    arr_1 = SymbolicFourierApproximation(**params).fit(X, y).transform(X)
    arr_2 = SymbolicFourierApproximation(**params).fit_transform(X, y)
    np.testing.assert_array_equal(arr_1, arr_2)


@pytest.mark.parametrize('drop_sum', [False, True])
def test_norm_mean_drops_first_coefficient(drop_sum):
    """With ``norm_mean=True`` and ``anova=False``, the sum is dropped.

    Once each series is centered, the first coefficient is zero up to
    rounding errors, so it must not be binned.
    """
    rng = np.random.RandomState(0)
    X_int = rng.randint(0, 10, (20, 16)).astype('float64')
    sfa = SymbolicFourierApproximation(n_coefs=4, norm_mean=True, drop_sum=drop_sum)
    X_new = sfa.fit_transform(X_int)
    assert X_new.shape == (20, 4)
    expected = SymbolicFourierApproximation(n_coefs=4, norm_mean=True, drop_sum=True)
    np.testing.assert_array_equal(X_new, expected.fit_transform(X_int))
    # No bin edge is built from a zero-variance (rounding noise) coefficient.
    assert np.all(np.ptp(sfa.bin_edges_, axis=1) > 1e-6)


def test_norm_mean_with_anova_keeps_selection():
    """ANOVA already discards the constant first coefficient."""
    rng = np.random.RandomState(0)
    X_rand, y_rand = rng.randn(20, 16), np.repeat([0, 1], 10)
    sfa = SymbolicFourierApproximation(n_coefs=4, norm_mean=True, anova=True)
    sfa.fit(X_rand, y_rand)
    dft = DiscreteFourierTransform(n_coefs=4, norm_mean=True, anova=True)
    dft.fit(X_rand, y_rand)
    np.testing.assert_array_equal(sfa.support_, dft.support_)
