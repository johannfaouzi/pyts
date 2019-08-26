"""Testing for Symbolic Fourier Approximation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
from sklearn.feature_selection import f_classif
from pyts.approximation import MultipleCoefficientBinning
from pyts.approximation import SymbolicFourierApproximation


rng = np.random.RandomState(42)
n_samples, n_timestamps = 5, 8
X = rng.randn(n_samples, n_timestamps)
y = rng.randint(2, size=n_samples)


def _compute_expected_results(X, y=None, n_coefs=None, n_bins=4,
                              strategy='quantile', drop_sum=False, anova=False,
                              norm_mean=False, norm_std=False, alphabet=None):
    """Compute the expected results."""
    X = np.asarray(X)
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

    mcb = MultipleCoefficientBinning(n_bins=n_bins, strategy=strategy,
                                     alphabet=alphabet)
    arr_desired = mcb.fit_transform(X_fft)
    return arr_desired


@pytest.mark.parametrize(
    'params',
    [({}),
     ({'n_coefs': 3}),
     ({'n_bins': 2}),
     ({'strategy': 'uniform'}),
     ({'drop_sum': True}),
     ({'anova': True}),
     ({'norm_mean': True, 'drop_sum': True}),
     ({'norm_std': True}),
     ({'norm_mean': True, 'norm_std': True, 'drop_sum': True}),
     ({'n_coefs': 2, 'drop_sum': True, 'anova': True})]
)
def test_actual_results(params):
    """Test that the actual results are the expected ones."""
    arr_actual = SymbolicFourierApproximation(**params).fit_transform(X, y)
    arr_desired = _compute_expected_results(X, y, **params)
    np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'params',
    [({}),
     ({'n_coefs': 3}),
     ({'n_bins': 2}),
     ({'strategy': 'uniform'}),
     ({'drop_sum': True}),
     ({'anova': True}),
     ({'norm_mean': True, 'drop_sum': True}),
     ({'norm_std': True}),
     ({'norm_mean': True, 'norm_std': True, 'drop_sum': True}),
     ({'n_coefs': 2, 'drop_sum': True, 'anova': True})]
)
def test_fit_transform(params):
    """Test that fit and transform yield the same results as fit_transform."""
    arr_1 = SymbolicFourierApproximation(**params).fit(X, y).transform(X)
    arr_2 = SymbolicFourierApproximation(**params).fit_transform(X, y)
    np.testing.assert_array_equal(arr_1, arr_2)
