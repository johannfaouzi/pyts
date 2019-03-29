"""Testing for Symbolic Fourier Approximation."""

import numpy as np
from ..sfa import SymbolicFourierApproximation


def test_SymbolicFourierApproximation():
    """Test 'SymbolicFourierApproximation' class."""
    # Test 1
    rng = np.random.RandomState(41)
    X = rng.randn(5, 8)

    sfa = SymbolicFourierApproximation(
        n_coefs=4, drop_sum=True, anova=False, norm_mean=False, norm_std=False,
        n_bins=2, strategy='uniform', alphabet='ordinal')
    X_fft = np.fft.rfft(X)
    X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
    X_fft = X_fft.reshape(5, 10, order='F')
    X_fft = X_fft[:, 2:6]

    feature_min, feature_max = np.min(X_fft, axis=0), np.max(X_fft, axis=0)
    bin_edges = np.empty((4, 1))
    for i in range(4):
        bin_edges[i] = np.linspace(feature_min[i], feature_max[i], 3)[1:-1]
    arr_desired = np.empty((5, 4))
    for i in range(4):
        arr_desired[:, i] = np.digitize(X_fft[:, i], bin_edges[i])
    np.testing.assert_array_equal(sfa.fit_transform(X), arr_desired)
    np.testing.assert_array_equal(sfa.fit(X).transform(X), arr_desired)

    # Test 2
    rng = np.random.RandomState(41)
    X = rng.randn(5, 8)

    sfa = SymbolicFourierApproximation(
        n_coefs=4, drop_sum=True, anova=False, norm_mean=False, norm_std=False,
        n_bins=4, strategy='uniform', alphabet='ordinal')
    X_fft = np.fft.rfft(X)
    X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
    X_fft = X_fft.reshape(5, 10, order='F')
    X_fft = X_fft[:, 2:6]

    feature_min, feature_max = np.min(X_fft, axis=0), np.max(X_fft, axis=0)
    bin_edges = np.empty((4, 3))
    for i in range(4):
        bin_edges[i] = np.linspace(feature_min[i], feature_max[i], 5)[1:-1]
    arr_desired = np.empty((5, 4))
    for i in range(4):
        arr_desired[:, i] = np.digitize(X_fft[:, i], bin_edges[i])
    np.testing.assert_array_equal(sfa.fit_transform(X), arr_desired)
    np.testing.assert_array_equal(sfa.fit(X).transform(X), arr_desired)
