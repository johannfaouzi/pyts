"""Testing for Singular Spectrum Analysis."""

import numpy as np
from ..ssa import SingularSpectrumAnalysis


def test_SingularSpectrumAnalysis():
    """Testing 'SingularSpectrumAnalysis' class."""
    rng = np.random.RandomState(41)
    X = rng.randn(10, 48)

    # Test 1
    ssa = SingularSpectrumAnalysis(window_size=8)
    arr_actual = ssa.fit_transform(X).sum(axis=1)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 2
    ssa = SingularSpectrumAnalysis(window_size=2)
    arr_actual = ssa.fit_transform(X).sum(axis=1)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 3
    ssa = SingularSpectrumAnalysis(window_size=8)
    arr_actual = ssa.fit_transform(X).sum(axis=1)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 4
    ssa = SingularSpectrumAnalysis(window_size=8)
    arr_actual = ssa.fit_transform(X).sum(axis=1)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 5: window_size
    for new_window_size in range(1, 11):
        arr_actual = ssa.fit_transform(X).sum(axis=1)
        np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 6: groups (None)
    ssa = SingularSpectrumAnalysis(window_size=2, groups=None)
    arr_actual = arr_actual = ssa.fit_transform(X).sum(axis=1)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 7: groups (integer)
    for groups in range(1, 11):
        ssa = SingularSpectrumAnalysis(window_size=10, groups=groups)
        if groups == 1:
            arr_actual = ssa.fit_transform(X)
        else:
            arr_actual = ssa.fit_transform(X).sum(axis=1)
        np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)

    # Test 8: groups (array-like)
    groups = [[0, 1, 2], [3, 4]]
    ssa = SingularSpectrumAnalysis(window_size=5, groups=groups)
    arr_actual = ssa.fit_transform(X).sum(axis=1)
    np.testing.assert_allclose(arr_actual, X, atol=1e-5, rtol=0.)
