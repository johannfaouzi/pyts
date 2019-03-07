"""Testing for Singular Spectrum Analysis."""

import numpy as np
import pytest
import re
from ..ssa import _outer_dot, _diagonal_averaging, SingularSpectrumAnalysis


def test_outer_dot():
    """Test 'outer_dot' function."""
    v = np.arange(9).reshape(1, 3, 3).astype('float64')
    X = np.asarray([0, 1, 2]).reshape(1, 3, 1).astype('float64')
    n_samples, window_size, n_windows = 1, 3, 1
    arr_actual = _outer_dot(v, X, n_samples, window_size, n_windows)
    arr_desired = np.empty((1, 3, 3, 1))
    for i in range(3):
        np.dot(np.outer(v[0, :, i], v[0, :, i]), X[0])
        arr_desired[0, i] = np.dot(np.outer(v[0, :, i], v[0, :, i]), X[0])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_diagonal_averaging():
    """Test '_diagonal_averaging' function."""
    X = np.arange(6).reshape(1, 1, 2, 3).astype('float64')
    n_samples, n_timestamps, window_size, n_windows = 1, 4, 2, 3
    grouping_size, gap = 1, 3

    arr_actual = _diagonal_averaging(
        X, n_samples, n_timestamps, window_size,
        n_windows, grouping_size, gap
    )
    arr_desired = np.array([0, (1 + 3) / 2, (2 + 4) / 2, 5])
    arr_desired = arr_desired.reshape(1, 1, n_timestamps)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_SingularSpectrumAnalysis():
    """Test 'SingularSpectrumAnalysis' class."""
    rng = np.random.RandomState(41)
    X = rng.randn(10, 48)

    # Parameter check
    msg_error = "'window_size' must be an integer or a float."
    with pytest.raises(TypeError, match=msg_error):
        ssa = SingularSpectrumAnalysis(window_size=None)
        ssa.fit_transform(X)

    msg_error = "'groups' must be either None, an integer or array-like."
    with pytest.raises(TypeError, match=msg_error):
        ssa = SingularSpectrumAnalysis(window_size=4, groups="3")
        ssa.fit_transform(X)

    msg_error = re.escape(
        "If 'window_size' is an integer, it must be greater "
        "than or equal to 2 and lower than or equal to "
        "n_timestamps (got {0}).".format(1)
    )
    with pytest.raises(ValueError, match=msg_error):
        ssa = SingularSpectrumAnalysis(window_size=1, groups=None)
        ssa.fit_transform(X)

    msg_error = re.escape(
        "If 'window_size' is a float, it must be greater "
        "than 0 and lower than or equal to 1 "
        "(got {0}).".format(0.)
    )
    with pytest.raises(ValueError, match=msg_error):
        ssa = SingularSpectrumAnalysis(window_size=0., groups=None)
        ssa.fit_transform(X)

    msg_error = (
        "If 'groups' is an integer, it must be greater than or equal to 1 "
        "and lower than or equal to 'window_size'."
    )
    with pytest.raises(ValueError, match=msg_error):
        ssa = SingularSpectrumAnalysis(window_size=5, groups=6)
        ssa.fit_transform(X)

    msg_error = re.escape(
        "If 'groups' is array-like, all the values in 'groups' "
        "must be integers between 0 and ('window_size' - 1)."
    )
    with pytest.raises(ValueError, match=msg_error):
        ssa = SingularSpectrumAnalysis(window_size=5, groups=[[0, 2, 5]])
        ssa.fit_transform(X)

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
