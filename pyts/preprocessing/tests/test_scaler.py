"""Testing for scalers."""

import numpy as np
from ..scaler import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


def test_StandardScaler():
    """Test 'StandardScaler' class."""
    X = np.arange(30).reshape(3, 10)

    # Test 1
    arr_actual = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    arr_desired = (X - X.mean(axis=1)[:, None]) / X.std(axis=1)[:, None]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    arr_actual = StandardScaler(with_mean=False,
                                with_std=True).fit_transform(X)
    arr_desired = X / X.std(axis=1)[:, None]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 3
    arr_actual = StandardScaler(with_mean=True,
                                with_std=False).fit_transform(X)
    arr_desired = X - X.mean(axis=1)[:, None]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 4
    arr_actual = StandardScaler(with_mean=False,
                                with_std=False).fit_transform(X)
    arr_desired = X
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_MinMaxScaler():
    """Test 'MinMaxScaler' class."""
    X = np.arange(30).reshape(3, 10)

    # Test 1
    min_, max_ = 0, 1
    arr_actual = MinMaxScaler(sample_range=(min_, max_)).fit_transform(X)
    arr_desired = (X - X.min(axis=1)[:, None])
    arr_desired = arr_desired / (X.max(axis=1) - X.min(axis=1))[:, None]
    arr_desired = arr_desired * (max_ - min_) + min_
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    min_, max_ = -1, 3
    arr_actual = MinMaxScaler(sample_range=(min_, max_)).fit_transform(X)
    arr_desired = (X - X.min(axis=1)[:, None])
    arr_desired = arr_desired / (X.max(axis=1) - X.min(axis=1))[:, None]
    arr_desired = arr_desired * (max_ - min_) + min_
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_MaxAbsScaler():
    """Test 'MaxAbsScaler' class."""
    X = np.arange(30).reshape(3, 10)
    arr_actual = MaxAbsScaler().fit_transform(X)
    arr_desired = X / np.abs(X).max(axis=1)[:, None]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_RobustScaler():
    """Test 'RobustScaler' class."""
    # Test 1
    X = np.arange(15).reshape(3, 5)
    arr_actual = RobustScaler(
        with_centering=False, with_scaling=False, quantile_range=(25., 75.)
    ).fit_transform(X)
    arr_desired = X
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    X = np.arange(15).reshape(3, 5)
    arr_actual = RobustScaler(
        with_centering=True, with_scaling=False, quantile_range=(25., 75.)
    ).fit_transform(X)
    arr_desired = X - X.mean(axis=1)[:, None]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 3
    X = np.arange(15).reshape(3, 5)
    arr_actual = RobustScaler(
        with_centering=False, with_scaling=True, quantile_range=(25., 75.)
    ).fit_transform(X)
    scale = np.diff(np.percentile(X, (25., 75.), axis=1), axis=0).T
    arr_desired = X / scale
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 4
    X = np.arange(15).reshape(3, 5)
    arr_actual = RobustScaler(
        with_centering=True, with_scaling=True, quantile_range=(25., 75.)
    ).fit_transform(X)
    scale = np.diff(np.percentile(X, (25., 75.), axis=1), axis=0).T
    arr_desired = (X - X.mean(axis=1)[:, None]) / scale
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 5
    X = np.arange(33).reshape(3, 11)
    arr_actual = RobustScaler(
        with_centering=True, with_scaling=True, quantile_range=(10., 90.)
    ).fit_transform(X)
    scale = np.diff(np.percentile(X, (10., 90.), axis=1), axis=0).T
    arr_desired = (X - X.mean(axis=1)[:, None]) / scale
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
