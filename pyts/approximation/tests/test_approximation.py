"""Tests for :mod:`pyts.approximation` module."""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
from itertools import product
import numpy as np
from ..approximation import PAA, DFT


standard_library.install_aliases()


def test_PAA():
    """Testing 'PAA'."""
    # Parameter
    X = np.arange(30).astype('float64')

    # Test 1
    paa = PAA(window_size=2, overlapping=True)
    arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
    arr_desired = np.arange(0.5, 30, 2)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    paa = PAA(window_size=3, overlapping=False)
    arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
    arr_desired = np.arange(1, 30, 3)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 3
    paa = PAA(window_size=5)
    arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
    arr_desired = np.arange(2, 30, 5)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 4
    paa = PAA(output_size=10, overlapping=True)
    arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
    arr_desired = np.arange(1, 30, 3)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 5
    paa = PAA(output_size=10, overlapping=False)
    arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
    arr_desired = np.arange(1, 30, 3)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 6
    paa = PAA(window_size=4, overlapping=True)
    arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
    arr_desired = np.array([1.5, 4.5, 8.5, 12.5, 15.5, 19.5, 23.5, 27.5])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_DFT():
    """Testing 'DFT'."""
    # Parameters
    X = np.arange(40).reshape(4, 10).astype('float64')
    y = np.repeat(np.arange(2), 2)
    X_fft = np.fft.fft(X)
    X_fft = np.vstack([np.real(X_fft),
                       np.imag(X_fft)]).reshape(4, -1, order='F')

    # Test 1
    dft = DFT(n_coefs=None, anova=False, norm_mean=False, norm_std=False)
    arr_actual = dft.fit_transform(X, y)
    np.testing.assert_allclose(arr_actual, X_fft, atol=1e-5, rtol=0.)

    # Test 2
    dft = DFT(n_coefs=None, anova=False, norm_mean=True, norm_std=False)
    arr_actual = dft.fit_transform(X)
    X_mean = X - X.mean(axis=1).reshape(4, 1)
    X_mean_fft = np.fft.fft(X_mean)
    X_mean_fft = np.vstack([np.real(X_mean_fft),
                            np.imag(X_mean_fft)]).reshape(4, -1, order='F')
    np.testing.assert_allclose(arr_actual, X_mean_fft, atol=1e-5, rtol=0.)

    # Test 3
    dft = DFT(n_coefs=None, anova=False, norm_mean=False, norm_std=True)
    arr_actual = dft.fit_transform(X)
    X_std = X / X.std(axis=1).reshape(4, 1)
    X_std_fft = np.fft.fft(X_std)
    X_std_fft = np.vstack([np.real(X_std_fft),
                           np.imag(X_std_fft)]).reshape(4, -1, order='F')
    np.testing.assert_allclose(arr_actual, X_std_fft, atol=1e-5, rtol=0.)

    # Test 4
    dft = DFT(n_coefs=None, anova=False, norm_mean=True, norm_std=True)
    arr_actual = dft.fit_transform(X)
    X_mean_std = X_mean / X.std(axis=1).reshape(4, 1)
    X_mean_std_fft = np.fft.fft(X_mean_std)
    X_mean_std_fft = np.vstack([np.real(X_mean_std_fft),
                                np.imag(X_mean_std_fft)]
                               ).reshape(4, -1, order='F')
    np.testing.assert_allclose(arr_actual, X_mean_std_fft, atol=1e-5, rtol=0.)

    # Test 5
    dft = DFT(n_coefs=6, anova=False, norm_mean=False, norm_std=False)
    arr_actual = dft.fit_transform(X)
    np.testing.assert_allclose(arr_actual, X_fft[:, :6], atol=1e-5, rtol=0.)

    # Test 6
    dft = DFT(n_coefs=6, anova=False, norm_mean=False, norm_std=False)
    arr_actual = dft.fit_transform(X)
    np.testing.assert_allclose(arr_actual, X_fft[:, :6], atol=1e-5, rtol=0.)

    # Test 7
    dft = DFT(n_coefs=2, anova=False, norm_mean=True, norm_std=False)
    arr_actual = dft.fit_transform(X)
    np.testing.assert_allclose(arr_actual, X_fft[:, 2:4], atol=1e-5, rtol=0.)

    # Test: loop
    rng = np.random.RandomState(41)
    X_noise = X + (rng.randn(4, 10) / 100)
    y = [0, 1, 0, 1]
    n_coefs_list = [None, 2]
    anova_list = [True, False]
    norm_mean_list = [True, False]
    norm_std_list = [True, False]
    for (n_coefs, anova, norm_mean,
         norm_std) in product(*[n_coefs_list, anova_list,
                                norm_mean_list, norm_std_list]):
        dft = DFT(n_coefs, anova, norm_mean, norm_std)
        dft.fit_transform(X_noise, y)
        dft.fit(X_noise, y).transform(X_noise)
