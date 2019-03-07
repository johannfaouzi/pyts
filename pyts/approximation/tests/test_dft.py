"""Testing for Discrete Fourier Transform."""

import numpy as np
import pytest
import re
from ..dft import DiscreteFourierTransform


def test_DiscreteFourierTransform():
    """Test 'DiscreteFourierTransform' class."""
    rng = np.random.RandomState(42)
    X = rng.randn(5, 8)
    y = [0, 0, 0, 1, 1]

    # Parameter check
    msg_error = "'n_coefs' must be None, an integer or a float."
    with pytest.raises(TypeError, match=msg_error):
        dft = DiscreteFourierTransform(
            n_coefs='-1', drop_sum=False, anova=False, norm_mean=False,
            norm_std=False)
        dft.fit(X, y).transform(X)

    msg_error = ("If 'n_coefs' is an integer, it must be greater than or "
                 "equal to 1 and lower than or equal to n_timestamps if "
                 "'drop_sum=False'.")
    with pytest.raises(ValueError, match=msg_error):
        dft = DiscreteFourierTransform(
            n_coefs=0, drop_sum=False, anova=False, norm_mean=False,
            norm_std=False)
        dft.fit(X, y).transform(X)

    msg_error = re.escape(
        "If 'n_coefs' is an integer, it must be greater than or "
        "equal to 1 and lower than or equal to (n_timestamps - 1) "
        "if 'drop_sum=True'."
    )
    with pytest.raises(ValueError, match=msg_error):
        dft = DiscreteFourierTransform(
            n_coefs=10, drop_sum=True, anova=False, norm_mean=False,
            norm_std=False)
        dft.fit(X, y).transform(X)

    msg_error = ("If 'n_coefs' is a float, it must be greater "
                 "than 0 and lower than or equal to 1.")
    with pytest.raises(ValueError, match=msg_error):
        dft = DiscreteFourierTransform(
            n_coefs=2., drop_sum=False, anova=False, norm_mean=False,
            norm_std=False)
        dft.fit(X, y).transform(X)

    # Test 1
    X = rng.randn(5, 8)
    n_coefs, drop_sum, anova = None, True, False
    norm_mean, norm_std = False, False
    dft = DiscreteFourierTransform(n_coefs, drop_sum, anova,
                                   norm_mean, norm_std)
    arr_actual = dft.fit_transform(X)
    X_fft = np.fft.rfft(X)
    X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
    X_fft = X_fft.reshape(5, 10, order='F')
    arr_desired = X_fft[:, 2:-1]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2
    n_coefs, drop_sum, anova = None, True, False
    norm_mean, norm_std = False, False
    dft = DiscreteFourierTransform(n_coefs, drop_sum, anova,
                                   norm_mean, norm_std)
    arr_actual = dft.fit_transform(X)
    X_fft = np.fft.rfft(X)
    X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
    X_fft = X_fft.reshape(5, 10, order='F')
    arr_desired = X_fft[:, 2:-1]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 3
    n_coefs, drop_sum, anova = None, False, False
    norm_mean, norm_std = False, False
    dft = DiscreteFourierTransform(n_coefs, drop_sum, anova,
                                   norm_mean, norm_std)
    arr_actual = dft.fit_transform(X)
    X_fft = np.fft.rfft(X)
    X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
    X_fft = X_fft.reshape(5, 10, order='F')
    arr_desired = np.hstack([X_fft[:, :1], X_fft[:, 2:-1]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 4
    n_coefs, drop_sum, anova = None, False, False
    norm_mean, norm_std = True, False
    dft = DiscreteFourierTransform(n_coefs, drop_sum, anova,
                                   norm_mean, norm_std)
    arr_actual = dft.fit_transform(X)
    X = X - X.mean(axis=1)[:, None]
    X_fft = np.fft.rfft(X)
    X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
    X_fft = X_fft.reshape(5, 10, order='F')
    arr_desired = np.hstack([X_fft[:, :1], X_fft[:, 2:-1]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 5
    n_coefs, drop_sum, anova = None, False, False
    norm_mean, norm_std = True, True
    dft = DiscreteFourierTransform(n_coefs, drop_sum, anova,
                                   norm_mean, norm_std)
    arr_actual = dft.fit_transform(X)
    X = (X - X.mean(axis=1)[:, None]) / X.std(axis=1)[:, None]
    X_fft = np.fft.rfft(X)
    X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
    X_fft = X_fft.reshape(5, 10, order='F')
    arr_desired = np.hstack([X_fft[:, :1], X_fft[:, 2:-1]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 6
    n_coefs, drop_sum, anova = 3, False, False
    norm_mean, norm_std = False, False
    dft = DiscreteFourierTransform(n_coefs, drop_sum, anova,
                                   norm_mean, norm_std)
    arr_actual = dft.fit_transform(X)
    X_fft = np.fft.rfft(X)
    X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
    X_fft = X_fft.reshape(5, 10, order='F')
    X_fft = np.hstack([X_fft[:, :1], X_fft[:, 2:-1]])
    arr_desired = X_fft[:, :3]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
