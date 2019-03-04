"""Testing for Discrete Fourier Transform."""

from itertools import product
import numpy as np
from ..dft import DiscreteFourierTransform


def test_DiscreteFourierTransform():
    """Test 'DiscreteFourierTransform' class."""
    rng = np.random.RandomState(42)
    X = rng.randn(5, 8)
    y = [0, 0, 0, 1, 1]

    # Parameter check
    def type_error_list():
        type_error_list_ = ["'n_coefs' must be None, an integer or a float."]
        return type_error_list_

    def value_error_list():
        value_error_list_ = [
            "If 'n_coefs' is an integer, it must be greater than or "
            "equal to 1 and lower than or equal to n_timestamps if "
            "'drop_sum=False'.",
            "If 'n_coefs' is an integer, it must be greater than or "
            "equal to 1 and lower than or equal to (n_timestamps - 1) "
            "if 'drop_sum=True'.",
            "If 'n_coefs' is a float, it must be greater "
            "than 0 and lower than or equal to 1."
        ]
        return value_error_list_

    n_coefs_list = [-1, -1., 0.5, 2, None, "str"]
    drop_sum_list = [True, False]
    anova_list = [True, False]
    norm_mean_list = [True, False]
    norm_std_list = [True, False]
    for (n_coefs, drop_sum, anova, norm_mean, norm_std) in product(
        n_coefs_list, drop_sum_list, anova_list, norm_mean_list, norm_std_list
    ):
        dft = DiscreteFourierTransform(
            n_coefs, drop_sum, anova, norm_mean, norm_std)
        try:
            dft.fit(X, y).transform(X)
            dft.fit_transform(X, y)
        except ValueError as e:
            if str(e) in value_error_list():
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) in type_error_list():
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))

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
