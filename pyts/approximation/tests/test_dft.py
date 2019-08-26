"""Testing for Discrete Fourier Transform."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from sklearn.feature_selection import f_classif
from pyts.approximation import DiscreteFourierTransform


rng = np.random.RandomState(42)
n_samples, n_timestamps = 5, 8
X = rng.randn(n_samples, n_timestamps)
y = rng.randint(2, size=n_samples)


def _compute_expected_results(X, y=None, n_coefs=None, drop_sum=False,
                              anova=False, norm_mean=False, norm_std=False):
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
    if n_coefs is None:
        return X_fft
    else:
        if anova:
            _, p = f_classif(X_fft, y)
            support = np.argsort(p)[:n_coefs]
            return X_fft[:, support]
        else:
            return X_fft[:, :n_coefs]


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_coefs': '3'}, TypeError,
      "'n_coefs' must be None, an integer or a float."),

     ({'n_coefs': 0}, ValueError,
      "If 'n_coefs' is an integer, it must be greater than or equal to 1 and "
      "lower than or equal to n_timestamps if 'drop_sum=False'."),

     ({'n_coefs': 8, 'drop_sum': True}, ValueError,
      "If 'n_coefs' is an integer, it must be greater than or equal to 1 and "
      "lower than or equal to (n_timestamps - 1) if 'drop_sum=True'."),

     ({'n_coefs': 2.}, ValueError,
      ("If 'n_coefs' is a float, it must be greater than 0 and lower than or "
       "equal to 1."))]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    dft = DiscreteFourierTransform(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        dft.fit(X, y)


@pytest.mark.parametrize(
    'params',
    [({}),
     ({'n_coefs': 3}),
     ({'drop_sum': True}),
     ({'anova': True}),
     ({'norm_mean': True}),
     ({'norm_std': True}),
     ({'norm_mean': True, 'norm_std': True}),
     ({'n_coefs': 2, 'drop_sum': True, 'anova': True})]
)
def test_actual_results(params):
    """Test that the actual results are the expected ones."""
    arr_actual = DiscreteFourierTransform(**params).fit_transform(X, y)
    arr_desired = _compute_expected_results(X, y, **params)
    np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'params',
    [({}),
     ({'n_coefs': 3}),
     ({'n_coefs': 0.5}),
     ({'drop_sum': True}),
     ({'anova': True}),
     ({'norm_mean': True}),
     ({'norm_std': True}),
     ({'norm_mean': True, 'norm_std': True}),
     ({'n_coefs': 2, 'drop_sum': True, 'anova': True})]
)
def test_fit_transform(params):
    """Test that fit and transform yield the same results as fit_transform."""
    arr_1 = DiscreteFourierTransform(**params).fit(X, y).transform(X)
    arr_2 = DiscreteFourierTransform(**params).fit_transform(X, y)
    np.testing.assert_array_equal(arr_1, arr_2)
