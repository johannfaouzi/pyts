"""Testing for scalers."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
from pyts.preprocessing import (StandardScaler, MinMaxScaler,
                                MaxAbsScaler, RobustScaler)


X = np.arange(10).reshape(2, 5)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'with_mean': True, 'with_std': True},
      (X - X.mean(axis=1)[:, None]) / X.std(axis=1)[:, None]),

     ({'with_mean': True, 'with_std': False}, X - X.mean(axis=1)[:, None]),

     ({'with_mean': False, 'with_std': True}, X / X.std(axis=1)[:, None]),

     ({'with_mean': False, 'with_std': False}, X)]
)
def test_actual_results_standard_scaler(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = StandardScaler(**params).transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'sample_range': (0, 1)}, [[0, 0.25, 0.5, 0.75, 1] for _ in range(2)]),

     ({'sample_range': (-1, 1)}, [[-1, -0.5, 0, 0.5, 1] for _ in range(2)]),

     ({'sample_range': (0, 4)}, [[0, 1, 2, 3, 4] for _ in range(2)]),

     ({'sample_range': (-10, 10)}, [[-10, -5, 0, 5, 10] for _ in range(2)])]
)
def test_actual_results_min_max_scaler(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = MinMaxScaler(**params).transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_actual_results_max_abs_scaler():
    """Test that the actual results are the expected ones."""
    arr_actual = MaxAbsScaler().transform(X)
    arr_desired = X / np.abs(X).max(axis=1)[:, None]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({}, [[-1, -0.5, 0, 0.5, 1] for _ in range(2)]),

     ({'quantile_range': (0., 100.)},
      [[-0.5, -0.25, 0, 0.25, 0.5] for _ in range(2)]),

     ({'with_centering': False}, X / 2),

     ({'with_scaling': False}, [[-2, -1, 0, 1, 2] for _ in range(2)]),

     ({'with_centering': False, 'with_scaling': False}, X)]
)
def test_actual_results_robust_scaler(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = RobustScaler(**params).fit_transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
