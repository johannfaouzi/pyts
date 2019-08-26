"""Testing for transformers."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from scipy.stats import boxcox, norm, yeojohnson
from pyts.preprocessing import StandardScaler
from pyts.preprocessing import PowerTransformer, QuantileTransformer


X = np.arange(1, 34).reshape(3, 11)


def test_actual_results_power_transformer_box_cox():
    """Test that the actual results are the expected ones."""
    for standardize in [True, False]:
        pt = PowerTransformer(method='box-cox', standardize=standardize)
        arr_actual = pt.transform(X)
        arr_desired = [boxcox(X[i])[0] for i in range(3)]
        if standardize:
            arr_desired = StandardScaler().transform(arr_desired)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_actual_results_power_transformer_yeo_johnson():
    """Test that the actual results are the expected ones."""
    for standardize in [True, False]:
        pt = PowerTransformer(method='yeo-johnson', standardize=standardize)
        arr_actual = pt.transform(X)
        arr_desired = [yeojohnson(X[i].astype('float64'))[0] for i in range(3)]
        if standardize:
            arr_desired = StandardScaler().transform(arr_desired)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_actual_results_quantile_transformer_uniform():
    """Test that the actual results are the expected ones."""
    transformer = QuantileTransformer(n_quantiles=11)
    arr_actual = transformer.fit_transform(X)
    arr_desired = [np.linspace(0, 1, 11) for _ in range(3)]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_actual_results_quantile_transformer_normal():
    """Test that the actual results are the expected ones."""
    X_ppf = norm.ppf(np.linspace(0, 1, 1000)[1:-1])
    weights = np.round(norm.pdf(X_ppf) * 1000).astype('int64')
    X = []
    for value, weight in zip(X_ppf, weights):
        X += [value] * weight
    X = np.asarray(X).reshape(1, -1)
    transformer = QuantileTransformer(n_quantiles=11,
                                      output_distribution='normal')
    arr_actual = transformer.transform(X)
    arr_desired = X
    atol = 0.01 * X.shape[1]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=atol, rtol=0.)
