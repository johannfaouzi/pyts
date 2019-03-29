"""Testing for transformers."""

import numpy as np
from itertools import product
from scipy.stats import norm
from ..transformer import PowerTransformer, QuantileTransformer


def test_PowerTransformer():
    """Test 'PowerTransformer' class."""
    X = np.arange(30).reshape(3, 10)
    method_list = ['yeo-johnson', 'box-cox']
    standardize_list = [True, False]
    for (method, standardize) in product(
        method_list, standardize_list
    ):
        PowerTransformer().fit_transform(X)


def test_QuantileTransformer():
    """Test 'QuantileTransformer' class."""
    # Test 1: uniform distribution
    X = np.arange(33).reshape(3, 11)
    arr_actual = QuantileTransformer(
        n_quantiles=10, output_distribution='uniform'
    ).fit_transform(X)
    arr_desired = (X - X.min(axis=1)[:, None])
    arr_desired = arr_desired / (X.max(axis=1) - X.min(axis=1))[:, None]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Test 2: normal distribution
    X_ppf = norm.ppf(np.linspace(0, 1, 1000)[1:-1])
    weights = np.round(norm.pdf(X_ppf) * 1000).astype('int64')
    X = []
    for value, weight in zip(X_ppf, weights):
        X += [value] * weight
    X = np.asarray(X).reshape(1, -1)
    arr_actual = QuantileTransformer(
        n_quantiles=100, output_distribution='normal'
    ).fit_transform(X)
    arr_desired = X
    atol = 0.01 * X.shape[1]
    np.testing.assert_allclose(arr_actual, arr_desired, atol=atol)
