"""Testing for WEASELMUSE class."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyts.multivariate.transformation import WEASELMUSE

n_samples, n_features, n_timestamps, n_classes = 40, 3, 100, 2
rng = np.random.RandomState(42)
X = rng.randn(n_samples, n_features, n_timestamps)
y = rng.randint(n_classes, size=n_samples)


@pytest.mark.parametrize(
    'params, type_desired',
    [({'sparse': True}, csr_matrix), ({'sparse': False}, np.ndarray)],
)
def test_output_dtype(params, type_desired):
    """Check that the output dtype is the expected one."""
    transformer = WEASELMUSE(**params)
    output = transformer.fit_transform(X, y)
    assert isinstance(output, type_desired)


@pytest.mark.parametrize('params', [{'sparse': True}, {'sparse': False}])
def test_output_ndim(params):
    """Check that the number of dimensions is always 2."""
    transformer = WEASELMUSE(**params)
    ndim_actual = transformer.fit_transform(X, y).ndim
    assert ndim_actual == 2


def test_n_estimators():
    """Check that the number of estimators is the number of features."""
    transformer = WEASELMUSE().fit(X, y)
    assert len(transformer._estimators) == n_features
    assert len(transformer._estimators_diff) == n_features


@pytest.mark.parametrize('params', [{'sparse': True}, {'sparse': False}])
def test_fit_transform(params):
    """Check that fit and transform and fit_transform yield same results."""
    transformer = WEASELMUSE(**params)
    arr_1 = transformer.fit(X, y).transform(X)
    arr_2 = transformer.fit_transform(X, y)
    if transformer.sparse:
        assert (arr_1 != arr_2).nnz == 0
    else:
        np.testing.assert_array_equal(arr_1, arr_2)


@pytest.mark.parametrize('dtype', [np.uint8, np.int8])
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('method', ['fit', 'transform', 'fit_transform'])
def test_integer_differences(dtype, sparse, method):
    """Convert integers before differencing in each public entry point."""
    rng = np.random.RandomState(42)
    X_integer = rng.randint(0, 256, size=(12, 2, 12)).astype(dtype)
    X_original = X_integer.copy()
    X_float = X_integer.astype(np.float64)
    y = np.arange(12) % 2
    params = {
        'word_size': 2,
        'n_bins': 2,
        'window_sizes': [4],
        'strategy': 'uniform',
        'chi2_threshold': 0.1,
        'sparse': sparse,
    }
    reference = WEASELMUSE(**params)
    expected = reference.fit_transform(X_float, y)
    transformer = WEASELMUSE(**params)
    if method == 'fit':
        actual = transformer.fit(X_integer, y).transform(X_float)
    elif method == 'transform':
        actual = transformer.fit(X_float, y).transform(X_integer)
    else:
        actual = transformer.fit_transform(X_integer, y)

    assert transformer.vocabulary_ == reference.vocabulary_
    if sparse:
        actual, expected = actual.toarray(), expected.toarray()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(X_integer, X_original)
