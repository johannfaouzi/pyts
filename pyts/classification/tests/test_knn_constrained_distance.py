"""Constrained DTW neighbors respect their pointwise distance."""

import numpy as np
import pytest
from numba import njit
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

from pyts.classification import KNeighborsClassifier
from pyts.metrics import dtw


@njit
def absolute_distance(x, y):
    return abs(x - y)


@pytest.mark.parametrize(
    'method, options',
    [('sakoechiba', {'window_size': 0.5}), ('itakura', {'max_slope': 3.0})],
)
@pytest.mark.parametrize('dist', ['absolute', absolute_distance, 'square'])
def test_constrained_pointwise_distance(method, options, dist):
    rng = np.random.RandomState(42)
    X = rng.normal(size=(6, 8))
    X_test = rng.normal(size=(3, 8))
    y = np.array(['a', 'b', 'b', 'a', 'b', 'a'])
    params = dict(options, dist=dist)
    classifier = KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        metric='dtw_' + method,
        metric_params=params,
    ).fit(X, y)
    train_distances = np.array(
        [[dtw(x, z, method=method, options=options, dist=dist) for z in X] for x in X]
    )
    test_distances = np.array(
        [
            [dtw(x, z, method=method, options=options, dist=dist) for z in X]
            for x in X_test
        ]
    )
    reference = SklearnKNN(n_neighbors=3, weights='distance', metric='precomputed').fit(
        train_distances, y
    )
    distances, indices = classifier._clf.kneighbors(X_test)
    expected_distances, expected_indices = reference.kneighbors(test_distances)
    np.testing.assert_allclose(distances, expected_distances)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(
        classifier.predict_proba(X_test), reference.predict_proba(test_distances)
    )
    np.testing.assert_array_equal(
        classifier.predict(X_test), reference.predict(test_distances)
    )
    assert params == dict(options, dist=dist)
