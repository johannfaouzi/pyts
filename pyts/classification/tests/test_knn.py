"""Testing for k-nearest-neighbors."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import inspect

import numpy as np
import pytest
from numba import njit
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

from pyts.classification import KNeighborsClassifier
from pyts.metrics import dtw

X = np.arange(40).reshape(8, 5)
y = np.array([0, 0, 0, 1, 1, 0, 1, 1])
y_proba = np.vstack([1 - y, y]).T
X_test = X + 0.1


@pytest.mark.parametrize(
    'params',
    [
        ({'metric': 'euclidean'}),
        ({'metric': 'manhattan'}),
        ({'metric': 'dtw'}),
        ({'metric': 'dtw_classic'}),
        ({'metric': 'dtw_sakoechiba'}),
        ({'metric': 'dtw_sakoechiba', 'metric_params': {}}),
        ({'metric': 'dtw_sakoechiba', 'metric_params': {'window_size': 0.5}}),
        ({'metric': 'dtw_itakura'}),
        ({'metric': 'dtw_itakura', 'metric_params': {}}),
        ({'metric': 'dtw_itakura', 'metric_params': {'max_slope': 3.0}}),
        ({'metric': 'dtw_multiscale'}),
        ({'metric': 'dtw_fast'}),
        ({'metric': 'boss'}),
    ],
)
def test_actual_results(params):
    """Test that the actual results are the expected ones."""
    knn = KNeighborsClassifier(n_neighbors=1, **params)
    proba_actual = knn.fit(X, y).predict_proba(X_test)
    pred_actual = knn.predict(X_test)
    np.testing.assert_array_equal(proba_actual, y_proba)
    np.testing.assert_array_equal(pred_actual, y)


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


def test_unknown_parameter_is_rejected_at_construction():
    """Unknown parameters fail immediately instead of at fit time.

    They used to be accepted, then rejected by scikit-learn's own classifier
    when fitting, and silently dropped by ``clone`` and ``get_params``.
    """
    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        KNeighborsClassifier(foo=1)


def test_parameters_match_scikit_learn():
    """Every scikit-learn parameter is exposed and survives ``clone``."""
    sklearn_params = set(inspect.signature(SklearnKNN.__init__).parameters) - {'self'}
    clf = KNeighborsClassifier(n_neighbors=3, leaf_size=10, p=1, n_jobs=2)
    assert sklearn_params <= set(clf.get_params())
    assert clone(clf).get_params() == clf.get_params()
