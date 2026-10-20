"""Testing for MultivariateClassifier."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import re

import numpy as np
import pytest

from pyts.classification import BOSSVS, SAXVSM
from pyts.multivariate.classification import MultivariateClassifier
from pyts.transformation import BOSS

n_samples, n_features, n_timestamps, n_classes = 40, 3, 30, 2
rng = np.random.RandomState(42)
X = rng.randn(n_samples, n_features, n_timestamps)
y = rng.randint(n_classes, size=n_samples)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [
        (
            {'estimator': [SAXVSM(), SAXVSM(), BOSS()]},
            ValueError,
            "Estimator 2 must be a classifier.",
        ),
        (
            {'estimator': [SAXVSM()]},
            ValueError,
            "If 'estimator' is a list, its length must be equal to "
            "the number of features (1 != 3)",
        ),
        (
            {'estimator': None},
            TypeError,
            "'estimator' must be a classifier that inherits from "
            "sklearn.base.BaseEstimator or a list thereof.",
        ),
    ],
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    clf = MultivariateClassifier(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        clf.fit(X, y)


@pytest.mark.parametrize(
    'params, X, error, err_msg',
    [
        (
            {'estimator': BOSS()},
            X[0, 0],
            ValueError,
            "X must be 3-dimensional (got 1).",
        ),
        (
            {'estimator': BOSS()},
            X[0],
            ValueError,
            "X must be 3-dimensional (got 2).",
        ),
    ],
)
def test_input_check(params, X, error, err_msg):
    """Test input data validation."""
    clf = MultivariateClassifier(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        clf.fit(X, y)


@pytest.mark.parametrize(
    'params',
    [
        {'estimator': SAXVSM()},
        {'estimator': [SAXVSM() for _ in range(n_features)]},
        {'estimator': [SAXVSM(), SAXVSM(), BOSSVS()]},
    ],
)
def test_actual_results_without_weights(params):
    """Test that the actual results are the expected ones."""
    arr_actual = MultivariateClassifier(**params).fit(X, y).predict(X)
    predictions = []
    if isinstance(params['estimator'], list):
        for i in range(n_features):
            predictions.append(params['estimator'][i].fit(X[:, i], y).predict(X[:, i]))
    else:
        for i in range(n_features):
            predictions.append(params['estimator'].fit(X[:, i], y).predict(X[:, i]))
    predictions = np.asarray(predictions)
    arr_desired = []
    for i in range(n_samples):
        arr_desired.append(np.argmax(np.bincount(predictions[:, i])))
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    'params',
    [
        {'estimator': SAXVSM(), 'weights': [0.1, 0.7, 0.2]},
        {
            'estimator': [SAXVSM() for _ in range(n_features)],
            'weights': [0.1, 0.7, 0.2],
        },
        {
            'estimator': [SAXVSM(), SAXVSM(), BOSSVS()],
            'weights': [0.1, 0.7, 0.2],
        },
    ],
)
def test_actual_results_with_weights(params):
    """Test that the actual results are the expected ones."""
    arr_actual = MultivariateClassifier(**params).fit(X, y).predict(X)
    predictions = []
    if isinstance(params['estimator'], list):
        for i in range(n_features):
            predictions.append(params['estimator'][i].fit(X[:, i], y).predict(X[:, i]))
    else:
        for i in range(n_features):
            predictions.append(params['estimator'].fit(X[:, i], y).predict(X[:, i]))
    predictions = np.asarray(predictions)
    arr_desired = []
    for i in range(n_samples):
        arr_desired.append(np.argmax(np.bincount(predictions[:, i], params['weights'])))
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.0)


def test_shared_estimator_instance_is_cloned():
    """The same instance listed for every feature is fitted per feature."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    from pyts.classification import KNeighborsClassifier

    rng = np.random.RandomState(0)
    X_sep = rng.randn(30, 3, 20)
    y_sep = np.repeat([0, 1], 15)
    # Only the first two features separate the classes; the third is noise.
    X_sep[y_sep == 1, :2] += 2
    knn = KNeighborsClassifier(n_neighbors=1)

    shared = MultivariateClassifier([knn] * 3).fit(X_sep, y_sep)
    separate = MultivariateClassifier(
        [KNeighborsClassifier(n_neighbors=1) for _ in range(3)]
    ).fit(X_sep, y_sep)

    np.testing.assert_array_equal(shared.predict(X_sep), separate.predict(X_sep))
    assert len({id(estimator) for estimator in shared.estimators_}) == 3
    # The estimator passed by the user is left unfitted.
    with pytest.raises(NotFittedError):
        check_is_fitted(knn)
