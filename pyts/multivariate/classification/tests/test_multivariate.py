"""Testing for MultivariateClassifier."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.classification import SAXVSM
from pyts.multivariate.classification import MultivariateClassifier
from pyts.transformation import BOSS


n_samples, n_features, n_timestamps, n_classes = 40, 3, 30, 2
rng = np.random.RandomState(42)
X = rng.randn(n_samples, n_features, n_timestamps)
y = rng.randint(n_classes, size=n_samples)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'estimator': [SAXVSM(), SAXVSM(), BOSS()]},
      ValueError, "Estimator 2 must be a classifier."),

     ({'estimator': [SAXVSM()]}, ValueError,
      "If 'estimator' is a list, its length must be equal to "
      "the number of features (1 != 3)"),

     ({'estimator': None}, TypeError,
      "'estimator' must be a classifier that inherits from "
      "sklearn.base.BaseEstimator or a list thereof.")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    clf = MultivariateClassifier(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        clf.fit(X, y)


@pytest.mark.parametrize(
    'params, X, error, err_msg',
    [({'estimator': BOSS()}, X[0, 0], ValueError,
      "X must be 3-dimensional (got 1)."),
     ({'estimator': BOSS()}, X[0], ValueError,
      "X must be 3-dimensional (got 2).")]
)
def test_input_check(params, X, error, err_msg):
    """Test input data validation."""
    clf = MultivariateClassifier(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        clf.fit(X, y)


def test_actual_results_without_weights():
    """Test that the actual results are the expected ones."""
    params = {'estimator': SAXVSM()}
    arr_actual = MultivariateClassifier(**params).fit(X, y).predict(X)
    predictions = []
    for i in range(n_features):
        predictions.append(
            params['estimator'].fit(X[:, i], y).predict(X[:, i]))
    predictions = np.asarray(predictions)
    arr_desired = []
    for i in range(n_samples):
        arr_desired.append(np.argmax(np.bincount(predictions[:, i])))
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_actual_results_with_weights():
    """Test that the actual results are the expected ones."""
    params = {'estimator': SAXVSM(), 'weights': [0.1, 0.7, 0.2]}
    arr_actual = MultivariateClassifier(**params).fit(X, y).predict(X)
    predictions = []
    for i in range(n_features):
        predictions.append(
            params['estimator'].fit(X[:, i], y).predict(X[:, i]))
    predictions = np.asarray(predictions)
    arr_desired = []
    for i in range(n_samples):
        arr_desired.append(np.argmax(
            np.bincount(predictions[:, i], params['weights'])))
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
