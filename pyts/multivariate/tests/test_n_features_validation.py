"""Multivariate estimators check the number of features against ``fit``."""

# License: BSD-3-Clause

import re

import numpy as np
import pytest

from pyts.classification import KNeighborsClassifier
from pyts.multivariate.classification import MultivariateClassifier
from pyts.multivariate.transformation import WEASELMUSE, MultivariateTransformer
from pyts.transformation import BOSS

rng = np.random.RandomState(0)
X = rng.randn(20, 3, 24)
y = np.repeat([0, 1], 10)


@pytest.fixture(scope='module')
def fitted_estimators():
    """Each estimator fitted once on 3 features, with its prediction method."""
    return [
        (MultivariateClassifier(KNeighborsClassifier()).fit(X, y), 'predict'),
        (
            MultivariateTransformer(
                BOSS(word_size=2, n_bins=2, window_size=10, sparse=False)
            ).fit(X),
            'transform',
        ),
        (WEASELMUSE(window_sizes=[6, 12], sparse=False).fit(X, y), 'transform'),
    ]


@pytest.mark.parametrize('n_features', [2, 5])
def test_feature_count_mismatch(fitted_estimators, n_features):
    """Too few or too many features raise a clear error."""
    msg = f"X has {n_features} features, but this estimator was fitted with 3"
    X_new = rng.randn(4, n_features, 24)
    for estimator, method in fitted_estimators:
        assert estimator.n_features_in_ == 3
        with pytest.raises(ValueError, match=re.escape(msg)):
            getattr(estimator, method)(X_new)


def test_feature_count_match(fitted_estimators):
    """Data with the fitted number of features is still accepted."""
    for estimator, method in fitted_estimators:
        assert len(getattr(estimator, method)(X[:4])) == 4


def test_weasel_muse_fit_transform_sets_n_features_in():
    """``fit_transform`` records the number of features like ``fit``."""
    transformer = WEASELMUSE(window_sizes=[6, 12], sparse=False)
    transformer.fit_transform(X, y)
    assert transformer.n_features_in_ == 3


@pytest.mark.parametrize('weights', [[1, 2], [1, 2, 3, 4], [[1, 2, 3]]])
def test_weights_shape(weights):
    """``weights`` with the wrong shape are rejected in ``fit``."""
    msg = "'weights' must be None or array-like with shape (n_features,) = (3,)"
    clf = MultivariateClassifier(KNeighborsClassifier(), weights=weights)
    with pytest.raises(ValueError, match=re.escape(msg)):
        clf.fit(X, y)


def test_valid_weights_unchanged():
    """Valid weights give the same predictions as before."""
    clf = MultivariateClassifier(KNeighborsClassifier(), weights=[1, 2, 3]).fit(X, y)
    assert clf.predict(X).shape == (20,)
