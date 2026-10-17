"""Integer classifier weights follow the documented weighted vote."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from pyts.multivariate.classification import MultivariateClassifier


@pytest.mark.parametrize('dtype', [np.int16, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('as_list', [False, True])
def test_weighted_vote_numeric_weights(dtype, as_list):
    """One weighted classifier can outweigh two classifiers voting for another class."""
    estimators = [
        DummyClassifier(strategy='constant', constant=1),
        DummyClassifier(strategy='constant', constant=0),
        DummyClassifier(strategy='constant', constant=0),
    ]
    weights = np.asarray([3, 1, 1], dtype=dtype)
    if as_list:
        weights = weights.tolist()
    X = np.arange(24).reshape(2, 3, 4)
    classifier = MultivariateClassifier(estimators, weights=weights).fit(X, ['a', 'b'])
    np.testing.assert_array_equal(classifier.predict(X), ['b', 'b'])
    np.testing.assert_array_equal(weights, [3, 1, 1])
