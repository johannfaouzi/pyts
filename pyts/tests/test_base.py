"""Testing for base classes."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.utils import get_tags

from pyts.approximation import SymbolicFourierApproximation
from pyts.classification import (
    BOSSVS,
    SAXVSM,
    TSBF,
    KNeighborsClassifier,
    LearningShapelets,
    TimeSeriesForest,
)
from pyts.datasets import load_basic_motions, load_gunpoint
from pyts.multivariate.classification import MultivariateClassifier
from pyts.multivariate.image import JointRecurrencePlot

X_uni, _, y_uni, _ = load_gunpoint(return_X_y=True)
X_multi, _, y_multi, _ = load_basic_motions(return_X_y=True)


@pytest.mark.parametrize(
    'estimator, X, y',
    [
        (SymbolicFourierApproximation(n_bins=2), X_uni, None),
        (
            SymbolicFourierApproximation(n_bins=2, strategy='entropy'),
            X_uni,
            y_uni,
        ),
    ],
)
def test_univariate_transformer_mixin(estimator, X, y):
    sfa_1 = clone(estimator)
    sfa_2 = clone(estimator)
    np.testing.assert_array_equal(
        sfa_1.fit_transform(X, y), sfa_2.fit(X, y).transform(X)
    )


@pytest.mark.parametrize(
    'estimator, X, y',
    [
        (JointRecurrencePlot(), X_multi, None),
        (JointRecurrencePlot(), X_multi, y_multi),
    ],
)
def test_multivariate_transformer_mixin(estimator, X, y):
    jrp_1 = clone(estimator)
    jrp_2 = clone(estimator)
    np.testing.assert_allclose(jrp_1.fit_transform(X, y), jrp_2.fit(X, y).transform(X))


@pytest.mark.parametrize(
    'sample_weight',
    [None, np.ones_like(y_uni), np.random.uniform(size=y_uni.size)],
)
def test_univariate_classifier_mixin(sample_weight):
    clf = SAXVSM().fit(X_uni, y_uni)
    assert isinstance(clf.score(X_uni, y_uni, sample_weight), (float, np.floating))


@pytest.mark.parametrize(
    'sample_weight',
    [None, np.ones(y_multi.size), np.random.uniform(size=y_multi.size)],
)
def test_multivariate_classifier_mixin(sample_weight):
    clf = MultivariateClassifier(SAXVSM()).fit(X_multi, y_multi)
    assert isinstance(clf.score(X_multi, y_multi, sample_weight), (float, np.floating))


@pytest.mark.parametrize(
    'estimator',
    [
        KNeighborsClassifier(),
        BOSSVS(),
        SAXVSM(),
        LearningShapelets(),
        TSBF(),
        TimeSeriesForest(),
        MultivariateClassifier(KNeighborsClassifier()),
    ],
    ids=lambda estimator: type(estimator).__name__,
)
def test_classifier_tags(estimator):
    """scikit-learn recognizes pyts classifiers as classifiers.

    This is what makes, for instance, ``cross_val_score`` and ``GridSearchCV``
    use stratified folds when ``cv`` is an integer.
    """
    assert is_classifier(estimator)
    assert get_tags(estimator).estimator_type == 'classifier'
    assert isinstance(
        check_cv(3, [0, 1] * 6, classifier=is_classifier(estimator)), StratifiedKFold
    )
