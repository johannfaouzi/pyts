"""Utility class for multivariate time series classification."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from typing import Any, Literal, Protocol, Self

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._available_if import available_if
from sklearn.utils.validation import check_is_fitted

from pyts._base import MultivariateClassifierMixin
from pyts.multivariate.utils import check_3d_array
from pyts.multivariate.utils._utils import _check_n_features


class _SupportsFitPredict(Protocol):
    """Structural type for a per-feature classifier used internally.

    Unlike ``pyts._base._SupportsPredict``, this also requires ``fit``,
    since each per-feature classifier is fitted here (not by an external
    mixin) before being used for prediction.
    """

    # The ``...`` body is PEP 544 ``Protocol`` stub syntax, not a no-op
    # statement.
    # codeql[py/ineffectual-statement]
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self: ...

    # codeql[py/ineffectual-statement]
    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]: ...


@njit(
    [
        "int64[:](int64[:,:], float64[:])",
        "int64[:](int64[:,:], none)",
    ]
)
def _hard_vote(
    y_pred: npt.NDArray[np.int64], weights: npt.NDArray[np.float64] | None
) -> npt.NDArray[np.int64]:
    n_samples, _n_features = y_pred.shape
    maj = np.empty(n_samples, dtype=np.int64)
    for i in prange(n_samples):
        maj[i] = np.argmax(np.bincount(y_pred[i], weights))
    return maj


def _check_soft_voting(clf: 'MultivariateClassifier') -> bool:
    """Check that ``predict_proba`` is available (soft voting)."""
    if clf.voting != 'soft':
        raise AttributeError(
            f"predict_proba is not available when voting={clf.voting!r}."
        )
    return True


class MultivariateClassifier(MultivariateClassifierMixin, BaseEstimator):
    """Classifier for multivariate time series.

    It provides a convenient class to classify multivariate time series with
    classifier that can only deal with univariate time series. A classifier is
    fitted for each feature, and the labels are predicted by combining the
    predictions for each feature with hard or soft voting.

    Parameters
    ----------
    estimator : estimator object or list thereof
        Classifier. If one estimator is provided, it is cloned and each clone
        performs prediction for one feature. If a list of estimators is
        provided, each estimator performs prediction for one feature.

    weights : array-like, shape = (n_classifiers,) or None (default=None)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (hard voting) or the class probabilities before
        averaging (soft voting). Uses uniform weights if None.

    voting : {'hard', 'soft'} (default = 'hard')
        If 'hard', uses the predicted class labels for majority rule voting.
        If 'soft', predicts the class label based on the argmax of the
        (weighted) average of the class probabilities predicted for each
        feature, which requires every classifier to implement
        ``predict_proba``. ``predict_proba`` is only available with soft
        voting.

    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        An array of class labels known to the classifier.

    estimators_ : list of estimator objects
        The collection of fitted classifiers.

    n_features_in_ : int
        Number of features seen during ``fit``. ``predict`` requires the
        same number of features.

    Examples
    --------
    >>> from pyts.classification import BOSSVS
    >>> from pyts.datasets import load_basic_motions
    >>> from pyts.multivariate.classification import MultivariateClassifier
    >>> X_train, X_test, y_train, y_test = load_basic_motions(return_X_y=True)
    >>> clf = MultivariateClassifier(BOSSVS())
    >>> clf.fit(X_train, y_train)
    MultivariateClassifier(...)
    >>> clf.score(X_test, y_test)
    1.0

    """

    def __init__(
        self,
        estimator: _SupportsFitPredict | list[_SupportsFitPredict],
        weights: npt.ArrayLike | None = None,
        voting: Literal['hard', 'soft'] = 'hard',
    ) -> None:
        self.estimator = estimator
        self.weights = weights
        self.voting = voting

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """Fit each classifier.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        y : None or array-like, shape = (n_samples,)
            Class labels.

        Returns
        -------
        self : object

        """
        X = check_3d_array(X)
        _, n_features, _ = X.shape
        self._check_params(n_features)

        if self.weights is None:
            self._weights = None
        else:
            self._weights = np.asarray(self.weights, dtype=np.float64)
            if self._weights.shape != (n_features,):
                raise ValueError(
                    "'weights' must be None or array-like with shape "
                    f"(n_features,) = ({n_features},) "
                    f"(got shape {self._weights.shape})."
                )
        self.n_features_in_ = n_features

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_ind = self._le.transform(y)
        for i, clf in enumerate(self.estimators_):
            clf.fit(X[:, i, :], y_ind)
            if self.voting == 'soft' and not hasattr(clf, 'predict_proba'):
                raise ValueError(
                    "voting='soft' requires every classifier to implement "
                    f"predict_proba (classifier {i} does not)."
                )
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Predict class labels using hard or soft voting.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        Returns
        -------
        y_pred : array, shape = (n_samples,)
            Predicted class labels.

        """
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)
            return self._le.inverse_transform(maj)

        X = check_3d_array(X)
        check_is_fitted(self, 'estimators_')
        _check_n_features(X, self.n_features_in_)
        n_samples, n_features, _ = X.shape

        y_pred = np.empty((n_samples, n_features))
        for i, clf in enumerate(self.estimators_):
            y_pred[:, i] = clf.predict(X[:, i, :])

        maj = _hard_vote(y_pred.astype('int64'), self._weights)
        return self._le.inverse_transform(maj)

    @available_if(_check_soft_voting)
    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute probabilities of possible outcomes for samples in X.

        Only available with soft voting. The class probabilities are the
        (weighted) average of the class probabilities predicted for each
        feature.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        Returns
        -------
        y_proba : array, shape = (n_samples, n_classes)
            Class probabilities of each sample. The order of the classes is
            given by the attribute ``classes_``.

        """
        X = check_3d_array(X)
        check_is_fitted(self, 'estimators_')
        _check_n_features(X, self.n_features_in_)
        probas = np.asarray(
            [clf.predict_proba(X[:, i, :]) for i, clf in enumerate(self.estimators_)],
            dtype=np.float64,
        )
        return np.average(probas, axis=0, weights=self._weights)

    def _check_params(self, n_features: int) -> None:
        """Check parameters."""
        if self.voting not in ('hard', 'soft'):
            raise ValueError(
                f"'voting' must be either 'hard' or 'soft' (got {self.voting!r})."
            )
        if isinstance(self.estimator, BaseEstimator) and hasattr(
            self.estimator, 'predict'
        ):
            self.estimators_ = [clone(self.estimator) for _ in range(n_features)]

        elif isinstance(self.estimator, list):
            if len(self.estimator) != n_features:
                raise ValueError(
                    "If 'estimator' is a list, its length must "
                    "be equal to the number of features "
                    f"({len(self.estimator)} != {n_features})"
                )
            # Clone each estimator so that the same instance listed several
            # times is fitted independently on each feature, and so that the
            # estimators passed by the user are left unfitted.
            estimators = []
            for i, estimator in enumerate(self.estimator):
                if not (
                    isinstance(estimator, BaseEstimator)
                    and hasattr(estimator, 'predict')
                ):
                    raise ValueError(f"Estimator {i} must be a classifier.")
                estimators.append(clone(estimator))
            self.estimators_ = estimators

        else:
            raise TypeError(
                "'estimator' must be a classifier that inherits from "
                "sklearn.base.BaseEstimator or a list thereof."
            )
