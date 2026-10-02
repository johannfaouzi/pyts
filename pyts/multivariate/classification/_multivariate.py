"""Utility class for multivariate time series classification."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from pyts._base import MultivariateClassifierMixin
from pyts.multivariate.utils import check_3d_array


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


class MultivariateClassifier(BaseEstimator, MultivariateClassifierMixin):
    """Classifier for multivariate time series.

    It provides a convenient class to classify multivariate time series with
    classifier that can only deal with univariate time series. The labels are
    predicted in a hard voting fashion using the predictions for each feature.

    Parameters
    ----------
    estimator : estimator object or list thereof
        Classifier. If one estimator is provided, it is cloned and each clone
        performs prediction for one feature. If a list of estimators is
        provided, each estimator performs prediction for one feature.

    weights : array-like, shape = (n_classifiers,) or None (default=None)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels. Uses uniform weights if None.

    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        An array of class labels known to the classifier.

    estimators_ : list of estimator objects
        The collection of fitted classifiers.

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
    ) -> None:
        self.estimator = estimator
        self.weights = weights

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
            self._weights = np.asarray(self.weights)

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_ind = self._le.transform(y)
        for i, clf in enumerate(self.estimators_):
            clf.fit(X[:, i, :], y_ind)
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Predict class labels using hard voting.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        Returns
        -------
        y_pred : array, shape = (n_samples,)
            Predicted class labels.

        """
        X = check_3d_array(X)
        check_is_fitted(self, 'estimators_')
        n_samples, n_features, _ = X.shape

        y_pred = np.empty((n_samples, n_features))
        for i, clf in enumerate(self.estimators_):
            y_pred[:, i] = clf.predict(X[:, i, :])

        maj = _hard_vote(y_pred.astype('int64'), self._weights)
        return self._le.inverse_transform(maj)

    def _check_params(self, n_features: int) -> None:
        """Check parameters."""
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
            for i, estimator in enumerate(self.estimator):
                if not (
                    isinstance(estimator, BaseEstimator)
                    and hasattr(estimator, 'predict')
                ):
                    raise ValueError(f"Estimator {i} must be a classifier.")
            self.estimators_ = self.estimator

        else:
            raise TypeError(
                "'estimator' must be a classifier that inherits from "
                "sklearn.base.BaseEstimator or a list thereof."
            )
