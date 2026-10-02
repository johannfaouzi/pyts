"""Base classes for all estimators."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score


class _SupportsFitTransform(Protocol):
    """Structural type for what the ``*TransformerMixin`` classes need.

    The mixins below implement ``fit_transform`` in terms of ``fit`` and
    ``transform``, which are provided by whatever concrete estimator class
    they are mixed into (alongside ``sklearn.base.BaseEstimator``), not by
    the mixin itself. This ``Protocol`` documents that expectation for the
    type checker instead of it reporting a missing-attribute error.
    """

    # Not all concrete ``fit`` implementations in pyts accept arbitrary
    # ``**fit_params`` (many stateless preprocessing steps just take
    # ``X``/``y``), so it is deliberately left out of this Protocol: adding
    # it would make those otherwise-compatible classes fail structural
    # conformance.
    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = ...,
        # The ``...`` body is PEP 544 ``Protocol`` stub syntax, not a
        # no-op statement.
        # codeql[py/ineffectual-statement]
    ) -> Self: ...

    # Different concrete transformers return different dtypes (e.g. float64
    # for most, but int64 for KBinsDiscretizer and its callers), so this is
    # intentionally left as ``Any`` rather than a specific dtype.
    # codeql[py/ineffectual-statement]
    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]: ...


class _SupportsPredict(Protocol):
    """Structural type for what the ``*ClassifierMixin`` classes need."""

    # codeql[py/ineffectual-statement]
    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]: ...


class UnivariateTransformerMixin:
    """Mixin class for all univariate transformers in pyts."""

    def fit_transform(
        self: _SupportsFitTransform,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> npt.NDArray[Any]:
        """Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.

        y : None or array-like, shape = (n_samples,) (default = None)
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : array
            Transformed array.

        """
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class MultivariateTransformerMixin:
    """Mixin class for all multivariate transformers in pyts."""

    def fit_transform(
        self: _SupportsFitTransform,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> npt.NDArray[Any]:
        """Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        y : None or array-like, shape = (n_samples,) (default = None)
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : array
            Transformed array.

        """
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class UnivariateClassifierMixin:
    """Mixin class for all univariate classifiers in pyts."""

    _estimator_type = "classifier"

    def score(
        self: _SupportsPredict,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
    ) -> float | np.floating:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.

        y : array-like, shape = (n_samples,)
            True labels for `X`.

        sample_weight : None or array-like, shape = (n_samples,) (default = None)
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` with regards to `y`.

        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class MultivariateClassifierMixin:
    """Mixin class for all multivariate classifiers in pyts."""

    _estimator_type = "classifier"

    def score(
        self: _SupportsPredict,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
    ) -> float | np.floating:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        y : array-like, shape = (n_samples,)
            True labels for `X`.

        sample_weight : None or array-like, shape = (n_samples,) (default = None)
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` with regards to `y`.

        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
