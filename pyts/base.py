"""Base classes for all estimators."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from sklearn.metrics import accuracy_score


class UnivariateTransformerMixin:
    """Mixin class for all univariate transformers in pyts."""

    def fit_transform(self, X, y=None, **fit_params):
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

        """  # noqa: E501
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class MultivariateTransformerMixin:
    """Mixin class for all multivariate transformers in pyts."""

    def fit_transform(self, X, y=None, **fit_params):
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

        """  # noqa: E501
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class UnivariateClassifierMixin:
    """Mixin class for all univariate classifiers in pyts."""

    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
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

        """  # noqa: E501
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class MultivariateClassifierMixin:
    """Mixin class for all multivariate classifiers in pyts."""

    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
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

        """  # noqa: E501
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
