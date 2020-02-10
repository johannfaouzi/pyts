"""Utility class for multivariate time series transformation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from ..utils import check_3d_array


class MultivariateTransformer(BaseEstimator, TransformerMixin):
    r"""Transformer for multivariate time series.

    It provides a convenient class to transform multivariate time series with
    transformers that can only deal with univariate time series.

    Parameters
    ----------
    estimator : estimator object or list thereof
        Transformer. If one estimator is provided, it is cloned and each clone
        transforms one feature. If a list of estimators is provided, each
        estimator transforms one feature.

    flatten : bool (default = True)
        Affect shape of transform output. If True, ``transform``
        returns an array with shape (n_samples, \*). If False, the output of
        ``transform`` from each estimator must have the same shape and
        ``transform`` returns an array with shape (n_samples, n_features, \*).
        Ignored if the transformers return sparse matrices.

    Attributes
    ----------
    estimators_ : list of estimator objects
        The collection of fitted transformers.

    Examples
    --------
    >>> from pyts.datasets import load_basic_motions
    >>> from pyts.multivariate.transformation import MultivariateTransformer
    >>> from pyts.image import GramianAngularField
    >>> X, _, _, _ = load_basic_motions(return_X_y=True)
    >>> transformer = MultivariateTransformer(GramianAngularField(),
    ...                                       flatten=False)
    >>> X_new = transformer.fit_transform(X)
    >>> X_new.shape
    (40, 6, 100, 100)

    """

    def __init__(self, estimator, flatten=True):
        self.estimator = estimator
        self.flatten = flatten

    def fit(self, X, y=None):
        """Pass.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        y : None or array-like, shape = (n_samples,) (default = None)
            Class labels.

        Returns
        -------
        self : object

        """
        X = check_3d_array(X)
        _, n_features, _ = X.shape
        self._check_params(n_features)
        for i, transformer in enumerate(self.estimators_):
            transformer.fit(X[:, i, :], y)
        return self

    def transform(self, X):
        r"""Apply transform to each feature.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        Returns
        -------
        X_new : array, shape = (n_samples, *) or (n_samples, n_features, *)
            Transformed time series.

        """
        X = check_3d_array(X)
        n_samples, _, _ = X.shape
        check_is_fitted(self, 'estimators_')

        X_transformed = [transformer.transform(X[:, i, :])
                         for i, transformer in enumerate(self.estimators_)]
        all_sparse = np.all([isinstance(X_transformed_i, csr_matrix)
                             for X_transformed_i in X_transformed])
        if all_sparse:
            X_new = hstack(X_transformed)
        else:
            X_new = [self._convert_to_array(X_transformed_i)
                     for X_transformed_i in X_transformed]
            ndims = [X_new_i.ndim for X_new_i in X_new]
            shapes = [X_new_i.shape for X_new_i in X_new]
            one_dim = (np.unique(ndims).size == 1)
            if one_dim:
                one_shape = np.unique(shapes, axis=0).shape[0] == 1
            else:
                one_shape = False
            if (not one_shape) or self.flatten:
                X_new = [X_new_i.reshape(n_samples, -1) for X_new_i in X_new]
                X_new = np.concatenate(X_new, axis=1)
            else:
                X_new = np.asarray(X_new)
                axes = [1, 0] + [i for i in range(2, X_new.ndim)]
                X_new = np.transpose(X_new, axes=axes)
        return X_new

    def _check_params(self, n_features):
        """Check parameters."""
        transformer = (isinstance(self.estimator, BaseEstimator)
                       and hasattr(self.estimator, 'transform'))
        if transformer:
            self.estimators_ = [clone(self.estimator)
                                for _ in range(n_features)]

        elif isinstance(self.estimator, list):
            if len(self.estimator) != n_features:
                raise ValueError(
                    "If 'estimator' is a list, its length must be equal to "
                    "the number of features ({0} != {1})"
                    .format(len(self.estimator), n_features)
                )
            for i, estimator in enumerate(self.estimator):
                if not (isinstance(estimator, BaseEstimator)
                        and hasattr(estimator, 'transform')):
                    raise ValueError("Estimator {} must be a transformer."
                                     .format(i))
            self.estimators_ = self.estimator

        else:
            raise TypeError(
                "'estimator' must be a transformer that inherits from "
                "sklearn.base.BaseEstimator or a list thereof.")

    @staticmethod
    def _convert_to_array(X):
        """Convert the input data to an array if necessary."""
        if isinstance(X, csr_matrix):
            return X.A
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError('Unexpected type for X: {}.'
                             .format(type(X).__name__))
