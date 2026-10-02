"""Utility class for multivariate time series transformation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from typing import Any, Self

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from pyts._base import MultivariateTransformerMixin, _SupportsFitTransform
from pyts.multivariate.utils import check_3d_array


class MultivariateTransformer(BaseEstimator, MultivariateTransformerMixin):
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

    def __init__(
        self,
        estimator: _SupportsFitTransform | list[_SupportsFitTransform],
        flatten: bool = True,
    ) -> None:
        self.estimator = estimator
        self.flatten = flatten

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> Self:
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

    def fit_transform(  # pyrefly: ignore[bad-override]
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> npt.NDArray[Any] | csr_matrix:
        r"""Fit to data, then transform it.

        Unlike most transformers, ``transform`` may return a sparse matrix
        (when the wrapped per-feature transformers do), so this method is
        overridden instead of relying on
        ``MultivariateTransformerMixin.fit_transform``, whose declared
        return type is a plain array.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        y : None or array-like, shape = (n_samples,) (default = None)
            Class labels.

        Returns
        -------
        X_new : array, shape = (n_samples, \*) or (n_samples, n_features, \*)
            Transformed time series. May be a sparse matrix instead when
            the wrapped per-feature transformers return one.

        """
        return self.fit(X, y).transform(X)

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any] | csr_matrix:
        r"""Apply transform to each feature.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        Returns
        -------
        X_new : array, shape = (n_samples, \*) or (n_samples, n_features, \*)
            Transformed time series.

        """
        X = check_3d_array(X)
        n_samples, _, _ = X.shape
        check_is_fitted(self, 'estimators_')

        X_transformed = [
            transformer.transform(X[:, i, :])
            for i, transformer in enumerate(self.estimators_)
        ]
        all_sparse = np.all(
            [
                isinstance(X_transformed_i, csr_matrix)
                for X_transformed_i in X_transformed
            ]
        )
        if all_sparse:
            # ``hstack`` does not guarantee the ``csr_matrix`` format (its
            # default output format depends on its inputs), so normalize
            # explicitly to match the declared return type and the
            # convention used by ``WEASELMUSE.transform``.
            X_new = csr_matrix(hstack(X_transformed))
        else:
            X_new = [
                self._convert_to_array(X_transformed_i)
                for X_transformed_i in X_transformed
            ]
            ndims = [X_new_i.ndim for X_new_i in X_new]
            shapes = [X_new_i.shape for X_new_i in X_new]
            one_dim = np.unique(ndims).size == 1
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

    def _check_params(self, n_features: int) -> None:
        """Check parameters."""
        if isinstance(self.estimator, BaseEstimator) and hasattr(
            self.estimator, 'transform'
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
                    and hasattr(estimator, 'transform')
                ):
                    raise ValueError(f"Estimator {i} must be a transformer.")
            self.estimators_ = self.estimator

        else:
            raise TypeError(
                "'estimator' must be a transformer that inherits from "
                "sklearn.base.BaseEstimator or a list thereof."
            )

    @staticmethod
    def _convert_to_array(
        X: npt.NDArray[Any] | csr_matrix,
    ) -> npt.NDArray[Any]:
        """Convert the input data to an array if necessary."""
        if isinstance(X, csr_matrix):
            return X.toarray()
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError(f'Unexpected type for X: {type(X).__name__}.')
