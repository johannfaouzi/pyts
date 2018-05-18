"""The :mod:`pyts.decomposition` module includes decomposition algorithms.

Implemented algorithms are:
- Singular Spectrum Analysis
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


standard_library.install_aliases()


class SSA(BaseEstimator, TransformerMixin):
    """Singular Spectrum Analysis.

    Parameters
    ----------
    window_size : int
        The size of the sliding window.

    grouping : None, int or array-like (default = None)
        The way the elementary matrices are grouped. If None,
        no grouping is performed. If an integer, the number of
        groups is equal to this integer. If array-like, each element
        must be a array-like containing the indices for each group.

    """

    def __init__(self, window_size, grouping=None):
        self.window_size = window_size
        self.grouping = grouping

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            ignored

        y
            Ignored

        """
        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array-like, shape = [n_samples, n_splits, n_features]
            Transformed data. ``n_splits`` value depends on the value of
            ``grouping``. If ``grouping=None``, ``n_splits`` is equal to
            ``window_size``. If ``grouping`` is an integer, ``n_splits`` is
            equal to ``grouping``. If ``grouping`` is array-like, ``n_splits``
            is equal to the length of ``grouping``.

        """
        # Check input data
        X = check_array(X)

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters
        if not isinstance(self.window_size, int):
            raise TypeError("'window_size' must be an integer.")
        if self.window_size < 1:
            raise ValueError("'window_size' must be greater or equal than 1.")
        if self.window_size > n_features:
            raise ValueError("'window_size' must be lower or equal than "
                             "the size of each time series.")
        if not (self.grouping is None or
                isinstance(self.grouping, (int, list, tuple, np.ndarray))):
            raise TypeError("'grouping' must be either None, an integer "
                            "or array-like.")
        if isinstance(self.grouping, int) and self.grouping > self.window_size:
            raise ValueError("If 'grouping' is an integer, it must be "
                             "lower than or equal to 'window_size'.")
        if isinstance(self.grouping, (list, tuple, np.ndarray)):
            idx = np.concatenate(self.grouping)
            diff = np.setdiff1d(idx, np.arange(self.window_size))
            if diff.size > 0:
                raise ValueError("If 'grouping' is array-like, all values in "
                                 "'grouping' must be lower than 'window_size'."
                                 " {0} is not lower than "
                                 "{1}.".format(diff[0], self.window_size))

        return np.apply_along_axis(self._ssa, 1, X, n_features,
                                   self.window_size, self.grouping)

    def _ssa(self, ts, ts_size, window_size, grouping):

        n_lags = ts_size - window_size + 1

        # First step: create a matrix of lagged vectors
        X = np.array([ts[i: i + window_size] for i in range(n_lags)]).T

        # Second step: compute the eigenvalues and eigenvectors of this
        # matrix multiplied by its transposed matrix then compute
        # elementary matrices
        w, v = np.linalg.eigh(X.dot(X.T))
        w, v = w[::-1], v[:, ::-1]

        elementary_matrices = np.empty((window_size, window_size, n_lags))
        for i in range(window_size):
            elementary_matrices[i] = np.outer(v[:, i], v[:, i]).dot(X)

        # Third step: group elementary matrices
        if grouping is None:
            grouping_size = window_size
            grouped_matrices = elementary_matrices

        elif isinstance(grouping, int):
            grouping = np.array_split(np.arange(window_size), grouping)
            grouping_size = len(grouping)
            grouped_matrices = np.zeros((grouping_size, window_size, n_lags))
            for i, group in enumerate(grouping):
                grouped_matrices[i] = elementary_matrices[group].sum(axis=0)
        else:
            grouping_size = len(grouping)
            grouped_matrices = np.zeros((grouping_size, window_size, n_lags))
            for i, group in enumerate(grouping):
                grouped_matrices[i] = elementary_matrices[group].sum(axis=0)

        # Fourth step: decompose the time series in several time series
        if window_size >= n_lags:
            grouped_matrices = np.transpose(grouped_matrices, axes=[0, 2, 1])
            gap = window_size
        else:
            gap = n_lags
        y = np.zeros((grouping_size, ts_size))
        first_row = [(0, col) for col in range(n_lags)]
        last_col = [(row, n_lags - 1) for row in range(1, window_size)]
        indices = first_row + last_col
        for group in range(grouping_size):
            for i, j in indices:
                y[group, i + j] = np.diag(np.fliplr(grouped_matrices[group]),
                                          gap - i - j - 1).mean()

        return y
