"""Code for Singular Spectrum Analysis."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import ceil
from numba import njit, prange
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from ..utils.utils import _windowed_view


@njit
def _outer_dot(v, X, n_samples, window_size, n_windows):
    X_new = np.empty((n_samples, window_size, window_size, n_windows))
    for i in prange(n_samples):
        for j in prange(window_size):
            X_new[i, j] = np.dot(np.outer(v[i, :, j], v[i, :, j]), X[i])
    return X_new


@njit
def _diagonal_averaging(X, n_samples, n_timestamps, window_size,
                        n_windows, grouping_size, gap):
    X_new = np.empty((n_samples, grouping_size, n_timestamps))
    first_row = [(0, col) for col in range(n_windows)]
    last_col = [(row, n_windows - 1) for row in range(1, window_size)]
    indices = first_row + last_col
    for i in prange(n_samples):
        for group in prange(grouping_size):
            for (j, k) in indices:
                X_new[i, group, j + k] = np.diag(
                    X[i, group, :, ::-1], gap - j - k - 1
                ).mean()
    return X_new


class SingularSpectrumAnalysis(BaseEstimator, TransformerMixin):
    """Singular Spectrum Analysis.

    Parameters
    ----------
    window_size : int or float (default = 4)
        Size of the sliding window (i.e. the size of each word). If float, it
        represents the percentage of the size of each time series and must be
        between 0 and 1. The window size will be computed as
        ``max(2, ceil(window_size * n_timestamps))``.

    groups : None, int or array-like (default = None)
        The way the elementary matrices are grouped. If None, no grouping is
        performed. If an integer, it represents the number of groups and the
        bounds of the groups are computed as
        ``np.linspace(0, window_size, groups + 1).astype('int64')``.
        If array-like, each element must be array-like and contain the indices
        for each group.

    References
    ----------
    .. [1] N. Golyandina, and A. Zhigljavsky, "Singular Spectrum Analysis for
           Time Series". Springer-Verlag Berlin Heidelberg (2013).

    Examples
    --------
    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.decomposition import SingularSpectrumAnalysis
    >>> X, _, _, _ = load_gunpoint(return_X_y=True)
    >>> transformer = SingularSpectrumAnalysis(window_size=5)
    >>> X_new = transformer.transform(X)
    >>> X_new.shape
    (50, 5, 150)

    """

    def __init__(self, window_size=4, groups=None):
        self.window_size = window_size
        self.groups = groups

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            ignored

        y
            Ignored

        Returns
        -------
        self : object

        """
        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)

        Returns
        -------
        X_new : array-like, shape = (n_samples, n_splits, n_timestamps)
            Transformed data. ``n_splits`` value depends on the value of
            ``groups``. If ``groups=None``, ``n_splits`` is equal to
            ``window_size``. If ``groups`` is an integer, ``n_splits`` is
            equal to ``groups``. If ``groups`` is array-like, ``n_splits``
            is equal to the length of ``groups``. If ``n_split=1``, ``X_new``
            is squeezed and its shape is (n_samples, n_timestamps).

        """
        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape
        window_size = self._check_params(n_timestamps)
        n_windows = n_timestamps - window_size + 1

        X_window = np.transpose(
            _windowed_view(X, n_samples, n_timestamps,
                           window_size, window_step=1), axes=(0, 2, 1)
        ).copy()
        X_tranpose = np.matmul(X_window,
                               np.transpose(X_window, axes=(0, 2, 1)))
        w, v = np.linalg.eigh(X_tranpose)
        w, v = w[:, ::-1], v[:, :, ::-1]

        X_elem = _outer_dot(v, X_window, n_samples, window_size, n_windows)
        X_groups, grouping_size = self._grouping(
            X_elem, n_samples, window_size, n_windows
        )
        if window_size >= n_windows:
            X_groups = np.transpose(X_groups, axes=(0, 1, 3, 2))
            gap = window_size
        else:
            gap = n_windows

        X_ssa = _diagonal_averaging(
            X_groups, n_samples, n_timestamps, window_size,
            n_windows, grouping_size, gap
        )
        return np.squeeze(X_ssa)

    def _grouping(self, X, n_samples, window_size, n_windows):
        if self.groups is None:
            grouping_size = window_size
            X_new = X
        elif isinstance(self.groups, int):
            grouping = np.linspace(0, window_size,
                                   self.groups + 1).astype('int64')
            grouping_size = len(grouping) - 1
            X_new = np.zeros((n_samples, grouping_size,
                              window_size, n_windows))
            for i, (j, k) in enumerate(zip(grouping[:-1], grouping[1:])):
                X_new[:, i] = X[:, j:k].sum(axis=1)
        else:
            grouping_size = len(self.groups)
            X_new = np.zeros((n_samples, grouping_size,
                              window_size, n_windows))
            for i, group in enumerate(self.groups):
                X_new[:, i] = X[:, group].sum(axis=1)
        return X_new, grouping_size

    def _check_params(self, n_timestamps):
        if not isinstance(self.window_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_size' must be an integer or a float.")
        if isinstance(self.window_size, (int, np.integer)):
            if not 2 <= self.window_size <= n_timestamps:
                raise ValueError(
                    "If 'window_size' is an integer, it must be greater "
                    "than or equal to 2 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.window_size)
                )
            window_size = self.window_size
        else:
            if not 0 < self.window_size <= 1:
                raise ValueError(
                    "If 'window_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.window_size)
                )
            window_size = max(2, ceil(self.window_size * n_timestamps))
        if not (self.groups is None
                or isinstance(self.groups, (int, list, tuple, np.ndarray))):
            raise TypeError("'groups' must be either None, an integer "
                            "or array-like.")
        if isinstance(self.groups, (int, np.integer)):
            if not 1 <= self.groups <= self.window_size:
                raise ValueError(
                    "If 'groups' is an integer, it must be greater than or "
                    "equal to 1 and lower than or equal to 'window_size'."
                )
        if isinstance(self.groups, (list, tuple, np.ndarray)):
            idx = np.concatenate(self.groups)
            diff = np.setdiff1d(idx, np.arange(self.window_size))
            flat_list = [item for group in self.groups for item in group]
            if ((diff.size > 0)
                or not (all(isinstance(x, (int, np.integer))
                            for x in flat_list))):
                raise ValueError(
                    "If 'groups' is array-like, all the values in 'groups' "
                    "must be integers between 0 and ('window_size' - 1)."
                )
        return window_size
