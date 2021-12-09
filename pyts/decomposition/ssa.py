"""Code for Singular Spectrum Analysis."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import ceil
from numba import njit, prange
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from ..base import UnivariateTransformerMixin
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


class SingularSpectrumAnalysis(BaseEstimator, UnivariateTransformerMixin):
    """Singular Spectrum Analysis.

    Parameters
    ----------
    window_size : int or float (default = 4)
        Size of the sliding window (i.e. the size of each word). If float, it
        represents the percentage of the size of each time series and must be
        between 0 and 1. The window size will be computed as
        ``max(2, ceil(window_size * n_timestamps))``.

    groups : None, int, 'auto', or array-like (default = None)
        The way the elementary matrices are grouped. If None, no grouping is
        performed. If an integer, it represents the number of groups and the
        bounds of the groups are computed as
        ``np.linspace(0, window_size, groups + 1).astype('int64')``.
        If 'auto', then three groups are determined, containing trend,
        seasonal, and residual. If array-like, each element must be array-like
        and contain the indices for each group.

    lower_frequency_bound : float (default = 0.075)
        The boundary of the periodogram to characterize trend, seasonal and
        residual components. It must be between 0 and 0.5.
        Ignored if ``groups`` is not set to 'auto'.

    lower_frequency_contribution : float (default = 0.85)
        The relative threshold to characterize trend, seasonal and
        residual components by considering the periodogram.
        It must be between 0 and 1. Ignored if ``groups`` is not set to 'auto'.

    chunksize : int or None (default = None)
        If int, the transformation of the whole dataset is performed using
        chunks (batches) and ``chunksize`` corresponds to the maximum size of
        each chunk (batch). If None, the transformation is performed on the
        whole dataset at once. Performing the transformation with chunks is
        likely to be a bit slower but requires less memory.

    n_jobs : None or int (default = None)
        The number of jobs to use for the computation. Only used if
        ``chunksize`` is set to an integer.

    References
    ----------
    .. [1] N. Golyandina, and A. Zhigljavsky, "Singular Spectrum Analysis for
           Time Series". Springer-Verlag Berlin Heidelberg (2013).

    .. [2] T. Alexandrov, "A Method of Trend Extraction Using Singular
           Spectrum Analysis", REVSTAT (2008).

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

    def __init__(self, window_size=4, groups=None,
                 lower_frequency_bound=0.075,
                 lower_frequency_contribution=0.85,
                 chunksize=None, n_jobs=1):
        self.window_size = window_size
        self.groups = groups
        self.lower_frequency_bound = lower_frequency_bound
        self.lower_frequency_contribution = lower_frequency_contribution
        self.chunksize = chunksize
        self.n_jobs = n_jobs

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
            equal to ``groups``. If ``groups='auto'``, ``n_splits`` is equal
            to three. If ``groups`` is array-like, ``n_splits`` is equal to
            the length of ``groups``. If ``n_splits=1``, ``X_new`` is squeezed
            and its shape is (n_samples, n_timestamps).

        """
        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape
        window_size, grouping_size = self._check_params(n_timestamps)
        n_windows = n_timestamps - window_size + 1

        try:
            # Get a rough estimation of the required memory
            max_array = np.zeros((
                n_samples + 1, window_size + grouping_size,
                window_size, n_windows
            ))
            del max_array
        except MemoryError:
            msg = "The required memory is greater than the available memory. "
            if self.chunksize is None:
                msg += (
                    "Set the `chunksize` parameter to an integer to perform "
                    "the transformation using chunks (batches) to decrease "
                    "the required memory."
                )
            else:
                msg += (
                    "Decrease the value of the `chunksize` parameter to "
                    "to decrease the required memory."
                )
            raise MemoryError(msg)

        if self.chunksize is not None:
            return self._transform(X)
        else:
            idx = np.r_[
                np.arange(0, n_samples, self.chunksize), n_samples
            ]
            return np.asarray(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(self._transform)(X[i:j])
                    for i, j in zip(idx[:-1], idx[1:])
                )
            )

    def _transform(self, X):

        n_samples, n_timestamps = X.shape
        window_size, grouping_size = self._check_params(n_timestamps)
        n_windows = n_timestamps - window_size + 1

        X_window = np.transpose(
            _windowed_view(X, n_samples, n_timestamps,
                           window_size, window_step=1), axes=(0, 2, 1)
        ).copy()
        X_tranpose = np.matmul(X_window,
                               np.transpose(X_window, axes=(0, 2, 1)))
        w, v = np.linalg.eigh(X_tranpose)
        w, v = w[:, ::-1], v[:, :, ::-1]

        del X_tranpose

        X_elem = _outer_dot(v, X_window, n_samples, window_size, n_windows)
        X_groups, grouping_size = self._grouping(
            X_elem, v, n_samples, window_size, n_windows, grouping_size,
        )
        if window_size >= n_windows:
            X_groups = np.transpose(X_groups, axes=(0, 1, 3, 2))
            gap = window_size
        else:
            gap = n_windows

        del X_elem

        X_ssa = _diagonal_averaging(
            X_groups, n_samples, n_timestamps, window_size,
            n_windows, grouping_size, gap
        )
        return np.squeeze(X_ssa)

    def _grouping(
        self, X, v, n_samples, window_size, n_windows, grouping_size
    ):
        if self.groups is None:
            X_new = X
        elif self.groups == "auto":
            f = np.arange(0, 1 + window_size // 2) / window_size
            Pxx = np.abs(np.fft.rfft(v, axis=1, norm='ortho')) ** 2
            if Pxx.shape[-1] % 2 == 0:
                Pxx[:, 1:-1, :] *= 2
            else:
                Pxx[:, 1:, :] *= 2

            Pxx_cumsum = np.cumsum(Pxx, axis=1)
            idx_trend = np.where(f < self.lower_frequency_bound)[0][-1]
            idx_resid = Pxx_cumsum.shape[1] // 2

            c = self.lower_frequency_contribution
            trend = Pxx_cumsum[:, idx_trend, :] / Pxx_cumsum[:, -1, :] > c
            resid = Pxx_cumsum[:, idx_resid, :] / Pxx_cumsum[:, -1, :] < c
            season = np.logical_and(~trend, ~resid)

            X_new = np.zeros((n_samples, grouping_size,
                              window_size, n_windows))
            for i in range(n_samples):
                for j, arr in enumerate((trend, season, resid)):
                    X_new[i, j] = X[i, arr[i]].sum(axis=0)
        elif isinstance(self.groups, (int, np.integer)):
            grouping = np.linspace(0, window_size,
                                   self.groups + 1).astype('int64')
            X_new = np.zeros((n_samples, grouping_size,
                              window_size, n_windows))
            for i, (j, k) in enumerate(zip(grouping[:-1], grouping[1:])):
                X_new[:, i] = X[:, j:k].sum(axis=1)
        else:
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
                or (isinstance(self.groups, str) and self.groups == "auto")
                or isinstance(self.groups, (int, list, tuple, np.ndarray))):
            raise TypeError("'groups' must be either None, an integer, "
                            "'auto' or array-like.")
        if self.groups is None:
            grouping_size = window_size
        elif (isinstance(self.groups, str) and self.groups == "auto"):
            grouping_size = 3
        elif isinstance(self.groups, (int, np.integer)):
            if not 1 <= self.groups <= self.window_size:
                raise ValueError(
                    "If 'groups' is an integer, it must be greater than or "
                    "equal to 1 and lower than or equal to 'window_size'."
                )
            grouping = np.linspace(
                0, window_size, self.groups + 1).astype('int64')
            grouping_size = len(grouping) - 1
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
            grouping_size = len(self.groups)

        if not isinstance(self.lower_frequency_bound, (float, np.floating)):
            raise TypeError("'lower_frequency_bound' must be a float.")
        else:
            if not 0 < self.lower_frequency_bound < 0.5:
                raise ValueError(
                    "'lower_frequency_bound' must be greater than 0 and "
                    "lower than 0.5."
                )

        if not isinstance(self.lower_frequency_contribution,
                          (float, np.floating)):
            raise TypeError("'lower_frequency_contribution' must be a float.")
        else:
            if not 0 < self.lower_frequency_contribution < 1:
                raise ValueError(
                    "'lower_frequency_contribution' must be greater than 0 "
                    "and lower than 1."
                )

        chunksize_int = isinstance(self.chunksize, (int, np.integer))
        if not (self.chunksize is None or chunksize_int):
            raise TypeError("'chunksize' must be None or an integer.")
        if chunksize_int and self.chunksize < 1:
            raise ValueError("If 'chunksize' is an integer, it must be "
                             "positive (got {})".format(self.chunksize))

        n_jobs_int = (isinstance(self.n_jobs, (int, np.integer)) and
                      self.n_jobs != 0)
        if not (self.n_jobs is None or n_jobs_int):
            raise ValueError("'n_jobs' must be None or an integer not equal "
                             "to zero (got {}).".format(self.n_jobs))
        return window_size, grouping_size
