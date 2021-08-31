"""Code for discretizers."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from warnings import warn
from ..base import UnivariateTransformerMixin


@njit()
def _uniform_bins(sample_min, sample_max, n_samples, n_bins):
    bin_edges = np.empty((n_bins - 1, n_samples))
    for i in prange(n_samples):
        bin_edges[:, i] = np.linspace(
            sample_min[i], sample_max[i], n_bins + 1)[1:-1]
    return bin_edges


@njit()
def _digitize_1d(X, bins, n_samples, n_timestamps):
    X_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_samples):
        X_digit[i] = np.searchsorted(bins, X[i], side='left')
    return X_digit


@njit()
def _digitize_2d(X, bins, n_samples, n_timestamps):
    X_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_samples):
        X_digit[i] = np.searchsorted(bins[i], X[i], side='left')
    return X_digit


def _digitize(X, bins):
    n_samples, n_timestamps = X.shape
    if bins.ndim == 1:
        X_binned = _digitize_1d(X, bins, n_samples, n_timestamps)
    else:
        X_binned = _digitize_2d(X, bins, n_samples, n_timestamps)
    return X_binned.astype('int64')


@njit
def _reshape_with_nan(X, n_samples, lengths, max_length):
    X_fill = np.full((n_samples, max_length), np.nan)
    for i in prange(n_samples):
        X_fill[i, :lengths[i]] = X[i]
    return X_fill


class KBinsDiscretizer(BaseEstimator, UnivariateTransformerMixin):
    """Bin continuous data into intervals sample-wise.

    Parameters
    ----------
    n_bins : int (default = 5)
        The number of bins to produce. The intervals for the bins are
        determined by the minimum and maximum of the input data. It must
        be greater than or equal to 2.

    strategy : 'uniform', 'quantile' or 'normal' (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution

    raise_warning : bool (default = True)
        If True, a warning is raised when the number of bins is smaller for
        at least one sample. In this case, you should consider decreasing the
        number of bins or removing these samples.

    Examples
    --------
    >>> from pyts.preprocessing import KBinsDiscretizer
    >>> X = [[0, 1, 0, 2, 3, 3, 2, 1],
    ...      [7, 0, 6, 1, 5, 3, 4, 2]]
    >>> discretizer = KBinsDiscretizer(n_bins=2)
    >>> print(discretizer.transform(X))
    [[0 0 0 1 1 1 1 0]
     [1 0 1 0 1 0 1 0]]

    """

    def __init__(self, n_bins=5, strategy='quantile', raise_warning=True):
        self.n_bins = n_bins
        self.strategy = strategy
        self.raise_warning = raise_warning

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        Returns
        -------
        self : object

        """
        return self

    def transform(self, X):
        """Bin the data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        Returns
        -------
        X_new : array-like, shape = (n_samples, n_timestamps)
            Binned data.

        """
        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape
        self._check_params(n_timestamps)

        bin_edges = self._compute_bins(
            X, n_samples, self.n_bins, self.strategy)
        X_new = _digitize(X, bin_edges)
        return X_new

    def _check_params(self, n_timestamps):
        if not isinstance(self.n_bins, (int, np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        if not 2 <= self.n_bins:
            raise ValueError(
                "'n_bins' must be greater than or equal to 2 (got {0})."
                .format(self.n_bins)
            )
        if self.strategy not in ['uniform', 'quantile', 'normal']:
            raise ValueError("'strategy' must be either 'uniform', 'quantile' "
                             "or 'normal' (got {0}).".format(self.strategy))

    def _compute_bins(self, X, n_samples, n_bins, strategy):
        if strategy == 'normal':
            bin_edges = norm.ppf(np.linspace(0, 1, self.n_bins + 1)[1:-1])
        elif strategy == 'uniform':
            sample_min, sample_max = np.min(X, axis=1), np.max(X, axis=1)
            bin_edges = _uniform_bins(
                sample_min, sample_max, n_samples, n_bins).T
        else:
            bin_edges = np.percentile(
                X, np.linspace(0, 100, self.n_bins + 1)[1:-1], axis=1
            ).T
            mask = np.c_[
                ~np.isclose(0, np.diff(bin_edges, axis=1), rtol=0, atol=1e-8),
                np.full((n_samples, 1), True)
            ]
            if (self.n_bins > 2) and np.any(~mask):
                samples = np.where(np.any(~mask, axis=0))[0]
                if self.raise_warning:
                    warn("Some quantiles are equal. The number of bins will "
                         "be smaller for sample {0}. Consider decreasing the "
                         "number of bins or removing these samples."
                         .format(samples))
                lengths = np.sum(mask, axis=1)
                max_length = np.max(lengths)

                bin_edges_ = List()
                for i in range(n_samples):
                    bin_edges_.append(bin_edges[i][mask[i]])

                bin_edges = _reshape_with_nan(bin_edges_, n_samples,
                                              lengths, max_length)
        return bin_edges
