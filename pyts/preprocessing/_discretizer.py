"""Code for discretizers."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from typing import Any, Self
from warnings import warn

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from numba.typed import List
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from pyts._base import UnivariateTransformerMixin


@njit(
    [
        "float64[:,:](float64[:], float64[:], int64, int64)",
        # The direct unit tests for this private helper exercise it with
        # plain integer arrays (unlike its real caller, which always
        # passes float64 after ``check_array``), so both are compiled
        # eagerly rather than only covering the production call site.
        "float64[:,:](int64[:], int64[:], int64, int64)",
    ]
)
def _uniform_bins(
    sample_min: npt.NDArray[np.float64],
    sample_max: npt.NDArray[np.float64],
    n_samples: int,
    n_bins: int,
) -> npt.NDArray[np.float64]:
    bin_edges = np.empty((n_bins - 1, n_samples))
    for i in prange(n_samples):
        bin_edges[:, i] = np.linspace(sample_min[i], sample_max[i], n_bins + 1)[1:-1]
    return bin_edges


@njit("float64[:,:](float64[:,:], float64[:], int64, int64)")
def _digitize_1d(
    X: npt.NDArray[np.float64],
    bins: npt.NDArray[np.float64],
    n_samples: int,
    n_timestamps: int,
) -> npt.NDArray[np.float64]:
    X_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_samples):
        X_digit[i] = np.searchsorted(bins, X[i], side='left')
    return X_digit


@njit("float64[:,:](float64[:,:], float64[:,:], int64, int64)")
def _digitize_2d(
    X: npt.NDArray[np.float64],
    bins: npt.NDArray[np.float64],
    n_samples: int,
    n_timestamps: int,
) -> npt.NDArray[np.float64]:
    X_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_samples):
        X_digit[i] = np.searchsorted(bins[i], X[i], side='left')
    return X_digit


def _digitize(
    X: npt.NDArray[np.float64], bins: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    n_samples, n_timestamps = X.shape
    if bins.ndim == 1:
        X_binned = _digitize_1d(X, bins, n_samples, n_timestamps)
    else:
        X_binned = _digitize_2d(X, bins, n_samples, n_timestamps)
    return X_binned.astype('int64')


# Deliberately left without an explicit signature: its real caller passes
# a ``numba.typed.List`` of float64 arrays, but its direct unit tests pass
# plain tuples of arrays of varying arity (2- and 3-tuples) and dtypes
# (int64 and float64) to exercise the padding logic conveniently. Each of
# those tuple shapes is a distinct numba type, so eagerly enumerating every
# combination used across both production and tests is impractical here;
# lazy compilation is the correct choice.
@njit
def _reshape_with_nan(
    X: Any, n_samples: int, lengths: npt.NDArray[np.int64], max_length: int
) -> npt.NDArray[np.float64]:
    X_fill = np.full((n_samples, max_length), np.nan)
    for i in prange(n_samples):
        X_fill[i, : lengths[i]] = X[i]
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

    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = 'quantile',
        raise_warning: bool = True,
    ) -> None:
        self.n_bins = n_bins
        self.strategy = strategy
        self.raise_warning = raise_warning

    def fit(
        self, X: npt.ArrayLike | None = None, y: npt.ArrayLike | None = None
    ) -> Self:
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

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.int64]:
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
        X = check_array(X, dtype=np.float64)
        n_samples, n_timestamps = X.shape
        self._check_params(n_timestamps)

        bin_edges = self._compute_bins(X, n_samples, self.n_bins, self.strategy)
        X_new = _digitize(X, bin_edges)
        return X_new

    def _check_params(self, n_timestamps: int) -> None:
        if not isinstance(self.n_bins, (int, np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        if not 2 <= self.n_bins:
            raise ValueError(
                f"'n_bins' must be greater than or equal to 2 (got {self.n_bins})."
            )
        if self.strategy not in ['uniform', 'quantile', 'normal']:
            raise ValueError(
                "'strategy' must be either 'uniform', 'quantile' "
                f"or 'normal' (got {self.strategy})."
            )

    def _compute_bins(
        self,
        X: npt.NDArray[np.float64],
        n_samples: int,
        n_bins: int,
        strategy: str,
    ) -> npt.NDArray[np.float64]:
        if strategy == 'normal':
            bin_edges = norm.ppf(np.linspace(0, 1, self.n_bins + 1)[1:-1])
        elif strategy == 'uniform':
            sample_min, sample_max = np.min(X, axis=1), np.max(X, axis=1)
            bin_edges = _uniform_bins(sample_min, sample_max, n_samples, n_bins).T
        else:
            bin_edges = np.percentile(
                X, np.linspace(0, 100, self.n_bins + 1)[1:-1], axis=1
            ).T
            mask = np.c_[
                ~np.isclose(0, np.diff(bin_edges, axis=1), rtol=0, atol=1e-8),
                np.full((n_samples, 1), True),
            ]
            if (self.n_bins > 2) and np.any(~mask):
                samples = np.where(np.any(~mask, axis=1))[0]
                if self.raise_warning:
                    warn(
                        "Some quantiles are equal. The number of bins "
                        f"will be smaller for sample {samples}. Consider "
                        "decreasing the number of bins or removing these "
                        "samples.",
                        UserWarning,
                        stacklevel=2,
                    )
                lengths = np.sum(mask, axis=1)
                max_length = np.max(lengths)

                bin_edges_ = List()
                for i in range(n_samples):
                    bin_edges_.append(bin_edges[i][mask[i]])

                bin_edges = _reshape_with_nan(
                    bin_edges_, n_samples, lengths, max_length
                )
        return bin_edges
