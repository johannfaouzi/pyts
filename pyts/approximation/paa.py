"""Code for Piecewise Aggregate Approximation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import ceil
from numba import njit, prange
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from ..utils import segmentation


@njit(parallel=True)
def _paa(X, n_samples, n_timestamps, start, end, n_timestamps_new):
    X_paa = np.empty((n_samples, n_timestamps_new))
    for i in prange(n_samples):
        for j in prange(n_timestamps_new):
            X_paa[i, j] = np.mean(X[i, start[j]:end[j]])
    return X_paa


class PiecewiseAggregateApproximation(BaseEstimator, TransformerMixin):
    """Piecewise Aggregate Approximation.

    Parameters
    ----------
    window_size : int, float or None (default = 1)
        Length of the sliding window. If float, it represents
        a percentage of the size of each time series and must be
        between 0 and 1.

    output_size : int, float or None (default = None)
        Size of the returned time series. If float, it represents
        a percentage of the size of each time series and must be
        between 0. and 1. Ignored if ``window_size`` is not None.
        It can't be None if ``window_size`` is None. If you want to use
        ``output_size`` over ``window_size``, you must set
        ``window_size=None``.

    overlapping : bool (default = True)
        When ``window_size=None``, ``output_size`` is used to derive the window
        size; the window size is fixed if ``overlapping=True`` and may vary
        if ``overlapping=False``. Ignored if ``window_size`` is specified.

    References
    ----------
    .. [1] E. Keogh, K. Chakrabarti, M. Pazzani, and S. Mehrotra,
           "Dimensionality reduction for fast similarity search in large
           time series databases". Knowledge and information Systems,
           3(3), 263-286 (2001).

    Examples
    --------
    >>> from pyts.approximation import PiecewiseAggregateApproximation
    >>> X = [[0, 4, 2, 1, 7, 6, 3, 5],
    ...      [2, 5, 4, 5, 3, 4, 2, 3]]
    >>> transformer = PiecewiseAggregateApproximation(window_size=2)
    >>> transformer.transform(X)
    array([[2. , 1.5, 6.5, 4. ],
           [3.5, 4.5, 3.5, 2.5]])

    """

    def __init__(self, window_size=1, output_size=None, overlapping=True):
        self.window_size = window_size
        self.output_size = output_size
        self.overlapping = overlapping

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
        self
            object

        """
        return self

    def transform(self, X):
        """Reduce the dimensionality of each time series.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)

        Returns
        -------
        X_new : array, shape = (n_samples, n_timestamps_new)

        """
        X = check_array(X)
        n_samples, n_timestamps = X.shape

        window_size, output_size = self._check_params(n_timestamps)
        if window_size == 1:
            return X
        else:
            start, end, n_timestamps_new = segmentation(
                n_timestamps, window_size, self.overlapping, output_size
            )
            X_paa = _paa(X, n_samples, n_timestamps,
                         start, end, n_timestamps_new)
            return X_paa

    def _check_params(self, n_timestamps):
        if (self.window_size is None and self.output_size is None):
            raise TypeError("'window_size' and 'output_size' cannot be "
                            "both None.")
        if self.window_size is not None:
            if not isinstance(self.window_size,
                              (int, np.integer, float, np.floating)):
                raise TypeError("If specified, 'window_size' must be an "
                                "integer or a float.")
            if isinstance(self.window_size, (int, np.integer)):
                if not 1 <= self.window_size <= n_timestamps:
                    raise ValueError(
                        "If 'window_size' is an integer, it must be greater "
                        "than or equal to 1 and lower than or equal to "
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
                window_size = ceil(self.window_size * n_timestamps)
            output_size = None
        else:
            if not isinstance(self.output_size,
                              (int, np.integer, float, np.floating)):
                raise TypeError("If specified, 'output_size' must be an "
                                "integer or a float.")
            if isinstance(self.output_size, (int, np.integer)):
                if not 1 <= self.output_size <= n_timestamps:
                    raise ValueError(
                        "If 'output_size' is an integer, it must be greater "
                        "than or equal to 1 and lower than or equal to "
                        "n_timestamps (got {0}).".format(self.output_size)
                    )
                output_size = self.output_size
            else:
                if not 0 < self.output_size <= 1.:
                    raise ValueError(
                        "If 'output_size' is a float, it must be greater "
                        "than 0 and lower than or equal to 1 "
                        "(got {0}).".format(self.output_size)
                    )
                output_size = ceil(self.output_size * n_timestamps)
            window_size, remainder = divmod(n_timestamps, output_size)
            if remainder != 0:
                window_size += 1
        return window_size, output_size
