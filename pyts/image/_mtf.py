"""Code for Markov Transition Field."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from math import ceil
from typing import Self

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from pyts._base import UnivariateTransformerMixin
from pyts.preprocessing import KBinsDiscretizer
from pyts.utils import segmentation


@njit("float64[:,:,:](int64[:,:], int64, int64, int64)")
def _markov_transition_matrix(
    X_binned: npt.NDArray[np.int64],
    n_samples: int,
    n_timestamps: int,
    n_bins: int,
) -> npt.NDArray[np.float64]:
    X_mtm = np.zeros((n_samples, n_bins, n_bins))
    for i in prange(n_samples):
        for j in prange(n_timestamps - 1):
            X_mtm[i, X_binned[i, j], X_binned[i, j + 1]] += 1
    return X_mtm


@njit(
    [
        "float64[:,:,:](int64[:,:], float64[:,:,:], int64, int64, int64)",
        # The direct unit tests for this private helper pass a plain
        # integer array for X_mtm (unlike the real caller, which always
        # passes the float64 output of ``_markov_transition_matrix``), so
        # both are compiled eagerly.
        "float64[:,:,:](int64[:,:], int64[:,:,:], int64, int64, int64)",
    ]
)
def _markov_transition_field(
    X_binned: npt.NDArray[np.int64],
    X_mtm: npt.NDArray[np.float64],
    n_samples: int,
    n_timestamps: int,
    n_bins: int,
) -> npt.NDArray[np.float64]:
    X_mtf = np.zeros((n_samples, n_timestamps, n_timestamps))
    for i in prange(n_samples):
        for j in prange(n_timestamps):
            for k in prange(n_timestamps):
                X_mtf[i, j, k] = X_mtm[i, X_binned[i, j], X_binned[i, k]]
    return X_mtf


@njit(
    [
        "float64[:,:,:](float64[:,:,:], int64, int64, int64[:], int64[:])",
        # The direct unit tests for this private helper pass a plain
        # integer array for X_mtf (unlike the real caller, which always
        # passes the float64 output of ``_markov_transition_field``), so
        # both are compiled eagerly.
        "float64[:,:,:](int64[:,:,:], int64, int64, int64[:], int64[:])",
    ]
)
def _aggregated_markov_transition_field(
    X_mtf: npt.NDArray[np.float64],
    n_samples: int,
    image_size: int,
    start: npt.NDArray[np.int64],
    end: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    X_amtf = np.empty((n_samples, image_size, image_size))
    for i in prange(n_samples):
        for j in prange(image_size):
            for k in prange(image_size):
                X_amtf[i, j, k] = np.mean(
                    X_mtf[i, start[j] : end[j], start[k] : end[k]]
                )
    return X_amtf


# ``X_binned`` (from ``KBinsDiscretizer``) is always int64 and ``X_mtm``
# (from ``_markov_transition_matrix``) is always float64 in this function's
# only call site (``MarkovTransitionField.transform``), so a single
# signature suffices -- unlike ``_aggregated_markov_transition_field``
# above, this one is not directly unit-tested with other dtype
# combinations.
@njit("float64[:,:,:](int64[:,:], float64[:,:,:], int64, int64, int64[:], int64[:])")
def _aggregated_markov_transition_field_fused(
    X_binned: npt.NDArray[np.int64],
    X_mtm: npt.NDArray[np.float64],
    n_samples: int,
    image_size: int,
    start: npt.NDArray[np.int64],
    end: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Aggregate the Markov transition field without ever materializing
    the full (n_samples, n_timestamps, n_timestamps) field that
    ``_markov_transition_field`` would build: each aggregated pixel is a
    mean over ``X_mtm[i, X_binned[i, row], X_binned[i, col]]`` for
    ``row``/``col`` in its window, computed directly instead of first
    writing every (row, col) entry into a full-size intermediate and then
    reducing it away.
    """
    X_amtf = np.empty((n_samples, image_size, image_size))
    for i in prange(n_samples):
        for j in prange(image_size):
            for k in prange(image_size):
                total = 0.0
                count = 0
                for row in range(start[j], end[j]):
                    binned_row = X_binned[i, row]
                    for col in range(start[k], end[k]):
                        total += X_mtm[i, binned_row, X_binned[i, col]]
                        count += 1
                X_amtf[i, j, k] = total / count
    return X_amtf


class MarkovTransitionField(BaseEstimator, UnivariateTransformerMixin):
    """Markov Transition Field.

    Parameters
    ----------
    image_size : int or float (default = 1.)
        Shape of the output images. If float, it represents a percentage
        of the size of each time series and must be between 0 and 1. Output
        images are square, thus providing the size of one dimension is enough.

    n_bins : int (default = 5)
        Number of bins (also known as the size of the alphabet)

    strategy : 'uniform', 'quantile' or 'normal' (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution


    overlapping : bool (default = False)
        If False, reducing the image with the blurring kernel
        will be applied on non-overlapping rectangles. If True,
        it will be applied on possibly overlapping squares.

    flatten : bool (default = False)
        If True, images are flattened to be one-dimensional.

    References
    ----------
    .. [1] Z. Wang and T. Oates, "Encoding Time Series as Images for Visual
           Inspection and Classification Using Tiled Convolutional Neural
           Networks." AAAI Workshop (2015).

    Examples
    --------
    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.image import MarkovTransitionField
    >>> X, _, _, _ = load_gunpoint(return_X_y=True)
    >>> transformer = MarkovTransitionField()
    >>> X_new = transformer.transform(X)
    >>> X_new.shape
    (50, 150, 150)

    """

    def __init__(
        self,
        image_size: int | float = 1.0,
        n_bins: int = 8,
        strategy: str = 'quantile',
        overlapping: bool = False,
        flatten: bool = False,
    ) -> None:
        self.image_size = image_size
        self.n_bins = n_bins
        self.strategy = strategy
        self.overlapping = overlapping
        self.flatten = flatten

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

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Transform each time series into a MTF image.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Input data

        Returns
        -------
        X_new : array-like, shape = (n_samples, image_size, image_size)
            Transformed data. If ``flatten=True``, the shape is
            `(n_samples, image_size * image_size)`.

        """
        X = check_array(X)
        n_samples, n_timestamps = X.shape
        image_size = self._check_params(n_timestamps)

        discretizer = KBinsDiscretizer(n_bins=self.n_bins, strategy=self.strategy)
        X_binned = discretizer.fit_transform(X)

        X_mtm = _markov_transition_matrix(
            X_binned, n_samples, n_timestamps, self.n_bins
        )
        sum_mtm = X_mtm.sum(axis=2)
        np.place(sum_mtm, sum_mtm == 0, 1)
        X_mtm /= sum_mtm[:, :, None]

        # The full (n_samples, n_timestamps, n_timestamps) Markov transition
        # field is never materialized: ``_aggregated_markov_transition_field_
        # fused`` computes each aggregated pixel directly from ``X_binned``
        # and ``X_mtm``.
        window_size, remainder = divmod(n_timestamps, image_size)
        if remainder == 0:
            # Equivalent to reshaping the full field into
            # (n_samples, image_size, window_size, image_size, window_size)
            # and averaging over axes (2, 4): each aggregated pixel (j, k)
            # is the mean over the non-overlapping window
            # [j * window_size, (j + 1) * window_size) x
            # [k * window_size, (k + 1) * window_size).
            start = np.arange(0, n_timestamps, window_size, dtype='int64')
            end = start + window_size
        else:
            window_size += 1
            start, end, _ = segmentation(
                n_timestamps, window_size, self.overlapping, image_size
            )
        X_amtf = _aggregated_markov_transition_field_fused(
            X_binned, X_mtm, n_samples, image_size, start, end
        )
        if self.flatten:
            return X_amtf.reshape(n_samples, -1)
        return X_amtf

    def _check_params(self, n_timestamps: int) -> int:
        if not isinstance(self.image_size, (int, np.integer, float, np.floating)):
            raise TypeError("'image_size' must be an integer or a float.")
        if isinstance(self.image_size, (int, np.integer)):
            if self.image_size < 1 or self.image_size > n_timestamps:
                raise ValueError(
                    "If 'image_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    f"n_timestamps (got {self.image_size})."
                )
            image_size = self.image_size
        else:
            if self.image_size < 0.0 or self.image_size > 1.0:
                raise ValueError(
                    "If 'image_size' is a float, it must be greater than 0 "
                    f"and lower than or equal to 1 (got {self.image_size})."
                )
            image_size = ceil(self.image_size * n_timestamps)
        if not isinstance(self.n_bins, (int, np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        if not self.n_bins >= 2:
            raise ValueError("'n_bins' must be greater than or equal to 2.")
        if self.strategy not in ['uniform', 'quantile', 'normal']:
            raise ValueError("'strategy' must be 'uniform', 'quantile' or 'normal'.")
        return image_size
