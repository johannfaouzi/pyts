"""Code for Gramian Angular Field."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import ceil
from numba import njit, prange
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from ..approximation import PiecewiseAggregateApproximation
from ..preprocessing import MinMaxScaler


@njit()
def _gasf(X_cos, X_sin, n_samples, image_size):
    X_gasf = np.empty((n_samples, image_size, image_size))
    for i in prange(n_samples):
        X_gasf[i] = np.outer(X_cos[i], X_cos[i]) - np.outer(X_sin[i], X_sin[i])
    return X_gasf


@njit()
def _gadf(X_cos, X_sin, n_samples, image_size):
    X_gadf = np.empty((n_samples, image_size, image_size))
    for i in prange(n_samples):
        X_gadf[i] = np.outer(X_sin[i], X_cos[i]) - np.outer(X_cos[i], X_sin[i])
    return X_gadf


class GramianAngularField(BaseEstimator, TransformerMixin):
    """Gramian Angular Field.

    Parameters
    ----------
    image_size : int or float (default = 1.)
        Shape of the output images. If float, it represents a percentage
        of the size of each time series and must be between 0 and 1. Output
        images are square, thus providing the size of one dimension is enough.

    sample_range : None or tuple (min, max) (default = (-1, 1))
        Desired range of transformed data. If None, no scaling is performed
        and all the values of the input data must be between -1 and 1.
        If tuple, each sample is scaled between min and max; min must be
        greater than or equal to -1 and max must be lower than or equal to 1.

    method : 'summation' or 'difference' (default = 'summation')
        Type of Gramian Angular Field. 's' can be used for 'summation'
        and 'd' for 'difference'.

    overlapping : bool (default = False)
        If True, reduce the size of each time series using PAA with possible
        overlapping windows.

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
    >>> from pyts.image import GramianAngularField
    >>> X, _, _, _ = load_gunpoint(return_X_y=True)
    >>> transformer = GramianAngularField()
    >>> X_new = transformer.transform(X)
    >>> X_new.shape
    (50, 150, 150)

    """

    def __init__(self, image_size=1., sample_range=(-1, 1),
                 method='summation', overlapping=False, flatten=False):
        self.image_size = image_size
        self.sample_range = sample_range
        self.method = method
        self.overlapping = overlapping
        self.flatten = flatten

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
        """Transform each time series into a GAF image.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)

        Returns
        -------
        X_new : array-like, shape = (n_samples, image_size, image_size)
            Transformed data. If ``flatten=True``, the shape is
            `(n_samples, image_size * image_size)`.

        """
        X = check_array(X)
        n_samples, n_timestamps = X.shape
        image_size = self._check_params(n_timestamps)

        paa = PiecewiseAggregateApproximation(
            window_size=None, output_size=image_size,
            overlapping=self.overlapping
        )
        X_paa = paa.fit_transform(X)
        if self.sample_range is None:
            X_min, X_max = np.min(X_paa), np.max(X_paa)
            if (X_min < -1) or (X_max > 1):
                raise ValueError("If 'sample_range' is None, all the values "
                                 "of X must be between -1 and 1.")
            X_cos = X_paa
        else:
            scaler = MinMaxScaler(sample_range=self.sample_range)
            X_cos = scaler.fit_transform(X_paa)
        X_sin = np.sqrt(np.clip(1 - X_cos ** 2, 0, 1))
        if self.method in ['s', 'summation']:
            X_new = _gasf(X_cos, X_sin, n_samples, image_size)
        else:
            X_new = _gadf(X_cos, X_sin, n_samples, image_size)

        if self.flatten:
            return X_new.reshape(n_samples, -1)
        return X_new

    def _check_params(self, n_timestamps):
        if not isinstance(self.image_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'image_size' must be an integer or a float.")
        if isinstance(self.image_size, (int, np.integer)):
            if self.image_size < 1 or self.image_size > n_timestamps:
                raise ValueError(
                    "If 'image_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.image_size)
                )
            image_size = self.image_size
        else:
            if not (0 < self.image_size <= 1.):
                raise ValueError(
                    "If 'image_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {0})."
                    .format(self.image_size)
                )
            image_size = ceil(self.image_size * n_timestamps)
        if not ((self.sample_range is None)
                or (isinstance(self.sample_range, tuple))):
            raise TypeError("'sample_range' must be None or a tuple.")
        if isinstance(self.sample_range, tuple):
            if len(self.sample_range) != 2:
                raise ValueError("If 'sample_range' is a tuple, its length "
                                 "must be equal to 2.")
            if not -1 <= self.sample_range[0] < self.sample_range[1] <= 1:
                raise ValueError(
                    "If 'sample_range' is a tuple, it must satisfy "
                    "-1 <= sample_range[0] < sample_range[1] <= 1."
                )
        if self.method not in ['s', 'd', 'summation', 'difference']:
            raise ValueError("'method' must be either 'summation', 's', "
                             "'difference' or 'd'.")
        return image_size
