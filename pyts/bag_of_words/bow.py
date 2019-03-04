"""Code for Bag-of-Words."""

import numpy as np
from math import ceil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from ..utils import windowed_view


class BagOfWwords(BaseEstimator, TransformerMixin):
    """Transform time series into bag of words.

    Parameters
    ----------
    window_size : int or float (default = 0.1)
        Size of the sliding window (i.e. the size of each word). If float, it
        represents the percentage of the size of each time series and must be
        between 0 and 1. The window size will be computed as
        ``ceil(window_size * n_timestamps)``.

    window_step : int or float (default = 1)
        Step of the sliding window. If float, it
        represents the percentage of the size of each time series and must be
        between 0 and 1. The window size will be computed as
        ``ceil(window_step * n_timestamps)``.

    numerosity_reduction : bool (default = True)
        If True, delete sample-wise all but one occurence of back to back
        identical occurences of the same words.

    """

    def __init__(self, window_size=0.1, window_step=1,
                 numerosity_reduction=True):
        self.window_size = window_size
        self.window_step = window_step
        self.numerosity_reduction = numerosity_reduction

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
        """Transform time series into sequences of words.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)

        Returns
        -------
        X_new : array, shape = (n_samples, )
            Transformed data. Each row is a string consisting of words
            separated by a whitespace.

        """
        X = check_array(X, dtype=None)
        n_samples, n_timestamps = X.shape
        window_size, window_step = self._check_params(n_timestamps)
        n_windows = (n_timestamps - window_size + window_step) // window_step

        X_window = windowed_view(X, window_size, window_step)
        X_word = np.asarray([[''.join(X_window[i, j])
                              for j in range(n_windows)]
                             for i in range(n_samples)])

        if self.numerosity_reduction:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1],
                              np.full(n_samples, True)]
            X_bow = np.asarray([' '.join(X_word[i, not_equal[i]])
                                for i in range(n_samples)])
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])
        return X_bow

    def _check_params(self, n_timestamps):
        if not isinstance(self.window_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_size' must be an integer or a float.")
        if isinstance(self.window_size, (int, np.integer)):
            if not 1 <= self.window_size <= n_timestamps:
                raise ValueError(
                    "If 'window_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to the "
                    "size of each time series (i.e. the size of the last "
                    "dimension of X) (got {0}).".format(self.window_size)
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
        if not isinstance(self.window_step,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_step' must be an integer or a float.")
        if isinstance(self.window_step, (int, np.integer)):
            if not 1 <= self.window_step <= n_timestamps:
                raise ValueError(
                    "If 'window_step' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to the "
                    "size of each time series (i.e. the size of the last "
                    "dimension of X) (got {0}).".format(self.window_step)
                )
            window_step = self.window_step
        else:
            if not 0 < self.window_step <= 1:
                raise ValueError(
                    "If 'window_step' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.window_step)
                )
                window_step = ceil(self.window_step * n_timestamps)
        return window_size, window_step
