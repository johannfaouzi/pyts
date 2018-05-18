"""The :mod:`pyts.bop` module includes bag-of-words algorithms.

Implemented algorithms are:
- Bag-Of-Words
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
from ..utils import numerosity_reduction


standard_library.install_aliases()


class BOW(BaseEstimator, TransformerMixin):
    """Bag Of Words.

    Parameters
    ----------
    window_size : int (default = 4)
        Size of the window (i.e. the size of each word).

    numerosity_reduction : bool (default = True)
        If True, delete all but one occurence of back to back
        identical occurences of the same words.

    """

    def __init__(self, window_size=4, numerosity_reduction=True):
        self.window_size = window_size
        self.numerosity_reduction = numerosity_reduction

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
        """Transform a sequence of letters into a sequence of words.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array-like, size = [n_samples]
            Transformed data.

        """
        # Check input data
        X = check_array(X)

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters
        if not isinstance(self.window_size, int):
            raise TypeError("'window_size' must be an integer.")
        if self.window_size < 1:
            raise ValueError("'window_size' must be greater than or equal "
                             "to 1.")
        if self.window_size > n_features:
            raise ValueError("'window_size' must be lower than or equal to "
                             "the size of each time series.")
        if not isinstance(self.numerosity_reduction, (int, float)):
            raise TypeError("'numerosity_reduction' must be a boolean.")

        n_windows = n_features - self.window_size + 1
        X_window = np.asarray([X[:, i: i + self.window_size]
                               for i in range(n_windows)])
        X_window = X_window.reshape(n_samples * n_windows, -1, order='F')
        X_bow = np.apply_along_axis(lambda x: ''.join(x),
                                    1,
                                    X_window).reshape(n_samples, -1)
        if self.numerosity_reduction:
            X_bow = np.apply_along_axis(numerosity_reduction, 1, X_bow)
        else:
            X_bow = np.apply_along_axis(lambda x: ' '.join(x), 1, X_bow)
        return X_bow
