"""The :mod:`pyts.preprocessing` module includes preprocessing algorithms.

Implemented algorithms are:
- StandardScaler
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler as SS
from sklearn.utils.validation import check_array, check_is_fitted


standard_library.install_aliases()


class StandardScaler(BaseEstimator, TransformerMixin):
    """Standardize time series by removing mean and scaling to unit variance.

    Parameters
    ----------
    norm_mean : bool (default = True)
        If True, center the data before scaling.

    norm_std : bool (default = True)
        If True, scale the data to unit variance.

    """

    def __init__(self, norm_mean=True, norm_std=True):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        """
        self._ss = SS(with_mean=self.norm_mean, with_std=self.norm_std)
        return self

    def transform(self, X):
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        y
            Ignored

        Returns
        -------
        X_new : array-like, shape = [n_samples, n_features]
            Standardized data.

        """
        check_is_fitted(self, '_ss')

        # Check input data
        X = check_array(X)

        # Check parameters
        if not isinstance(self.norm_mean, (int, float)):
            raise TypeError("'norm_mean' must be a boolean.")
        if not isinstance(self.norm_std, (int, float)):
            raise TypeError("'norm_std' must be a boolean.")

        return self._ss.fit_transform(X.T).T
