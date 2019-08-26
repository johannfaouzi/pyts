"""Code for imputers."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import MissingIndicator
from sklearn.utils.validation import check_array


class InterpolationImputer(BaseEstimator, TransformerMixin):
    """Impute missing values using interpolation.

    Parameters
    ----------
    missing_values : None, np.nan, integer or float (default = np.nan)
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. If an integer or a float,
        the input data must not contain NaN or infinity values.

    strategy : str or int (default = 'linear')
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
        refer to a spline interpolation of zeroth, first, second or third
        order; 'previous' and 'next' simply return the previous or next value
        of the point) or as an integer specifying the order of the spline
        interpolator to use. Default is 'linear'.

    Examples
    --------
    >>> import numpy as np
    >>> from pyts.preprocessing import InterpolationImputer
    >>> X = [[1, None, 3, 4], [8, None, 4, None]]
    >>> imputer = InterpolationImputer()
    >>> imputer.transform(X)
    array([[1., 2., 3., 4.],
           [8., 6., 4., 2.]])

    """

    def __init__(self, missing_values=np.nan, strategy='linear'):
        self.missing_values = missing_values
        self.strategy = strategy

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
        """Perform imputation using interpolation.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data with missing values.

        Returns
        -------
        X_new : array-like, shape = (n_samples, n_timestamps)
            Data without missing values.

        """
        missing_values, force_all_finite = self._check_params()
        X = check_array(X, dtype='float64', force_all_finite=force_all_finite)
        n_samples, n_timestamps = X.shape

        indicator = MissingIndicator(
            missing_values=missing_values, features='all', sparse=False,
        )
        non_missing_idx = ~(indicator.fit_transform(X))
        x_new = np.arange(n_timestamps)
        X_imputed = np.asarray(
            [self._impute_one_sample(X[i], non_missing_idx[i], x_new)
             for i in range(n_samples)]
        )
        return X_imputed

    def _check_params(self):
        if self.missing_values is None:
            missing_values = np.nan
            force_all_finite = 'allow-nan'
        elif isinstance(self.missing_values,
                        (int, np.integer, float, np.floating)):
            if np.isinf(self.missing_values):
                raise ValueError("'missing_values' cannot be infinity.")
            elif np.isnan(self.missing_values):
                force_all_finite = 'allow-nan'
                missing_values = np.nan
            else:
                force_all_finite = True
                missing_values = self.missing_values
        else:
            raise ValueError(
                "'missing_values' must be an integer, a float, None or "
                "np.nan (got {0!s})".format(self.missing_values)
            )
        strategy_str_values = [
            'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next'
        ]
        if not ((isinstance(self.strategy, (int, np.integer)))
                or (self.strategy in strategy_str_values)):
            raise ValueError(
                "'strategy' must be an integer or one of 'linear', 'nearest', "
                "'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next' "
                "(got {0})".format(self.strategy)
            )
        return missing_values, force_all_finite

    def _impute_one_sample(self, x, non_missing_idx, x_new):
        idx = x_new[non_missing_idx]
        f = interp1d(idx, x[non_missing_idx], kind=self.strategy,
                     copy=True, fill_value='extrapolate', assume_sorted=True)
        return f(x_new)
