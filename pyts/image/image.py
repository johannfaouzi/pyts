"""The :mod:`pyts.image` module includes imaging algorithms.

Implemented algorithms are:
- Gramian Angular Summation Field
- Gramian Angular Difference Field
- Markov Transition Field
- Recurrence Plots
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array
from ..approximation import PAA
from ..utils import segmentation


standard_library.install_aliases()


class GASF(BaseEstimator, TransformerMixin):
    """Gramian Angular Summation Field.

    Parameters
    ----------
    image_size : int (default = 32)
        Determine the shape of the output images: (image_size, image_size)

    overlapping : bool (default = False)
        If True, reduce the size of each time series using PAA with possible
        overlapping windows.

    scale : {-1, 0} (default = -1)
        The lower bound of the scaled time series.

    """

    def __init__(self, image_size=32, overlapping=False, scale=-1):
        self.image_size = image_size
        self.overlapping = overlapping
        self.scale = scale

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        """
        return self

    def transform(self, X):
        """Transform each time series into a GASF image.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array-like, shape = [n_samples, image_size, image_size]
            Transformed data.

        """
        # Check input data
        X = check_array(X)

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters
        if not isinstance(self.image_size, int):
            raise TypeError("'image_size' must be an integer.")
        if self.image_size < 2:
            raise ValueError("'image_size' must be greater than or equal "
                             "to 2.")
        if self.image_size > n_features:
            raise ValueError("'image_size' must be lower than or equal to "
                             "the size of each time series.")
        if not isinstance(self.overlapping, (float, int)):
            raise TypeError("'overlapping' must be a boolean.")
        if self.scale not in [0, -1]:
            raise ValueError("'scale' must be either 0 or -1.")

        paa = PAA(output_size=self.image_size, overlapping=self.overlapping)
        X_paa = paa.fit_transform(X)
        scaler = MinMaxScaler(feature_range=(self.scale, 1))
        X_scaled = scaler.fit_transform(X_paa.T).T
        X_sin = np.sqrt(np.clip(1 - X_scaled**2, 0, 1))
        X_scaled_outer = np.apply_along_axis(self._outer, 1, X_scaled)
        X_sin_outer = np.apply_along_axis(self._outer, 1, X_sin)
        return X_scaled_outer - X_sin_outer

    def _outer(self, arr):
        return np.outer(arr, arr)


class GADF(BaseEstimator, TransformerMixin):
    """Gramian Angular Difference Field.

    Parameters
    ----------
    image_size : int (default = 32)
        Determine the shape of the output images: (image_size, image_size)

    overlapping : bool (default = False)
        If True, reducing the size of the time series with PAA is
        done with possible overlapping windows.

    scale : {-1, 0} (default = -1)
        The lower bound of the scaled time series.

    """

    def __init__(self, image_size=32, overlapping=False, scale=-1):
        self.image_size = image_size
        self.overlapping = overlapping
        self.scale = scale

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        """
        return self

    def transform(self, X):
        """Transform each time series into a GADF image.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array-like, shape = [n_samples, image_size, image_size]
            Transformed data.

        """
        # Check input data
        X = check_array(X)

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters
        if not isinstance(self.image_size, int):
            raise TypeError("'image_size' must be an integer.")
        if self.image_size < 2:
            raise ValueError("'image_size' must be greater or equal than 2.")
        if self.image_size > n_features:
            raise ValueError("'image_size' must be lower or equal than "
                             "the size of each time series.")
        if not isinstance(self.overlapping, (float, int)):
            raise TypeError("'overlapping' must be a boolean.")
        if self.scale not in [0, -1]:
            raise ValueError("'scale' must be either 0 or -1.")

        paa = PAA(output_size=self.image_size, overlapping=self.overlapping)
        X_paa = paa.fit_transform(X)
        n_features_new = X_paa.shape[1]
        scaler = MinMaxScaler(feature_range=(self.scale, 1))
        X_scaled = scaler.fit_transform(X_paa.T).T
        X_sin = np.sqrt(np.clip(1 - X_scaled**2, 0, 1))
        X_scaled_sin = np.hstack([X_scaled, X_sin])
        X_scaled_sin_outer = np.apply_along_axis(self._outer_stacked,
                                                 1,
                                                 X_scaled_sin,
                                                 n_features_new,
                                                 True)
        X_sin_scaled_outer = np.apply_along_axis(self._outer_stacked,
                                                 1,
                                                 X_scaled_sin,
                                                 n_features_new,
                                                 False)
        return X_sin_scaled_outer - X_scaled_sin_outer

    def _outer_stacked(self, arr, size, first=True):
        if first:
            return np.outer(arr[:size], arr[size:])
        else:
            return np.outer(arr[size:], arr[:size])


class MTF(BaseEstimator, TransformerMixin):
    """Markov Transition Field.

    Parameters
    ----------
    image_size : int (default = 32)
        Determine the shape of the output images: (image_size, image_size)

    n_bins : int (default = 4)
        Number of bins (also known as the size of the alphabet)

    quantiles : {'gaussian', 'empirical'} (default = 'gaussian')
        The way to compute quantiles. If 'gaussian', quantiles from a
        gaussian distribution N(0,1) are used. If 'empirical', empirical
        quantiles are used.

    overlapping : bool (default = False)
        If False, reducing the image with the blurring kernel
        will be applied on non-overlapping rectangles. If True,
        it will be applied on eventually overlapping squares.

    """

    def __init__(self, image_size=32, n_bins=4,
                 quantiles='empirical', overlapping=False):
        self.image_size = image_size
        self.n_bins = n_bins
        self.quantiles = quantiles
        self.overlapping = overlapping

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        """
        return self

    def transform(self, X):
        """Transform each time series into a MTF image.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array-like, shape = [n_samples, image_size, image_size]
            Transformed data.

        """
        # Check input data
        X = check_array(X)

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters
        if not isinstance(self.image_size, int):
            raise TypeError("'size' must be an integer.")
        if self.image_size < 2:
            raise ValueError("'image_size' must be greater or equal than 2.")
        if self.image_size > n_features:
            raise ValueError("'image_size' must be lower or equal than "
                             "the size of each time series.")
        if not isinstance(self.n_bins, int):
            raise TypeError("'n_bins' must be an integer.")
        if self.n_bins < 2:
            raise ValueError("'n_bins' must be greater or equal than 2.")
        if self.quantiles not in ['gaussian', 'empirical']:
            raise ValueError("'quantiles' must be either 'gaussian' or "
                             "'empirical'.")
        if not isinstance(self.overlapping, (float, int)):
            raise TypeError("'overlapping' must be a boolean.")

        if self.quantiles == 'gaussian':
            bins = scipy.stats.norm.ppf(np.linspace(0, 1,
                                                    self.n_bins + 1
                                                    )[1:-1])
            X_binned = np.apply_along_axis(np.digitize, 1, X, bins)
        else:
            bins = np.percentile(X,
                                 np.linspace(0, 100, self.n_bins + 1)[1:-1],
                                 axis=1)
            X_binned = np.array([np.digitize(X[i], bins[:, i])
                                 for i in range(n_samples)])

        window_size = n_features // self.image_size
        remainder = n_features % self.image_size
        return np.apply_along_axis(self._mtf, 1, X_binned, n_features,
                                   self.image_size, self.n_bins,
                                   self.overlapping, window_size,
                                   remainder)

    def _mtf(self, binned_ts, ts_size, image_size, n_bins, overlapping,
             window_size, remainder):
        # Compute Markov Transition Matrix
        MTM = np.zeros((n_bins, n_bins))
        lagged_ts = np.vstack([binned_ts[:-1], binned_ts[1:]])
        np.add.at(MTM, tuple(map(tuple, lagged_ts)), 1)

        non_zero_rows = np.where(MTM.sum(axis=1) != 0)[0]
        MTM = np.multiply(MTM[non_zero_rows][:, non_zero_rows].T,
                          np.sum(MTM[non_zero_rows], axis=1)**(-1)).T

        # Compute list of indices based on values
        list_values = [np.where(binned_ts == q) for q in non_zero_rows]

        # Compute Markov Transition Field
        MTF = np.zeros((ts_size, ts_size))
        for i in range(non_zero_rows.size):
            for j in range(non_zero_rows.size):
                MTF[np.meshgrid(list_values[i], list_values[j])] = MTM[i, j]

        # Compute Aggregated Markov Transition Field
        if remainder == 0:
            return np.reshape(MTF,
                              (image_size, window_size,
                               image_size, window_size)
                              ).mean(axis=(1, 3))
        else:
            window_size += 1
            start, end, _ = segmentation(ts_size, window_size, overlapping)
            AMTF = np.zeros((image_size, image_size))
            for i in range(image_size):
                for j in range(image_size):
                    AMTF[i, j] = MTF[start[i]:end[i], start[j]:end[j]].mean()

            return AMTF


class RecurrencePlots(BaseEstimator, TransformerMixin):
    """Recurrence Plots.

    Parameters
    ----------
    dimension : int (default = 1)
        Dimension of the trajectory.

    epsilon : float, 'percentage_points', 'percentage_distance' or None
    (default = None)
        Threshold for the minimum distance.

    percentage : float (default = 10)
        Percentage of black points if ``epsilon='percentage_points'``
        or percentage of maximum distance for threshold if
        ``epsilon='percentage_distance'``.

    """

    def __init__(self, dimension=1, epsilon=None, percentage=10):
        self.dimension = dimension
        self.epsilon = epsilon
        self.percentage = percentage

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        """
        return self

    def transform(self, X):
        """Transform each time series into a recurrence plot.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array-like, shape = [n_samples, n_features-dimension+1,
                                     n_features-dimension+1]
            Transformed data.

        """
        # Check input data
        X = check_array(X)
        n_samples, n_features = X.shape

        # Check parameters
        if not isinstance(self.dimension, int):
            raise TypeError("'dimension' must be an integer.")
        if self.dimension <= 0:
            raise ValueError("'dimension' must be greater than or equal to 1.")
        if (self.epsilon is not None and
                self.epsilon not in ['percentage_points',
                                     'percentage_distance'] and
                not isinstance(self.epsilon, (int, float))):
            raise TypeError("'epsilon' must be either None, "
                            "'percentage_points', 'percentage_distance', "
                            "a float or an integer.")
        if (isinstance(self.epsilon, (int, float))) and (self.epsilon < 0):
            raise ValueError("if 'epsilon' is a float or an integer,"
                             "'epsilon' must be greater than or equal to 0.")
        if not isinstance(self.percentage, (int, float)):
            raise TypeError("'percentage' must be a float or an integer.")
        if (self.percentage < 0) or (self.percentage > 100):
            raise ValueError("'percentage' must be between 0 and 100.")

        n_windows = n_features - self.dimension + 1
        X_window = np.transpose(np.asarray([X[:, i: i + self.dimension]
                                for i in range(n_windows)]), axes=(1, 0, 2))
        X_normed = np.linalg.norm(X_window[:, None, :, :] -
                                  X_window[:, :, None, :], axis=3)
        if self.epsilon is None:
            recurrence_plot = X_normed
        elif self.epsilon == 'percentage_points':
            recurrence_plot = X_normed < np.percentile(X_normed,
                                                       self.percentage)
        elif self.epsilon == 'percentage_distance':
            threshold = self.percentage / 100 * np.max(X_normed)
            recurrence_plot = X_normed < threshold
        else:
            recurrence_plot = X_normed < self.epsilon
        return recurrence_plot.astype('float64')
