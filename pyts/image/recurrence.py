"""Code for Reccurence Plot."""

import numpy as np
from math import sqrt
from numba import prange, njit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


@njit()
def _trajectory_distances(X, n_samples, n_timestamps, dimension, image_size):
    X_dist = np.zeros((n_samples, image_size, image_size))
    for i in prange(n_samples):
        for j in prange(image_size):
            for k in prange(j + 1, image_size):
                value = sqrt(
                    np.sum(
                        (X[i, j: j + dimension] - X[i, k: k + dimension]) ** 2)
                )
                X_dist[i, j, k] = X_dist[i, k, j] = value
    return X_dist


class RecurrencePlot(BaseEstimator, TransformerMixin):
    """Recurrence Plot.

    Parameters
    ----------
    dimension : int or float (default = 1)
        Dimension of the trajectory.

    epsilon : float, 'percentage_points', 'percentage_distance' or None
    (default = None)
        Threshold for the minimum distance.

    percentage : int or float (default = 10)
        Percentage of black points if ``epsilon='percentage_points'``
        or percentage of maximum distance for threshold if
        ``epsilon='percentage_distance'``. Ignored if ``epsilon`` is
        a float or None.

    References
    ----------
    .. [1] J.-P Eckmann and S. Oliffson Kamphorst and D Ruelle, "Recurrence
           Plots of Dynamical Systems". Europhysics Letters (1987).

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

        Returns
        -------
        self : object

        """
        return self

    def transform(self, X):
        """Transform each time series into a recurrence plot.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)

        Returns
        -------
        X_new : array, shape = (n_samples, image_size, image_size)
            Transformed data. ``image_size`` is the number of
            trajectories and is equal to ``n_timetamps - dimension + 1``.

        """
        X = check_array(X)
        n_samples, n_timestamps = X.shape
        dimension = self._check_params(n_timestamps)

        image_size = n_timestamps - dimension + 1
        if dimension == 1:
            X_dist = np.abs(X[:, :, None] - X[:, None, :])
        else:
            X_dist = _trajectory_distances(
                X, n_samples, n_timestamps, dimension, image_size
            )
        if self.epsilon is None:
            X_rp = X_dist
        elif self.epsilon == 'percentage_points':
            percents = np.percentile(
                np.reshape(X_dist, (n_samples, image_size * image_size)),
                self.percentage,
                axis=1
            )
            X_rp = X_dist < percents[:, None, None]
        elif self.epsilon == 'percentage_distance':
            percents = self.percentage / 100 * np.max(X_dist, axis=(1, 2))
            X_rp = X_dist < percents[:, None, None]
        else:
            X_rp = X_dist < self.epsilon
        return X_rp.astype('float64')

    def _check_params(self, n_timestamps):
        if not isinstance(self.dimension,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'dimension' must be an integer or a float.")
        if isinstance(self.dimension, (int, np.integer)):
            if self.dimension < 1 or self.dimension > n_timestamps:
                raise ValueError(
                    "If 'dimension' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to the size "
                    "of each time series (i.e. the size of the last dimension "
                    "of X) (got {0}).".format(self.dimension)
                )
            dimension = self.dimension
        else:
            if self.dimension < 0. or self.dimension > 1.:
                raise ValueError(
                    "If 'dimension' is a float, it must be greater "
                    "than or equal to 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.dimension)
                )
            dimension = int(self.dimension * n_timestamps)
        if (self.epsilon is not None
            and self.epsilon not in ['percentage_points',
                                     'percentage_distance']
            and not isinstance(self.epsilon,
                               (int, np.integer, float, np.floating))):
            raise TypeError("'epsilon' must be either None, "
                            "'percentage_points', 'percentage_distance', "
                            "a float or an integer.")
        if ((isinstance(self.epsilon, (int, np.integer, float, np.floating)))
            and (self.epsilon < 0)):
            raise ValueError("If 'epsilon' is a float or an integer,"
                             "'epsilon' must be greater than or equal to 0.")
        if not isinstance(self.percentage,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'percentage' must be a float or an integer.")
        if not 0 <= self.percentage <= 100:
            raise ValueError("'percentage' must be between 0 and 100.")
        return dimension
