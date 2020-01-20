"""Code for Reccurence Plot."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import ceil
from numpy.lib.stride_tricks import as_strided
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


def _trajectories(X, dimension, time_delay):
    n_samples, n_timestamps = X.shape
    shape_new = (n_samples,
                 n_timestamps - (dimension - 1) * time_delay,
                 dimension)
    s0, s1 = X.strides
    strides_new = (s0, s1, time_delay * s1)
    return as_strided(X, shape=shape_new, strides=strides_new)


class RecurrencePlot(BaseEstimator, TransformerMixin):  # noqa: D207
    r"""Recurrence Plot.

    A recurrence plot is an image representing the distances between
    trajectories extracted from the original time series.

    Parameters
    ----------
    dimension : int or float (default = 1)
        Dimension of the trajectory. If float, If float, it represents
        a percentage of the size of each time series and must be between
        0 and 1.

    time_delay : int or float (default = 1)
        Time gap between two back-to-back points of the trajectory. If
        float, If float, it represents a percentage of the size of each
        time series and must be between 0 and 1.

    threshold : float, 'point', 'distance' or None (default = None)
        Threshold for the minimum distance. If None, the recurrence plots
        are not binarized. If 'point', the threshold is computed such as
        `percentage` percents of the points are smaller than the threshold.
        If 'distance', the threshold is computed as the `percentage` of the
        maximum distance.

    percentage : int or float (default = 10)
        Percentage of black points if ``threshold='point'`` or percentage of
        maximum distance for threshold if ``threshold='distance'``.
        Ignored if ``threshold`` is a float or None.

    flatten : bool (default = False)
        If True, images are flattened to be one-dimensional.

    Notes
    -----
    Given a time series :math:`(x_1, \ldots, x_n)`, the extracted
    trajectories are

    .. math::

        \vec{x}_i = (x_i, x_{i + \tau}, \ldots, x_{i + (m - 1)\tau}), \quad
        \forall i \in \{1, \ldots, n - (m - 1)\tau \}

    where :math:`m` is the ``dimension`` of the trajectories and :math:`\tau`
    is the ``time_delay``. The recurrence plot, denoted :math:`R`, is the
    pairwise distance between the trajectories

    .. math::

        R_{i, j} = \Theta(\varepsilon - \| \vec{x}_i - \vec{x}_j \|), \quad
        \forall i,j \in \{1, \ldots, n - (m - 1)\tau \}

    where :math:`\Theta` is the Heaviside function and :math:`\varepsilon`
    is the ``threshold``.

    References
    ----------
    .. [1] J.-P Eckmann, S. Oliffson Kamphorst and D Ruelle, "Recurrence
           Plots of Dynamical Systems". Europhysics Letters (1987).

    Examples
    --------
    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.image import RecurrencePlot
    >>> X, _, _, _ = load_gunpoint(return_X_y=True)
    >>> transformer = RecurrencePlot()
    >>> X_new = transformer.transform(X)
    >>> X_new.shape
    (50, 150, 150)

    """

    def __init__(self, dimension=1, time_delay=1,
                 threshold=None, percentage=10, flatten=False):
        self.dimension = dimension
        self.time_delay = time_delay
        self.threshold = threshold
        self.percentage = percentage
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
        """Transform each time series into a recurrence plot.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)

        Returns
        -------
        X_new : array, shape = (n_samples, image_size, image_size)
            Recurrence plots. ``image_size`` is the number of
            trajectories and is equal to
            ``n_timestamps - (dimension - 1) * time_delay``.
            If ``flatten=True``, the shape is
            `(n_samples, image_size * image_size)`.

        """
        X = check_array(X)
        n_samples, n_timestamps = X.shape
        dimension, time_delay = self._check_params(n_timestamps)

        if dimension == 1:
            X_dist = np.abs(X[:, :, None] - X[:, None, :])
        else:
            X_traj = _trajectories(X, dimension, time_delay)
            X_dist = np.sqrt(
                np.sum((X_traj[:, None, :, :] - X_traj[:, :, None, :]) ** 2,
                       axis=3)
            )
        if self.threshold is None:
            X_rp = X_dist
        elif self.threshold == 'point':
            image_size = n_timestamps - (dimension - 1) * time_delay
            percents = np.percentile(
                np.reshape(X_dist, (n_samples, image_size * image_size)),
                self.percentage, axis=1
            )
            X_rp = X_dist < percents[:, None, None]
        elif self.threshold == 'distance':
            percents = self.percentage / 100 * np.max(X_dist, axis=(1, 2))
            X_rp = X_dist < percents[:, None, None]
        else:
            X_rp = X_dist < self.threshold

        if self.flatten:
            return X_rp.reshape(n_samples, -1).astype('float64')
        return X_rp.astype('float64')

    def _check_params(self, n_timestamps):
        if not isinstance(self.dimension,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'dimension' must be an integer or a float.")
        if isinstance(self.dimension, (int, np.integer)):
            if not 1 <= self.dimension <= n_timestamps:
                raise ValueError(
                    "If 'dimension' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.dimension)
                )
            dimension = self.dimension
        else:
            if not 0 < self.dimension < 1.:
                raise ValueError(
                    "If 'dimension' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.dimension)
                )
            dimension = ceil(self.dimension * n_timestamps)

        if not isinstance(self.time_delay,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'time_delay' must be an integer or a float.")
        if isinstance(self.time_delay, (int, np.integer)):
            if not 1 <= self.time_delay <= n_timestamps:
                raise ValueError(
                    "If 'time_delay' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.time_delay)
                )
            time_delay = self.time_delay
        else:
            if not 0 < self.time_delay < 1.:
                raise ValueError(
                    "If 'time_delay' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.time_delay)
                )
            time_delay = ceil(self.time_delay * n_timestamps)

        if n_timestamps - (dimension - 1) * time_delay < 1:
            raise ValueError("The number of trajectories must be positive. "
                             "Consider trying with smaller values for "
                             "'dimension' and 'time_delay'.")

        if (self.threshold is not None
            and self.threshold not in ['point', 'distance']
            and not isinstance(self.threshold,
                               (int, np.integer, float, np.floating))):
            raise TypeError("'threshold' must be either None, 'point', "
                            "'distance', a float or an integer.")
        threshold_number = isinstance(self.threshold,
                                      (int, np.integer, float, np.floating))
        if threshold_number and (self.threshold <= 0):
            raise ValueError("If 'threshold' is a float or an integer, "
                             "it must be greater than or equal to 0.")

        if not isinstance(self.percentage,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'percentage' must be a float or an integer.")
        if not 0 <= self.percentage <= 100:
            raise ValueError("'percentage' must be between 0 and 100.")

        return dimension, time_delay
