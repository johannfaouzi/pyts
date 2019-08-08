"""Joint Recurrence Plots."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ...image import RecurrencePlot
from ..utils import check_3d_array


class JointRecurrencePlot(BaseEstimator, TransformerMixin):
    r"""Joint Recurrence Plot.

    A recurrence plot is an image representing the distances between
    trajectories extracted from the original time series.

    A joint recurrence plot is an extension of recurrence plots for
    multivariate time series: it is the Hadamard of the recurrence
    plots obtained for each feature of the multivariate time series.

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

    threshold : float, 'point', 'distance' or None or list thereof (default = None)
        Threshold for the minimum distance. If None, the recurrence plots
        are not binarized. If 'point', the threshold is computed such as
        `percentage` percents of the points are smaller than the threshold.
        If 'distance', the threshold is computed as the `percentage` of the
        maximum distance.

    percentage : int, float or list thereof (default = 10)
        Percentage of black points if ``threshold='point'`` or percentage of
        maximum distance for threshold if ``threshold='distance'``.
        Ignored if ``threshold`` is a float or None.

    References
    ----------
    .. [1] M. Romano, M. Thiel, J. Kurths and W. con Bloh, "Multivariate
           Recurrence Plots". Physics Letters A (2004)

    """  # noqa: E501

    def __init__(self, dimension=1, time_delay=1, threshold=None,
                 percentage=10):
        self.dimension = dimension
        self.time_delay = time_delay
        self.threshold = threshold
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
        """Transform each time series into a joint recurrence plot.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        Returns
        -------
        X_new : array, shape = (n_samples, image_size, image_size)
            Joint Recurrence plots. ``image_size`` is the number of
            trajectories and is equal to
            ``n_timestamps - (dimension - 1) * time_delay``.

        """
        X = check_3d_array(X)
        _, n_features, _ = X.shape
        thresholds_, percentages_ = self._check_params(n_features)

        X_rp = [self._joint_recurrence_plot(
            X[:, i, :], self.dimension, self.time_delay,
            thresholds_[i], percentages_[i]) for i in range(n_features)]
        X_jrp = np.product(X_rp, axis=0)
        return X_jrp

    @staticmethod
    def _joint_recurrence_plot(X, dimension, time_delay,
                               threshold, percentage):
        recurrence_plot = RecurrencePlot(
            dimension, time_delay, threshold, percentage)
        return recurrence_plot.transform(X)

    def _check_params(self, n_features):
        if isinstance(self.threshold, (tuple, list, np.ndarray)):
            if len(self.threshold) != n_features:
                raise ValueError(
                    "If 'threshold' is a list, its length must be equal to "
                    "n_features ({0} != {1})."
                    .format(len(self.threshold), n_features)
                )
            thresholds_ = self.threshold
        else:
            thresholds_ = [self.threshold for _ in range(n_features)]
        if isinstance(self.percentage, (tuple, list, np.ndarray)):
            if len(self.percentage) != n_features:
                raise ValueError(
                    "If 'percentage' is a list, its length must be equal to "
                    "n_features ({0} != {1})."
                    .format(len(self.percentage), n_features)
                )
            percentages_ = self.percentage
        else:
            percentages_ = [self.percentage for _ in range(n_features)]
        return thresholds_, percentages_
