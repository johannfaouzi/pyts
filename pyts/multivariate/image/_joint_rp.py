"""Joint Recurrence Plots."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from typing import Any, Self

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from pyts._base import MultivariateTransformerMixin
from pyts.image import RecurrencePlot
from pyts.multivariate.utils import check_3d_array


class JointRecurrencePlot(BaseEstimator, MultivariateTransformerMixin):
    r"""Joint Recurrence Plot.

    A recurrence plot is an image representing the distances between
    trajectories extracted from the original time series.

    A joint recurrence plot is an extension of recurrence plots for
    multivariate time series: it is the Hadamard of the recurrence
    plots obtained for each feature of the multivariate time series.

    Parameters
    ----------
    dimension : int or float (default = 1)
        Dimension of the trajectory. If float, it represents a percentage of
        the size of each time series and must be between 0 and 1.

    time_delay : int or float (default = 1)
        Time gap between two back-to-back points of the trajectory. If
        float, it represents a percentage of the size of each time series and
        must be between 0 and 1.

    threshold : float, 'point', 'distance' or None or list thereof (default = None)
        Threshold for the minimum distance. If None, the recurrence plots
        are not binarized. If ``threshold='point'``, the threshold is computed
        such as ``percentage`` percents of the points are smaller than the
        threshold. If ``threshold='distance'``, the threshold is computed as
        the ``percentage`` of the maximum distance.

    percentage : int, float or list thereof (default = 10)
        Percentage of black points if ``threshold='point'`` or percentage of
        maximum distance for threshold if ``threshold='distance'``.
        Ignored if ``threshold`` is a float or None. Note that the percentage
        is calculated for each recurrence plot independently, which implies
        that there will probably be less than `percentage` percents of black
        points in the joint recurrence plot.

    References
    ----------
    .. [1] M. Romano, M. Thiel, J. Kurths and W. con Bloh, "Multivariate
           Recurrence Plots". Physics Letters A (2004)

    Examples
    --------
    >>> from pyts.datasets import load_basic_motions
    >>> from pyts.multivariate.image import JointRecurrencePlot
    >>> X, _, _, _ = load_basic_motions(return_X_y=True)
    >>> transformer = JointRecurrencePlot()
    >>> X_new = transformer.transform(X)
    >>> X_new.shape
    (40, 100, 100)

    """

    def __init__(
        self,
        dimension: int | float = 1,
        time_delay: int | float = 1,
        threshold: float | str | npt.ArrayLike | None = None,
        percentage: int | float | npt.ArrayLike = 10,
    ) -> None:
        self.dimension = dimension
        self.time_delay = time_delay
        self.threshold = threshold
        self.percentage = percentage

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

        # An in-place running product avoids two forms of waste that
        # ``np.prod([...], axis=0)`` on a Python list would incur: keeping
        # every one of the ``n_features`` per-feature recurrence plots
        # alive simultaneously, and ``np.prod`` first stacking that list
        # into one contiguous ``(n_features, n_samples, image_size,
        # image_size)`` array before reducing it — an extra
        # ``n_features``-fold intermediate on top of arrays that are
        # already large (see ``RecurrencePlot``).
        X_jrp = self._joint_recurrence_plot(
            X[:, 0, :],
            self.dimension,
            self.time_delay,
            thresholds_[0],
            percentages_[0],
        )
        for i in range(1, n_features):
            X_jrp *= self._joint_recurrence_plot(
                X[:, i, :],
                self.dimension,
                self.time_delay,
                thresholds_[i],
                percentages_[i],
            )
        return X_jrp

    @staticmethod
    def _joint_recurrence_plot(
        X: npt.NDArray[np.float64],
        dimension: int | float,
        time_delay: int | float,
        threshold: float | str | None,
        percentage: int | float,
    ) -> npt.NDArray[np.float64]:
        recurrence_plot = RecurrencePlot(dimension, time_delay, threshold, percentage)
        return recurrence_plot.transform(X)

    def _check_params(
        self, n_features: int
    ) -> tuple[
        list[Any] | npt.NDArray[Any] | tuple[Any, ...],
        list[Any] | npt.NDArray[Any] | tuple[Any, ...],
    ]:
        if isinstance(self.threshold, (tuple, list, np.ndarray)):
            if len(self.threshold) != n_features:
                raise ValueError(
                    "If 'threshold' is a list, its length must be equal to "
                    f"n_features ({len(self.threshold)} != {n_features})."
                )
            thresholds_ = self.threshold
        else:
            thresholds_ = [self.threshold for _ in range(n_features)]
        if isinstance(self.percentage, (tuple, list, np.ndarray)):
            if len(self.percentage) != n_features:
                raise ValueError(
                    "If 'percentage' is a list, its length must be equal to "
                    f"n_features ({len(self.percentage)} != {n_features})."
                )
            percentages_ = self.percentage
        else:
            percentages_ = [self.percentage for _ in range(n_features)]
        return thresholds_, percentages_
