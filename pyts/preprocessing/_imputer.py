"""Code for imputers."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_interp_spline
from sklearn.base import BaseEstimator
from sklearn.impute import MissingIndicator
from sklearn.utils.validation import check_array

from pyts._base import UnivariateTransformerMixin

#: Spline order for each string strategy that is interpolated with
#: ``scipy.interpolate.make_interp_spline`` (the modern replacement for
#: ``scipy.interpolate.interp1d``, which is now considered legacy).
_SPLINE_ORDERS: dict[str, int] = {
    'zero': 0,
    'linear': 1,
    'slinear': 1,
    'quadratic': 2,
    'cubic': 3,
}

#: Strategies that pick an existing value rather than fitting a spline,
#: computed with ``numpy.searchsorted`` instead.
_STEP_STRATEGIES = frozenset({'previous', 'next', 'nearest'})


class InterpolationImputer(BaseEstimator, UnivariateTransformerMixin):
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

        Each strategy requires a minimum number of non-missing values per
        sample to be well-defined: 1 for 'zero', 'previous', 'next' and
        'nearest'; 2 for 'linear' and 'slinear'; 3 for 'quadratic'; 4 for
        'cubic'; and ``order + 1`` for an integer spline order. A
        ``ValueError`` is raised if a sample does not have enough
        non-missing values for the selected strategy.

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

    def __init__(
        self,
        missing_values: int | float | None = np.nan,
        strategy: str | int = 'linear',
    ) -> None:
        self.missing_values = missing_values
        self.strategy = strategy

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
        missing_values, ensure_all_finite = self._check_params()
        # check_array is wrapped by sklearn's @validate_params decorator,
        # which hides its real keyword-only parameters (including
        # ensure_all_finite) from pyrefly's source-based inference.
        X = check_array(
            X,
            dtype=np.float64,
            ensure_all_finite=ensure_all_finite,  # pyrefly: ignore[unexpected-keyword]
        )
        n_samples, n_timestamps = X.shape

        indicator = MissingIndicator(
            missing_values=missing_values,
            features='all',
            sparse=False,
        )
        # With sparse=False, fit_transform always returns a dense ndarray,
        # but pyrefly infers a spmatrix-only return type from the source.
        non_missing_idx = ~(  # pyrefly: ignore[unsupported-operation]
            indicator.fit_transform(X)
        )

        min_points = self._min_points_required()
        n_points_per_sample = non_missing_idx.sum(axis=1)
        too_few = np.flatnonzero(n_points_per_sample < min_points)
        if too_few.size > 0:
            i = too_few[0]
            raise ValueError(
                f"Sample {i} has {n_points_per_sample[i]} non-missing "
                f"value(s), but strategy={self.strategy!r} requires at "
                f"least {min_points}."
            )

        x_new = np.arange(n_timestamps)
        X_imputed = np.asarray(
            [
                self._impute_one_sample(X[i], non_missing_idx[i], x_new)
                for i in range(n_samples)
            ]
        )
        return X_imputed

    def _check_params(self) -> tuple[float, bool | str]:
        if self.missing_values is None:
            missing_values = np.nan
            ensure_all_finite = 'allow-nan'
        elif isinstance(self.missing_values, (int, np.integer, float, np.floating)):
            if np.isinf(self.missing_values):
                raise ValueError("'missing_values' cannot be infinity.")
            elif np.isnan(self.missing_values):
                ensure_all_finite = 'allow-nan'
                missing_values = np.nan
            else:
                ensure_all_finite = True
                missing_values = self.missing_values
        else:
            raise ValueError(
                "'missing_values' must be an integer, a float, None or "
                f"np.nan (got {self.missing_values!s})"
            )
        strategy_str_values = [
            'linear',
            'nearest',
            'zero',
            'slinear',
            'quadratic',
            'cubic',
            'previous',
            'next',
        ]
        if not (
            (isinstance(self.strategy, (int, np.integer)))
            or (self.strategy in strategy_str_values)
        ):
            raise ValueError(
                "'strategy' must be an integer or one of 'linear', 'nearest', "
                "'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next' "
                f"(got {self.strategy})"
            )
        return missing_values, ensure_all_finite

    def _min_points_required(self) -> int:
        """Minimum number of non-missing values required by ``strategy``.

        'zero', 'previous', 'next' and 'nearest' only ever look up an
        existing value, so a single non-missing value is enough. The other
        string strategies are spline interpolations of a fixed order (1 for
        'linear'/'slinear', 2 for 'quadratic', 3 for 'cubic'), and an
        integer strategy is the order of the spline interpolator to use;
        a spline of order ``k`` requires at least ``k + 1`` points to be
        well-defined.
        """
        if self.strategy in _STEP_STRATEGIES:
            return 1
        if isinstance(self.strategy, (int, np.integer)):
            return self.strategy + 1
        return _SPLINE_ORDERS[self.strategy] + 1

    def _impute_one_sample(
        self,
        x: npt.NDArray[np.float64],
        non_missing_idx: npt.NDArray[np.bool_],
        x_new: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        idx = x_new[non_missing_idx]
        values = x[non_missing_idx]

        if self.strategy in _STEP_STRATEGIES:
            return _step_interpolate(idx, values, x_new, self.strategy)

        k = (
            self.strategy
            if isinstance(self.strategy, (int, np.integer))
            else _SPLINE_ORDERS[self.strategy]
        )
        # ``BSpline`` (returned by ``make_interp_spline``) extrapolates by
        # default, reproducing ``interp1d``'s ``fill_value='extrapolate'``.
        spline = make_interp_spline(idx, values, k=k)
        return spline(x_new)


def _step_interpolate(
    idx: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
    x_new: npt.NDArray[np.int64],
    strategy: str,
) -> npt.NDArray[np.float64]:
    """Look up the previous, next or nearest known value for each point.

    Modern replacement for ``interp1d(idx, values, kind=strategy,
    fill_value='extrapolate')`` for the three non-spline string strategies,
    following the ``numpy.searchsorted``-based recipe from scipy's own
    "1-D interpolation" tutorial (the documented replacement for
    ``interp1d``, which is considered legacy). Points before the first (or
    after the last) known index extrapolate by holding the boundary value,
    matching ``interp1d``'s behavior for these strategies.
    """
    last = len(idx) - 1
    if strategy == 'previous':
        pos = np.clip(np.searchsorted(idx, x_new, side='right') - 1, 0, last)
    elif strategy == 'next':
        pos = np.clip(np.searchsorted(idx, x_new, side='left'), 0, last)
    else:  # strategy == 'nearest'
        pos_right = np.clip(np.searchsorted(idx, x_new, side='left'), 0, last)
        pos_left = np.clip(pos_right - 1, 0, last)
        left_dist = np.abs(x_new - idx[pos_left])
        right_dist = np.abs(idx[pos_right] - x_new)
        # Ties go to the lower index, matching interp1d(kind='nearest').
        pos = np.where(right_dist < left_dist, pos_right, pos_left)
    return values[pos]
