"""Code for Lower Bounds of Dynamic Time Warping."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from math import sqrt

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from sklearn.utils import check_array


def _check_consistent_lengths(
    X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]
) -> None:
    n_timestamps_X, n_timestamps_Y = X.shape[-1], Y.shape[-1]
    if not n_timestamps_X == n_timestamps_Y:
        raise ValueError(
            "Found input variables with inconsistent numbers of "
            f"timestamps: [{n_timestamps_X}, {n_timestamps_Y}]"
        )


_LOWER_BOUND_YI_X_Y_SIGNATURES = [
    "float64(float64[:], float64, float64, float64[:], float64, float64)",
    # The direct unit tests for this private helper pass plain integer
    # arrays and scalars (unlike the real caller, which always passes
    # float64 after ``check_array``), so both are compiled eagerly.
    "float64(int64[:], int64, int64, int64[:], int64, int64)",
]


@njit(_LOWER_BOUND_YI_X_Y_SIGNATURES)
def _lower_bound_yi_x_y(
    x: npt.NDArray[np.float64],
    x_min: np.float64,
    x_max: np.float64,
    y: npt.NDArray[np.float64],
    y_min: np.float64,
    y_max: np.float64,
) -> float:
    if x_max >= y_max:
        if x_min < y_min:
            sum1 = np.sum(np.square(x[x > y_max] - y_max))
            sum2 = np.sum(np.square(x[x < y_min] - y_min))
            return sqrt(sum1 + sum2)
        elif x_min > y_max:
            return sqrt(max(np.sum(np.square(x - y_max)), np.sum(np.square(y - x_min))))
        else:
            sum1 = np.sum(np.square(x[x > y_max] - y_max))
            sum2 = np.sum(np.square(y[y < x_min] - x_min))
            return sqrt(sum1 + sum2)
    else:
        if y_min < x_min:
            sum1 = np.sum(np.square(y[y > x_max] - x_max))
            sum2 = np.sum(np.square(y[y < x_min] - x_min))
            return sqrt(sum1 + sum2)
        elif y_min > x_max:
            return sqrt(max(np.sum(np.square(y - x_max)), np.sum(np.square(x - y_min))))
        else:
            sum1 = np.sum(np.square(y[y > x_max] - x_max))
            sum2 = np.sum(np.square(x[x < y_min] - y_min))
            return sqrt(sum1 + sum2)


@njit(
    [
        # Signature string split across two lines (implicit concatenation)
        # to stay within the line length, not a missing comma.
        # codeql[py/implicit-string-concatenation-in-list]
        "float64[:,:](float64[:,:], float64[:], float64[:], float64[:,:], "
        "float64[:], float64[:])",
        # Same rationale as ``_lower_bound_yi_x_y`` above: the direct unit
        # tests pass plain integer arrays.
        "float64[:,:](int64[:,:], int64[:], int64[:], int64[:,:], int64[:], int64[:])",
    ]
)
def _lower_bound_yi_X_Y(
    X: npt.NDArray[np.float64],
    X_min: npt.NDArray[np.float64],
    X_max: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
    Y_min: npt.NDArray[np.float64],
    Y_max: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    n_samples_X, _ = X.shape
    n_samples_Y, _ = Y.shape
    X_yi = np.empty((n_samples_X, n_samples_Y))
    for i in prange(n_samples_X):
        for j in prange(n_samples_Y):
            X_yi[i, j] = _lower_bound_yi_x_y(
                X[i], X_min[i], X_max[i], Y[j], Y_min[j], Y_max[j]
            )
    return X_yi


def lower_bound_yi(
    X_train: npt.ArrayLike, X_test: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Compute the "LB_Yi" lower bounds between two datasets.

    Parameters
    ----------
    X_train : array-like, shape = (n_samples_train, n_timestamps)
        Training set.

    X_test: : array-like, shape = (n_samples_test, n_timestamps)
        Test set.

    Returns
    -------
    lower_bounds : array, shape = (n_samples_test, n_samples_train)
        "LB_Yi" lower bounds.

    References
    ----------
    .. [1] B. K. Yi et al, "Efficient Retrieval of Similar Time Sequences
           Under Time Warping". International Conference on Data Engineering,
           201-208 (1998).

    Examples
    --------
    >>> X_train = [[5, 4, 3, 2, 1], [1, 8, 4, 3, 2], [6, 3, 5, 4, 7]]
    >>> X_test = [[2, 1, 8, 4, 5]]
    >>> lower_bound_yi(X_train, X_test)
    array([[3.        , 0.        , 2.44...]])

    """
    X_train = check_array(X_train)
    X_test = check_array(X_test)
    _check_consistent_lengths(X_train, X_test)
    X_train_min = np.min(X_train, axis=1)
    X_train_max = np.max(X_train, axis=1)
    X_test_min = np.min(X_test, axis=1)
    X_test_max = np.max(X_test, axis=1)
    lb_yi = _lower_bound_yi_X_Y(
        X_test, X_test_min, X_test_max, X_train, X_train_min, X_train_max
    )
    return lb_yi


def lower_bound_kim(
    X_train: npt.ArrayLike, X_test: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Compute the "LB_Kim" lower bounds between two datasets.

    Parameters
    ----------
    X_train : array-like, shape = (n_samples_train, n_timestamps)
        Training set.

    X_test: : array-like, shape = (n_samples_test, n_timestamps)
        Test set.

    Returns
    -------
    lower_bounds : array, shape = (n_samples_test, n_samples_train)
        "LB_Kim" lower bounds.

    References
    ----------
    .. [1] S. W. Kim et al, "An Index-Based Approach for Similarity Search
           Supporting Time Warping in Large Sequence Databases". International
           Conference on Data Engineering, 607-614 (2001).

    Examples
    --------
    >>> X_train = [[2, 1, 8, 4, 5], [1, 2, 3, 4, 5]]
    >>> X_test = [[5, 4, 3, 2, 1], [1, 8, 4, 3, 2], [6, 3, 5, 4, 7]]
    >>> lower_bound_kim(X_train, X_test)
    array([[4, 4],
           [3, 3],
           [4, 5]])

    """
    X_train = check_array(X_train)
    X_test = check_array(X_test)
    _check_consistent_lengths(X_train, X_test)
    first = np.abs(X_test[:, 0, None] - X_train[None, :, 0])
    last = np.abs(X_test[:, -1, None] - X_train[None, :, -1])
    max_ = np.abs(np.max(X_test, axis=1)[:, None] - np.max(X_train, axis=1)[None, :])
    min_ = np.abs(np.min(X_test, axis=1)[:, None] - np.min(X_train, axis=1)[None, :])
    lb_kim = np.max(np.asarray([first, last, max_, min_]), axis=0)
    return lb_kim


_WARPING_ENVELOPE_2D_SIGNATURES = [
    "UniTuple(float64[:,:], 2)(float64[:,:], int64, int64, int64[:,:])",
    # ``_warping_envelope`` (the public caller) validates X with a plain
    # ``check_array`` that does not force float64, and both this file's
    # own doctest (``lower_bound_keogh``) and its unit tests exercise it
    # with an integer-valued X, so that is compiled eagerly too. Its
    # output (built with ``np.empty``, no dtype argument) is always
    # float64 regardless of X's dtype.
    "UniTuple(float64[:,:], 2)(int64[:,:], int64, int64, int64[:,:])",
]


@njit(_WARPING_ENVELOPE_2D_SIGNATURES)
def _warping_envelope_2d(
    X: npt.NDArray[np.float64],
    n_samples: int,
    n_timestamps: int,
    region: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    lower = np.empty((n_samples, n_timestamps))
    upper = np.empty((n_samples, n_timestamps))
    for i in prange(n_samples):
        for j in prange(n_timestamps):
            sub_series = X[i, region[0, j] : region[1, j]]
            lower[i, j] = np.min(sub_series)
            upper[i, j] = np.max(sub_series)
    return lower, upper


@njit(
    [
        "UniTuple(float64[:,:,:], 2)(float64[:,:,:], int64, int64, int64, int64[:,:])",
        # Same rationale as ``_warping_envelope_2d`` above.
        "UniTuple(float64[:,:,:], 2)(int64[:,:,:], int64, int64, int64, int64[:,:])",
    ]
)
def _warping_envelope_3d(
    X: npt.NDArray[np.float64],
    n_samples_X: int,
    n_samples_Y: int,
    n_timestamps: int,
    region: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    lower = np.empty((n_samples_X, n_samples_Y, n_timestamps))
    upper = np.empty((n_samples_X, n_samples_Y, n_timestamps))
    for i in prange(n_samples_X):
        for j in prange(n_samples_Y):
            for k in prange(n_timestamps):
                sub_series = X[i, j, region[0, k] : region[1, k]]
                lower[i, j, k] = np.min(sub_series)
                upper[i, j, k] = np.max(sub_series)
    return lower, upper


def _warping_envelope(
    X: npt.NDArray[np.float64], region: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the warping envelope.

    Parameters
    ----------
    X : array
        Input data. It must be two- or three-dimensional.

    region : array, shape = (2, n_timestamps)
        Constraint region. The first row consists of the starting indices
        (included) and the second row consists of the ending indices (excluded)
        of the valid rows for each column.

    Returns
    -------
    lower : array
        The lower envelope.

    upper : array
        The upper envelope.

    """
    X = check_array(X, ensure_2d=False, allow_nd=True)
    region = check_array(region, ensure_min_samples=2)
    n_dims = X.ndim
    if n_dims not in (2, 3):
        raise ValueError("X must be a two- or three-dimensional.")
    if n_dims == 2:
        n_samples, n_timestamps = X.shape
        lower, upper = _warping_envelope_2d(X, n_samples, n_timestamps, region)
    else:
        n_samples_X, n_samples_Y, n_timestamps = X.shape
        lower, upper = _warping_envelope_3d(
            X, n_samples_X, n_samples_Y, n_timestamps, region
        )
    return lower, upper


_CLIP_2D_SIGNATURES = [
    "float64[:,:,:](float64[:,:], float64[:,:], float64[:,:], int64, int64, int64)",
    # The direct unit tests for this private helper (via the ``_clip``
    # wrapper) pass plain integer arrays for X, lower and upper together,
    # so that combination is compiled eagerly too. Its output (built with
    # ``np.empty``, no dtype argument) is always float64.
    "float64[:,:,:](int64[:,:], int64[:,:], int64[:,:], int64, int64, int64)",
    # ``lower_bound_improved`` chains ``_warping_envelope`` (always
    # float64 output) onto an X that ``check_array`` left as int64, so X
    # and (lower, upper) can genuinely have different dtypes in
    # production, not just matching pairs.
    "float64[:,:,:](int64[:,:], float64[:,:], float64[:,:], int64, int64, int64)",
]


@njit(_CLIP_2D_SIGNATURES)
def _clip_2d(
    X: npt.NDArray[np.float64],
    X_min: npt.NDArray[np.float64],
    X_max: npt.NDArray[np.float64],
    n_samples_X: int,
    n_samples_clip: int,
    n_timestamps: int,
) -> npt.NDArray[np.float64]:
    X_clipped = np.empty((n_samples_X, n_samples_clip, n_timestamps))
    for i in prange(n_samples_X):
        X_clipped[i] = np.minimum(np.maximum(X[i], X_min), X_max)
    return X_clipped


@njit(
    [
        # Signature string split across two lines (implicit concatenation)
        # to stay within the line length, not a missing comma.
        # codeql[py/implicit-string-concatenation-in-list]
        "float64[:,:,:](float64[:,:], float64[:,:,:], float64[:,:,:], "
        "int64, int64, int64)",
        # Same rationale as ``_clip_2d`` above.
        "float64[:,:,:](int64[:,:], int64[:,:,:], int64[:,:,:], int64, int64, int64)",
        # ``lower_bound_improved`` chains ``_warping_envelope`` (always
        # float64 output) onto an X that ``check_array`` left as int64
        # (see ``_clip_2d`` above for the identical, actually-exercised
        # scenario in the 2D case; this covers the same pattern in 3D).
        # codeql[py/implicit-string-concatenation-in-list]
        "float64[:,:,:](int64[:,:], float64[:,:,:], float64[:,:,:], "
        "int64, int64, int64)",
    ]
)
def _clip_3d(
    X: npt.NDArray[np.float64],
    X_min: npt.NDArray[np.float64],
    X_max: npt.NDArray[np.float64],
    n_samples_X: int,
    n_samples_clip: int,
    n_timestamps: int,
) -> npt.NDArray[np.float64]:
    X_clipped = np.empty((n_samples_X, n_samples_clip, n_timestamps))
    for i in prange(n_samples_X):
        X_clipped[i] = np.minimum(np.maximum(X[i], X_min[:, i]), X_max[:, i])
    return X_clipped


def _clip(
    X: npt.NDArray[np.float64],
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Clip an array.

    Parameters
    ----------
    X : array, shape = (n_samples, n_timestamps)
        Array to clip.

    lower : array
        Minimum values in the clipped array. It must be
        two- or three-dimensional.

    upper : array
        Maximum values in the clipped array. It must be
        two- or three-dimensional, and have the same shape
        as ``lower``.

    Returns
    -------
    X_clipped : array
        Clipped array.

    """
    X = check_array(X)
    lower = check_array(lower, ensure_2d=False, allow_nd=True)
    upper = check_array(upper, ensure_2d=False, allow_nd=True)
    n_dims = lower.ndim
    if n_dims not in (2, 3):
        raise ValueError("'lower' must be two- or three-dimensional.")
    if not lower.shape == upper.shape:
        raise ValueError(
            "'lower' and 'upper' must have the same shape "
            f"({lower.shape} != {upper.shape})"
        )
    if n_dims == 2:
        n_samples_X, n_timestamps = X.shape
        n_samples_clip, _ = lower.shape
        X_clipped = _clip_2d(X, lower, upper, n_samples_X, n_samples_clip, n_timestamps)
    else:
        n_samples_X, n_samples_Y, n_timestamps = lower.shape
        X_clipped = _clip_3d(X, lower, upper, n_samples_Y, n_samples_X, n_timestamps)
    return X_clipped


# ``lower_train``/``upper_train`` are always float64 (``_warping_envelope``'s
# output, built with ``np.empty`` and no dtype argument, regardless of
# ``X_train``'s own dtype). Only ``X_test`` genuinely varies in dtype
# (``check_array`` in ``lower_bound_keogh``/``lower_bound_improved`` does not
# force float64), so only that parameter needs a second variant.
_SQUARED_LB_KEOGH_SIGNATURES = [
    "float64[:,:](float64[:,:], float64[:,:], float64[:,:], int64, int64, int64)",
    "float64[:,:](int64[:,:], float64[:,:], float64[:,:], int64, int64, int64)",
]


@njit(_SQUARED_LB_KEOGH_SIGNATURES)
def _squared_lb_keogh(
    X_test: npt.NDArray[np.float64],
    lower_train: npt.NDArray[np.float64],
    upper_train: npt.NDArray[np.float64],
    n_samples_test: int,
    n_samples_train: int,
    n_timestamps: int,
) -> npt.NDArray[np.float64]:
    """Compute the squared "LB_Keogh" lower bounds without materializing
    the full (n_samples_test, n_samples_train, n_timestamps) projected
    tensor that a broadcasted ``X_test[:, None, :] - _clip(...)`` would
    require: each test/train pair's projected value is computed and
    immediately reduced into the running sum, one timestamp at a time.
    """
    squared_lb = np.empty((n_samples_test, n_samples_train))
    for i in prange(n_samples_test):
        for j in prange(n_samples_train):
            total = 0.0
            for k in range(n_timestamps):
                value = X_test[i, k]
                lower = lower_train[j, k]
                upper = upper_train[j, k]
                if value < lower:
                    projected = lower
                elif value > upper:
                    projected = upper
                else:
                    projected = value
                diff = value - projected
                total += diff * diff
            squared_lb[i, j] = total
    return squared_lb


# X_test and X_train can each independently be float64 or int64 (neither
# ``lower_bound_improved`` nor its ``check_array`` calls force a dtype), so
# all four combinations are compiled eagerly. ``lower_train``/``upper_train``
# are always float64, as in ``_squared_lb_keogh`` above. ``region`` is always
# int64, matching ``_warping_envelope_2d``'s own signature.
_SQUARED_LB_KEOGH_AND_IMPROVED_SIGNATURES = [
    # Each entry below is a single numba signature string, split across
    # two lines (implicit concatenation) only to stay within the line
    # length; it is not a missing comma between list elements.
    # codeql[py/implicit-string-concatenation-in-list]
    "UniTuple(float64[:,:], 2)(float64[:,:], float64[:,:], float64[:,:], "
    "float64[:,:], int64[:,:], int64, int64, int64)",
    # codeql[py/implicit-string-concatenation-in-list]
    "UniTuple(float64[:,:], 2)(int64[:,:], float64[:,:], float64[:,:], "
    "float64[:,:], int64[:,:], int64, int64, int64)",
    # codeql[py/implicit-string-concatenation-in-list]
    "UniTuple(float64[:,:], 2)(float64[:,:], int64[:,:], float64[:,:], "
    "float64[:,:], int64[:,:], int64, int64, int64)",
    # codeql[py/implicit-string-concatenation-in-list]
    "UniTuple(float64[:,:], 2)(int64[:,:], int64[:,:], float64[:,:], "
    "float64[:,:], int64[:,:], int64, int64, int64)",
]


@njit(_SQUARED_LB_KEOGH_AND_IMPROVED_SIGNATURES)
def _squared_lb_keogh_and_improved(
    X_test: npt.NDArray[np.float64],
    X_train: npt.NDArray[np.float64],
    lower_train: npt.NDArray[np.float64],
    upper_train: npt.NDArray[np.float64],
    region: npt.NDArray[np.int64],
    n_samples_test: int,
    n_samples_train: int,
    n_timestamps: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute both squared "LB_Keogh" and "LB_Improved" terms without ever
    materializing the (n_samples_test, n_samples_train, n_timestamps)
    projected tensors that the original broadcasted implementation built
    (once for the test-onto-train projection, once more for its own
    warping envelope, and once more for the train-onto-that projection).
    For each (test, train) pair, the small (n_timestamps,) projected
    series is kept only long enough to compute its own warping envelope
    (needed by "LB_Improved") and is then discarded.
    """
    squared_lb_keogh = np.empty((n_samples_test, n_samples_train))
    squared_lb_improved = np.empty((n_samples_test, n_samples_train))
    for i in prange(n_samples_test):
        for j in prange(n_samples_train):
            # LB_Keogh term: project X_test[i] onto X_train[j]'s envelope.
            projected = np.empty(n_timestamps)
            total_keogh = 0.0
            for k in range(n_timestamps):
                value = X_test[i, k]
                lower = lower_train[j, k]
                upper = upper_train[j, k]
                if value < lower:
                    proj_value = lower
                elif value > upper:
                    proj_value = upper
                else:
                    proj_value = value
                projected[k] = proj_value
                diff = value - proj_value
                total_keogh += diff * diff
            squared_lb_keogh[i, j] = total_keogh

            # LB_Improved term: warping envelope of the projected series,
            # then project X_train[j] onto it.
            total_improved = 0.0
            for k in range(n_timestamps):
                start, end = region[0, k], region[1, k]
                lower_env = projected[start]
                upper_env = projected[start]
                for m in range(start + 1, end):
                    if projected[m] < lower_env:
                        lower_env = projected[m]
                    if projected[m] > upper_env:
                        upper_env = projected[m]
                train_value = X_train[j, k]
                if train_value < lower_env:
                    train_proj = lower_env
                elif train_value > upper_env:
                    train_proj = upper_env
                else:
                    train_proj = train_value
                diff = train_value - train_proj
                total_improved += diff * diff
            squared_lb_improved[i, j] = total_improved
    return squared_lb_keogh, squared_lb_improved


def lower_bound_keogh(
    X_train: npt.ArrayLike, X_test: npt.ArrayLike, region: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    r"""Compute the "LB_Keogh" lower bounds between two datasets.

    Parameters
    ----------
    X_train : array-like, shape = (n_samples_train, n_timestamps)
        Training set. The warping envelopes are computed
        on this set.

    X_test: : array-like, shape = (n_samples_test, n_timestamps)
        Test set.

    region : array, shape = (2, n_timestamps)
        Constraint region. The first row consists of the starting indices
        (included) and the second row consists of the ending indices (excluded)
        of the valid rows for each column.

    Returns
    -------
    lower_bounds : array, shape = (n_samples_test, n_samples_train)
        "LB_Keogh" lower bounds.

    Notes
    -----
    The "LB_Keogh" lower bounds are computed as

    .. math:: LB_Keogh(X, Y) = \Vert X - H(X, Y) \Vert_{2}

    where :math:`X` is the test set (``X_test``), :math:`Y` is the
    training set (``X_train``), and :math:`H(X, Y)` is the projection
    of :math:`X` on :math:`Y`.

    References
    ----------
    .. [1] E. Keogh and C. A. Ratanamahatana, "Exact indexing of dynamic
           time warping". Knowledge and Information Systems, 7(3),
           358-386 (2005).

    Examples
    --------
    >>> X_train = [[0, 1, 2, 3], [1, 2, 3, 4]]
    >>> X_test = [[0, 2.5, 3.5, 6]]
    >>> region = [[0, 0, 1, 2], [2, 3, 4, 4]]
    >>> lower_bound_keogh(X_train, X_test, region)
    array([[3.08...  , 2.23...]])

    """
    X_train = check_array(X_train)
    X_test = check_array(X_test)
    _check_consistent_lengths(X_train, X_test)
    n_samples_test, n_timestamps = X_test.shape
    n_samples_train, _ = X_train.shape
    lower, upper = _warping_envelope(X_train, region)
    squared_lb_keogh = _squared_lb_keogh(
        X_test, lower, upper, n_samples_test, n_samples_train, n_timestamps
    )
    return np.sqrt(squared_lb_keogh)


def lower_bound_improved(
    X_train: npt.ArrayLike, X_test: npt.ArrayLike, region: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    r"""Compute the "LB_Improved" lower bounds between two datasets.

    Parameters
    ----------
    X_train : array-like, shape = (n_samples_train, n_timestamps)
        Training set.

    X_test: : array-like, shape = (n_samples_test, n_timestamps)
        Test set.

    region : array, shape = (2, n_timestamps)
        Constraint region. The first row consists of the starting indices
        (included) and the second row consists of the ending indices (excluded)
        of the valid rows for each column.

    Returns
    -------
    lower_bounds : array, shape = (n_samples_test, n_samples_train)
        "LB_Improved" lower bounds.

    Notes
    -----
    The "LB_Improved" lower bounds are computed as

    .. math::

        LB_Improved(X, Y) = \sqrt\left\( \Vert X - H(X, Y) \Vert_{2}^2
        + \Vert Y - H(Y, H(X, Y)) \Vert_{2}^2 \right\)

    where :math:`X` is the test set (``X_test``), :math:`Y` is the
    training set (``X_train``), and :math:`H(X, Y)` is the projection
    of :math:`X` on :math:`Y`.

    References
    ----------
    .. [1] D. Lemire, "Faster Retrieval with a Two-Pass Dynamic-Time-Warping
           Lower Bound". Pattern Recognition, 42(9), 2169-2180 (2009).

    Examples
    --------
    >>> X_train = [[0, 1, 2, 3], [1, 2, 3, 4]]
    >>> X_test = [[0, 2.5, 3.5, 3.3]]
    >>> region = [[0, 0, 1, 2], [2, 3, 4, 4]]
    >>> lower_bound_improved(X_train, X_test, region)
    array([[0.76..., 1.11...]])

    """
    X_train = check_array(X_train)
    X_test = check_array(X_test)
    _check_consistent_lengths(X_train, X_test)
    n_samples_test, n_timestamps = X_test.shape
    n_samples_train, _ = X_train.shape

    # ``_warping_envelope`` re-validates ``region`` itself (and is still
    # used standalone elsewhere), but the fused kernel below needs its own
    # int64-typed copy to call directly.
    region_arr = check_array(region, ensure_min_samples=2, dtype=np.int64)

    lower_train, upper_train = _warping_envelope(X_train, region)
    squared_lb_keogh, squared_lb_improved = _squared_lb_keogh_and_improved(
        X_test,
        X_train,
        lower_train,
        upper_train,
        region_arr,
        n_samples_test,
        n_samples_train,
        n_timestamps,
    )

    return np.sqrt(squared_lb_keogh + squared_lb_improved)
