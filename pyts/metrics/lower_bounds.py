"""Code for Lower Bounds of Dynamic Time Warping."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import sqrt
from numba import njit, prange
from sklearn.utils import check_array


def _check_consistent_lengths(X, Y):
    n_timestamps_X, n_timestamps_Y = X.shape[-1], Y.shape[-1]
    if not n_timestamps_X == n_timestamps_Y:
        raise ValueError(
            "Found input variables with inconsistent numbers of "
            "timestamps: [{0}, {1}]".format(n_timestamps_X, n_timestamps_Y)
        )


@njit()
def _lower_bound_yi_x_y(x, x_min, x_max, y, y_min, y_max):
    if x_max >= y_max:
        if x_min < y_min:
            sum1 = np.sum(np.square(x[x > y_max] - y_max))
            sum2 = np.sum(np.square(x[x < y_min] - y_min))
            return sqrt(sum1 + sum2)
        elif x_min > y_max:
            return sqrt(max(np.sum(np.square(x - y_max)),
                            np.sum(np.square(y - x_min))))
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
            return sqrt(max(np.sum(np.square(y - x_max)),
                            np.sum(np.square(x - y_min))))
        else:
            sum1 = np.sum(np.square(y[y > x_max] - x_max))
            sum2 = np.sum(np.square(x[x < y_min] - y_min))
            return sqrt(sum1 + sum2)


@njit()
def _lower_bound_yi_X_Y(X, X_min, X_max, Y, Y_min, Y_max):
    n_samples_X, _ = X.shape
    n_samples_Y, _ = Y.shape
    X_yi = np.empty((n_samples_X, n_samples_Y))
    for i in prange(n_samples_X):
        for j in prange(n_samples_Y):
            X_yi[i, j] = _lower_bound_yi_x_y(
                X[i], X_min[i], X_max[i], Y[j], Y_min[j], Y_max[j]
            )
    return X_yi


def lower_bound_yi(X_train, X_test):
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
    lb_yi = _lower_bound_yi_X_Y(X_test, X_test_min, X_test_max,
                                X_train, X_train_min, X_train_max)
    return lb_yi


def lower_bound_kim(X_train, X_test):
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
    max_ = np.abs(np.max(X_test, axis=1)[:, None]
                  - np.max(X_train, axis=1)[None, :])
    min_ = np.abs(np.min(X_test, axis=1)[:, None]
                  - np.min(X_train, axis=1)[None, :])
    lb_kim = np.max(np.asarray([first, last, max_, min_]), axis=0)
    return lb_kim


@njit()
def _warping_envelope_2d(X, n_samples, n_timestamps, region):
    lower = np.empty((n_samples, n_timestamps))
    upper = np.empty((n_samples, n_timestamps))
    for i in prange(n_samples):
        for j in prange(n_timestamps):
            sub_series = X[i, region[0, j]:region[1, j]]
            lower[i, j] = np.min(sub_series)
            upper[i, j] = np.max(sub_series)
    return lower, upper


@njit()
def _warping_envelope_3d(X, n_samples_X, n_samples_Y, n_timestamps, region):
    lower = np.empty((n_samples_X, n_samples_Y, n_timestamps))
    upper = np.empty((n_samples_X, n_samples_Y, n_timestamps))
    for i in prange(n_samples_X):
        for j in prange(n_samples_Y):
            for k in prange(n_timestamps):
                sub_series = X[i, j, region[0, k]:region[1, k]]
                lower[i, j, k] = np.min(sub_series)
                upper[i, j, k] = np.max(sub_series)
    return lower, upper


def _warping_envelope(X, region):
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
        lower, upper = _warping_envelope_2d(
            X, n_samples, n_timestamps, region
        )
    else:
        n_samples_X, n_samples_Y, n_timestamps = X.shape
        lower, upper = _warping_envelope_3d(
            X, n_samples_X, n_samples_Y, n_timestamps, region
        )
    return lower, upper


@njit()
def _clip_2d(X, X_min, X_max, n_samples_X, n_samples_clip, n_timestamps):
    X_clipped = np.empty((n_samples_X, n_samples_clip, n_timestamps))
    for i in prange(n_samples_X):
        X_clipped[i] = np.minimum(np.maximum(X[i], X_min), X_max)
    return X_clipped


@njit()
def _clip_3d(X, X_min, X_max, n_samples_X, n_samples_clip, n_timestamps):
    X_clipped = np.empty((n_samples_X, n_samples_clip, n_timestamps))
    for i in prange(n_samples_X):
        X_clipped[i] = np.minimum(np.maximum(X[i], X_min[:, i]), X_max[:, i])
    return X_clipped


def _clip(X, lower, upper):
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
            "({0} != {1})".format(lower.shape, upper.shape)
        )
    if n_dims == 2:
        n_samples_X, n_timestamps = X.shape
        n_samples_clip, _ = lower.shape
        X_clipped = _clip_2d(
            X, lower, upper, n_samples_X, n_samples_clip, n_timestamps
        )
    else:
        n_samples_X, n_samples_Y, n_timestamps = lower.shape
        X_clipped = _clip_3d(
            X, lower, upper, n_samples_Y, n_samples_X, n_timestamps
        )
    return X_clipped


def lower_bound_keogh(X_train, X_test, region):
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
    lower, upper = _warping_envelope(X_train, region)
    X_proj = _clip(X_test, lower, upper)
    squared_lb_keogh = np.sum(
        (X_test[:, None, :] - X_proj) ** 2, axis=-1
    )
    return np.sqrt(squared_lb_keogh)


def lower_bound_improved(X_train, X_test, region):
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

    # LB Keogh lower bounds
    lower_train, upper_train = _warping_envelope(X_train, region)
    X_test_proj = _clip(X_test, lower_train, upper_train)
    squared_lb_keogh = np.sum(
        (X_test[:, None, :] - X_test_proj) ** 2, axis=-1
    )

    # LB Improved lower bounds
    lower_test, upper_test = _warping_envelope(X_test_proj, region)
    X_train_proj = _clip(X_train, lower_test, upper_test)
    squared_lb_improved = np.sum(
        (X_train[:, None, :] - X_train_proj) ** 2, axis=-1
    )

    return np.sqrt(squared_lb_keogh + squared_lb_improved.T)
