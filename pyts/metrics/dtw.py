"""Code for Dynamic Time Warping and its variants."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import ceil, log2, sqrt
from numba import njit, prange
from sklearn.utils import check_array
from ..utils import deprecated


@njit()
def _square(x, y):
    return (x - y) ** 2


@njit()
def _absolute(x, y):
    return abs(x - y)


@njit()
def _cost_matrix_region(x, y, dist, region):
    n_timestamps_1, n_timestamps_2 = x.size, y.size
    cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
    for i in prange(n_timestamps_1):
        for j in prange(region[0, i], region[1, i]):
            cost_mat[i, j] = dist(x[i], y[j])
    return cost_mat


@njit()
def _project_cost_matrix_region(cost_mat, region):
    n_timestamps_1, n_timestamps_2 = cost_mat.shape
    cost_mat_projected = np.full((n_timestamps_1, n_timestamps_2), np.inf)
    for i in prange(n_timestamps_1):
        for j in prange(region[0, i], region[1, i]):
            cost_mat_projected[i, j] = cost_mat[i, j]
    return cost_mat_projected


@njit()
def _cost_matrix_no_region(x, y, dist):
    n_timestamps_1, n_timestamps_2 = x.size, y.size
    cost_mat = np.empty((n_timestamps_1, n_timestamps_2))
    for j in prange(n_timestamps_2):
        for i in prange(n_timestamps_1):
            cost_mat[i, j] = dist(x[i], y[j])
    return cost_mat


def _check_input_dtw(x, y, precomputed_cost, dist, method):
    if dist == "precomputed" and method in ["multiscale", "fast"]:
        raise ValueError("The method '{0}' cannot be used with "
                         "a precomputed cost. Provide the raw time series or"
                         "use one of the methods: 'classic', 'sakoechiba', "
                         "'itakura'".format(method))
    if dist == "precomputed":
        cost = check_array(precomputed_cost, ensure_min_samples=2,
                           ensure_min_features=2, ensure_2d=True,
                           force_all_finite=False, dtype='float64')
        n_timestamps_1, n_timestamps_2 = precomputed_cost.shape
    else:
        cost = None
        x = check_array(x, ensure_2d=False, dtype='float64')
        y = check_array(y, ensure_2d=False, dtype='float64')
        if x.ndim != 1:
            raise ValueError("'x' must be a one-dimensional array.")
        if y.ndim != 1:
            raise ValueError("'y' must be a one-dimensional array.")
        n_timestamps_1 = x.size
        n_timestamps_2 = y.size

    return x, y, cost, n_timestamps_1, n_timestamps_2


def _input_to_cost(x, y, dist, precomputed_cost, region):
    """Computes cost matrix from dtw input."""
    if dist == "precomputed":
        if region is not None:
            cost_mat = _project_cost_matrix_region(precomputed_cost, region)
        else:
            cost_mat = precomputed_cost.copy()
    else:
        cost_mat = cost_matrix(x, y, dist=dist, region=region)
    cost_mat = check_array(cost_mat, ensure_min_samples=2,
                           ensure_min_features=2, ensure_2d=True,
                           force_all_finite=False, dtype='float64')
    return cost_mat


def _check_region(region, n_timestamps_1, n_timestamps_2):
    """Project region on the feasible set."""
    region = np.clip(region[:, :n_timestamps_1], 0, n_timestamps_2)
    return region


def cost_matrix(x, y, dist='square', region=None):
    """Compute the cost matrix between two samples.

    Parameters
    ----------
    x : array-like, shape = (n_timestamps_1,)
        First sample.

    y : array-like, shape = (n_timestamps_2,)
        Second sample.

    dist : 'square', 'absolute' or callable (default = 'square')
        Distance used. If 'square', the squared difference is used.
        If 'absolute', the absolute difference is used. If callable,
        it must be a function with a numba.njit() decorator that takes
        as input two numbers (two arguments) and returns a number.

    region : None or array-like, shape = (2, n_timestamps_1) (default = None)
        Constraint region. If None, there is no contraint region.
        If array-like, the first row indicates the starting indices (included)
        and the second row the ending indices (excluded) of the valid rows
        for each column.

    Returns
    -------
    cost_matrix : array, shape = (n_timestamps_1, n_timestamps_2)
        Cost matrix.

    """
    if dist == 'square':
        dist_ = _square
    elif dist == 'absolute':
        dist_ = _absolute
    elif isinstance(dist, str):
        raise ValueError("'dist' must be either 'square', 'absolute' or "
                         "callable (got {0}).".format(dist))
    else:
        try:
            temp = dist(1, 1)
        except:  # noqa: E722
            raise ValueError("Calling dist(1, 1) did not work.")
        else:
            if not isinstance(temp, (int, float)):
                raise ValueError("Calling dist(1, 1) did not return a float "
                                 "or an integer.")
        dist_ = dist
    if region is not None:
        region = check_array(region)
        region_shape = region.shape
        if region_shape != (2, x.size):
            raise ValueError(
                "The shape of 'region' must be equal to (2, n_timestamps_1) "
                "(got {0}).".format(region_shape)
            )
    if region is None:
        cost_mat = _cost_matrix_no_region(x, y, dist_)
    else:
        cost_mat = _cost_matrix_region(x, y, dist_, region)
    return cost_mat


@njit()
def _accumulated_cost_matrix_region(cost_matrix, region):
    n_timestamps_1, n_timestamps_2 = cost_matrix.shape
    acc_cost_mat = np.ones((n_timestamps_1, n_timestamps_2)) * np.inf
    acc_cost_mat[0, 0: region[1, 0]] = np.cumsum(
        cost_matrix[0, 0: region[1, 0]]
    )
    acc_cost_mat[0: region[1, 0], 0] = np.cumsum(
        cost_matrix[0: region[1, 0], 0]
    )
    region_ = np.copy(region)
    region_[0] = np.maximum(region_[0], 1)
    for i in range(1, n_timestamps_1):
        for j in range(region_[0, i], region_[1, i]):
            acc_cost_mat[i, j] = cost_matrix[i, j] + min(
                acc_cost_mat[i - 1][j - 1],
                acc_cost_mat[i - 1][j],
                acc_cost_mat[i][j - 1]
            )
    return acc_cost_mat


@njit()
def _accumulated_cost_matrix_no_region(cost_matrix):
    n_timestamps_1, n_timestamps_2 = cost_matrix.shape
    acc_cost_mat = np.empty((n_timestamps_1, n_timestamps_2))
    acc_cost_mat[0] = np.cumsum(cost_matrix[0])
    acc_cost_mat[:, 0] = np.cumsum(cost_matrix[:, 0])
    for j in range(1, n_timestamps_2):
        for i in range(1, n_timestamps_1):
            acc_cost_mat[i, j] = cost_matrix[i, j] + min(
                acc_cost_mat[i - 1][j - 1],
                acc_cost_mat[i - 1][j],
                acc_cost_mat[i][j - 1]
            )
    return acc_cost_mat


def accumulated_cost_matrix(cost_mat, region=None):
    """Compute the accumulated cost matrix.

    Parameters
    ----------
    cost_mat : array-like, shape = (n_timestamps_1, n_timestamps_2)
        Cost matrix.

    region : None or tuple, shape = (2, n_timestamps_1) (default = None)
        Constraint region. If None, there is no contraint region.
        If array-like, the first row indicates the starting indices (included)
        and the second row the ending indices (excluded) of the valid rows
        for each column.

    Returns
    -------
    acc_cost_mat : array, shape = (n_timestamps_1, n_timestamps_2)
        Accumulated cost matrix.

    """
    cost_mat = check_array(cost_mat, ensure_min_samples=2,
                           ensure_min_features=2, ensure_2d=True,
                           force_all_finite=False, dtype='float64')
    cost_mat_shape = cost_mat.shape

    if region is None:
        acc_cost_mat = _accumulated_cost_matrix_no_region(cost_mat)
    else:
        region = check_array(region, dtype='int64')
        region_shape = region.shape
        if region_shape != (2, cost_mat_shape[0]):
            raise ValueError("The shape of 'region' must be equal to "
                             "(2, n_timestamps_1) "
                             "(got {0})".format(region_shape)
                             )
        acc_cost_mat = _accumulated_cost_matrix_region(cost_mat, region)
    return acc_cost_mat


@njit()
def _return_path(acc_cost_mat):
    n_timestamps_1, n_timestamps_2 = acc_cost_mat.shape
    path = [(n_timestamps_1 - 1, n_timestamps_2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = np.array([acc_cost_mat[i - 1][j - 1],
                            acc_cost_mat[i - 1][j],
                            acc_cost_mat[i][j - 1]])
            argmin = np.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return np.transpose(np.array(path)[::-1])


def _return_results(dtw_dist, cost_mat, acc_cost_mat,
                    return_cost=False, return_accumulated=False,
                    return_path=False):
    """Return the results according to the parameters."""
    res = (dtw_dist, )
    if return_cost:
        res += (cost_mat, )
    if return_accumulated:
        res += (acc_cost_mat, )
    if return_path:
        path = _return_path(acc_cost_mat)
        res += (path, )
    if len(res) == 1:
        return res[0]
    else:
        return res


def _dtw_classic(x=None, y=None, dist='square', precomputed_cost=None,
                 return_cost=False, return_accumulated=False,
                 return_path=False):
    """Classic Dynamic Time Warping (DTW) distance between two time series.

    The classic version of DTW has no constraint region, allowing for possibly
    very large time shifts. This can be an upside or a downside depending on
    the application. If you do not want to allow large time shifts,
    consider using another method with a constraint region.

    Options
    -------
    This method has no options.

    References
    ----------
    .. [1] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 26(1), 43-49 (1978).

    """
    x, y, precomputed_cost, n_timestamps_1, n_timestamps_2 = \
        _check_input_dtw(x, y, precomputed_cost, dist, method="classic")
    cost_mat = _input_to_cost(x, y, dist, precomputed_cost, region=None)
    acc_cost_mat = accumulated_cost_matrix(cost_mat, region=None)
    dtw_dist = acc_cost_mat[-1, -1]
    if dist == 'square':
        dtw_dist = sqrt(dtw_dist)

    res = _return_results(dtw_dist, cost_mat, acc_cost_mat,
                          return_cost, return_accumulated, return_path)
    return res


@deprecated("dtw_classic is deprecated in 0.11 and will be removed in 0.12. "
            "Use dtw(method='classic') instead.")
def dtw_classic(x=None, y=None, dist='square', precomputed_cost=None,
                return_cost=False, return_accumulated=False,
                return_path=False):
    """Classic Dynamic Time Warping (DTW) distance between two time series.

    .. deprecated:: 0.11
        This function is deprecated and will be removed in 0.12.
        Use :func:`dtw` with ``method='classic'`` instead.

    Parameters
    ----------
    x : array-like, shape = (n_timestamps_1,)
        First array. Ignored if ``dist == 'precomputed'``.

    y : array-like, shape = (n_timestamps_2,)
        Second array. Ignored if ``dist == 'precomputed'``.

    dist : 'square', 'absolute', 'precomputed' or callable (default = 'square')
        Distance used. If 'square', the squared difference is used.
        If 'absolute', the absolute difference is used. If 'precomputed',
        `precomputed_cost` must be the cost matrix. If callable,
        it must be a function with a numba.njit() decorator that takes
        as input two numbers (two arguments) and returns a number.

    precomputed_cost : array-like, shape = (n_timestamps_1, n_timestamps_2) \
            (default = None)
        Precomputed cost matrix between the time series.
        Ignored if ``dist != 'precomputed'``.

    return_cost : bool (default = False)
        If True, the cost matrix is returned.

    return_accumulated : bool (default = False)
        If True, the accumulated cost matrix is returned.

    return_path : bool (default = False)
        If True, the optimal path is returned.

    Returns
    -------
    dtw_dist : float
        The DTW distance between the two arrays.

    cost_mat : array, shape = (n_timestamps_1, n_timestamps_2)
        Cost matrix. Only returned if ``return_cost=True``.

    acc_cost_mat : array, shape = (n_timestamps_1, n_timestamps_2)
        Accumulated cost matrix. Only returned if ``return_accumulated=True``.

    path : array, shape = (2, path_length)
        The optimal path along the cost matrix. The first row consists
        of the indices of the optimal path for x while the second row
        consists of the indices of the optimal path for y. Only returned
        if ``return_path=True``.

    """
    return _dtw_classic(x, y, dist, precomputed_cost, return_cost,
                        return_accumulated, return_path)


def _dtw_region(x=None, y=None, dist='square', region=None,
                precomputed_cost=None, return_cost=False,
                return_accumulated=False, return_path=False):
    """Dynamic Time Warping (DTW) distance with a constraint region.

    This version of DTW allows users to provide their own constraint region.
    If not provided, no constraint region is used, which is exactly the
    classic DTW.

    Options
    -------
    region : None or array-like, shape = (2, n_timestamps_1) (default = None)
        Constraint region. If None, no constraint region is used. Otherwise,
        the first row consists of the starting indices (included) and the
        second row consists of the ending indices (excluded) of the valid rows
        for each column.

    """
    x, y, precomputed_cost, n_timestamps_1, n_timestamps_2 = \
        _check_input_dtw(x, y, precomputed_cost, dist, method="region")
    cost_mat = _input_to_cost(x, y, dist, precomputed_cost, region=region)
    acc_cost_mat = accumulated_cost_matrix(cost_mat, region)
    dtw_dist = acc_cost_mat[-1, -1]
    if dist == 'square':
        dtw_dist = sqrt(dtw_dist)

    res = _return_results(dtw_dist, cost_mat, acc_cost_mat,
                          return_cost, return_accumulated, return_path)
    return res


@deprecated("dtw_region is deprecated in 0.11 and will be removed in 0.12. "
            "Use dtw(method='region') instead.")
def dtw_region(x=None, y=None, dist='square', region=None,
               precomputed_cost=None, return_cost=False,
               return_accumulated=False, return_path=False):
    """Dynamic Time Warping (DTW) distance with a constraint region.

    .. deprecated:: 0.11
        This function is deprecated and will be removed in 0.12.
        Use :func:`dtw` with ``method='region'`` instead.

    Parameters
    ----------
    x : array-like, shape = (n_timestamps_1,)
        First array. Ignored if ``dist == 'precomputed'``.

    y : array-like, shape = (n_timestamps_2,)
        Second array. Ignored if ``dist == 'precomputed'``.

    dist : 'square', 'absolute', 'precomputed' or callable (default = 'square')
        Distance used. If 'square', the squared difference is used.
        If 'absolute', the absolute difference is used. If 'precomputed',
        `precomputed_cost` must be the cost matrix. If callable,
        it must be a function with a numba.njit() decorator that takes
        as input two numbers (two arguments) and returns a number.

    region : None or array-like, shape = (2, n_timestamps_1)
        Constraint region. If None, no constraint region is used. Otherwise,
        the first row consists of the starting indices (included) and the
        second row consists of the ending indices (excluded) of the valid rows
        for each column.

    precomputed_cost : array-like, shape = (n_timestamps_1, n_timestamps_2) \
            (default = None)
        Precomputed cost matrix between the time series.
        Ignored if ``dist != 'precomputed'``.

    return_cost : bool (default = False)
        If True, the cost matrix is returned.

    return_accumulated : bool (default = False)
        If True, the accumulated cost matrix is returned.

    return_path : bool (default = False)
        If True, the optimal path is returned.

    Returns
    -------
    dtw_dist : float
        The DTW distance between the two arrays.

    cost_mat : array, shape = (n_timestamps_1, n_timestamps_2)
        Cost matrix. Only returned if ``return_cost=True``.

    acc_cost_mat : array, shape = (n_timestamps_1 n_timestamps_2)
        Accumulated cost matrix. Only returned if ``return_accumulated=True``.

    path : array, shape = (2, path_length)
        The optimal path along the cost matrix. The first row consists
        of the indices of the optimal path for x while the second row
        consists of the indices of the optimal path for y. Only returned
        if ``return_path=True``.

    """
    return _dtw_region(x, y, dist, region, precomputed_cost, return_cost,
                       return_accumulated, return_path)


def _check_sakoe_chiba_params(n_timestamps_1, n_timestamps_2, window_size):
    """Check and set some parameters of the sakoe-chiba band."""
    if not isinstance(n_timestamps_1, (int, np.integer)):
        raise TypeError("'n_timestamps_1' must be an integer.")
    else:
        if not n_timestamps_1 >= 2:
            raise ValueError("'n_timestamps_1' must be an integer greater than"
                             " or equal to 2.")
    if not isinstance(window_size, (int, np.integer, float, np.floating)):
        raise TypeError("'window_size' must be an integer or a float.")
    n_timestamps = max(n_timestamps_1, n_timestamps_2)

    if isinstance(window_size, (float, np.floating)):
        if not 0. <= window_size <= 1.:
            raise ValueError("The given 'window_size' is a float, "
                             "it must be between "
                             "0. and 1. To set the size of the sakoe-chiba "
                             "manually, 'window_size' must be an integer.")
        window_size = ceil(window_size * (n_timestamps - 1))
    else:
        if not 0 <= window_size <= (n_timestamps - 1):
            raise ValueError(
                "The given 'window_size' is an integer, it must "
                "be greater "
                "than or equal to 0 and lower than max('n_timestamps_1', "
                "'n_timestamps_2')."
            )

    scale = (n_timestamps_2 - 1) / (n_timestamps_1 - 1)

    if n_timestamps_2 > n_timestamps_1:
        window_size = max(window_size, scale / 2)
        horizontal_shift = 0
        vertical_shift = window_size
    elif n_timestamps_1 > n_timestamps_2:
        window_size = max(window_size, 0.5 / scale)
        horizontal_shift = window_size
        vertical_shift = 0
    else:
        horizontal_shift = 0
        vertical_shift = window_size
    return scale, horizontal_shift, vertical_shift


def sakoe_chiba_band(n_timestamps_1, n_timestamps_2=None, window_size=0.1):
    """Compute the Sakoe-Chiba band.

    Parameters
    ----------
    n_timestamps_1 : int
        The size of the first time series.

    n_timestamps_2 : int (optional, default None)
        The size of the second time series. If None, set to `n_timestamps_1`.

    window_size : float or int (default = 0.1)
        The window size above and below the diagonale.
        If float, `window_size must be between 0 and
        1, and the actual window size will be computed as:
        ``ceil(window_size * max((n_timestamps_1, n_timestamps_2) - 1))``.
        If int, `window_size` must be the largest temporal shift allowed.
        Each cell whose distance with the diagonale is lower than or equal to
        'window_size' becomes a valid cell for the path.

    Returns
    -------
    region : array, shape = (2, n_timestamps_1)
        Constraint region. The first row consists of the starting indices
        (included) and the second row consists of the ending indices (excluded)
        of the valid rows for each column.

    References
    ----------
    .. [1] H. Sakoe and S. Chiba, “Dynamic programming algorithm optimization
           for spoken word recognition”. IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 26(1), 43-49 (1978).

    Examples
    --------
    >>> from pyts.metrics import sakoe_chiba_band
    >>> print(sakoe_chiba_band(5, window_size=0.5))
    [[0 0 0 1 2]
     [3 4 5 5 5]]

    """
    if n_timestamps_2 is None:
        n_timestamps_2 = n_timestamps_1
    scale, horizontal_shift, vertical_shift = \
        _check_sakoe_chiba_params(n_timestamps_1, n_timestamps_2, window_size)

    lower_bound = scale * (np.arange(n_timestamps_1) - horizontal_shift) \
        - vertical_shift
    lower_bound = np.round(lower_bound, 2)
    lower_bound = np.ceil(lower_bound)
    upper_bound = scale * (np.arange(n_timestamps_1) + horizontal_shift) \
        + vertical_shift
    upper_bound = np.round(upper_bound, 2)
    upper_bound = np.floor(upper_bound) + 1
    region = np.asarray([lower_bound, upper_bound]).astype('int64')
    region = _check_region(region, n_timestamps_1, n_timestamps_2)
    return region


def _dtw_sakoechiba(x=None, y=None, dist='square', window_size=0.1,
                    precomputed_cost=None, return_cost=False,
                    return_accumulated=False, return_path=False):
    """Dynamic Time Warping (DTW) distance with Sakoe-Chiba band constraint.

    The Sakoe-Chiba constraint region is a band centered around the main
    diagonal. Any cell whose distance to the diagonale is lower than a
    given value is valid for the path.

    Options
    -------
    window_size : float or int (default = 0.1)
        The window size above and below the diagonal.
        If float, `window_size` must be between 0 and
        1, and the actual window size will be computed as
        ``ceil(window_size * max((n_timestamps_1, n_timestamps_2) - 1))``.
        If int, `window_size` must be the largest temporal shift allowed.
        Each cell whose distance with the diagonal is lower than or equal to
        `window_size` becomes a valid cell for the path.

    References
    ----------
    .. [1] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 26(1), 43-49 (1978).


    """
    x, y, precomputed_cost, n_timestamps_1, n_timestamps_2 = \
        _check_input_dtw(x, y, precomputed_cost, dist, method="sakoechiba")
    region = sakoe_chiba_band(n_timestamps_1, n_timestamps_2, window_size)
    cost_mat = _input_to_cost(x, y, dist, precomputed_cost, region=region)

    acc_cost_mat = accumulated_cost_matrix(cost_mat, region)
    dtw_dist = acc_cost_mat[-1, -1]
    if dist == 'square':
        dtw_dist = sqrt(dtw_dist)

    res = _return_results(dtw_dist, cost_mat, acc_cost_mat,
                          return_cost, return_accumulated, return_path)
    return res


@deprecated("dtw_sakoechiba is deprecated in 0.11 and will be removed in "
            "0.12. Use dtw(method='sakoechiba') instead.")
def dtw_sakoechiba(x=None, y=None, dist='square', window_size=0.1,
                   precomputed_cost=None, return_cost=False,
                   return_accumulated=False, return_path=False):
    """Dynamic Time Warping (DTW) distance with Sakoe-Chiba band constraint.

    .. deprecated:: 0.11
        This function is deprecated and will be removed in 0.12.
        Use :func:`dtw` with ``method='sakoechiba'`` instead.

    Parameters
    ----------
    x : array-like, shape = (n_timestamps_1,)
        First array. Ignored if ``dist == 'precomputed'``.

    y : array-like, shape = (n_timestamps_2,)
        Second array. Ignored if ``dist == 'precomputed'``.

    dist : 'square', 'absolute', 'precomputed' or callable (default = 'square')
        Distance used. If 'square', the squared difference is used.
        If 'absolute', the absolute difference is used. If 'precomputed',
        `precomputed_cost` must be the cost matrix. If callable,
        it must be a function with a numba.njit() decorator that takes
        as input two numbers (two arguments) and returns a number.

    window_size : float or int (default = 0.1)
        The window size above and below the diagonale.
        If float, `window_size must be between 0 and
        1, and the actual window size will be computed as:
        ``ceil(window_size * max((n_timestamps_1, n_timestamps_2) - 1))``.
        If int, `window_size` must be the largest temporal shift allowed.
        Each cell whose distance with the diagonale is lower than or equal to
        'window_size' becomes a valid cell for the path.

    precomputed_cost : array-like, shape = (n_timestamps_1, n_timestamps_2) \
            (default = None)
        Precomputed cost matrix between the time series.
        Ignored if ``dist != 'precomputed'``.

    return_cost : bool (default = False)
        If True, the cost matrix is returned.

    return_accumulated : bool (default = False)
        If True, the accumulated cost matrix is returned.

    return_path : bool (default = False)
        If True, the optimal path is returned.

    Returns
    -------
    dtw_dist : float
        The DTW distance between the two arrays.

    cost_mat : array, shape = (n_timestamps_1, n_timestamps_2)
        Cost matrix. Only returned if ``return_cost=True``.

    acc_cost_mat : array, shape = (n_timestamps_1, n_timestamps_2)
        Accumulated cost matrix. Only returned if ``return_accumulated=True``.

    path : array, shape = (2, path_length)
        The optimal path along the cost matrix. The first row consists
        of the indices of the optimal path for x while the second row
        consists of the indices of the optimal path for y. Only returned
        if ``return_path=True``.

    """
    return _dtw_sakoechiba(x, y, dist, window_size, precomputed_cost,
                           return_cost, return_accumulated, return_path)


def _get_itakura_slopes(n_timestamps_1, n_timestamps_2, max_slope):
    """Compute the slopes of the parallelogram bounds."""
    if not isinstance(n_timestamps_1, (int, np.integer)):
        raise TypeError("'n_timestamps_1' must be an integer.")
    else:
        if not n_timestamps_1 >= 2:
            raise ValueError("'n_timestamps_1' must be an integer greater than"
                             " or equal to 2.")

    if not isinstance(max_slope, (int, np.integer, float, np.floating)):
        raise TypeError("'max_slope' must be an integer or a float.")
    else:
        if not max_slope >= 1:
            raise ValueError("'max_slope' must be a number greater "
                             "than or equal to 1.")

    min_slope = 1 / max_slope
    scale_max = (n_timestamps_2 - 1) / (n_timestamps_1 - 2)
    max_slope *= scale_max
    max_slope = max(1., max_slope)

    scale_min = (n_timestamps_2 - 2) / (n_timestamps_1 - 1)

    min_slope *= scale_min
    min_slope = min(1., min_slope)
    return min_slope, max_slope


def itakura_parallelogram(n_timestamps_1, n_timestamps_2=None, max_slope=2.):
    """Compute the Itakura parallelogram.

    Parameters
    ----------
    n_timestamps_1 : int
        The size of the first time series.

    n_timestamps_2 : int (optional, default None)
        The size of the second time series. If None, set to `n_timestamps_1`.

    max_slope : float (default = 2.)
        Maximum slope for the parallelogram. Must be >= 1.

    Returns
    -------
    region : array, shape = (2, n_timestamps_1)
        Constraint region. The first row consists of the starting indices
        (included) and the second row consists of the ending indices (excluded)
        of the valid rows for each column.

    References
    ----------
    .. [1] F. Itakura, "Minimum prediction residual principle applied to
           speech recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 23(1), 67–72 (1975).

    Examples
    --------
    >>> from pyts.metrics import itakura_parallelogram
    >>> print(itakura_parallelogram(5))
    [[0 1 1 2 4]
     [1 3 4 4 5]]

    """
    if n_timestamps_2 is None:
        n_timestamps_2 = n_timestamps_1
    min_slope_, max_slope_ = _get_itakura_slopes(
        n_timestamps_1, n_timestamps_2, max_slope)

    # Now we create the piecewise linear functions defining the parallelogram
    # lower_bound[0] = min_slope * x
    # lower_bound[1] = max_slope * (x - n_timestamps_1) + n_timestamps_2

    centered_scale = np.arange(n_timestamps_1) - n_timestamps_1 + 1
    lower_bound = np.empty((2, n_timestamps_1))
    lower_bound[0] = min_slope_ * np.arange(n_timestamps_1)
    lower_bound[1] = max_slope_ * centered_scale + n_timestamps_2 - 1

    # take the max of the lower linear funcs
    lower_bound = np.round(lower_bound, 2)
    lower_bound = np.ceil(np.max(lower_bound, axis=0))

    # upper_bound[0] = max_slope * x
    # upper_bound[1] = min_slope * (x - n_timestamps_1) + n_timestamps_2

    upper_bound = np.empty((2, n_timestamps_1))
    upper_bound[0] = max_slope_ * np.arange(n_timestamps_1) + 1
    upper_bound[1] = min_slope_ * centered_scale + n_timestamps_2

    # take the min of the upper linear funcs
    upper_bound = np.round(upper_bound, 2)
    upper_bound = np.floor(np.min(upper_bound, axis=0))

    # Little fix for max_slope = 1
    if max_slope == 1:
        if n_timestamps_2 > n_timestamps_1:
            upper_bound[:-1] = lower_bound[1:]
        else:
            upper_bound = lower_bound + 1

    region = np.asarray([lower_bound, upper_bound]).astype('int64')
    region = _check_region(region, n_timestamps_1, n_timestamps_2)
    return region


def _dtw_itakura(x=None, y=None, dist='square', max_slope=2.,
                 precomputed_cost=None, return_cost=False,
                 return_accumulated=False, return_path=False):
    """Dynamic Time Warping distance with Itakura parallelogram constraint.

    The Itakura constraint region is a parallelogram. Contrary to the
    Sakoe-Chiba band, whose width is constant, the Itakura parallelogram
    has a varying width, allowing for larger time shifts in the middle than
    at the first and last time points.

    Options
    -------
    max_slope : float (default = 2.)
        Maximum slope of the parallelogram.

    References
    ----------
    .. [1] F. Itakura, "Minimum prediction residual principle applied to
           speech recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 23(1), 67–72 (1975).

    """
    x, y, precomputed_cost, n_timestamps_1, n_timestamps_2 = \
        _check_input_dtw(x, y, precomputed_cost, dist, method="itakura")
    region = itakura_parallelogram(n_timestamps_1, n_timestamps_2, max_slope)
    cost_mat = _input_to_cost(x, y, dist, precomputed_cost, region=region)
    acc_cost_mat = accumulated_cost_matrix(cost_mat, region)
    dtw_dist = acc_cost_mat[-1, -1]
    if dist == 'square':
        dtw_dist = sqrt(dtw_dist)

    res = _return_results(dtw_dist, cost_mat, acc_cost_mat,
                          return_cost, return_accumulated, return_path)
    return res


@deprecated("dtw_itakura is deprecated in 0.11 and will be removed in 0.12. "
            "Use dtw(method='itakura') instead.")
def dtw_itakura(x=None, y=None, dist='square', max_slope=2.,
                precomputed_cost=None, return_cost=False,
                return_accumulated=False, return_path=False):
    """Dynamic Time Warping distance with Itakura parallelogram constraint.

    .. deprecated:: 0.11
        This function is deprecated and will be removed in 0.12.
        Use :func:`dtw` with ``method='itakura'`` instead.

    Parameters
    ----------
    x : array-like, shape = (n_timestamps_1,)
        First array. Ignored if ``dist == 'precomputed'``.

    y : array-like, shape = (n_timestamps_2,)
        Second array. Ignored if ``dist == 'precomputed'``.

    dist : 'square', 'absolute', 'precomputed' or callable (default = 'square')
        Distance used. If 'square', the squared difference is used.
        If 'absolute', the absolute difference is used. If 'precomputed',
        `precomputed_cost` must be the cost matrix. If callable,
        it must be a function with a numba.njit() decorator that takes
        as input two numbers (two arguments) and returns a number.

    max_slope : float (default = 2.)
        Maximum slope for the parallelogram.

    precomputed_cost : array-like, shape = (n_timestamps_1, n_timestamps_2) \
            (default = None)
        Precomputed cost matrix between the time series.
        Ignored if ``dist != 'precomputed'``.

    return_cost : bool (default = False)
        If True, the cost matrix is returned.

    return_accumulated : bool (default = False)
        If True, the accumulated cost matrix is returned.

    return_path : bool (default = False)
        If True, the optimal path is returned.

    Returns
    -------
    dtw_dist : float
        The DTW distance between the two arrays.

    cost_mat : ndarray, shape = (n_timestamps_1, n_timestamps_2)
        Cost matrix. Only returned if ``return_cost=True``.

    acc_cost_mat : ndarray, shape = (n_timestamps_1, n_timestamps_2)
        Accumulated cost matrix. Only returned if ``return_accumulated=True``.

    path : array, shape = (2, path_length)
        The optimal path along the cost matrix. The first row consists
        of the indices of the optimal path for x while the second row
        consists of the indices of the optimal path for y. Only returned
        if ``return_path=True``.

    """
    return _dtw_itakura(x, y, dist, max_slope, precomputed_cost, return_cost,
                        return_accumulated, return_path)


def _blurred_path_region(n_timestamps_1, n_timestamps_2, resolution_level,
                         n_timestamps_reduced_1, n_timestamps_reduced_2,
                         path, radius):
    path_length = path.shape[1]
    path_up = np.repeat(path, radius, axis=1)
    path_down = path_up.copy()
    path_left = path_up.copy()
    path_right = path_up.copy()

    for i in range(1, radius + 1):
        start = path_length * (i - 1)
        end = path_length * i
        path_up[0, start: end] += i
        path_down[0, start: end] -= i
        path_left[1, start: end] -= i
        path_right[1, start: end] += i

    path_radius = np.c_[path, path_up, path_down, path_left, path_right]
    path_radius[0] = np.clip(path_radius[0], 0, n_timestamps_reduced_1 - 1)
    path_radius[1] = np.clip(path_radius[1], 0, n_timestamps_reduced_2 - 1)

    region_reduced = np.empty((2, n_timestamps_reduced_1))
    for i in range(n_timestamps_reduced_1):
        arr = path_radius[1, path_radius[0] == i]
        min_, max_ = np.min(arr), np.max(arr)
        region_reduced[0, i] = min_ * resolution_level
        region_reduced[1, i] = (max_ + 1) * resolution_level

    region = np.repeat(region_reduced, resolution_level, axis=1)
    region = _check_region(region, n_timestamps_1, n_timestamps_2)

    return region.astype('int64')


def _multiscale_region(x, y, dist, resolution=2, radius=0):
    n_timestamps_1 = len(x)
    n_timestamps_2 = len(y)
    if not isinstance(resolution, (int, np.integer)):
        raise TypeError("'resolution' must be an integer.")
    if resolution < 1:
        raise ValueError("'resolution' must be a positive integer.")
    if not isinstance(radius, (int, np.integer)):
        raise TypeError("'radius' must be an integer.")
    if radius < 0:
        raise ValueError("'radius' must be a non-negative integer.")

    if resolution == 1:
        region = None
    else:
        remainder_1 = n_timestamps_1 % resolution
        remainder_2 = n_timestamps_2 % resolution

        if remainder_1 != 0:
            x_padded = np.append(x, np.repeat(x[-1], resolution - remainder_1))
            x_padded = x_padded.reshape(-1, resolution).mean(axis=1)
        else:
            x_padded = x.reshape(-1, resolution).mean(axis=1)
        if remainder_2 != 0:
            y_padded = np.append(y, np.repeat(y[-1], resolution - remainder_2))
            y_padded = y_padded.reshape(-1, resolution).mean(axis=1)
        else:
            y_padded = y.reshape(-1, resolution).mean(axis=1)
        cost_mat_res = cost_matrix(x_padded, y_padded, dist=dist, region=None)
        acc_cost_mat_res = accumulated_cost_matrix(cost_mat_res, region=None)
        path_res = _return_path(acc_cost_mat_res)
        region = _blurred_path_region(n_timestamps_1, n_timestamps_2,
                                      resolution, x_padded.size,
                                      y_padded.size, path_res,
                                      radius)
    return region


def _fast_region(x, y, dist, radius=0):
    if not isinstance(radius, (int, np.integer)):
        raise TypeError("'radius' must be an integer.")
    if radius < 0:
        raise ValueError("'radius' must be a non-negative integer.")
    n_timestamps_1, n_timestamps_2 = len(x), len(y)
    min_size = radius + 2
    region = None
    n_timestamps = min(n_timestamps_1, n_timestamps_2)
    if n_timestamps > min_size:
        n_recursions = ceil(log2(n_timestamps / min_size))
        for i in range(n_recursions):
            resolution = 2 ** (n_recursions - i)
            remainder_1 = n_timestamps_1 % resolution
            remainder_2 = n_timestamps_2 % resolution

            if remainder_1 != 0:
                x_padded = np.append(
                    x, np.repeat(x[-1], resolution - remainder_1)
                )
                x_padded = x_padded.reshape(-1, resolution).mean(axis=1)
            else:
                x_padded = x.reshape(-1, resolution).mean(axis=1)
            if remainder_2 != 0:
                y_padded = np.append(
                    y, np.repeat(y[-1], resolution - remainder_2)
                )
                y_padded = y_padded.reshape(-1, resolution).mean(axis=1)
            else:
                y_padded = y.reshape(-1, resolution).mean(axis=1)

            cost_mat_res = cost_matrix(x_padded, y_padded,
                                       dist=dist, region=region)
            acc_cost_mat_res = accumulated_cost_matrix(cost_mat_res,
                                                       region=region)
            path_res = _return_path(acc_cost_mat_res)
            n_timestamps_next_1 = ceil((2 * n_timestamps_1) / resolution)
            n_timestamps_next_2 = ceil((2 * n_timestamps_2) / resolution)

            region = _blurred_path_region(n_timestamps_next_1,
                                          n_timestamps_next_2,
                                          2, x_padded.size, y_padded.size,
                                          path_res, radius)

    return region


def _compute_region(n_timestamps_1, n_timestamps_2, method, dist,
                    x=None, y=None, **options):
    """Compute the region of feasible alignment paths."""

    if options is None:
        options = dict()

    if method == 'classic':
        region = None
    elif method == 'sakoechiba':
        region = sakoe_chiba_band(n_timestamps_1, n_timestamps_2, **options)
    elif method == 'itakura':
        region = itakura_parallelogram(n_timestamps_1, n_timestamps_2,
                                       **options)
    elif method == 'multiscale':
        region = _multiscale_region(x, y, dist, **options)
    elif method == 'fast':
        region = _fast_region(x, y, dist, **options)
    else:
        raise ValueError("'method' must be either 'classic', 'sakoechiba', "
                         "'itakura', 'multiscale' or 'fast'.")

    return region


def _dtw_multiscale(x, y, dist='square', resolution=2, radius=0,
                    return_cost=False, return_accumulated=False,
                    return_path=False):
    """Multiscale Dynamic Time Warping (DTW) distance.

    This version of DTW builds an adaptive constraint region. First both time
    series are downsampled, with the sizes of the original times series being
    divided by ``resolution``. Then the optimal path between the two
    downsampled time series is computed. Finally this optimal path is
    projected on the orginal scale and used as the constraint region.

    Options
    -------
    resolution : int (default = 2)
        The resolution level.

    radius : int (default = 0)
        The radius used to expand the constraint region. The optimal path
        computed at the resolution level is expanded with `radius` cells to the
        top, bottom, left and right of every cell belonging to the optimal
        path. It is computed at the resolution level.

    References
    ----------
    .. [1] M. Müller, H. Mattes and F. Kurth, "An efficient multiscale approach
           to audio synchronization". International Conference on Music
           Information Retrieval, 6(1), 192-197 (2006).

    """
    x, y, precomputed_cost, n_timestamps_1, n_timestamps_2 = \
        _check_input_dtw(x, y, precomputed_cost=None, dist=dist,
                         method="multiscale")

    region = _multiscale_region(x, y, dist, resolution=resolution,
                                radius=radius)
    cost_mat = cost_matrix(x, y, dist=dist, region=region)
    acc_cost_mat = accumulated_cost_matrix(cost_mat, region=region)
    dtw_dist = acc_cost_mat[-1, -1]
    if dist == 'square':
        dtw_dist = sqrt(dtw_dist)

    res = _return_results(dtw_dist, cost_mat, acc_cost_mat,
                          return_cost, return_accumulated, return_path)
    return res


@deprecated("dtw_multiscale is deprecated in 0.11 and will be removed in "
            "0.12. Use dtw(method='multiscale') instead.")
def dtw_multiscale(x, y, dist='square', resolution=2, radius=0,
                   return_cost=False, return_accumulated=False,
                   return_path=False):
    """Multiscale Dynamic Time Warping distance.

    .. deprecated:: 0.11
        This function is deprecated and will be removed in 0.12.
        Use :func:`dtw` with ``method='multiscale'`` instead.

    Parameters
    ----------
    x : array-like, shape = (n_timestamps_1,)
        First array.

    y : array-like, shape = (n_timestamps_2,)
        Second array.

    dist : 'square', 'absolute', 'precomputed' or callable (default = 'square')
        Distance used. If 'square', the squared difference is used.
        If 'absolute', the absolute difference is used. If callable,
        it must be a function with a numba.njit() decorator that takes
        as input two numbers (two arguments) and returns a number.

    resolution : int (default = 2)
        The resolution level.

    radius : int (default = 0)
        The radius used to expand the constraint region. The optimal path
        computed at the resolution level is expanded with `radius` cells to the
        top, bottom, left and right of every cell belonging to the optimal
        path. It is computed at the resolution level.

    return_cost : bool (default = False)
        If True, the cost matrix is returned.

    return_accumulated : bool (default = False)
        If True, the accumulated cost matrix is returned.

    return_path : bool (default = False)
        If True, the optimal path is returned.

    Returns
    -------
    dtw_dist : float
        The DTW distance between the two arrays.

    cost_mat : ndarray, shape = (n_timestamps_1, n_timestamps_2)
        Cost matrix. Only returned if ``return_cost=True``.

    acc_cost_mat : ndarray, shape = (n_timestamps_1, n_timestamps_2)
        Accumulated cost matrix. Only returned if ``return_accumulated=True``.

    path : array, shape = (2, path_length)
        The optimal path along the cost matrix. The first row consists
        of the indices of the optimal path for x while the second row
        consists of the indices of the optimal path for y. Only returned
        if ``return_path=True``.

    """
    return _dtw_multiscale(x, y, dist, resolution, radius, return_cost,
                           return_accumulated, return_path)


def _dtw_fast(x, y, dist='square', radius=0, return_cost=False,
              return_accumulated=False, return_path=False):
    """Fast Dynamic Time Warping distance.

    This version of DTW builds an adaptive constraint region. The constraint
    region is created recursively by downsampling the time series, computing
    the optimal path and projecting it to a higher resolution. This process
    is repeated until the resolution matches the original resolution.

    Options
    -------
    radius : int (default = 0)
        The radius used to expand the constraint region. The optimal path
        computed at the resolution level is expanded with `radius` cells to the
        top, bottom, left and right of every cell belonging to the optimal
        path. It is computed at the resolution level.

    References
    ----------
    .. [1] S. Salvador ans P. Chan, "FastDTW: Toward Accurate Dynamic Time
           Warping in Linear Time and Space". KDD Workshop on Mining Temporal
           and Sequential Data, 70–80 (2004).

    """
    x, y, precomputed_cost, n_timestamps_1, n_timestamps_2 = \
        _check_input_dtw(x, y, precomputed_cost=None, dist=dist, method="fast")
    region = _fast_region(x, y, dist, radius=radius)
    cost_mat = cost_matrix(x, y, dist=dist, region=region)
    acc_cost_mat = accumulated_cost_matrix(cost_mat, region=region)
    dtw_dist = acc_cost_mat[-1, -1]
    if dist == 'square':
        dtw_dist = sqrt(dtw_dist)

    res = _return_results(dtw_dist, cost_mat, acc_cost_mat,
                          return_cost, return_accumulated, return_path)
    return res


@deprecated("dtw_fast is deprecated in 0.11 and will be removed in 0.12. "
            "Use dtw(method='fast') instead.")
def dtw_fast(x, y, dist='square', radius=0, return_cost=False,
             return_accumulated=False, return_path=False):
    """Fast Dynamic Time Warping distance.

    .. deprecated:: 0.11
        This function is deprecated and will be removed in 0.12.
        Use :func:`dtw` with ``method='fast'`` instead.

    Parameters
    ----------
    x : array-like, shape = (n_timestamps_1,)
        First array.

    y : array-like, shape = (n_timestamps_2,)
        Second array.

    dist : 'square', 'absolute', 'precomputed' or callable (default = 'square')
        Distance used. If 'square', the squared difference is used.
        If 'absolute', the absolute difference is used. If 'precomputed',
        `precomputed_cost` must be the cost matrix. If callable,
        it must be a function with a numba.njit() decorator that takes
        as input two numbers (two arguments) and returns a number.

    radius : int (default = 0)
        The radius used to expand the constraint region. The optimal path
        computed at the resolution level is expanded with `radius` cells to the
        top, bottom, left and right of every cell belonging to the optimal
        path. It is computed at the resolution level.

    return_cost : bool (default = False)
        If True, the cost matrix is returned.

    return_accumulated : bool (default = False)
        If True, the accumulated cost matrix is returned.

    return_path : bool (default = False)
        If True, the optimal path is returned.

    Returns
    -------
    dtw_dist : float
        The DTW distance between the two arrays.

    cost_mat : ndarray, shape = (n_timestamps_1, n_timestamps_2)
        Cost matrix. Only returned if ``return_cost=True``.

    acc_cost_mat : ndarray, shape = (n_timestamps_1, n_timestamps_2)
        Accumulated cost matrix. Only returned if ``return_accumulated=True``.

    path : ndarray, shape = (2, path_length)
        The optimal path along the cost matrix. The first row consists
        of the indices of the optimal path for x while the second row
        consists of the indices of the optimal path for y. Only returned
        if ``return_path=True``.

    """
    return _dtw_fast(x, y, dist, radius, return_cost, return_accumulated,
                     return_path)


def dtw(x=None, y=None, dist='square', method='classic', options=None,
        precomputed_cost=None, return_cost=False,
        return_accumulated=False, return_path=False):
    """Dynamic Time Warping (DTW) distance between two samples.

    Parameters
    ----------
    x : array-like, shape = (n_timestamps_1,)
        First array. Ignored if ``dist == 'precomputed'``.

    y : array-like, shape = (n_timestamps_2,)
        Second array. Ignored if ``dist == 'precomputed'``.

    dist : 'square', 'absolute', 'precomputed' or callable (default = 'square')
        Distance used. If 'square', the squared difference is used.
        If 'absolute', the absolute difference is used. If callable,
        it must be a function with a numba.njit() decorator that takes
        as input two numbers (two arguments) and returns a number.
        If 'precomputed', ``precomputed_cost`` must be the cost matrix and
        ``method`` must be 'classic', 'sakoechiba' or 'itakura'.

    method : str (default = 'classic')
        Method used. Should be one of

            - 'classic': :ref:`(see here) <metrics.dtw-classic>`
            - 'sakoechiba': :ref:`(see here) <metrics.dtw-sakoechiba>`
            - 'itakura': :ref:`(see here) <metrics.dtw-itakura>`
            - 'region': :ref:`(see here) <metrics.dtw-region>`
            - 'multiscale': :ref:`(see here) <metrics.dtw-multiscale>`
            - 'fast': :ref:`(see here) <metrics.dtw-fast>`

    options : None or dict (default = None)
        Dictionary of method options. Here is a quick summary of the available
        options for each method:

            - 'classic': None
            - 'sakoechiba': window_size (int or float)
            - 'itakura': max_slope (float)
            - 'region' : region (array-like)
            - 'multiscale': resolution (int) and radius (int)
            - 'fast': radius (int)

        For more information on these options, see :func:`show_options()`.

    precomputed_cost : array-like, shape = (n_timestamps_1, n_timestamps_2) \
            (default = None)
        Precomputed cost matrix between the time series.
        Ignored if ``dist != 'precomputed'``.

    return_cost : bool (default = False)
        If True, the cost matrix is returned.

    return_accumulated : bool (default = False)
        If True, the accumulated cost matrix is returned.

    return_path : bool (default = False)
        If True, the optimal path is returned.

    Returns
    -------
    dist : float
        The DTW distance between the two arrays.

    cost_mat : ndarray, shape = (n_timestamps_1, n_timestamps_2)
        Cost matrix. Only returned if ``return_cost=True``.

    acc_cost_mat : ndarray, shape = (n_timestamps_1, n_timestamps_2)
        Accumulated cost matrix. Only returned if ``return_accumulated=True``.

    path : ndarray, shape = (2, path_length)
        The optimal path along the cost matrix. The first row consists
        of the indices of the optimal path for x while the second row
        consists of the indices of the optimal path for y. Only returned
        if ``return_path=True``.

    Notes
    -----
    This section describes the available versions of Dynamic Time Warping (DTW)
    that can be selected by the `method` parameter.

    **Unconstrained path**

    Method :ref:`'classic'<metrics.dtw-classic>` computes the original DTW
    score between two time series with no constraint region: each cell is a
    valid cell to find the optimal path minimizing the total cost.

    **Global constraint region**

    Using no constraint region allows for very large distortion between both
    time series, which may be ill-suited for some cases. One possible solution
    to this issue is to use a global constraint region for the optimal path.
    A cell can be part of the optimal path if and only if this cell is in the
    constraint region. Another advantage of using a global constraint region is
    to decrease the computational complexity to find the optimal path.

    Method :ref:`'sakoechiba'<metrics.dtw-sakoechiba>` uses a band as the
    constraint region, allowing for a maximum fixed time shift at every time
    point.

    Method :ref:`'itakura'<metrics.dtw-itakura>` uses a parallelogram as
    the constraint region, allowing for time shifts of varying length: small
    time shifts at the beginning and at the end of the time series, larger
    time shifts in the middle.

    **Adaptative constraint region**

    One drawback of global constraint regions is that they do not depend on
    the time series and might be ill-suited for some time series. Adaptative
    constraint regions are regions that are specific to the two provided
    time series. However, their computational complexities are larger that
    the ones of global regions.

    Method :ref:`'multiscale'<metrics.dtw-multiscale>` downsamples both time
    series by a given factor, computes the optimal path with no constraint
    region for the downsampled time series, and projects the optimal path on
    the original scale to define the constraint region.

    Method :ref:`'fast'<metrics.dtw-fast>` can be seen as a recursive version
    of 'multiscale': the time series are downsampled so that the new time
    series are very small, the optimal path for these time series is computed
    and is projected at the scale that is two times larger. This process is
    repeated until the original scale of the time series is reached.

    References
    ----------
    .. [1] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 26(1), 43-49 (1978).

    .. [2] F. Itakura, "Minimum prediction residual principle applied to
           speech recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 23(1), 67–72 (1975).

    .. [3] M. Müller, H. Mattes and F. Kurth, "An efficient multiscale approach
           to audio synchronization". International Conference on Music
           Information Retrieval, 6(1), 192-197 (2006).

    .. [4] S. Salvador ans P. Chan, "FastDTW: Toward Accurate Dynamic Time
           Warping in Linear Time and Space". KDD Workshop on Mining Temporal
           and Sequential Data, 70–80 (2004).

    Examples
    --------
    >>> from pyts.metrics import dtw
    >>> x = [0, 1, 1]
    >>> y = [2, 0, 1]
    >>> dtw(x, y, method='sakoechiba', options={'window_size': 0.5})
    2.0

    """
    if options is None:
        options = dict()
    x, y, precomputed_cost, n_timestamps_1, n_timestamps_2 = \
        _check_input_dtw(x, y, precomputed_cost, dist, method)
    region = _compute_region(n_timestamps_1, n_timestamps_2, method, dist, x=x,
                             y=y, **options)
    cost_mat = _input_to_cost(x, y, dist, precomputed_cost, region)
    acc_cost_mat = accumulated_cost_matrix(cost_mat, region=region)
    dtw_dist = acc_cost_mat[-1, -1]
    if dist == 'square':
        dtw_dist = sqrt(dtw_dist)

    res = _return_results(dtw_dist, cost_mat, acc_cost_mat,
                          return_cost, return_accumulated, return_path)
    return res


def show_options(method=None, disp=True):
    """Show documentation for additional options of Dynamic Time Warping
    methods.

    These are method-specific options that can be supplied through the
    ``options`` dict.

    Parameters
    ----------
    method : None or str (default = None)
        If None, shows all methods. Otherwise, shows only the options for the
        specified method. If str, it must be either 'classic', 'sakoechiba',
        'itakura', 'region', 'multiscale' or 'fast'.

    disp : bool (default = True)
        Whether to print the result rather than returning it.

    Returns
    -------
    text : None or str
        Either None (if ``disp=True``) or the text string (if ``disp=False``).

    Notes
    -----
    The available methods are:

    - :ref:`'classic' <metrics.dtw-classic>`
    - :ref:`'sakoechiba' <metrics.dtw-sakoechiba>`
    - :ref:`'itakura' <metrics.dtw-itakura>`
    - :ref:`'region' <metrics.dtw-region>`
    - :ref:`'multiscale' <metrics.dtw-multiscale>`
    - :ref:`'fast' <metrics.dtw-fast>`

    """
    import textwrap

    def get_options(string):
        start = '    Options'
        end = '    References'
        str_lines = string.splitlines()
        start_idx = str_lines.index(start)
        end_idx = str_lines.index(end)
        doc = textwrap.dedent('\n'.join(str_lines[:start_idx])) + '\n'
        doc += textwrap.dedent('\n'.join(str_lines[start_idx:end_idx])) + '\n'
        return doc

    text = """\n"""
    if method is None:
        methods = ('classic', 'sakoechiba', 'itakura', 'region',
                   'multiscale', 'fast')
        for met in methods:
            text += show_options(met, disp=False)
    elif method == 'classic':
        text += "classic\n=======\n\n"
        text += get_options(_dtw_classic.__doc__)
    elif method == 'sakoechiba':
        text += "sakoechiba\n==========\n\n"
        text += get_options(_dtw_sakoechiba.__doc__)
    elif method == 'itakura':
        text += "itakura\n=======\n\n"
        text += get_options(_dtw_itakura.__doc__)
    elif method == 'region':
        text += "region\n=======\n\n"
        text += get_options(_dtw_itakura.__doc__)
    elif method == 'multiscale':
        text += "multiscale\n==========\n\n"
        text += get_options(_dtw_multiscale.__doc__)
    elif method == 'fast':
        text += "fast\n====\n\n"
        text += get_options(_dtw_fast.__doc__)
    else:
        raise ValueError("'method' must be either None, 'classic', "
                         "'sakoechiba', 'itakura', 'region', 'multiscale' "
                         "or 'fast'.")
    if disp:
        print(text)
        return
    else:
        return text
