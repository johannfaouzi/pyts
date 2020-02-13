"""Code for utility tools."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import njit
from sklearn.utils import check_array


def segmentation(ts_size, window_size, overlapping=False, n_segments=None):
    """Compute the indices for Piecewise Agrgegate Approximation.

    Parameters
    ----------
    ts_size : int
        The size of the time series.

    window_size : int
        The size of the window.

    overlapping : bool (default = False)
        If True, overlapping windows may be used. If False, non-overlapping
        are used.

    n_segments : int or None (default = None)
        The number of windows. If None, the number is automatically
        computed using ``window_size``.

    Returns
    -------
    start : array
        The lower bound for each window.

    end : array
        The upper bound for each window.

    size : int
        The size of ``start``.

    Examples
    --------
    >>> from pyts.utils import segmentation
    >>> start, end, size = segmentation(ts_size=12, window_size=3)
    >>> print(start)
    [0 3 6 9]
    >>> print(end)
    [ 3  6  9 12]
    >>> size
    4

    """
    if not isinstance(ts_size, (int, np.integer)):
        raise TypeError("'ts_size' must be an integer.")
    if not ts_size >= 2:
        raise ValueError("'ts_size' must be an integer greater than or equal "
                         "to 2 (got {0}).".format(ts_size))
    if not isinstance(window_size, (int, np.integer)):
        raise TypeError("'window_size' must be an integer.")
    if not window_size >= 1:
        raise ValueError("'window_size' must be an integer greater than or "
                         "equal to 1 (got {0}).".format(window_size))
    if not window_size <= ts_size:
        raise ValueError("'window_size' must be lower than or equal to "
                         "'ts_size' ({0} > {1}).".format(window_size, ts_size))
    if not (n_segments is None or isinstance(n_segments, (int, np.integer))):
        raise TypeError("'n_segments' must be None or an integer.")
    if isinstance(n_segments, (int, np.integer)):
        if not n_segments >= 2:
            raise ValueError(
                "If 'n_segments' is an integer, it must be greater than or "
                "equal to 2 (got {0}).".format(n_segments)
            )
        if not n_segments <= ts_size:
            raise ValueError(
                "If 'n_segments' is an integer, it must be lower than or "
                "equal to 'ts_size' ({0} > {1}).".format(n_segments, ts_size)
            )

    if n_segments is None:
        quotient, remainder = divmod(ts_size, window_size)
        n_segments = quotient if remainder == 0 else quotient + 1

    if not overlapping:
        bounds = np.linspace(0, ts_size, n_segments + 1).astype('int64')
        start = bounds[:-1]
        end = bounds[1:]
        size = start.size
        return start, end, size
    else:
        n_overlapping = (n_segments * window_size) - ts_size
        n_overlaps = n_segments - 1
        overlaps = np.linspace(0, n_overlapping,
                               n_overlaps + 1).astype('int64')
        bounds = np.arange(0, (n_segments + 1) * window_size, window_size)
        start = bounds[:-1] - overlaps
        end = bounds[1:] - overlaps
        size = start.size
        return start, end, size


@njit()
def _windowed_view(X, n_samples, n_timestamps, window_size, window_step):
    overlap = window_size - window_step
    shape_new = (n_samples,
                 (n_timestamps - overlap) // window_step,
                 window_size // 1)
    s0, s1 = X.strides
    strides_new = (s0, window_step * s1, s1)
    return as_strided(X, shape=shape_new, strides=strides_new)


def windowed_view(X, window_size, window_step=1):
    """Return a windowed view of a 2D array.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_timestamps)
        Input data.

    window_size : int
        The size of the window. It must be between 1 and ``n_timestamps``.

    window_step : int (default = 1)
        The step of the sliding window

    Returns
    -------
    X_new : array, shape = (n_samples, n_windows, window_size)
        Windowed view of the input data. ``n_windows`` is computed as
        ``(n_timestamps - window_size + window_step) // window_step``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyts.utils import windowed_view
    >>> windowed_view(np.arange(6).reshape(1, -1), window_size=2)
    array([[[0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5]]])

    """
    X = check_array(X, dtype=None)
    n_samples, n_timestamps = X.shape

    if not isinstance(window_size, (int, np.integer)):
        raise TypeError("'window_size' must be an integer.")
    if not 1 <= window_size <= n_timestamps:
        raise ValueError("'window_size' must be an integer between 1 and "
                         "n_timestamps.")
    if not isinstance(window_step, (int, np.integer)):
        raise TypeError("'window_step' must be an integer.")
    if not 1 <= window_step <= n_timestamps:
        raise ValueError("'window_step' must be an integer between 1 and "
                         "n_timestamps.")

    return _windowed_view(X, n_samples, n_timestamps, window_size, window_step)
