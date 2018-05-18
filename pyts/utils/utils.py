"""The :mod:`pyts.utils` module includes utility functions."""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
import numpy as np


standard_library.install_aliases()


def segmentation(ts_size, window_size, overlapping, n_segments=None):
    """Compute the indices for Piecewise Agrgegate Approximation.

    Parameters
    ----------
    ts_size : int
        The size of the time series.

    window_size : int
        The size of the window.

    overlapping : bool
        If True, overlapping windows may be used. If False, non-overlapping
        are used.

    n_segments : int or None (default = None)
        The number of windows. If None, the number is automatically
        computed using `window_size`.

    Returns
    -------
    start : array
        The lower bound for each window.

    end : array
        The upper bound for each window.

    size : int
        The size of `start`.

    """
    if n_segments is None:
        quotient = ts_size // window_size
        remainder = ts_size % window_size
        n_segments = quotient if remainder == 0 else quotient + 1

    bounds = np.linspace(0, ts_size,
                         n_segments + 1, endpoint=True).astype('int64')

    start = bounds[:-1]
    end = bounds[1:]
    size = start.size

    if not overlapping:
        return start, end, size
    else:
        correction = window_size - end + start
        half_size = size // 2
        new_start = start.copy()
        new_start[half_size:] = start[half_size:] - correction[half_size:]
        new_end = end.copy()
        new_end[:half_size] = end[:half_size] + correction[:half_size]
        return new_start, new_end, size


def numerosity_reduction(arr):
    """Perform numerosity reduction.

    Parameters
    ----------
    arr : array-like, shape [n_samples]

    Returns
    -------
    res : str
        string with each word separated with a whitespace.

    """
    not_equal = np.array(arr[1:] != arr[:-1])
    return ' '.join(np.append(arr[np.where(not_equal)], arr[-1]))


def dtw(x, y, dist='absolute', return_path=False, **kwargs):
    """Dynamic Time Warping.

    Parameters
    ----------
    x : array-like, shape [n1]
        First array.

    y : array-like, shape [n2]
        Second array

    dist : {'absolute', 'square' or callable} (default = 'absolue')
        The distance metric used. If callable, the first two arguments
        must be float numbers.

    return_path : bool (default = False)
        If true, the path along the accumulated cost matrix is returned.

    kwargs
        Additional keyword arguments for `dist` if `dist` is callable.
        Ignored otherwise.

    """
    x_size = x.size

    # Cost matrix
    C = [[] for _ in range(x_size)]

    if dist == 'absolute':
        for i in range(x_size):
            for j in range(x_size):
                C[i].append(abs(x[i] - y[j]))

    elif dist == 'square':
        for i in range(x_size):
            for j in range(x_size):
                C[i].append((x[i] - y[j])**2)

    else:
        for i in range(x_size):
            for j in range(x_size):
                C[i].append(dist(x[i], y[j], **kwargs))

    # Accumulated cost matrix
    D = [[0] for _ in range(x_size)]

    # Compute first row
    D[0] = np.cumsum(C[0]).tolist()

    # Compute first column
    for j in range(1, x_size):
        D[j][0] = D[j - 1][0] + C[j][0]

    # Compute the remaining cells recursively
    for j in range(1, x_size):
        for i in range(1, x_size):
            D[i].append(C[i][j] + min(D[i - 1][j - 1],
                                      D[i - 1][j],
                                      D[i][j - 1]))

    if not return_path:
        return D[x_size - 1][x_size - 1]

    else:
        path = [(x_size - 1, x_size - 1)]
        while path[-1] != (0, 0):
            i, j = path[-1]
            if i == 0:
                path.append((0, j - 1))
            elif j == 0:
                path.append((i - 1, 0))
            else:
                List = [D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]]
                argmin_List = List.index(min(List))
                if argmin_List == 0:
                    path.append((i - 1, j - 1))
                elif argmin_List == 1:
                    path.append((i - 1, j))
                else:
                    path.append((i, j - 1))

        return D, path[::-1]


def fast_dtw(x, y, window_size, approximation=True,
             dist='absolute', return_path=False, **kwargs):
    """Fast Dynamic Time Warping.

    Parameters
    ----------
    x : array-like, shape [n1]
        First array.

    y : array-like, shape [n2]
        Second array

    window_size : int
        The size of the window for the PAA algorithm.

    approximation : bool (default = True)
        If True, compute Dynamic Time Warping between the shrunk time
        series. If False, compute Dynamic Time Warping on the original
        time series with a constraint region based on the path of the
        Dynamic Time Warping of the shrunk time series.

    dist : {'absolute', 'square' or callable} (default = 'absolue')
        The distance metric used. If callable, the first two arguments
        must be float numbers.

    return_path : bool (default = False)
        If true, the path along the accumulated cost matrix is returned.

    kwargs
        Additional keyword arguments for `dist` if `dist` is callable.
        Ignored otherwise.

    """
    x_size = x.size

    # Compute path for shrunk time series
    remainder = x_size % window_size

    if remainder != 0:
        x_copy = np.append(x, [x[-1] for _ in range(window_size - remainder)])
        y_copy = np.append(y, [y[-1] for _ in range(window_size - remainder)])
    else:
        x_copy = x.copy()
        y_copy = y.copy()

    x_shrunk_size = x_copy.size // window_size
    x_shrunk = x_copy.reshape(x_shrunk_size, window_size).mean(axis=1)
    y_shrunk = y_copy.reshape(x_shrunk_size, window_size).mean(axis=1)

    if approximation:
        return dtw(x_shrunk, y_shrunk, dist, return_path, **kwargs)

    else:
        _, fast_path = dtw(x_shrunk, y_shrunk, dist, True, **kwargs)

        # Region of constraints
        region = {}
        for i, j in fast_path:
            first_value = i * window_size
            second_value = min((i + 1) * window_size, x_size)
            for a in range(window_size):
                key = j * window_size + a
                if key < x_size:
                    if key not in region.keys():
                        region[key] = np.arange(first_value, second_value)
                    else:
                        region[key] = np.append(region[key],
                                                np.arange(first_value,
                                                          second_value)
                                                )
        # Cost matrix
        C = [[] for _ in range(x_size)]
        if dist == 'absolute':
            for i in range(x_size):
                for j in range(x_size):
                    C[i].append(abs(x[i] - y[j]))
        elif dist == 'square':
            for i in range(x_size):
                for j in range(x_size):
                    C[i].append((x[i] - y[j])**2)
        else:
            for i in range(x_size):
                for j in range(x_size):
                    C[i].append(dist(x[i], y[j], **kwargs))

        # Accumulated cost matrix
        D = np.zeros((x_size, x_size)) + np.inf
        D[0, :window_size] = np.cumsum(np.asarray(C[0][:window_size]))
        for j in range(1, window_size):
            D[j, 0] = D[j - 1, 0] + C[j][0]
        for j in range(1, x_size):
            for i in region[j]:
                D[i, j] = C[i][j] + min(D[i - 1][j - 1],
                                        D[i - 1][j],
                                        D[i][j - 1])
        if not return_path:
            return D[x_size - 1][x_size - 1]
        else:
            path = [(x_size - 1, x_size - 1)]
            while path[-1] != (0, 0):
                i, j = path[-1]
                if i == 0:
                    path.append((0, j - 1))
                elif j == 0:
                    path.append((i - 1, 0))
                else:
                    liste = [D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]]
                    argmin_liste = np.argmin(liste)
                    if argmin_liste == 0:
                        path.append((i - 1, j - 1))
                    elif argmin_liste == 1:
                        path.append((i - 1, j))
                    else:
                        path.append((i, j - 1))

        return region, D, path[::-1]
