from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import numpy as np
from math import log


def bin_allocation_integers(x, n_bins, quantiles):

    for i in range(n_bins - 1):
        if x < quantiles[i]:
            return i

    return n_bins - 1


def bin_allocation_alphabet(x, n_bins, alphabet, quantiles):

    for i in range(n_bins - 1):
        if x < quantiles[i]:
            return alphabet[i]

    return alphabet[n_bins - 1]


def segmentation(bounds, window_size, overlapping):

    start = bounds[:-1]
    end = bounds[1:]

    if not overlapping:
        return np.array([arange(np.array([start, end])[:, i]) for i in range(start.size)])

    else:
        correction = window_size - end + start

        new_start = start.copy()
        new_start[start.size // 2:] = start[start.size // 2:] - correction[start.size // 2:]

        new_end = end.copy()
        new_end[:end.size // 2] = end[:end.size // 2] + correction[:end.size // 2]

        return np.apply_along_axis(arange, 1, np.array([new_start, new_end]).T)


def mean(ts, indices, overlapping):

    if not overlapping:
        return np.array([ts[indices[i]].mean() for i in range(indices.shape[0])])

    else:
        return np.mean(ts[indices], axis=1)


def arange(array):

    return np.arange(array[0], array[1])


def paa(ts, ts_size, window_size, overlapping, n_segments=None, plot=False):

    if (n_segments == ts_size or window_size == 1):
        return ts

    if n_segments is None:
        quotient = ts_size // window_size
        remainder = ts_size % window_size
        n_segments = quotient if remainder == 0 else quotient + 1
    bounds = np.linspace(0, ts_size, n_segments + 1, endpoint=True).astype('int16')
    indices = segmentation(bounds, window_size, overlapping)

    if not plot:
        return mean(ts, indices, overlapping)
    else:
        return indices, mean(ts, indices, overlapping)


def sax(ts, n_bins, quantiles, alphabet, plot=False):

        # Alphabet
        alphabet_short = alphabet[:n_bins]

        return_quantiles = False

        # Compute empirical quantiles if quantiles == 'empirical'
        if type(quantiles) == str:
            return_quantiles = True
            quantiles = np.percentile(ts, np.linspace(0, 100, n_bins + 1)[1:])

        # Compute binned time series
        binned_ts = [bin_allocation_alphabet(x, n_bins, alphabet_short, quantiles) for x in ts]

        if (plot and return_quantiles):
            # Return joined string
            return quantiles, ''.join(x for x in binned_ts)
        else:
            # Return joined string
            return ''.join(x for x in binned_ts)


def num_red(array):

    indices = []
    array_size = len(array)
    index = 1
    while index < array_size:
        if array[index - 1] == array[index]:
            indices.append(index)
        index += 1

    return np.delete(array, indices).tolist()


def vsm(ts_sax, ts_sax_size, window_size, numerosity_reduction=True):

    ts_vsm = [ts_sax[i:i + window_size] for i in range(ts_sax_size - window_size)]

    if numerosity_reduction:
        return num_red(ts_vsm)

    else:
        return ts_vsm


def gaf(ts, ts_size, image_size, overlapping, method, scale):

    # Compute aggregated time series
    window_size = ts_size // image_size
    window_size += 0 if ts_size % image_size == 0 else 1
    aggregated_ts = paa(ts, ts_size, window_size, overlapping, image_size)

    # Rescaling aggregated time series
    min_ts, max_ts = np.min(aggregated_ts), np.max(aggregated_ts)
    if scale == '0':
        rescaled_ts = (aggregated_ts - min_ts) / (max_ts - min_ts)
    if scale == '-1':
        rescaled_ts = (2 * aggregated_ts - max_ts - min_ts) / (max_ts - min_ts)

    # Compute GAF
    sin_ts = np.sqrt(np.clip(1 - rescaled_ts**2, 0, 1))
    if method == 's':
        return np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
    if method == 'd':
        return np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)


def mtf(ts, ts_size, image_size, n_bins, quantiles, overlapping):

    # Compute empirical quantiles if quantiles == 'empirical'
    if type(quantiles) == str:
        quantiles = np.percentile(ts, np.linspace(0, 100, n_bins + 1)[1:])

    # Compute binned time series
    binned_ts = np.array([bin_allocation_integers(x, n_bins, quantiles) for x in ts])

    # Compute Markov Transition Matrix
    MTM = np.zeros((n_bins, n_bins))
    for i in range(ts_size - 1):
        MTM[binned_ts[i], binned_ts[i + 1]] += 1
    non_zero_rows = np.where(MTM.sum(axis=1) != 0)[0]
    MTM = np.multiply(MTM[non_zero_rows][:, non_zero_rows].T, np.sum(MTM[non_zero_rows], axis=1)**(-1)).T

    # Compute list of indices based on values
    list_values = [np.where(binned_ts == q) for q in non_zero_rows]

    # Compute Markov Transition Field
    MTF = np.zeros((ts_size, ts_size))
    for i in range(non_zero_rows.size):
        for j in range(non_zero_rows.size):
            MTF[np.meshgrid(list_values[i], list_values[j])] = MTM[i, j]

    # Compute Aggregated Markov Transition Field
    window_size, remainder = ts_size // image_size, ts_size % image_size
    if remainder == 0:
        return np.reshape(MTF, (image_size, window_size, image_size, window_size)).mean(axis=(1, 3))

    else:
        window_size += 1
        bounds = np.linspace(0, ts_size, image_size + 1, endpoint=True).astype('int16')
        indices = segmentation(bounds, window_size, overlapping)

        AMTF = np.zeros((image_size, image_size))
        for i in range(image_size):
            for j in range(image_size):
                AMTF[i, j] = MTF[indices[i]][:, indices[j]].mean()

        return AMTF


def idf_func(x, num_classes):
    if x > 0:
        return log(num_classes / x)
    else:
        return 0


def idf_smooth_func(x, num_classes):
    if x > 0:
        return log(1 + num_classes / x)
    else:
        return 0


def dtw(x, y, dist='absolute', return_path=False, **kwargs):

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
            D[i].append(C[i][j] + min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]))

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
                        region[key] = np.append(region[key], np.arange(first_value, second_value))

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

        # Compute first row
        D[0, :window_size] = np.cumsum(np.asarray(C[0][:window_size]))

        # Compute first column
        for j in range(1, window_size):
            D[j, 0] = D[j - 1, 0] + C[j][0]

        # Compute the remaining cells recursively
        for j in range(1, x_size):
            for i in region[j]:
                D[i, j] = C[i][j] + min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1])

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

        return region, D, path[::-1]
