"""
=====================
Itakura parallelogram
=====================

This example shows how to set the parameters of the Itakura parallelogram when
computing the Dynamic Time Warping (DTW) between two time series.
It is implemented in :func:`pyts.metrics.dtw.itakura_parallelogram`.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
#         Hicham Janati <hicham.janati@inria.fr>
# License: BSD-3 Clause

import numpy as np
import matplotlib.pyplot as plt
from pyts.metrics import itakura_parallelogram
from pyts.metrics.dtw import _get_itakura_slopes

# We write a function to visualize the sakoe-chiba band for different
# time series lengths.


def plot_itakura(n_timestamps_1, n_timestamps_2, max_slope=1.,
                 ax=None):

    region = itakura_parallelogram(n_timestamps_1, n_timestamps_2, max_slope)
    max_slope, min_slope = _get_itakura_slopes(n_timestamps_1,
                                               n_timestamps_2,
                                               max_slope)

    if ax is None:
        _, ax = ax.subplots(1, 1, figsize=(n_timestamps_1 / 2,
                                           n_timestamps_2 / 2))
    mask = np.zeros((n_timestamps_2, n_timestamps_1))
    for i, (j, k) in enumerate(region.T):
        mask[j:k, i] = 1.

    ax.imshow(mask, origin='lower', cmap='Wistia')

    sz = max(n_timestamps_1, n_timestamps_2)
    x = np.arange(-1, sz + 1)

    low_max_line = ((n_timestamps_2 - 1) - max_slope * (n_timestamps_1 - 1)) +\
        max_slope * np.arange(-1, sz + 1)
    up_min_line = ((n_timestamps_2 - 1) - min_slope * (n_timestamps_1 - 1)) +\
        min_slope * np.arange(-1, sz + 1)
    diag = (n_timestamps_2 - 1) / (n_timestamps_1 - 1) * np.arange(-1, sz + 1)
    ax.plot(x, diag, 'black', lw=6)
    ax.plot(x, max_slope * np.arange(-1, sz + 1), 'b', lw=6)
    ax.plot(x, min_slope * np.arange(-1, sz + 1), 'r', lw=6)
    ax.plot(x, low_max_line, 'g', lw=6)
    ax.plot(x, up_min_line, 'y', lw=6)

    for i in range(n_timestamps_1):
        for j in range(n_timestamps_2):
            ax.plot(i, j, 'o', color='green', ms=7)

    ax.set_xticks(np.arange(-.5, max(n_timestamps_1, n_timestamps_2), 1),
                  minor=True)
    ax.set_yticks(np.arange(-.5, max(n_timestamps_1, n_timestamps_2), 1),
                  minor=True)
    ax.grid(which='minor', color='b', linestyle='--', linewidth=2)

    ax.set_xlim((-0.5, n_timestamps_1 - 0.5))
    ax.set_ylim((-0.5, n_timestamps_2 - 0.5))

    return ax

###########################################################################
# When `relative_window_size == True`, the `window_size` arg should be a
# fraction (0-1) of the time series lengths. If `relative_window_size == False`
# `window_size` is the max allowed temporal shift between the time series.


slopes = [1., 1.5, 3.]
rc = {"font.size": 35, "axes.titlesize": 45,
      "xtick.labelsize": 28, "ytick.labelsize": 28}
plt.rcParams.update(rc)
n_timestamps_1 = 15
n_timestamps_2 = 15
f, axes = plt.subplots(1, 3, figsize=(n_timestamps_1 * 3, n_timestamps_2))
for ax, slope in zip(axes, slopes):
    plot_itakura(n_timestamps_1, n_timestamps_2, max_slope=slope, ax=ax)
    title = "Slope = {} ".format(slope)
    ax.set_title(title)
plt.show()


####################################
# We show the same plot with n1 < n2

n_timestamps_1 = 15
n_timestamps_2 = 20

f, axes = plt.subplots(1, 3, figsize=(n_timestamps_1 * 3, n_timestamps_2))
for ax, slope in zip(axes, slopes):
    plot_itakura(n_timestamps_1, n_timestamps_2, max_slope=slope, ax=ax)
    title = "Slope = {} ".format(slope)
    ax.set_title(title)
plt.show()

####################################
# We show the same plot with n2 < n1

n_timestamps_1 = 20
n_timestamps_2 = 15

f, axes = plt.subplots(1, 3, figsize=(n_timestamps_1 * 3, n_timestamps_2))
for ax, slope in zip(axes, slopes):
    plot_itakura(n_timestamps_1, n_timestamps_2, max_slope=slope, ax=ax)
    title = "Slope = {} ".format(slope)
    ax.set_title(title)
plt.show()
