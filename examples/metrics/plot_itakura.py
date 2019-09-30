"""
=====================
Itakura parallelogram
=====================

This example explains how to set the `max_slope` parameter of the itakura
parallelogram when computing the Dynamic Time Warping (DTW) with
`method` == "itakura". The Itakura parallelogram is defined through a
`max_slope` parameter which determines the slope of the steeper side. It is
implemented in :func:`pyts.metrics.dtw.itakura_parallelogram`. The slope of the
other side is set to "1 / max_slope". For a feasible region, `max_slope`
must be larger than 1. This example visualizes the itakura parallelogram with
different slopes and temporal dimensions.
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyts.metrics import itakura_parallelogram
from pyts.metrics.dtw import _get_itakura_slopes

# #####################################################################
# We write a function to visualize the itakura parallelogram for different
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
    ax.plot(x, diag, 'black', lw=1)
    ax.plot(x, max_slope * np.arange(-1, sz + 1), 'b', lw=1.5)
    ax.plot(x, min_slope * np.arange(-1, sz + 1), 'r', lw=1.5)
    ax.plot(x, low_max_line, 'g', lw=1.5)
    ax.plot(x, up_min_line, 'y', lw=1.5)

    for i in range(n_timestamps_1):
        for j in range(n_timestamps_2):
            ax.plot(i, j, 'o', color='green', ms=1)

    ax.set_xticks(np.arange(-.5, max(n_timestamps_1, n_timestamps_2), 1),
                  minor=True)
    ax.set_yticks(np.arange(-.5, max(n_timestamps_1, n_timestamps_2), 1),
                  minor=True)
    ax.grid(which='minor', color='b', linestyle='--', linewidth=1)

    ax.set_xlim((-0.5, n_timestamps_1 - 0.5))
    ax.set_ylim((-0.5, n_timestamps_2 - 0.5))

    return ax


slopes = [1., 1.5, 3.]
rc = {"font.size": 20, "axes.titlesize": 16,
      "xtick.labelsize": 14, "ytick.labelsize": 14}
plt.rcParams.update(rc)


lengths = [(20, 20), (20, 10), (10, 20)]
fig = plt.figure(constrained_layout=True, figsize=(16, 16))
gs = gridspec.GridSpec(12, 12, figure=fig)
spans = [(4, 4), (2, 4), (4, 2)]
locs = [[[0, 0], [0, 4], [0, 8]],
        [[5, 0], [5, 4], [5, 8]],
        [[8, 1], [8, 5], [8, 9]]]
for i, (n1, n2) in enumerate(lengths):
    for j, slope in enumerate(slopes):
        loc_x, loc_y = locs[i][j]
        ax = fig.add_subplot(gs[loc_x: loc_x + spans[i][0],
                                loc_y: loc_y + spans[i][1]])
        if i == 0:
            ax.set_title("Slope = {}".format(slope))
        plot_itakura(n1, n2, max_slope=slope, ax=ax)
        ax.set_xticks(np.arange(0, n1, 2))
        ax.set_yticks(np.arange(0, n2, 2))
plt.show()
