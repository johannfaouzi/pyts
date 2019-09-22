"""
================
Sakoe-Chiba band
================

This example explains how to set the `window_size` parameter of the Sakoe-Chiba
band when computing the Dynamic Time Warping (DTW) with
`method` == "sakoechiba". The Sakoe-Chiba region is defined through a
`window_size` parameter which determines the largest temporal shift allowed
from the diagonal in the direction of the longest time series. It is
implemented in :func:`pyts.metrics.dtw.sakoe_chiba_band`. The window size can
be either set relatively to the length of the longest time series as a ratio
between 0 and 1, or manually if an integer is given.

This example visualizes the Sakoe-Chiba band in different scenarios: the
degenerate case `window_size` = 0, a relative size `window_size` 0.2 and the
absolute `window_size` = 4. The last two cases are equivalent since
0.2 * 20 = 4.
"""


# Author: Hicham Janati <hicham.janati@inria.fr>
#         Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pyts.metrics import sakoe_chiba_band
from pyts.metrics.dtw import _check_sakoe_chiba_params


# #####################################################################
# We write a function to visualize the sakoe-chiba band for different
# time series lengths.


def plot_sakoe_chiba(n_timestamps_1, n_timestamps_2, window_size=0.5,
                     ax=None):

    region = sakoe_chiba_band(n_timestamps_1, n_timestamps_2, window_size)
    scale, horizontal_shift, vertical_shift = \
        _check_sakoe_chiba_params(n_timestamps_1, n_timestamps_2, window_size)
    mask = np.zeros((n_timestamps_2, n_timestamps_1))
    for i, (j, k) in enumerate(region.T):
        mask[j:k, i] = 1.

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(n_timestamps_1 / 2,
                                            n_timestamps_2 / 2))
    ax.imshow(mask, origin='lower', cmap='Wistia', vmin=0, vmax=1)

    sz = max(n_timestamps_1, n_timestamps_2)
    x = np.arange(-1, sz + 1)
    lower_bound = scale * (x - horizontal_shift) - vertical_shift
    upper_bound = scale * (x + horizontal_shift) + vertical_shift
    ax.plot(x, lower_bound, 'b', lw=2)
    ax.plot(x, upper_bound, 'g', lw=2)
    diag = (n_timestamps_2 - 1) / (n_timestamps_1 - 1) * np.arange(-1, sz + 1)
    ax.plot(x, diag, 'black', lw=1)

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


window_sizes = [0, 0.2, 4]

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
    for j, window_size in enumerate(window_sizes):
        loc_x, loc_y = locs[i][j]
        ax = fig.add_subplot(gs[loc_x: loc_x + spans[i][0],
                                loc_y: loc_y + spans[i][1]])
        if i == 0:
            ax.set_title("Window size = {}".format(window_size))
        plot_sakoe_chiba(n1, n2, window_size, ax=ax)
        ax.set_xticks(np.arange(0, n1, 2))
        ax.set_yticks(np.arange(0, n2, 2))
plt.show()
