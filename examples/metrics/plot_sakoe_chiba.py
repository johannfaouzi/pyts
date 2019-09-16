"""
================
Sakoe-Chiba band
================

This example shows how to set the parameters of the Sakoe-Chiba band when
computing the Dynamic Time Warping (DTW) between two time series.
It is implemented in :func:`pyts.metrics.dtw.sakoe_chiba_band`.
"""


import numpy as np
import matplotlib.pyplot as plt
from pyts.metrics import sakoe_chiba_band
from pyts.metrics.dtw import _check_sakoe_chiba_params

# We write a function to visualize the sakoe-chiba band for different
# time series lengths.


def plot_sakoe_chiba(n_timestamps_1, n_timestamps_2, window_size=0.5,
                     relative_window_size=True, ax=None):

    region = sakoe_chiba_band(n_timestamps_1, n_timestamps_2, window_size,
                              relative_window_size)
    scale, window_size = \
        _check_sakoe_chiba_params(n_timestamps_1, n_timestamps_2, window_size,
                                  relative_window_size)
    mask = np.zeros((n_timestamps_2, n_timestamps_1))
    for i, (j, k) in enumerate(region.T):
        mask[j:k, i] = 1.

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(n_timestamps_1 / 2,
                                            n_timestamps_2 / 2))
    ax.imshow(mask, origin='lower', cmap='Wistia', vmin=0, vmax=1)

    sz = max(n_timestamps_1, n_timestamps_2)
    x = np.arange(-1, sz + 1)

    ax.plot(x, scale * np.arange(-1, sz + 1) - window_size, 'b', lw=6)
    ax.plot(x, scale * np.arange(-1, sz + 1) + window_size, 'g', lw=6)
    diag = (n_timestamps_2 - 1) / (n_timestamps_1 - 1) * np.arange(-1, sz + 1)
    ax.plot(x, diag, 'black', lw=6)

    for i in range(n_timestamps_1):
        for j in range(n_timestamps_2):
            ax.plot(i, j, 'o', color='green', ms=7)

    # ax = ax.gca()
    ax.set_xticks(np.arange(-.5, max(n_timestamps_1, n_timestamps_2), 1),
                  minor=True)
    ax.set_yticks(np.arange(-.5, max(n_timestamps_1, n_timestamps_2), 1),
                  minor=True)
    ax.grid(which='minor', color='b', linestyle='--', linewidth=2)

    ax.set_xlim((-0.5, n_timestamps_1 - 0.5))
    ax.set_ylim((-0.5, n_timestamps_2 - 0.5))

    return ax

################################
# relative window_size

# When `relative_window_size == True`, the `window_size` arg should be a
# fraction (0-1) of the time series lengths. `relative_window_size == False`
# `window_size` is the max allowed temporal shift between the time series.


rc = {"font.size": 35, "axes.titlesize": 45,
      "xtick.labelsize": 28, "ytick.labelsize": 28}
plt.rcParams.update(rc)

n_timestamps_1 = 15
n_timestamps_2 = 15
params = [dict(window_size=0, relative_window_size=True),
          dict(window_size=0.5, relative_window_size=True),
          dict(window_size=2, relative_window_size=False)]
f, axes = plt.subplots(1, 3, figsize=(n_timestamps_1 * 3, n_timestamps_2))
for ax, params_dict in zip(axes, params):
    plot_sakoe_chiba(n_timestamps_1, n_timestamps_2, ax=ax, **params_dict)
    title = "Window size = {} ".format(params_dict["window_size"])
    title += "\n Relative = {} ".format(params_dict["relative_window_size"])
    ax.set_title(title)
plt.show()


###########
# We show the same plot with unbalanced proportions

n_timestamps_1 = 15
n_timestamps_2 = 20
params = [dict(window_size=0, relative_window_size=True),
          dict(window_size=0.5, relative_window_size=True),
          dict(window_size=4, relative_window_size=False)]
f, axes = plt.subplots(1, 3, figsize=(n_timestamps_1 * 3, n_timestamps_2))
for ax, params_dict in zip(axes, params):
    plot_sakoe_chiba(n_timestamps_1, n_timestamps_2, ax=ax, **params_dict)
    title = "Window size = {} ".format(params_dict["window_size"])
    title += "\n Relative = {} ".format(params_dict["relative_window_size"])
    ax.set_title(title)
plt.show()

###########
# We show the same plot with reversed unbalanced proportions

n_timestamps_1 = 20
n_timestamps_2 = 15
params = [dict(window_size=0, relative_window_size=True),
          dict(window_size=0.5, relative_window_size=True),
          dict(window_size=4, relative_window_size=False)]
f, axes = plt.subplots(1, 3, figsize=(n_timestamps_1 * 3, n_timestamps_2))
for ax, params_dict in zip(axes, params):
    plot_sakoe_chiba(n_timestamps_1, n_timestamps_2, ax=ax, **params_dict)
    title = "Window size = {} ".format(params_dict["window_size"])
    title += "\n Relative = {} ".format(params_dict["relative_window_size"])
    ax.set_title(title)

plt.show()
