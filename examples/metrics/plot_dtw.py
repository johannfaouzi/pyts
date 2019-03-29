"""
====================
Dynamic Time Warping
====================

This example shows how to compute and visualize the optimal path
when computing Dynamic Time Warping (DTW) between two time series and
compare the results with different variants of DTW. It is implemented
as :func:`pyts.metrics.dtw`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.metrics import dtw, itakura_parallelogram, sakoe_chiba_band
from pyts.metrics.dtw import (cost_matrix, accumulated_cost_matrix,
                              _return_path, _multiscale_region)

# Parameters
n_samples, n_timestamps = 2, 48

# Toy dataset
rng = np.random.RandomState(41)
x, y = rng.randn(n_samples, n_timestamps)

plt.figure(figsize=(13, 14))
timestamps = np.arange(n_timestamps + 1)

# Dynamic Time Warping: classic
dtw_classic, path_classic = dtw(x, y, dist='square',
                                method='classic', return_path=True)
matrix_classic = np.zeros((n_timestamps + 1, n_timestamps + 1))
matrix_classic[tuple(path_classic)[::-1]] = 1.

plt.subplot(2, 2, 1)
plt.pcolor(timestamps, timestamps, matrix_classic,
           edgecolors='k', cmap='Greys')
plt.xlabel('x', fontsize=20, labelpad=-10)
plt.ylabel('y', fontsize=20, labelpad=-10)
plt.title("{0}\nDTW(x, y) = {1:.2f}".format('classic', dtw_classic),
          fontsize=16)

# Dynamic Time Warping: sakoechiba
dtw_sakoechiba, path_sakoechiba = dtw(
    x, y, dist='square', method='sakoechiba',
    options={'window_size': 5}, return_path=True
)
band = sakoe_chiba_band(n_timestamps, window_size=5)
matrix_sakoechiba = np.zeros((n_timestamps + 1, n_timestamps + 1))
for i in range(n_timestamps):
    matrix_sakoechiba[i, np.arange(*band[:, i])] = 0.5
matrix_sakoechiba[tuple(path_sakoechiba)[::-1]] = 1.

plt.subplot(2, 2, 2)
plt.pcolor(timestamps, timestamps, matrix_sakoechiba,
           edgecolors='k', cmap='Greys')
plt.xlabel('x', fontsize=20, labelpad=-10)
plt.ylabel('y', fontsize=20, labelpad=-10)
plt.title("{0}\nDTW(x, y) = {1:.2f}".format('sakoechiba', dtw_sakoechiba),
          fontsize=16)

# Dynamic Time Warping: itakura
dtw_itakura, path_itakura = dtw(
    x, y, dist='square', method='itakura',
    options={'max_slope': 2.}, return_path=True
)
parallelogram = itakura_parallelogram(n_timestamps, max_slope=2.)
matrix_itakura = np.zeros((n_timestamps + 1, n_timestamps + 1))
for i in range(n_timestamps):
    matrix_itakura[i, np.arange(*parallelogram[:, i])] = 0.5
matrix_itakura[tuple(path_itakura)[::-1]] = 1.

plt.subplot(2, 2, 3)
plt.pcolor(timestamps, timestamps, matrix_itakura,
           edgecolors='k', cmap='Greys')
plt.xlabel('x', fontsize=20, labelpad=-10)
plt.ylabel('y', fontsize=20, labelpad=-10)
plt.title("{0}\nDTW(x, y) = {1:.2f}".format('itakura', dtw_itakura),
          fontsize=16)

# Dynamic Time Warping: multiscale
dtw_multiscale, path_multiscale = dtw(
    x, y, dist='square', method='multiscale',
    options={'resolution': 3, 'radius': 1}, return_path=True
)
x_padded = x.reshape(-1, 3).mean(axis=1)
y_padded = y.reshape(-1, 3).mean(axis=1)
cost_mat_res = cost_matrix(x_padded, y_padded, dist='square', region=None)
acc_cost_mat_res = accumulated_cost_matrix(cost_mat_res)
path_res = _return_path(acc_cost_mat_res)
multiscale_region = _multiscale_region(
    n_timestamps, 3, x_padded.size, path_res, radius=1
)
matrix_multiscale = np.zeros((n_timestamps + 1, n_timestamps + 1))
for i in range(n_timestamps):
    matrix_multiscale[i, np.arange(*multiscale_region[:, i])] = 0.5
matrix_multiscale[tuple(path_multiscale)[::-1]] = 1.

plt.subplot(2, 2, 4)
plt.pcolor(timestamps, timestamps, matrix_multiscale,
           edgecolors='k', cmap='Greys')
plt.xlabel('x', fontsize=20, labelpad=-10)
plt.ylabel('y', fontsize=20, labelpad=-10)
plt.title("{0}\nDTW(x, y) = {1:.2f}".format('multiscale', dtw_multiscale),
          fontsize=16)

plt.suptitle("Dynamic Time Warping", fontsize=22, y=0.95)
plt.show()
