"""
=================================
Piecewise Aggregate Approximation
=================================

Time series with a high sampling rate can be very noisy. In order to reduce
noise, a technique called *Piecewise Aggregate Approximation* was invented,
consisting in taking the mean over back-to-back points. This decreases the
number of points and reduces noise while preserving the trend of the time
series. This example shows how to approximate a time series using
:class:`pyts.approximation.PiecewiseAggregateApproximation` and illustrates
the transformation.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation

# Parameters
n_samples, n_timestamps = 100, 48

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)

# PAA transformation
window_size = 6
paa = PiecewiseAggregateApproximation(window_size=window_size)
X_paa = paa.transform(X)

# Show the results for the first time series
plt.figure(figsize=(12, 8))
plt.plot(X[0], 'o--', label='Original')
plt.plot(np.arange(window_size // 2,
                   n_timestamps + window_size // 2,
                   window_size), X_paa[0], 'o--', label='PAA')
plt.vlines(np.arange(0, n_timestamps, window_size) - 0.5,
           X[0].min(), X[0].max(), color='g', linestyles='--', linewidth=0.5)
plt.legend(loc='best', fontsize=14)
plt.title('Piecewise Aggregate Approximation', fontsize=16)
plt.show()
