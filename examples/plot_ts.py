"""
======================
Plotting a time series
======================

This example shows how you can plot a single time series.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_samples, n_timestamps = 100, 48

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)

# Plot the first time series
plt.plot(X[0], 'o-')
plt.tight_layout()
plt.show()
