"""
======================
Plotting a time series
======================

This example shows how you can plot a single time series.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_samples, n_timestamps = 100, 48

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)

# Plot the first time series
plt.plot(X[0])
plt.show()
