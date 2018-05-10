"""
======================
Plotting a time series
======================

Plotting a time series.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters
n_samples = 100
n_features = 48
rng = np.random.RandomState(41)
delta = 0.5
dt = 1

# Generate a toy dataset
X = (norm.rvs(scale=delta**2 * dt,
     size=n_samples * n_features,
     random_state=rng).reshape((n_samples, n_features)))
X[:, 0] = 0
X = np.cumsum(X, axis=1)

# Plot the first sample
plt.plot(X[0])
plt.show()
