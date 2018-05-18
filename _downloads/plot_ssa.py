"""
==========================
Singular Spectrum Analysis
==========================

This example shows how you can decompose a time series into several time series
using :class:`pyts.decomposition.SSA`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.decomposition import SSA

# Parameters
n_samples, n_features = 100, 48

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_features)

# SSA transformation
window_size = 15
grouping = [[0, 1]]
ssa = SSA(window_size, grouping)
X_ssa = ssa.fit_transform(X)

# Show the results for the first time series
plt.figure(figsize=(12, 8))
plt.plot(X[0], 'o-', label='Original')
plt.plot(X_ssa[0, 0], 'o--', label='SSA')
plt.legend(loc='best', fontsize=14)
plt.show()
