"""
==========================
Preprocessing transformers
==========================

This example shows the different transforming algorithms available in
:mod:`pyts.preprocessing`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.preprocessing import PowerTransformer, QuantileTransformer

# Parameters
n_samples, n_timestamps = 100, 48

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)

# Scale the data with different scaling algorithms
X_power = PowerTransformer().transform(X)
X_quantile = QuantileTransformer().transform(X)

# Show the results for the first time series
plt.figure(figsize=(16, 6))

ax1 = plt.subplot(121)
ax1.plot(X[0], 'o-', label='Original')
ax1.set_title('Original time series')
ax1.legend(loc='best')

ax2 = plt.subplot(122)
ax2.plot(X_power[0], 'o--', color='C1', label='PowerTransformer')
ax2.plot(X_quantile[0], 'o--', color='C2', label='QuantileTransformer')
ax2.set_title('Transformed time series')
ax2.legend(loc='best')

plt.show()
