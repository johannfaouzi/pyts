"""
============
Transformers
============

This example shows the different transforming algorithms available in
:mod:`pyts.preprocessing`.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
from pyts.datasets import load_gunpoint
from pyts.preprocessing import PowerTransformer, QuantileTransformer

X, _, _, _ = load_gunpoint(return_X_y=True)
n_timestamps = X.shape[1]

# Scale the data with different scaling algorithms
X_power = PowerTransformer().transform(X)
X_quantile = QuantileTransformer(n_quantiles=n_timestamps).transform(X)

# Show the results for the first time series
plt.figure(figsize=(8, 6))
plt.plot(X[0], '--', label='Original')
plt.plot(X_power[0], '--', label='PowerTransformer')
plt.plot(X_quantile[0], '--', label='QuantileTransformer')
plt.legend(loc='best', fontsize=12)
plt.title('Transforming time series', fontsize=16)
plt.tight_layout()
plt.show()
