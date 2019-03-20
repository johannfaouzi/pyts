"""
=======
Imputer
=======

This example shows how one can impute missing values using
:class:`pyts.preprocessing.InterpolationImputer`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.preprocessing import InterpolationImputer

# Parameters
n_samples, n_timestamps = 100, 48

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)
missing_idx = rng.choice(np.arange(1, 47), size=14, replace=False)
X[:, missing_idx] = np.nan

# Show the results for different strategies for the first time series
plt.figure(figsize=(16, 10))
for i, strategy in enumerate(['linear', 'quadratic', 'cubic', 'nearest']):
    imputer = InterpolationImputer(strategy=strategy)
    X_imputed = imputer.transform(X)

    plt.subplot(2, 2, i + 1)
    plt.plot(X_imputed[0], 'o--', color='C1', label='Imputed')
    plt.plot(X[0], 'o--', color='C0', label='Original')
    plt.title("{0} Interpolation".format(strategy.capitalize()))
    plt.legend(loc='best')

plt.show()
