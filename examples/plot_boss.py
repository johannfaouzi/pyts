"""
==================
Bag-of-SFA Symbols
==================

This example shows how the BOSS algorithm transforms a time series of real
numbers into a sequence of frequencies of words. It is implemented as
:class:`pyts.transformation.BOSS`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.transformation import BOSS

# Parameters
n_samples, n_features = 100, 144

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_features)

# BOSS transformation
boss = BOSS(n_coefs=2, window_size=12)
X_boss = boss.fit_transform(X).toarray()

# Visualize the transformation for the first time series
plt.figure(figsize=(12, 8))
plt.bar(np.arange(X_boss[0].size), X_boss[0])
plt.xticks(np.arange(X_boss[0].size),
           np.vectorize(boss.vocabulary_.get)(np.arange(X_boss[0].size)),
           fontsize=14)
plt.xlabel("Words", fontsize=18)
plt.ylabel("Frequencies", fontsize=18)
plt.show()
