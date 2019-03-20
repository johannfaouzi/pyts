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
n_samples, n_timestamps = 100, 144

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)

# BOSS transformation
boss = BOSS(word_size=2, n_bins=4, window_size=12)
X_boss = boss.fit_transform(X)

# Visualize the transformation for the first time series
plt.figure(figsize=(12, 8))
vocabulary_length = len(boss.vocabulary_)
width = 0.3
plt.bar(np.arange(vocabulary_length) - width / 2, X_boss[0],
        width=width, label='First time series')
plt.bar(np.arange(vocabulary_length) + width / 2, X_boss[1],
        width=width, label='Second time series')
plt.xticks(np.arange(vocabulary_length),
           np.vectorize(boss.vocabulary_.get)(np.arange(X_boss[0].size)),
           fontsize=12)
plt.yticks(np.arange(np.max(X_boss[:2] + 1)), fontsize=12)
plt.xlabel("Words", fontsize=18)
plt.ylabel("Frequencies", fontsize=18)
plt.title("BOSS transformation", fontsize=20)
plt.legend(loc='best')
plt.show()
