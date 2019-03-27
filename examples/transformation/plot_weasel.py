"""
=======================================================
Word ExtrAction for time SEries cLassification (WEASEL)
=======================================================

This example shows how the WEASEL algorithm transforms a time series of
real numbers into a sequence of frequencies of words. It is implemented
as :class:`pyts.transformation.WEASEL`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.transformation import WEASEL

# Parameters
n_samples, n_timestamps = 100, 300
n_classes = 2

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)
y = rng.randint(n_classes, size=n_samples)

# WEASEL transformation
weasel = WEASEL(word_size=2, n_bins=2, window_sizes=[12, 36])
X_weasel = weasel.fit_transform(X, y).toarray()

# Visualize the transformation for the first time series
plt.figure(figsize=(12, 8))
vocabulary_length = len(weasel.vocabulary_)
width = 0.3
plt.bar(np.arange(vocabulary_length) - width / 2, X_weasel[0],
        width=width, label='First time series')
plt.bar(np.arange(vocabulary_length) + width / 2, X_weasel[1],
        width=width, label='Second time series')
plt.xticks(np.arange(vocabulary_length),
           np.vectorize(weasel.vocabulary_.get)(np.arange(X_weasel[0].size)),
           fontsize=12, rotation=60)
plt.yticks(np.arange(np.max(X_weasel[:2] + 1)), fontsize=12)
plt.xlabel("Words", fontsize=18)
plt.ylabel("Frequencies", fontsize=18)
plt.title("WEASEL transformation", fontsize=20)
plt.legend(loc='best')
plt.show()
