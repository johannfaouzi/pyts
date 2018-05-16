"""
==============================================
Word ExtrAction for time SEries cLassification
==============================================

This example shows how the WEASEL algorithm transforms a time series of real
numbers into a sequence of frequencies of words. It is implemented as
:class:`pyts.transformation.WEASEL`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.transformation import WEASEL

# Parameters
n_samples, n_features = 100, 144
n_classes = 2

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_features)
y = rng.randint(n_classes, size=n_samples)

# WEASEL transformation
weasel = WEASEL(n_coefs=2, window_sizes=[12, 24, 36], pvalue_threshold=0.2)
X_weasel = weasel.fit_transform(X, y).toarray()

# Visualize the transformation for the first time series
plt.figure(figsize=(12, 8))
plt.bar(np.arange(X_weasel[0].size), X_weasel[0])
plt.xticks(np.arange(X_weasel[0].size),
           np.vectorize(weasel.vocabulary_.get)(np.arange(X_weasel[0].size)),
           fontsize=12, rotation=60)
plt.xlabel("Words", fontsize=18)
plt.ylabel("Frequencies", fontsize=18)
plt.show()
