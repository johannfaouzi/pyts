"""
=======================
Markov Transition Field
=======================

This example shows how you can transform a time series into a Markov Transition
Field using :class:`pyts.image.MTF`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.image import MTF

# Parameters
n_samples, n_features = 100, 144

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_features)

# MTF transformations
image_size = 24
mtf = MTF(image_size)
X_mtf = mtf.fit_transform(X)

# Show the results for the first time series
plt.figure(figsize=(8, 8))
plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
plt.show()
