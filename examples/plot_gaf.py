"""
=====================
Gramian Angular Field
=====================

This example shows how you can transform a time series into a Gramian Angular
Field using :class:`pyts.image.GASF` for Gramian Angular Summation Field and
:class:`pyts.image.GADF` for Gramian Angular Difference Field.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GASF, GADF

# Parameters
n_samples, n_features = 100, 144

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_features)

# GAF transformations
image_size = 24
gasf = GASF(image_size)
X_gasf = gasf.fit_transform(X)
gadf = GADF(image_size)
X_gadf = gadf.fit_transform(X)

# Show the results for the first time series
plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.imshow(X_gasf[0], cmap='rainbow', origin='lower')
plt.title("GASF", fontsize=16)
plt.subplot(122)
plt.imshow(X_gadf[0], cmap='rainbow', origin='lower')
plt.title("GADF", fontsize=16)
plt.show()
