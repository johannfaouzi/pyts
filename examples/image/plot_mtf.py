"""
=======================
Markov Transition Field
=======================

This example shows how you can transform a time series into a Markov
Transition Field using :class:`pyts.image.MarkovTransitionField`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField

# Parameters
n_samples, n_timestamps = 100, 144

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)

# MTF transformation
mtf = MarkovTransitionField(image_size=24)
X_mtf = mtf.fit_transform(X)

# Show the image for the first time series
plt.figure(figsize=(6, 6))
plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
plt.title('Markov Transition Field', fontsize=14)
plt.colorbar(fraction=0.0457, pad=0.04)
plt.show()
