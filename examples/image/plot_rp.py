"""
===============
Recurrence Plot
===============

This example shows how you can transform a time series into a Recurrence
Plot using :class:`pyts.image.RecurrencePlot`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot

# Parameters
n_samples, n_timestamps = 100, 144

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)

# Recurrence plot transformation
rp = RecurrencePlot(dimension=7, time_delay=3,
                    threshold='percentage_points',
                    percentage=30)
X_rp = rp.fit_transform(X)

# Show the results for the first time series
plt.figure(figsize=(6, 6))
plt.imshow(X_rp[0], cmap='binary', origin='lower')
plt.title('Recurrence Plot', fontsize=14)
plt.show()
