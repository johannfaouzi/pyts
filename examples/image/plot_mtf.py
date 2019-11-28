"""
=======================
Markov Transition Field
=======================

A Markov Transition Field is an image obtained from a time series, representing
a field of transition probabilities for a discretized time series. Different
strategies can be used to bin time series. his example illustrates the
transformation on the first sample of the *GunPoint* dataset.
It is implemented as :class:`pyts.image.MarkovTransitionField`.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField
from pyts.datasets import load_gunpoint

X, _, _, _ = load_gunpoint(return_X_y=True)

# MTF transformation
mtf = MarkovTransitionField(image_size=24)
X_mtf = mtf.fit_transform(X)

# Show the image for the first time series
plt.figure(figsize=(5, 5))
plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
plt.title('Markov Transition Field', fontsize=18)
plt.colorbar(fraction=0.0457, pad=0.04)
plt.tight_layout()
plt.show()
