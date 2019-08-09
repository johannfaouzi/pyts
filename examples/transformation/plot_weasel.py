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
from pyts.datasets import load_gunpoint
from pyts.transformation import WEASEL

# Toy dataset
X_train, _, y_train, _ = load_gunpoint(return_X_y=True)

# WEASEL transformation
weasel = WEASEL(word_size=2, n_bins=2, window_sizes=[12, 36], sparse=False)
X_weasel = weasel.fit_transform(X_train, y_train)

# Visualize the transformation for the first time series
plt.figure(figsize=(8, 5))
vocabulary_length = len(weasel.vocabulary_)
width = 0.3
plt.bar(np.arange(vocabulary_length) - width / 2, X_weasel[y_train == 1][0],
        width=width, label='First time series in class 1')
plt.bar(np.arange(vocabulary_length) + width / 2, X_weasel[y_train == 2][0],
        width=width, label='First time series in class 2')
plt.xticks(np.arange(vocabulary_length),
           np.vectorize(weasel.vocabulary_.get)(np.arange(X_weasel[0].size)),
           fontsize=12, rotation=60)
y_max = np.max(np.concatenate([X_weasel[y_train == 1][0],
                               X_weasel[y_train == 2][0]]))
plt.yticks(np.arange(y_max + 1), fontsize=12)
plt.xlabel("Words", fontsize=16)
plt.ylabel("Frequencies", fontsize=16)
plt.title("WEASEL transformation", fontsize=20)
plt.legend(loc='best', fontsize=12)

plt.subplots_adjust(bottom=0.27)
plt.show()
