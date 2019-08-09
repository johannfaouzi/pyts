"""
=========================
Bag-of-SFA Symbols (BOSS)
=========================

This example shows how the BOSS algorithm transforms a time series of real
numbers into a sequence of frequencies of words. It is implemented as
:class:`pyts.transformation.BOSS`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.datasets import load_gunpoint
from pyts.transformation import BOSS

# Toy dataset
X_train, _, y_train, _ = load_gunpoint(return_X_y=True)

# BOSS transformation
boss = BOSS(word_size=2, n_bins=4, window_size=12, sparse=False)
X_boss = boss.fit_transform(X_train)

# Visualize the transformation for the first time series
plt.figure(figsize=(8, 5))
vocabulary_length = len(boss.vocabulary_)
width = 0.3
plt.bar(np.arange(vocabulary_length) - width / 2, X_boss[y_train == 1][0],
        width=width, label='First time series in class 1')
plt.bar(np.arange(vocabulary_length) + width / 2, X_boss[y_train == 2][0],
        width=width, label='First time series in class 2')
plt.xticks(np.arange(vocabulary_length),
           np.vectorize(boss.vocabulary_.get)(np.arange(X_boss[0].size)),
           fontsize=12)
y_max = np.max(np.concatenate([X_boss[y_train == 1][0],
                               X_boss[y_train == 2][0]]))
plt.yticks(np.arange(y_max + 1), fontsize=12)
plt.xlabel("Words", fontsize=16)
plt.ylabel("Frequencies", fontsize=16)
plt.title("BOSS transformation", fontsize=20)
plt.legend(loc='best', fontsize=12)
plt.show()
