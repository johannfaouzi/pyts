"""
===========================================
Bag-of-SFA Symbols using Spatial Pyramids (BOSS-SP)
===========================================

This example shows how the BOSS-SP algorithm transforms a dataset
consisting of time series into a histogram of words, applying the
BOSS algorithm on the original data as well as sub-series of the
data and combining them.
It is implemented as :class:`pyts.classification.BOSSSP`.
"""

# Author: Sven Barray
# License: BSD-3-Clause

import matplotlib.pyplot as plt
from pyts.classification import BOSSSP
from pyts.datasets import load_gunpoint

# Toy dataset
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

# BOSSSP transformation
bosssp = BOSSSP(word_size=4, n_bins=3, window_size=10)
bosssp.fit(X_train, y_train)
ts_index_to_plot = 0

# Visualize the transformation
plt.figure(figsize=(14, 5))
width = 0.4
plt.bar(bosssp._word_count[ts_index_to_plot].keys(),
        bosssp._word_count[ts_index_to_plot].values(), width)
plt.xlabel("Words", fontsize=14)
plt.ylabel("Occurences", fontsize=14)
plt.title("Number of occurence of each word after BOSS-SP transformation",
          fontsize=15)
plt.xticks(rotation=90)
plt.show()
