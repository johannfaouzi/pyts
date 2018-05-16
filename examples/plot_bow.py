"""
===
BOW
===

Illustration of Bag of Words.
"""

import numpy as np
from pyts.bow import BOW

# Parameters
n_samples = 100
n_features = 30
n_bins = 4
window_size = 1
alphabet = np.array([chr(i) for i in range(97, 97 + n_bins)])

# Toy dataset
rng = np.random.RandomState(41)
X = alphabet[rng.randint(n_bins, size=(n_samples, n_features))]

# Bag-of-words transformation
bow = BOW(window_size, numerosity_reduction=False)
X_bow = bow.fit_transform(X)
bow_num = BOW(window_size, numerosity_reduction=True)
X_bow_num = bow_num.fit_transform(X)

print("Original time series:")
print(X[0])
print("\n")
print("Bag of words without numerosity reduction:")
print("{", X_bow[0].replace(" ", ", "), "}")
print("\n")
print("Bag of words with numerosity reduction:")
print("{", X_bow_num[0].replace(" ", ", "), "}")
