"""
============
Bag of Words
============

This example shows how you can transform a discretized time series
(i.e. a time series represented as a sequence of letters) into a bag
of words using :class:`pyts.bag_of_words.BagOfWords`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.bag_of_words import BagOfWords

# Parameters
n_samples, n_timestamps = 100, 48
n_bins = 4

# Toy dataset
rng = np.random.RandomState(42)
alphabet = np.array(['a', 'b', 'c', 'd'])
X_ordinal = rng.randint(n_bins, size=(n_samples, n_timestamps))
X_alphabet = alphabet[X_ordinal]

# Bag-of-words transformation
bow = BagOfWords(window_size=2, numerosity_reduction=False)
X_bow = bow.transform(X_alphabet)
words = np.asarray(X_bow[0].split(' '))
different_words_idx = np.r_[True, words[1:] != words[:-1]]

# Show the results
plt.figure(figsize=(16, 4))
plt.suptitle('Transforming a discretized time series into a bag of words',
             fontsize=18, y=1.05)

plt.subplot(121)
plt.plot(X_ordinal[0], 'o')
plt.yticks(np.arange(4), alphabet)
plt.xticks([], [])
plt.title('Without numerosity reduction', fontsize=14)
for i, word in enumerate(words):
    plt.text(i, - 0.4 - (i % 5) / 4, word, fontsize=17, color='C0')

plt.subplot(122)
plt.plot(X_ordinal[0], 'o')
plt.yticks(np.arange(4), alphabet)
plt.xticks([], [])
plt.title('With numerosity reduction', fontsize=14)
for i, (word, different_word) in enumerate(zip(words, different_words_idx)):
    if different_word:
        plt.text(i, - 0.4 - (i % 5) / 4, word, fontsize=17, color='C0')
    else:
        plt.text(i, - 0.4 - (i % 5) / 4, word, fontsize=17, color='C0',
                 alpha=0.2)

plt.show()
