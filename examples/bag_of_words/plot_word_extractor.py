"""
==============
Word Extractor
==============

Time series are often transformed into sequences of symbols. Bag-of-words
approaches are then used to extract features from these sequences.
Identical back-to-back words can be discarded or kept using the
``numerosity_reduction`` parameter.

This example illustrates the transformation and the impact of this parameter.
It is implemented as :class:`pyts.bag_of_words.WordExtractor`.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
from pyts.bag_of_words import WordExtractor

# Parameters
n_samples, n_timestamps = 100, 48
n_bins = 4

# Toy dataset
rng = np.random.RandomState(42)
alphabet = np.array(['a', 'b', 'c', 'd'])
X_ordinal = rng.randint(n_bins, size=(n_samples, n_timestamps))
X_alphabet = alphabet[X_ordinal]

# Bag-of-words transformation
word = WordExtractor(window_size=2, numerosity_reduction=False)
X_bow = word.transform(X_alphabet)
words = np.asarray(X_bow[0].split(' '))
different_words_idx = np.r_[True, words[1:] != words[:-1]]

# Show the results
plt.figure(figsize=(16, 7))
plt.suptitle('Extracting words from a discretized time series',
             fontsize=20, y=0.9)

plt.subplot(121)
plt.plot(X_ordinal[0], 'o', scalex=0.2)
plt.yticks(np.arange(4), alphabet)
plt.xticks([], [])
plt.yticks(fontsize=16)
plt.title('Without numerosity reduction', fontsize=16)

for i, word in enumerate(words):
    plt.text(i, - 0.4 - (i % 5) / 4, word, fontsize=17, color='C0')

plt.subplot(122)
plt.plot(X_ordinal[0], 'o')
plt.yticks(np.arange(4), alphabet)
plt.xticks([], [])
plt.yticks(fontsize=16)
plt.title('With numerosity reduction', fontsize=16)

for i, (word, different_word) in enumerate(zip(words, different_words_idx)):
    if different_word:
        plt.text(i, - 0.4 - (i % 5) / 4, word, fontsize=17, color='C0')
    else:
        plt.text(i, - 0.4 - (i % 5) / 4, word, fontsize=17, color='C0',
                 alpha=0.2)

plt.tight_layout()
plt.subplots_adjust(bottom=0.3, top=0.8)
plt.show()
