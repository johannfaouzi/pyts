"""Symbols of any length and type are handled consistently."""

# License: BSD-3-Clause

import re

import numpy as np
import pytest

from pyts.bag_of_words import BagOfWords, WordExtractor

rng = np.random.RandomState(0)
# Each series is flat at the start, so some windows fall below
# ``threshold_std`` while the others do not.
X_flat = np.c_[np.zeros((3, 10)), np.sin(np.linspace(0, 6, 20)) + rng.randn(3, 20)]


@pytest.mark.parametrize('threshold_std', [None, 0.01])
def test_multi_character_symbols(threshold_std):
    """Symbols longer than one character are not truncated."""
    alphabet = ['lo', 'mid', 'hi']
    bow = BagOfWords(
        window_size=4,
        word_size=2,
        n_bins=3,
        alphabet=alphabet,
        threshold_std=threshold_std,
        numerosity_reduction=False,
    )
    pattern = re.compile(r'^(lo|mid|hi){2}$')
    for document in bow.transform(X_flat):
        words = document.split(' ')
        assert len(words) == 27
        assert all(pattern.match(word) for word in words), document


@pytest.mark.parametrize('threshold_std', [None, 0.01])
def test_single_symbol_words(threshold_std):
    """Words of a single symbol are built from the mean of each window."""
    bow = BagOfWords(window_size=4, word_size=1, threshold_std=threshold_std)
    for document in bow.transform(X_flat):
        assert all(len(word) == 1 for word in document.split(' '))


def test_word_extractor_requires_string_symbols():
    """Integer codes are rejected rather than ambiguously concatenated."""
    msg = "The symbols of 'X' must be strings (got an array with dtype"
    with pytest.raises(TypeError, match=re.escape(msg)):
        WordExtractor(window_size=2).transform([[0, 1, 1, 2], [2, 1, 0, 0]])


def test_word_extractor_object_strings():
    """An object array of strings is handled like a string array."""
    X = [['a', 'b', 'b', 'c'], ['c', 'b', 'a', 'a']]
    expected = WordExtractor(window_size=2).transform(X)
    actual = WordExtractor(window_size=2).transform(np.array(X, dtype=object))
    np.testing.assert_array_equal(actual, expected)
