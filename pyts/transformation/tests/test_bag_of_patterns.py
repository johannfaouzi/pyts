"""Testing for Bag-of-Patterns."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
from pyts.transformation import BagOfPatterns


X = [[0, 2, 1, 3, 4, 2, 1, 0, 3, 1, 2, 0],
     [2, 0, 1, 3, 2, 4, 1, 2, 0, 1, 3, 2]]


@pytest.mark.parametrize(
    'params, vocab_desired, arr_desired',
    [({'window_size': 4, 'word_size': 4, 'sparse': False},
      {0: 'abdc', 1: 'acbd', 2: 'acdb', 3: 'adbc', 4: 'bacd', 5: 'badb',
       6: 'bdab', 7: 'cabd', 8: 'cbad', 9: 'cbda', 10: 'cdba', 11: 'dbca',
       12: 'dcba'},
      [[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
       [2, 1, 0, 0, 0, 0, 2, 2, 0, 1, 0, 1, 0]]),

     ({'window_size': 4, 'n_bins': 2, 'word_size': 4},
      {0: 'aaba', 1: 'aabb', 2: 'abaa', 3: 'abab', 4: 'abba',
       5: 'baab', 6: 'baba', 7: 'bbaa'},
      [[1, 1, 0, 2, 1, 1, 1, 1], [0, 2, 2, 1, 0, 2, 2, 0]]),

     ({'window_size': 6, 'n_bins': 2, 'word_size': 2},
      {0: 'ab', 1: 'ba'}, [[2, 2], [2, 1]]),

     ({'window_size': 6, 'n_bins': 2, 'word_size': 2,
       'numerosity_reduction': False},
      {0: 'ab', 1: 'ba'}, [[3, 4], [4, 3]])]
)
def test_actual_results(params, vocab_desired, arr_desired):
    """Test that the actual results are the expected ones."""
    bop = BagOfPatterns(**params)
    arr_actual = bop.fit_transform(X)
    assert bop.vocabulary_ == vocab_desired
    if isinstance(arr_actual, np.ndarray):
        np.testing.assert_array_equal(arr_actual, arr_desired)
    else:
        np.testing.assert_array_equal(arr_actual.A, arr_desired)


@pytest.mark.parametrize(
    'params',
    [{'window_size': 4, 'word_size': 4},
     {'window_size': 4, 'word_size': 4, 'sparse': False},
     {'window_size': 4, 'word_size': 4, 'n_bins': 2},
     {'window_size': 4, 'word_size': 4, 'numerosity_reduction': False},
     {'window_size': 4, 'word_size': 4, 'norm_mean': 1, 'norm_std': 1},
     {'window_size': 4, 'word_size': 4, 'overlapping': False},
     {'window_size': 4, 'word_size': 4, 'strategy': 'normal'},
     {'window_size': 4, 'word_size': 4, 'window_step': 2},
     {'window_size': 4, 'word_size': 4, 'alphabet': ['d', 'c', 'b', 'a']}]
)
def test_fit_transform(params):
    """Test that fit_transform and fit then transform yield same results."""
    bop_1, bop_2 = BagOfPatterns(**params), BagOfPatterns(**params)
    arr_1 = bop_1.fit_transform(X)
    arr_2 = bop_2.fit(X).transform(X)
    assert bop_1.vocabulary_ == bop_2.vocabulary_
    if isinstance(arr_1, np.ndarray):
        np.testing.assert_array_equal(arr_1, arr_2)
    else:
        np.testing.assert_array_equal(arr_1.A, arr_2.A)
