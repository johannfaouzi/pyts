"""Testing for Bag-of-Words."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.bag_of_words import BagOfWords, WordExtractor

# ######################### Tests for WordExtractor #########################

X_word = [['a', 'a', 'a', 'b', 'a'],
          ['a', 'a', 'b', 'b', 'a'],
          ['b', 'b', 'b', 'b', 'a']]


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'window_size': '4'}, TypeError,
      "'window_size' must be an integer or a float."),
     ({'window_step': [0, 1]}, TypeError,
      "'window_step' must be an integer or a float."),
     ({'window_size': 0}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps (got 0)."),
     ({'window_size': 2.}, ValueError,
      "If 'window_size' is a float, it must be greater than 0 and lower "
      "than or equal to 1 (got {0}).".format(2.)),
     ({'window_step': 0}, ValueError,
      "If 'window_step' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps (got 0)."),
     ({'window_step': 2.}, ValueError,
      "If 'window_step' is a float, it must be greater than 0 and lower "
      "than or equal to 1 (got {0}).".format(2.))]
)
def test_parameter_check_word_extractor(params, error, err_msg):
    """Test parameter validation."""
    bow = WordExtractor(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        bow.transform(X_word)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({}, ['a b a', 'a b a', 'b a']),
     ({'numerosity_reduction': False},
      ['a a a b a', 'a a b b a', 'b b b b a']),
     ({'window_size': 1, 'numerosity_reduction': False},
      ['a a a b a', 'a a b b a', 'b b b b a']),
     ({'window_size': 2, 'numerosity_reduction': False},
      ['aa aa ab ba', 'aa ab bb ba', 'bb bb bb ba']),
     ({'window_size': 0.4, 'window_step': 0.2, 'numerosity_reduction': False},
      ['aa aa ab ba', 'aa ab bb ba', 'bb bb bb ba']),
     ({'window_size': 2, 'window_step': 1, 'numerosity_reduction': True},
      ['aa ab ba', 'aa ab bb ba', 'bb ba']),
     ({'window_size': 3, 'window_step': 2, 'numerosity_reduction': False},
      ['aaa aba', 'aab bba', 'bbb bba']),
     ({'window_size': 3, 'window_step': 2, 'numerosity_reduction': True},
      ['aaa aba', 'aab bba', 'bbb bba'])]
)
def test_actual_results_word_extractor(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = WordExtractor(**params).fit_transform(X_word)
    np.testing.assert_array_equal(arr_actual, arr_desired)


# ######################### Tests for BagOfWords #########################

X_bow = np.arange(20).reshape(2, 10)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'window_size': '4'}, TypeError,
      "'window_size' must be an integer or a float."),

     ({'window_size': 0}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps (got 0)."),

     ({'window_size': 2.}, ValueError,
      "If 'window_size' is a float, it must be greater than 0 and lower "
      "than or equal to 1 (got {0}).".format(2.)),

     ({'word_size': []}, TypeError,
      "'word_size' must be an integer or a float."),

     ({'word_size': 0}, ValueError,
      "If 'word_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to window_size (got 0)."),

     ({'window_size': 4, 'word_size': 5}, ValueError,
      "If 'word_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to window_size (got 5)."),

     ({'word_size': 2.}, ValueError,
      "If 'word_size' is a float, it must be greater than 0 and lower "
      "than or equal to 1 (got {0}).".format(2.)),

     ({'n_bins': '3'}, TypeError, "'n_bins' must be an integer."),

     ({'n_bins': 1}, ValueError,
      "'n_bins' must be greater than or equal to 2 and lower than "
      "or equal to min(word_size, 26) (got 1)."),

     ({'window_size': 6, 'word_size': 4, 'n_bins': 2, 'strategy': 'whoops'},
      ValueError,
      "'strategy' must be either 'uniform', 'quantile' or 'normal' "
      "(got {0})".format('whoops')),

     ({'window_size': 6, 'word_size': 4, 'n_bins': 2, 'window_step': [0, 1]},
      TypeError,
      "'window_step' must be an integer or a float."),

     ({'window_size': 6, 'word_size': 4, 'n_bins': 2, 'window_step': 0},
      ValueError,
      "If 'window_step' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps (got 0)."),

     ({'window_size': 6, 'word_size': 4, 'n_bins': 2, 'window_step': 2.},
      ValueError,
      "If 'window_step' is a float, it must be greater than 0 and lower "
      "than or equal to 1 (got {0}).".format(2.)),

     ({'window_size': 6, 'word_size': 4, 'n_bins': 2, 'alphabet': 'whoops'},
      TypeError,
      "'alphabet' must be None or array-like with shape (n_bins,) "
      "(got {0}).".format('whoops')),

     ({'window_size': 6, 'word_size': 4, 'n_bins': 2,
       'alphabet': ['a', 'b', 'c']}, ValueError,
      "If 'alphabet' is array-like, its shape must be equal to (n_bins,).")]
)
def test_parameter_check_bag_of_words(params, error, err_msg):
    """Test parameter validation."""
    bow = BagOfWords(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        bow.transform(X_bow)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'window_size': 6, 'word_size': 4}, ['abcd', 'abcd']),

     ({'window_size': 6, 'word_size': 4, 'numerosity_reduction': False},
      ['abcd abcd abcd abcd abcd', 'abcd abcd abcd abcd abcd']),

     ({'window_size': 6, 'word_size': 4, 'alphabet': ['y', 'o', 'l', 'o']},
      ['yolo', 'yolo']),

     ({'window_size': 0.5, 'word_size': 4}, ['abcd', 'abcd']),

     ({'window_size': 4, 'word_size': 1., 'numerosity_reduction': False},
      ['abcd abcd abcd abcd abcd abcd abcd',
       'abcd abcd abcd abcd abcd abcd abcd']),

     ({'window_size': 4, 'word_size': 4, 'window_step': 2,
       'numerosity_reduction': False},
      ['abcd abcd abcd abcd', 'abcd abcd abcd abcd']),

     ({'window_size': 4, 'word_size': 4, 'window_step': 0.2,
       'numerosity_reduction': False},
      ['abcd abcd abcd abcd', 'abcd abcd abcd abcd']),

     ({'window_size': 4, 'word_size': 4, 'window_step': 0.5},
      ['abcd', 'abcd'])]
)
def test_actual_results_bag_of_words(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = BagOfWords(**params).fit_transform(X_bow)
    np.testing.assert_array_equal(arr_actual, arr_desired)
