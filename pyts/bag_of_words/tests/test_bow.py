"""Testing for Bag-of-Words."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.bag_of_words import BagOfWords


X = [['a', 'a', 'a', 'b', 'a'],
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
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    bow = BagOfWords(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        bow.transform(X)


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
def test_actual_results(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = BagOfWords(**params).fit_transform(X)
    np.testing.assert_array_equal(arr_actual, arr_desired)
