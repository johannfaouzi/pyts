"""Testing for Bag-of-Words."""

import numpy as np
import pytest
import re
from ..bow import BagOfWords


def test_BagOfWords():
    """Test 'BagOfWords' class."""
    X = [['a', 'a', 'a', 'b', 'a'],
         ['a', 'a', 'b', 'b', 'a'],
         ['b', 'b', 'b', 'b', 'a']]

    # Parameter check
    msg_error = ("'window_size' must be an integer or a float.")
    with pytest.raises(TypeError, match=msg_error):
        bow = BagOfWords(window_size="1", window_step=1,
                         numerosity_reduction=True)
        bow.fit_transform(X)

    msg_error = ("'window_step' must be an integer or a float.")
    with pytest.raises(TypeError, match=msg_error):
        bow = BagOfWords(window_size=2, window_step="1",
                         numerosity_reduction=True)
        bow.fit_transform(X)

    msg_error = re.escape(
        "If 'window_size' is an integer, it must be greater "
        "than or equal to 1 and lower than or equal to the "
        "size of each time series (i.e. the size of the last "
        "dimension of X) (got {0}).".format(0)
    )
    with pytest.raises(ValueError, match=msg_error):
        bow = BagOfWords(window_size=0, window_step=1,
                         numerosity_reduction=True)
        bow.fit_transform(X)

    msg_error = re.escape(
        "If 'window_size' is a float, it must be greater than 0 and lower "
        "than or equal to 1 (got {0}).".format(2.)
    )
    with pytest.raises(ValueError, match=msg_error):
        bow = BagOfWords(window_size=2., window_step=1,
                         numerosity_reduction=True)
        bow.fit_transform(X)

    msg_error = re.escape(
        "If 'window_step' is an integer, it must be greater "
        "than or equal to 1 and lower than or equal to the "
        "size of each time series (i.e. the size of the last "
        "dimension of X) (got {0}).".format(0)
    )
    with pytest.raises(ValueError, match=msg_error):
        bow = BagOfWords(window_size=2, window_step=0,
                         numerosity_reduction=True)
        bow.fit_transform(X)

    msg_error = re.escape(
        "If 'window_step' is a float, it must be greater than 0 and lower "
        "than or equal to 1 (got {0}).".format(2.)
    )
    with pytest.raises(ValueError, match=msg_error):
        bow = BagOfWords(window_size=2, window_step=2.,
                         numerosity_reduction=True)
        bow.fit_transform(X)

    # Test 1
    bow = BagOfWords(window_size=2, window_step=1, numerosity_reduction=False)
    arr_actual = bow.fit_transform(X)
    arr_desired = ["aa aa ab ba",
                   "aa ab bb ba",
                   "bb bb bb ba"]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    bow = BagOfWords(window_size=0.4, window_step=0.2,
                     numerosity_reduction=False)
    arr_actual = bow.fit_transform(X)
    arr_desired = ["aa aa ab ba",
                   "aa ab bb ba",
                   "bb bb bb ba"]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 3
    bow = BagOfWords(window_size=2, window_step=1, numerosity_reduction=True)
    arr_actual = bow.fit_transform(X)
    arr_desired = ["aa ab ba",
                   "aa ab bb ba",
                   "bb ba"]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 4
    bow = BagOfWords(window_size=3, window_step=2, numerosity_reduction=False)
    arr_actual = bow.fit_transform(X)
    arr_desired = ["aaa aba",
                   "aab bba",
                   "bbb bba"]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 5
    bow = BagOfWords(window_size=3, window_step=2, numerosity_reduction=True)
    arr_actual = bow.fit_transform(X)
    arr_desired = ["aaa aba",
                   "aab bba",
                   "bbb bba"]
    np.testing.assert_array_equal(arr_actual, arr_desired)
