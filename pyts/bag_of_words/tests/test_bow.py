"""Testing for Bag-of-Words."""

import numpy as np
from itertools import product
from ..bow import BagOfWwords


def test_BagOfWwords():
    """Test 'BagOfWwords' class."""
    X = [['a', 'a', 'a', 'b', 'a'],
         ['a', 'a', 'b', 'b', 'a'],
         ['b', 'b', 'b', 'b', 'a']]

    # Parameter check
    def type_error_list():
        type_error_list_ = ["'window_size' must be an integer or a float.",
                            "'window_step' must be an integer or a float."]
        return type_error_list_

    def value_error_list(window_size, window_step):
        value_error_list_ = [
            "If 'window_size' is an integer, it must be greater "
            "than or equal to 1 and lower than or equal to the "
            "size of each time series (i.e. the size of the last "
            "dimension of X) (got {0}).".format(window_size),
            "If 'window_size' is a float, it must be greater than 0 and lower "
            "than or equal to 1 (got {0}).".format(window_size),
            "If 'window_step' is an integer, it must be greater "
            "than or equal to 1 and lower than or equal to the "
            "size of each time series (i.e. the size of the last "
            "dimension of X) (got {0}).".format(window_step),
            "If 'window_step' is a float, it must be greater than 0 and lower "
            "than or equal to 1 (got {0}).".format(window_step)
        ]
        return value_error_list_

    window_size_list = [1., 2., -1, 2, 3, None]
    window_step_list = [-1, 0, 2, None]
    numerosity_reduction_list = [True, False]

    for (window_size, window_step, numerosity_reduction) in product(
        window_size_list, window_step_list, numerosity_reduction_list
    ):
        bow = BagOfWwords(window_size, window_step, numerosity_reduction)
        try:
            bow.fit_transform(X)
        except ValueError as e:
            if str(e) in value_error_list(window_size, window_step):
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) in type_error_list():
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))

    # Test 1
    bow = BagOfWwords(window_size=2, window_step=1, numerosity_reduction=False)
    arr_actual = bow.fit_transform(X)
    arr_desired = ["aa aa ab ba",
                   "aa ab bb ba",
                   "bb bb bb ba"]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    bow = BagOfWwords(window_size=2, window_step=1, numerosity_reduction=True)
    arr_actual = bow.fit_transform(X)
    arr_desired = ["aa ab ba",
                   "aa ab bb ba",
                   "bb ba"]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 3
    bow = BagOfWwords(window_size=3, window_step=2, numerosity_reduction=False)
    arr_actual = bow.fit_transform(X)
    arr_desired = ["aaa aba",
                   "aab bba",
                   "bbb bba"]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 4
    bow = BagOfWwords(window_size=3, window_step=2, numerosity_reduction=True)
    arr_actual = bow.fit_transform(X)
    arr_desired = ["aaa aba",
                   "aab bba",
                   "bbb bba"]
    np.testing.assert_array_equal(arr_actual, arr_desired)
