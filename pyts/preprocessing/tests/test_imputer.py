"""Testing for imputers."""

import numpy as np
from itertools import product
from ..imputer import InterpolationImputer


def test_InterpolationImputer():
    """Test 'InterpolationImputer' class."""
    # Parameter check
    X = [[None, 0, 1, 2, None, 4],
         [0, 2, 4, None, 8, np.nan]]

    def value_error_list(missing_values, strategy):
        value_error_list_ = [
            "'missing_values' cannot be infinity.",
            "'missing_values' must be an integer, a float, None or "
            "np.nan (got {0!s})".format(missing_values),
            "'strategy' must be an integer or one of 'linear', 'nearest', "
            "'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next' "
            "(got {0})".format(strategy)
        ]
        return value_error_list_

    missing_values_list = [-3, None, np.inf, np.nan]
    strategy_list = ['linear', 2, None]
    for (missing_values, strategy) in product(
        missing_values_list, strategy_list
    ):
        try:
            imputer = InterpolationImputer(
                missing_values=missing_values, strategy=strategy
            )
        except ValueError as e:
            if str(e) in value_error_list(missing_values, strategy):
                pass
            else:
                raise ValueError("Unexpected ValueError: {0}".format(e))

    # Test 1
    X = [[None, 0, 1, 2, None, 4],
         [0, 2, 4, None, 8, np.nan]]
    imputer = InterpolationImputer(missing_values=None, strategy='linear')
    arr_actual = imputer.fit_transform(X)
    arr_desired = [[-1, 0, 1, 2, 3, 4],
                   [0, 2, 4, 6, 8, 10]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    X = [[None, 0, 1, 2, None, 4],
         [0, 2, 4, None, 8, np.nan]]
    imputer = InterpolationImputer(missing_values=np.nan, strategy='linear')
    arr_actual = imputer.fit_transform(X)
    arr_desired = [[-1, 0, 1, 2, 3, 4],
                   [0, 2, 4, 6, 8, 10]]
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 3
    X = [[-10, 0, 1, 2, -10, 4],
         [0, 2, 4, -10, 8, -10]]
    imputer = InterpolationImputer(missing_values=-10, strategy='linear')
    arr_actual = imputer.fit_transform(X)
    arr_desired = [[-1, 0, 1, 2, 3, 4],
                   [0, 2, 4, 6, 8, 10]]
    np.testing.assert_array_equal(arr_actual, arr_desired)
