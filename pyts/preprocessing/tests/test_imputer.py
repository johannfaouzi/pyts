"""Testing for imputers."""

import numpy as np
import pytest
import re
from ..imputer import InterpolationImputer


def test_InterpolationImputer():
    """Test 'InterpolationImputer' class."""
    # Parameter check
    X = [[None, 0, 1, 2, None, 4],
         [0, 2, 4, None, 8, np.nan]]

    msg_error = "'missing_values' cannot be infinity."
    with pytest.raises(ValueError, match=msg_error):
        imputer = InterpolationImputer(
            missing_values=np.inf, strategy='linear'
        )
        imputer.fit_transform(X)

    msg_error = re.escape(
        "'missing_values' must be an integer, a float, None or "
        "np.nan (got {0!s})".format("3")
    )
    with pytest.raises(ValueError, match=msg_error):
        imputer = InterpolationImputer(
            missing_values="3", strategy='linear'
        )
        imputer.fit_transform(X)

    msg_error = re.escape(
        "'strategy' must be an integer or one of 'linear', 'nearest', "
        "'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next' "
        "(got {0})".format('whoops')
    )
    with pytest.raises(ValueError, match=msg_error):
        imputer = InterpolationImputer(
            missing_values=np.nan, strategy='whoops'
        )
        imputer.fit_transform(X)

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
