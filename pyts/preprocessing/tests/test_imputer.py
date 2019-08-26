"""Testing for imputers."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.preprocessing import InterpolationImputer


X = [[np.nan, 1, 2, 3, np.nan, 5, 6, np.nan]]


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'missing_values': np.inf}, ValueError,
      "'missing_values' cannot be infinity."),

     ({'missing_values': "3"}, ValueError,
      "'missing_values' must be an integer, a float, None or np.nan "
      "(got {0!s})".format("3")),

     ({'strategy': 'whoops'}, ValueError,
      "'strategy' must be an integer or one of 'linear', 'nearest', "
      "'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next' "
      "(got {0})".format('whoops'))]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    imputer = InterpolationImputer(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        imputer.transform(X)


@pytest.mark.parametrize(
    'params, X, arr_desired',
    [({'missing_values': None}, [[None, 10, 8, None, 4, 2, None]],
      [[12, 10, 8, 6, 4, 2, 0]]),

     ({'missing_values': np.nan}, [[np.nan, 10, 8, np.nan, 4, 2, np.nan]],
      [[12, 10, 8, 6, 4, 2, 0]]),

     ({'missing_values': 45.}, [[45., 10, 8, 45., 4, 2, 45.]],
      [[12, 10, 8, 6, 4, 2, 0]]),

     ({'missing_values': 78}, [[78, 10, 8, 78, 4, 2, 78]],
      [[12, 10, 8, 6, 4, 2, 0]]),

     ({'missing_values': None, 'strategy': 'quadratic'},
      [[None, 9, 4, None, 0, 1, None]], [[16, 9, 4, 1, 0, 1, 4]]),

     ({'missing_values': None, 'strategy': 'previous'},
      [[5, 9, 4, None, 0, 1, None]], [[5, 9, 4, 4, 0, 1, 1]]),

     ({'missing_values': None, 'strategy': 'next'},
      [[None, 9, 4, None, 0, 1, 8]], [[9, 9, 4, 0, 0, 1, 8]])]
)
def test_actual_results(params, X, arr_desired):
    """Test that the actual results are the expected ones."""
    imputer = InterpolationImputer(**params)
    arr_actual = imputer.fit_transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, rtol=0, atol=1e-5)
