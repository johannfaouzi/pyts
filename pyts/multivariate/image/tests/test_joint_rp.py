"""Testing for JointRecurrencePlot."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.image import RecurrencePlot
from pyts.multivariate.image import JointRecurrencePlot


n_samples, n_features, n_timestamps = 40, 3, 30
rng = np.random.RandomState(42)
X = rng.randn(n_samples, n_features, n_timestamps)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'threshold': [None, None]}, ValueError,
      "If 'threshold' is a list, its length must be equal to n_features "
      "(2 != 3)."),

     ({'percentage': [0, 2, 4, 5]}, ValueError,
      "If 'percentage' is a list, its length must be equal to n_features "
      "(4 != 3).")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    transformer = JointRecurrencePlot(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        transformer.transform(X)


@pytest.mark.parametrize(
    'params, shape_desired',
    [({}, (40, 30, 30)),
     ({'dimension': 3}, (40, 28, 28)),
     ({'dimension': 9}, (40, 22, 22)),
     ({'time_delay': 3}, (40, 30, 30)),
     ({'dimension': 3, 'time_delay': 4}, (40, 22, 22)),
     ({'dimension': 6, 'time_delay': 3}, (40, 15, 15))]
)
def test_shapes(params, shape_desired):
    """Test that the shape of the output is the expected one."""
    transformer = JointRecurrencePlot(**params)
    assert transformer.fit(X).transform(X).shape == shape_desired
    assert transformer.fit_transform(X).shape == shape_desired


@pytest.mark.parametrize(
    'params',
    [{},
     {'dimension': 3},
     {'time_delay': 3},
     {'dimension': 3, 'time_delay': 4},
     {'dimension': 6, 'time_delay': 3},
     {'threshold': 3},
     {'percentage': 30}]
)
def test_actual_results_single_value(params):
    """Test that the actual results are the expected ones."""
    arr_actual = JointRecurrencePlot(**params).transform(X)
    arr_desired = []
    for i in range(n_features):
        arr_desired.append(RecurrencePlot(**params).transform(X[:, i]))
    arr_desired = np.prod(arr_desired, axis=0)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params',
    [{'threshold': [0.5, 0.9, 2], 'percentage': [10, 30, 50]},
     {'threshold': [None, 'distance', 'point'], 'percentage': [10, 30, 50]},
     {'threshold': [0.5, 0.9, None], 'percentage': [10., 30., 90.]}]
)
def test_actual_results_lists(params):
    """Test that the actual results are the expected ones."""
    arr_actual = JointRecurrencePlot(**params).transform(X)
    arr_desired = []
    for i, (threshold, percentage) in enumerate(
        zip(params['threshold'], params['percentage'])
    ):
        arr_desired.append(RecurrencePlot(
            threshold=threshold, percentage=percentage).transform(X[:, i]))
    arr_desired = np.prod(arr_desired, axis=0)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
