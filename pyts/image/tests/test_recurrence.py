"""Testing for Recurrence Plot."""

import numpy as np
import pytest
import re
from math import sqrt
from ..recurrence import _trajectory_distances, RecurrencePlot


def test_trajectory_distances():
    """Test '_trajectory_distances' function."""
    n_samples, n_timestamps, dimension = 2, 6, 1
    image_size = 6
    X = np.asarray([[0, 1, 1, 0, 2, 3],
                    [1, 0, 3, 2, 1, 0]])
    arr_actual = _trajectory_distances(
        X, n_samples, n_timestamps, dimension, image_size
    )
    arr_desired = np.asarray([[[0, 1, 1, 0, 2, 3],
                               [1, 0, 0, 1, 1, 2],
                               [1, 0, 0, 1, 1, 2],
                               [0, 1, 1, 0, 2, 3],
                               [2, 1, 1, 2, 0, 1],
                               [3, 2, 2, 3, 1, 0]],
                              [[0, 1, 2, 1, 0, 1],
                               [1, 0, 3, 2, 1, 0],
                               [2, 3, 0, 1, 2, 3],
                               [1, 2, 1, 0, 1, 2],
                               [0, 1, 2, 1, 0, 1],
                               [1, 0, 3, 2, 1, 0]]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_ReccurencePlot():
    """Test 'ReccurencePlot' class."""
    X = np.asarray([[0, 1, 2, 3],
                    [1, 0, 1, 1]])

    # Parameter check
    msg_error = "'dimension' must be an integer or a float."
    with pytest.raises(TypeError, match=msg_error):
        rp = RecurrencePlot(
            dimension="3", epsilon=None, percentage=None
        )
        rp.fit_transform(X)

    msg_error = (
        "'epsilon' must be either None, 'percentage_points', "
        "'percentage_distance', a float or an integer."
    )
    with pytest.raises(TypeError, match=msg_error):
        rp = RecurrencePlot(
            dimension=2, epsilon="3", percentage=None
        )
        rp.fit_transform(X)

    msg_error = "'percentage' must be a float or an integer."
    with pytest.raises(TypeError, match=msg_error):
        rp = RecurrencePlot(
            dimension=2, epsilon=None, percentage="3"
        )
        rp.fit_transform(X)

    msg_error = re.escape(
        "If 'dimension' is an integer, it must be greater "
        "than or equal to 1 and lower than or equal to "
        "n_timestamps (got {0}).".format(10)
    )
    with pytest.raises(ValueError, match=msg_error):
        rp = RecurrencePlot(
            dimension=10, epsilon=None, percentage=10
        )
        rp.fit_transform(X)

    msg_error = re.escape(
        "If 'dimension' is a float, it must be greater "
        "than 0 and lower than or equal to 1 "
        "(got {0}).".format(2.)
    )
    with pytest.raises(ValueError, match=msg_error):
        rp = RecurrencePlot(
            dimension=2., epsilon=None, percentage=10
        )
        rp.fit_transform(X)

    msg_error = (
        "If 'epsilon' is a float or an integer,"
        "it must be greater than or equal to 0."
    )
    with pytest.raises(ValueError, match=msg_error):
        rp = RecurrencePlot(
            dimension=2, epsilon=-2, percentage=10
        )
        rp.fit_transform(X)

    msg_error = "'percentage' must be between 0 and 100."
    with pytest.raises(ValueError, match=msg_error):
        rp = RecurrencePlot(
            dimension=2, epsilon=None, percentage=200
        )
        rp.fit_transform(X)

    # Accurate result (dimension = 1) check
    dimension = 1
    rp = RecurrencePlot(dimension, epsilon=None, percentage=10)
    arr_actual = rp.fit_transform(X)
    arr_desired = np.asarray([[[0, 1, 2, 3],
                               [1, 0, 1, 2],
                               [2, 1, 0, 1],
                               [3, 2, 1, 0]],
                              [[0, 1, 0, 0],
                               [1, 0, 1, 1],
                               [0, 1, 0, 0],
                               [0, 1, 0, 0]]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Accurate result (dimension > 1) check
    dimension = 3
    rp = RecurrencePlot(dimension=dimension, epsilon=None, percentage=10)
    arr_actual = rp.fit_transform(X)
    arr_desired = np.asarray([[[0, sqrt(3)],
                               [sqrt(3), 0]],
                              [[0, sqrt(2)],
                               [sqrt(2), 0]]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
