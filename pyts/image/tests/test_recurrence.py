"""Testing for Recurrence Plot."""

import numpy as np
import pytest
import re
from math import sqrt
from ..recurrence import _trajectories, RecurrencePlot


def test_trajectories():
    """Test '_trajectories' function."""
    X = np.asarray([[0, 1, 1, 0, 2, 3],
                    [1, 0, 3, 2, 1, 0]])
    arr_actual = _trajectories(
        X, dimension=3, time_delay=2
    )
    arr_desired = np.asarray([[[0, 1, 2],
                               [1, 0, 3]],
                              [[1, 3, 1],
                               [0, 2, 0]]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_ReccurencePlot():
    """Test 'ReccurencePlot' class."""
    X = np.asarray([[0, 1, 2, 3],
                    [1, 0, 1, 1]])

    # Parameter check
    msg_error = "'dimension' must be an integer or a float."
    with pytest.raises(TypeError, match=msg_error):
        rp = RecurrencePlot(
            dimension="3", time_delay=1, threshold=None, percentage=None
        )
        rp.fit_transform(X)

    msg_error = "'time_delay' must be an integer or a float."
    with pytest.raises(TypeError, match=msg_error):
        rp = RecurrencePlot(
            dimension=3, time_delay="1", threshold=None, percentage=None
        )
        rp.fit_transform(X)

    msg_error = (
        "'threshold' must be either None, 'percentage_points', "
        "'percentage_distance', a float or an integer."
    )
    with pytest.raises(TypeError, match=msg_error):
        rp = RecurrencePlot(
            dimension=2, time_delay=1, threshold="3", percentage=None
        )
        rp.fit_transform(X)

    msg_error = "'percentage' must be a float or an integer."
    with pytest.raises(TypeError, match=msg_error):
        rp = RecurrencePlot(
            dimension=2, time_delay=1, threshold=None, percentage="3"
        )
        rp.fit_transform(X)

    msg_error = re.escape(
        "If 'dimension' is an integer, it must be greater "
        "than or equal to 1 and lower than or equal to "
        "n_timestamps (got {0}).".format(10)
    )
    with pytest.raises(ValueError, match=msg_error):
        rp = RecurrencePlot(
            dimension=10, time_delay=1, threshold=None, percentage=10
        )
        rp.fit_transform(X)

    msg_error = re.escape(
        "If 'dimension' is a float, it must be greater "
        "than 0 and lower than or equal to 1 "
        "(got {0}).".format(2.)
    )
    with pytest.raises(ValueError, match=msg_error):
        rp = RecurrencePlot(
            dimension=2., time_delay=1, threshold=None, percentage=10
        )
        rp.fit_transform(X)

    msg_error = re.escape(
        "If 'time_delay' is an integer, it must be greater "
        "than or equal to 1 and lower than or equal to "
        "n_timestamps (got {0}).".format(6)
    )
    with pytest.raises(ValueError, match=msg_error):
        rp = RecurrencePlot(
            dimension=3, time_delay=6, threshold=None, percentage=10
        )
        rp.fit_transform(X)

    msg_error = re.escape(
        "If 'time_delay' is a float, it must be greater "
        "than 0 and lower than or equal to 1 "
        "(got {0}).".format(2.)
    )
    with pytest.raises(ValueError, match=msg_error):
        rp = RecurrencePlot(
            dimension=3, time_delay=2., threshold=None, percentage=10
        )
        rp.fit_transform(X)

    msg_error = (
        "If 'threshold' is a float or an integer,"
        "it must be greater than or equal to 0."
    )
    with pytest.raises(ValueError, match=msg_error):
        rp = RecurrencePlot(
            dimension=2, time_delay=1, threshold=-2, percentage=10
        )
        rp.fit_transform(X)

    msg_error = "'percentage' must be between 0 and 100."
    with pytest.raises(ValueError, match=msg_error):
        rp = RecurrencePlot(
            dimension=2, threshold=None, percentage=200
        )
        rp.fit_transform(X)

    # Accurate result (dimension = 1) check
    dimension = 1
    rp = RecurrencePlot(dimension, threshold=None, percentage=10)
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
    dimension, time_delay = 2, 2
    rp = RecurrencePlot(dimension, time_delay, threshold=None, percentage=10)
    arr_actual = rp.fit_transform(X)
    arr_desired = np.asarray([[[0, sqrt(2)],
                               [sqrt(2), 0]],
                              [[0, 1],
                               [1, 0]]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Accurate result (dimension and time_delay float) check
    dimension, time_delay = 0.5, 0.5
    rp = RecurrencePlot(dimension, time_delay, threshold=None, percentage=10)
    arr_actual = rp.fit_transform(X)
    arr_desired = np.asarray([[[0, sqrt(2)],
                               [sqrt(2), 0]],
                              [[0, 1],
                               [1, 0]]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Accurate result (epsilon='percentage_points') check
    dimension, time_delay = 3, 1
    rp = RecurrencePlot(dimension, time_delay, threshold='percentage_points',
                        percentage=50)
    arr_actual = rp.fit_transform(X)
    arr_desired = np.asarray([[[1, 0],
                               [0, 1]],
                              [[1, 0],
                               [0, 1]]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Accurate result (epsilon='percentage_distance') check
    dimension, time_delay = 3, 1
    rp = RecurrencePlot(dimension=dimension, threshold='percentage_distance',
                        percentage=50)
    arr_actual = rp.fit_transform(X)
    arr_desired = np.asarray([[[1, 0],
                               [0, 1]],
                              [[1, 0],
                               [0, 1]]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
