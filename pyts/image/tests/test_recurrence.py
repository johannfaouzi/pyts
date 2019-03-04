"""Testing for Recurrence Plot."""

import numpy as np
from math import sqrt
from itertools import product
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
    def type_error_list():
        type_error_list_ = [
            "'dimension' must be an integer or a float.",
            "'epsilon' must be either None, "
            "'percentage_points', 'percentage_distance', "
            "a float or an integer.",
            "'percentage' must be a float or an integer."
        ]
        return type_error_list_

    def value_error_list(dimension):
        value_error_list_ = [
            "If 'dimension' is an integer, it must be greater "
            "than or equal to 1 and lower than or equal to the size "
            "of each time series (i.e. the size of the last dimension "
            "of X) (got {0}).".format(dimension),
            "If 'dimension' is a float, it must be greater "
            "than or equal to 0 and lower than or equal to 1 "
            "(got {0}).".format(dimension),
            "If 'epsilon' is a float or an integer,"
            "'epsilon' must be greater than or equal to 0.",
            "'percentage' must be between 0 and 100."
        ]
        return value_error_list_

    dimension_list = [-1, -1., 1, 0.5, None]
    epsilon_list = [None, 'percentage_points', 'percentage_distance', 0, 1]
    percentage_list = [50, 150, None]
    for (dimension, epsilon, percentage) in product(
        dimension_list, epsilon_list, percentage_list
    ):
        rp = RecurrencePlot(dimension, epsilon, percentage)
        try:
            rp.fit_transform(X)
        except ValueError as e:
            if str(e) in value_error_list(dimension):
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) in type_error_list():
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))

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
