"""Testing for Gramian Angular Field."""

import numpy as np
from itertools import product
from ..gaf import _gasf, _gadf, GramianAngularField


def test_gasf():
    """Test '_gasf' function."""
    X_cos = np.asarray([[-1, 0, 1]])
    X_sin = np.asarray([[1, 0, 1]])
    arr_actual = _gasf(X_cos, X_sin, n_samples=1, image_size=3)
    arr_desired = np.asarray([[[1, 0, -1], [0, 0, 0], [-1, 0, 1]]])
    arr_desired = arr_desired - np.asarray([[[1, 0, 1], [0, 0, 0], [1, 0, 1]]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_gadf():
    """Test '_gadf' function."""
    X_cos = np.asarray([[-1, 0, 1]])
    X_sin = np.asarray([[1, 0, 1]])
    arr_actual = _gadf(X_cos, X_sin, n_samples=1, image_size=3)
    arr_desired = np.asarray([[[-1, 0, 1], [0, 0, 0], [-1, 0, 1]]])
    arr_desired -= np.asarray([[[-1, 0, -1], [0, 0, 0], [1, 0, 1]]])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_GramianAngularField():
    """Test 'GramianAngularField' class."""
    X = np.arange(9).reshape(1, 9)

    # Parameter check
    def type_error_list():
        type_error_list_ = ["'image_size' must be an integer or a float.",
                            "'sample_range' must be None or a tuple."]
        return type_error_list_

    def value_error_list(image_size):
        value_error_list_ = [
            "If 'image_size' is an integer, it must be greater "
            "than or equal to 1 and lower than or equal to the size "
            "of each time series (i.e. the size of the last dimension "
            "of X) (got {0}).".format(image_size),
            "If 'image_size' is a float, it must be greater "
            "than or equal to 0 and lower than or equal to 1 "
            "(got {0}).".format(image_size),
            "If 'sample_range' is a tuple, its length must be equal to 2.",
            "If 'sample_range' is a tuple, it must satisfy "
            "-1 <= sample_range[0] < sample_range[1] <= 1.",
            "'method' must be either 'summation', 's', 'difference' or 'd'.",
            "If 'sample_range' is None, all the values of X must be between "
            "-1 and 1."
        ]
        return value_error_list_

    image_size_list = [1., 2., -1, 2, 3, None]
    sample_range_list = [(-2, 2), (0, ), 0, None, (-0.5, 0.5)]
    method_list = ['s', 'd', None]
    overlapping_list = [True, False]

    for (image_size, sample_range, method, overlapping) in product(
        image_size_list, sample_range_list, method_list, overlapping_list
    ):
        gaf = GramianAngularField(image_size, sample_range,
                                  method, overlapping)
        try:
            gaf.fit_transform(X)
        except ValueError as e:
            if str(e) in value_error_list(image_size):
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) in type_error_list():
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))

    # Accurate result for method='s' check
    arccos = [np.pi, np.pi / 2, 0]
    gaf = GramianAngularField(
        image_size=3, sample_range=(-1, 1), method='s', overlapping=False)
    arr_actual = gaf.fit_transform(X)[0]
    arr_desired = np.cos([[x + y for x in arccos] for y in arccos])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Accurate result for method='d' check
    gaf = GramianAngularField(
        image_size=3, sample_range=(-1, 1), method='d', overlapping=False)
    arr_actual = gaf.fit_transform(X)[0]
    arr_desired = np.sin([[x - y for y in arccos] for x in arccos])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
