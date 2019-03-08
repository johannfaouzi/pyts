"""Testing for Gramian Angular Field."""

import numpy as np
import pytest
import re
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
    msg_error = "'image_size' must be an integer or a float."
    with pytest.raises(TypeError, match=msg_error):
        gaf = GramianAngularField(
            image_size="4", sample_range=(-1, 1), method='s', overlapping=False
        )
        gaf.fit_transform(X)

    msg_error = "'sample_range' must be None or a tuple."
    with pytest.raises(TypeError, match=msg_error):
        gaf = GramianAngularField(
            image_size=4, sample_range=[0, 1], method='s', overlapping=False
        )
        gaf.fit_transform(X)

    msg_error = re.escape(
        "If 'image_size' is an integer, it must be greater "
        "than or equal to 1 and lower than or equal to the size "
        "of each time series (i.e. the size of the last dimension "
        "of X) (got {0}).".format(0)
    )
    with pytest.raises(ValueError, match=msg_error):
        gaf = GramianAngularField(
            image_size=0, sample_range=(-1, 1), method='s', overlapping=False
        )
        gaf.fit_transform(X)

    msg_error = re.escape(
        "If 'image_size' is a float, it must be greater "
        "than or equal to 0 and lower than or equal to 1 "
        "(got {0}).".format(2.)
    )
    with pytest.raises(ValueError, match=msg_error):
        gaf = GramianAngularField(
            image_size=2., sample_range=(-1, 1), method='s', overlapping=False
        )
        gaf.fit_transform(X)

    msg_error = (
        "If 'sample_range' is a tuple, its length must be equal to 2."
    )
    with pytest.raises(ValueError, match=msg_error):
        gaf = GramianAngularField(
            image_size=4, sample_range=(-1, 0, 1),
            method='s', overlapping=False
        )
        gaf.fit_transform(X)

    msg_error = re.escape(
        "If 'sample_range' is a tuple, it must satisfy "
        "-1 <= sample_range[0] < sample_range[1] <= 1."
    )
    with pytest.raises(ValueError, match=msg_error):
        gaf = GramianAngularField(
            image_size=4, sample_range=(-2, 2), method='s', overlapping=False
        )
        gaf.fit_transform(X)

    msg_error = (
        "'method' must be either 'summation', 's', 'difference' or 'd'."
    )
    with pytest.raises(ValueError, match=msg_error):
        gaf = GramianAngularField(
            image_size=4, sample_range=(-1, 1), method='a', overlapping=False
        )
        gaf.fit_transform(X)

    msg_error = (
        "If 'sample_range' is None, all the values of X must be between "
        "-1 and 1."
    )
    with pytest.raises(ValueError, match=msg_error):
        gaf = GramianAngularField(
            image_size=4, sample_range=None, method='s', overlapping=False
        )
        gaf.fit_transform(X)

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

    # Accurate result for image_size float check
    arccos = [np.pi, np.pi / 2, 0]
    gaf = GramianAngularField(
        image_size=0.33, sample_range=(-1, 1), method='s', overlapping=False)
    arr_actual = gaf.fit_transform(X)[0]
    arr_desired = np.cos([[x + y for x in arccos] for y in arccos])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    # Accurate result for sample_range=None check
    arccos = [np.pi, np.pi / 2, 0]
    gaf = GramianAngularField(
        image_size=3, sample_range=None, method='s', overlapping=False)
    arr_actual = gaf.fit_transform(np.linspace(-1, 1, 3).reshape(1, 3))[0]
    arr_desired = np.cos([[x + y for x in arccos] for y in arccos])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
