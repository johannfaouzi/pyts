"""Testing the boundary values accepted by the image transformers."""

# License: BSD-3-Clause

import re

import numpy as np
import pytest

from pyts.image import MarkovTransitionField, RecurrencePlot

X = np.random.RandomState(0).randn(3, 12)


def test_mtf_zero_float_image_size():
    """A zero float image size is rejected with the intended error."""
    msg = (
        "If 'image_size' is a float, it must be greater than 0 "
        "and lower than or equal to 1 (got 0.0)."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MarkovTransitionField(image_size=0.0).transform(X)


@pytest.mark.parametrize(
    'params, n_trajectories',
    [
        ({'dimension': 1, 'time_delay': 1.0}, 12),
        ({'dimension': 1.0, 'time_delay': 1}, 1),
        ({'dimension': 1.0, 'time_delay': 0.01}, 1),
    ],
)
def test_recurrence_plot_float_bounds_of_one(params, n_trajectories):
    """A float of 1 (the whole series length) is accepted."""
    X_new = RecurrencePlot(**params).transform(X)
    assert X_new.shape == (X.shape[0], n_trajectories, n_trajectories)
    if isinstance(params['dimension'], int):
        np.testing.assert_allclose(X_new, RecurrencePlot().transform(X))


def test_recurrence_plot_float_bounds_of_one_too_long():
    """Accepted bounds still fail when no trajectory fits in the series."""
    msg = "The number of trajectories must be positive."
    with pytest.raises(ValueError, match=re.escape(msg)):
        RecurrencePlot(dimension=1.0, time_delay=2).transform(X)


@pytest.mark.parametrize('threshold', [0, 0.0, -1])
def test_recurrence_plot_non_positive_threshold(threshold):
    """The message matches the check: the threshold must be positive."""
    msg = "If 'threshold' is a float or an integer, it must be greater than 0."
    with pytest.raises(ValueError, match=re.escape(msg)):
        RecurrencePlot(threshold=threshold).transform(X)
