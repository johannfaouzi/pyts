"""Parameter and input validation of ROCKET."""

# License: BSD-3-Clause

import re

import numpy as np
import pytest

from pyts.transformation import ROCKET

rng = np.random.RandomState(0)
X = rng.randn(5, 20)


@pytest.mark.parametrize('n_kernels', [0, -1])
def test_n_kernels_must_be_positive(n_kernels):
    """A non-positive number of kernels is rejected with a clear error."""
    err_msg = f"'n_kernels' must be a positive integer (got {n_kernels})."
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        ROCKET(n_kernels=n_kernels).fit(X)


@pytest.mark.parametrize('kernel_sizes', [(1,), (1, 7), [1, 9, 11]])
def test_kernel_sizes_must_be_at_least_two(kernel_sizes):
    """Length-1 kernels would produce constant features."""
    err_msg = (
        "All the values in 'kernel_sizes' must be greater than or equal to 2 (1 < 2)."
    )
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        ROCKET(kernel_sizes=kernel_sizes).fit(X)


def test_kernel_size_two_gives_informative_features():
    """The smallest allowed kernel size gives non-constant features."""
    X_new = ROCKET(n_kernels=20, kernel_sizes=(2,), random_state=0).fit_transform(X)
    assert np.all(X_new.std(axis=0)[::2] > 0)


def test_transform_rejects_series_too_short_for_the_kernels():
    """A series shorter than a dilated kernel gets a clear error.

    It used to fail with "zero-size array to reduction operation maximum".
    """
    rocket = ROCKET(n_kernels=10, kernel_sizes=(9,), random_state=0).fit(X)
    min_n_timestamps = np.max(
        (rocket.length_ - 1) * rocket.dilation_ + 1 - 2 * rocket.padding_
    )
    assert min_n_timestamps > 5
    err_msg = (
        "The kernels learned during `fit` require time series with at least "
        f"{min_n_timestamps} timestamps (got 5)."
    )
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        rocket.transform(rng.randn(3, 5))


@pytest.mark.parametrize('n_timestamps', [20, 30])
def test_transform_accepts_long_enough_series(n_timestamps):
    """Series at least as long as the training ones can still be transformed."""
    rocket = ROCKET(n_kernels=10, random_state=0).fit(X)
    assert rocket.transform(rng.randn(3, n_timestamps)).shape == (3, 20)
