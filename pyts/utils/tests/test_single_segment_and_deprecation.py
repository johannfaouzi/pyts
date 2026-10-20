"""Single-segment segmentation and deprecation docstrings."""

# License: BSD-3-Clause

import numpy as np
import pytest

from pyts.image import GramianAngularField
from pyts.utils import deprecated, segmentation


@pytest.mark.parametrize('overlapping', [False, True])
def test_segmentation_single_segment(overlapping):
    """One segment spans the whole series."""
    start, end, size = segmentation(10, 10, overlapping=overlapping, n_segments=1)
    np.testing.assert_array_equal(start, [0])
    np.testing.assert_array_equal(end, [10])
    assert size == 1


@pytest.mark.parametrize('image_size', [1, 0.1])
def test_gramian_angular_field_single_pixel(image_size):
    """An image size of one gives one finite pixel per series."""
    X = np.random.RandomState(0).randn(4, 10)
    X_new = GramianAngularField(image_size=image_size).transform(X)
    assert X_new.shape == (4, 1, 1)
    assert np.all(np.isfinite(X_new))


def test_deprecated_updates_docstrings():
    """The deprecation notice is prepended to the docstring."""

    @deprecated('use g instead')
    def f():
        """Do f."""

    @deprecated()
    class C:
        """A class."""

    class P:
        @deprecated('gone')
        @property
        def x(self):
            """The x."""
            return 3

    assert f.__doc__ == 'DEPRECATED: use g instead\n\nDo f.'
    assert C.__doc__ == 'DEPRECATED\n\nA class.'
    assert P.x.__doc__ == 'DEPRECATED: gone\n\nThe x.'
    with pytest.warns(DeprecationWarning, match='gone'):
        assert P().x == 3
