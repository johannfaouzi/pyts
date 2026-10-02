import pytest

from pyts.utils._deprecation import deprecated

# Author: <hicham.janati@inria.fr>
# Adapted from sklearn.utils.tests.test_deprecation


@deprecated('qwerty')
class MockClass1:
    pass


class MockClass2:
    @deprecated('mockclass2_method')
    def method(self):
        pass


class MockClass3:
    @deprecated()
    def __init__(self):
        pass


@deprecated()
def mock_function():
    return 10


def test_deprecated():
    with pytest.warns(DeprecationWarning, match="qwerty"):
        MockClass1()

    with pytest.warns(DeprecationWarning, match="mockclass2_method"):
        MockClass2().method()

    with pytest.warns(DeprecationWarning, match="deprecated"):
        MockClass3()

    with pytest.warns(DeprecationWarning):
        val = mock_function()
        assert val == 10
