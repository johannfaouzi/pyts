import pytest

from pyts.utils.deprecation import _is_deprecated
from pyts.utils.deprecation import deprecated

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


class MockClass4:
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


def test_is_deprecated():
    # Test if _is_deprecated helper identifies wrapping via deprecated
    # NOTE it works only for class methods and functions
    assert _is_deprecated(MockClass1.__init__)
    assert _is_deprecated(MockClass2().method)
    assert _is_deprecated(MockClass3.__init__)
    assert not _is_deprecated(MockClass4.__init__)
    assert _is_deprecated(mock_function)
