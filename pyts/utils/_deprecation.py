import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

# Author: <hicham.janati@inria.fr>
# Adapted from sklearn.utils.deprecation


__all__ = ["deprecated"]

F = TypeVar("F", bound=Callable[..., Any])


class deprecated:
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    >>> from pyts.utils import deprecated

    >>> @deprecated()
    ... def some_function(): pass

    Parameters
    ----------
    extra : string
          to be added to the deprecation messages
    """

    def __init__(self, extra: str = "") -> None:
        self.extra = extra

    def __call__(self, obj: Any) -> Any:
        """Call method

        Parameters
        ----------
        obj : object
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        elif isinstance(obj, property):
            # Note that this is only triggered properly if the `property`
            # decorator comes before the `deprecated` decorator, like so:
            #
            # @deprecated(msg)
            # @property
            # def deprecated_attribute_(self):
            #     ...
            return self._decorate_property(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls: type) -> type:
        msg = f"Class {cls.__name__} is deprecated"
        if self.extra:
            msg += f"; {self.extra}"

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, stacklevel=2, category=DeprecationWarning)
            return init(*args, **kwargs)

        cls.__init__ = wrapped

        wrapped.__name__ = "__init__"
        # `deprecated_original` is a custom attribute this decorator adds
        # to the wrapper at runtime; ordinary function objects don't have
        # one, so the type checker cannot know about it statically.
        wrapped.deprecated_original = init  # pyrefly: ignore[missing-attribute]

        return cls

    def _decorate_fun(self, fun: F) -> F:
        """Decorate function fun"""

        msg = f"Function {fun.__name__} is deprecated"
        if self.extra:
            msg += f"; {self.extra}"

        @functools.wraps(fun)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, stacklevel=2, category=DeprecationWarning)
            return fun(*args, **kwargs)

        # Add a reference to the wrapped function so that we can introspect
        # on function arguments in Python 2 (already works in Python 3)
        wrapped.__wrapped__ = fun  # pyrefly: ignore[missing-attribute]

        return wrapped  # type: ignore[return-value]

    def _decorate_property(self, prop: property) -> property:
        msg = self.extra

        @property
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, stacklevel=2, category=DeprecationWarning)
            return prop.fget(*args, **kwargs)  # type: ignore[misc]

        return wrapped  # pyrefly: ignore[bad-return]
