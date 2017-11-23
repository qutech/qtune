import itertools
from typing import Iterable, Any, Callable


__all__ = ['nth']


def nth(iterable: Iterable[Any], n: int) -> Any:
    """Returns the nth item or a default value"""
    return next(itertools.islice(iterable, n, None))


def static_vars(**kwargs) -> Callable[[Callable], Callable]:
    def decorate(func: Callable) -> Callable:
        for key, value in kwargs.items():
            setattr(func, key, value)
        return func
    return decorate
