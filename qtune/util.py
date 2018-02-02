import itertools
import datetime
from typing import Iterable, Any, Callable
import numpy as np


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


def time_string() -> str:
    return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def find_lead_transition(data: np.ndarray, center: float, scan_range: float, npoints: int, width: float = .2e-3) -> float:
    if len(data.shape) == 2:
        y = np.mean(data, 0)
    elif len(data.shape) == 1:
        y = data
    else:
        print('data must be a one or two dimensional array!')
        return np.nan

    x = np.linspace(center - scan_range, center + scan_range, npoints)

    n = int(width/scan_range*npoints)
    for i in range(0, len(y)-n-1):
        y[i] -= y[i+n]

    y_red = y[0:len(y) - n - 1]
    x_red = x[0:len(y) - n - 1]

    y_red = np.absolute(y_red)
    max_index = int(np.argmax(y_red) + int(round(n / 2)))

    return x[max_index]

