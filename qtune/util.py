import itertools
import datetime
from typing import Iterable, Any, Callable, Sequence

import numpy as np
import sympy as sp


__all__ = ['nth']


class EvaluationError(RuntimeError):
    """This exception is raised if a fit or evaluation fails. It should be catched at the right position"""


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
    return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')


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


def nth_diff(data:np.ndarray, n: int):
    data_diff = np.ndarray((len(data) - n, ))
    for i in range(0, len(data) - n):
        data_diff[i] = data[i] - data[i+n]
    return data_diff


def moving_average_filter(data: np.ndarray, width) -> np.ndarray:
    data = data.squeeze()
    n_points = data.size
    smoothed = np.zeros((n_points, ))
    for i in range(width):
        smoothed[i] = sum(data[0:i + 1])
        smoothed[i] *= 1. / float(i + 1)
    for i in range(width, n_points, 1):
        smoothed[i] = sum(data[i - width + 1:i + 1])
        smoothed[i] *= 1. / width
    return smoothed


def find_stepes_point_sensing_dot(data: np.ndarray, scan_range=5e-3, npoints=1280) -> float:
    data = moving_average_filter(data, 200)
    data = nth_diff(data, 30)
    data = moving_average_filter(data, 30)
    max_index = np.argmin(data) + 15
    detuning = (float(max_index) - float(npoints) / 2.) * scan_range / (float(npoints) / 2.)
    return detuning


#def gradient_min_evaluations(parameters: List(np.ndarray, ...), voltage_points: List(np.ndarray, ...)):
def gradient_min_evaluations(parameters, voltage_points):

    """
    Uses finite differences and basis transformations to compute the gradient.
    :param parameters: A list of paramters belonging to the voltages
    :param voltage_points: List of voltage points. Either
    :return:
    """
    n_points = len(voltage_points)
    assert(len(voltage_points) == len(parameters))
    n_gates = voltage_points[0].size

    if n_points == n_gates + 1:
        voltage_diff = (np.stack(voltage_points[1:]) - voltage_points[0]).T
        parameter_diff = (np.stack(parameters[1:]) - parameters[0]).T
    elif n_points == 2 * n_gates:
        voltage_diff = (np.stack(voltage_points[1::2]) - np.stack(voltage_points[::2])).T
        parameter_diff = (np.stack(parameters[1::2]) - np.stack(parameters[::2])).T
    else:
        raise RuntimeError("Invalid number of points", parameters, voltage_points)

    try:
        gradient = np.dot(parameter_diff, np.linalg.inv(voltage_diff))
    except np.linalg.LinAlgError as err:
        raise EvaluationError() from err
    return gradient


def calculate_gradient_non_orthogonal(positions: Sequence[np.ndarray],
                                      values: Sequence[float],
                                      variances: Sequence[float]=None):
    n_points = len(values)
    assert len(values) == len(positions)
    n_dim = positions[0].size

    if n_points == n_dim + 1:
        voltage_diff = np.stack(positions[1:]) - positions[0]
        parameter_diff = np.stack(values[1:]) - values[0]
    elif n_points == 2 * n_dim:
        voltage_diff = np.stack(positions[1::2]) - np.stack(positions[::2])
        parameter_diff = np.stack(values[1::2]) - np.stack(values[::2])
    else:
        raise RuntimeError("Invalid number of points", positions, values)

    try:
        inverted_volt_diffs = np.linalg.inv(voltage_diff)
    except np.linalg.LinAlgError as err:
        raise EvaluationError() from err

    gradient = inverted_volt_diffs @ parameter_diff

    if variances:
        if n_points == n_dim + 1:
            diff_variances = np.stack(variances[1:]) + variances[0]
        else:
            diff_variances = np.stack(variances[1::2]) + np.stack(variances[::2])

        gradient_covariance = inverted_volt_diffs @ np.diag(diff_variances) @ inverted_volt_diffs.T
        return gradient, gradient_covariance

    return gradient


def get_orthogonal_vector(vectors: Sequence[np.ndarray]):
    """Return a vector orthogonal to the given ones"""
    ov, *_ = sp.Matrix(vectors).nullspace()
    ov = np.asarray(ov, dtype=float)
    return ov / np.linalg.norm(ov)
