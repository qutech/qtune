import itertools
import datetime
import time
import os
import subprocess
from typing import Iterable, Any, Callable, Sequence, Optional, Dict
import numbers
import matplotlib.axes
import matplotlib.pyplot as plt

import numpy as np
import sympy as sp
import pandas as pd


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


def high_res_datetime():
    now_perf = time.perf_counter()
    start_perf, start_datetime = high_res_datetime.start

    return start_datetime + datetime.timedelta(seconds=now_perf - start_perf)


high_res_datetime.start = (time.perf_counter(), datetime.datetime.now())


def time_string() -> str:
    return high_res_datetime().strftime('%Y_%m_%d_%H_%M_%S_%f')


def new_find_lead_transition_index(data: np.ndarray, width_in_index_points: int) -> int:
    data = data.copy().squeeze()
    for i in range(0, len(data)-width_in_index_points-1):
        data[i] -= data[i+width_in_index_points]
    return int(np.argmax(np.abs(data)[:-width_in_index_points-1]) + round(width_in_index_points / 2))


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


# def gradient_min_evaluations(parameters: List(np.ndarray, ...), voltage_points: List(np.ndarray, ...)):
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
        voltage_diff = (np.stack(positions[1:]) - np.asarray(positions[0])).T
        parameter_diff = (np.stack(values[1:]) - values[0]).T
    elif n_points == 2 * n_dim:
        voltage_diff = (np.stack(positions[1::2]) - np.stack(positions[::2])).T
        parameter_diff = (np.stack(values[1::2]) - np.stack(values[::2])).T
    else:
        raise RuntimeError("Invalid number of points", positions, values)

    try:
        inverted_volt_diffs = np.linalg.inv(voltage_diff)
    except np.linalg.LinAlgError as err:
        raise EvaluationError() from err

    gradient = parameter_diff @ inverted_volt_diffs

    if variances:
        if n_points == n_dim + 1:
            diff_variances = np.stack(variances[1:]) + variances[0]
        else:
            diff_variances = np.stack(variances[1::2]) + np.stack(variances[::2])

        gradient_covariance = inverted_volt_diffs.T @ np.diag(diff_variances) @ inverted_volt_diffs
        return gradient, gradient_covariance

    return gradient


def get_orthogonal_vector(vectors: Sequence[np.ndarray]):
    """Return a vector orthogonal to the given ones"""
    ov, *_ = sp.Matrix(vectors).nullspace()
    # ov = np.asarray(ov, dtype=float)
    ov = np.array(ov).astype(float)
    ov = np.squeeze(ov)
    return ov / np.linalg.norm(ov)


def plot_raw_data_fit(y_data: np.ndarray, x_data: Optional[np.ndarray], fit_function=None,
                      function_args: Optional[Dict[str, numbers.Number]]=None,
                      initial_arguments: Optional[Dict[str, numbers.Number]]=None,
                      ax: Optional[matplotlib.axes.Axes]=None):
    if ax is None:
        ax = plt.gca()
    if y_data is None:
        return ax
    y_data = y_data.squeeze()
    if len(y_data) == 2:
        y_data = np.nanmean(y_data)
    if x_data is None:
        x_data = np.arange(0, y_data.shape[0])
    if isinstance(function_args, pd.Series):
        function_args = dict(function_args)
    if isinstance(initial_arguments, pd.Series):
        initial_arguments = dict(initial_arguments)

    for data in [x_data, y_data]:
        if len(data.shape) > 1:
            raise RuntimeError('Data has too many dimensions and therefore can not be plotted')

    ax.plot(x_data, y_data, 'b.', label='Raw Data')
    if fit_function:
        if function_args:
            ax.plot(x_data, fit_function(x_data, **function_args), 'r', label='Fit')
        if initial_arguments:
            ax.plot(x_data, fit_function(x_data, **initial_arguments), 'k--', label='Initial Guess')
    return ax


def plot_raw_data_vertical_marks(y_data, x_data, marking_position, ax):
    if ax is None:
        ax = plt.gca()
    if y_data is None:
        return ax
    y_data = y_data.squeeze()
    if len(y_data) == 2:
        y_data = np.nanmean(y_data)
    if x_data is None:
        x_data = np.arange(0, y_data.shape[0])

    ax.plot(x_data, y_data, 'b.', label='Raw Data')
    ax.vlines(x=marking_position, ymin=min(y_data), ymax=max(y_data), label='Transition Position')
    ax.legend()
    return ax


def plot_raw_data_2_dim_marks(y_data, x_data, ax, marking_position):
    if ax is None:
        ax = plt.gca()
    if y_data is None:
        return ax
    y_data = y_data.squeeze()
    x, y = np.meshgrid(x_data[1], x_data[0])
    image = ax.pcolormesh(x, y, y_data)
    ax.hlines(y=marking_position.iloc[0], xmin=min(x_data[1]), xmax=max(x_data[1]))
    ax.vlines(x=marking_position.iloc[1], ymin=min(x_data[0]), ymax=max(x_data[0]))
    # plt.colorbar()
    return ax


def get_git_info():
    if os.path.isdir(os.path.join(os.path.dirname(__file__), '..', '.git')):
        git_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        try:
            import git
            repo = git.Repo(git_root)
            commit_time = repo.head.object.committed_date
            commit_hash = git.Repo(git_root).head.object.hexsha
            return commit_time, commit_hash
        except (ImportError, RuntimeError):
            pass

        try:
            commit_time = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', 'HEAD'],
                                                      timeout=3).strip().decode('ascii'))
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], timeout=3).strip().decode('ascii')
            return commit_time, commit_hash
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

    return None, None


def get_version():
    import qtune
    base = qtune.__version__

    commit_time, commit_hash = get_git_info()

    if commit_time:
        return '%s+%d+%s' % (base, commit_time, commit_hash)

    return base
