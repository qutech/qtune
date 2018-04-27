from typing import Tuple, Sequence

import numpy as np
import pandas as pd

import scipy.optimize

from qtune.gradient import GradientEstimator
from qtune.storage import HDF5Serializable


def make_target(desired: pd.Series=np.nan,
                maximum: pd.Series=np.nan,
                minimum: pd.Series=np.nan,
                tolerance: pd.Series=np.nan):
    for ser in (desired, maximum, minimum, tolerance):
        if isinstance(ser, pd.Series):
            names = ser.index
            break
    else:
        raise RuntimeError('Could not extract values names from arguments')

    def to_series(arg):
        if not isinstance(arg, pd.Series):
            return pd.Series(arg, index=names)
        else:
            return arg[names]

    return pd.DataFrame({'desired': to_series(desired),
                         'minimum': to_series(minimum),
                         'maximum': to_series(maximum),
                         'tolerance': to_series(tolerance)},
                        index=names)


class Solver(metaclass=HDF5Serializable):
    """
    The solver class implements an algorithm to minimise the difference of the values to the target values.
    """
    def suggest_next_position(self) -> pd.Series:
        raise NotImplementedError()

    def update_after_step(self, position: pd.Series, values: pd.Series, variances: pd.Series):
        raise NotImplementedError()

    def to_hdf5(self):
        raise NotImplementedError()

    @property
    def target(self) -> pd.DataFrame:
        raise NotImplementedError()


class NewtonSolver(Solver):
    """This solver uses (an estimate of) the jacobian and solves by inverting it.(Newton's method)

    The jacobian is put together from the given gradient estimators"""
    def __init__(self, target: pd.DataFrame,
                 gradient_estimators: Sequence[GradientEstimator],
                 current_position: pd.Series=None,
                 current_values: pd.Series=None):
        self._target = target
        self._gradient_estimators = list(gradient_estimators)
        assert len(self._target) == len(self._gradient_estimators)

        self._current_position = current_position
        if current_values:
            self._current_values = current_values[self._target.index]
        else:
            self._current_values = pd.Series(np.nan, index=self._target.index)

    @property
    def target(self) -> pd.DataFrame:
        return self._target

    @property
    def jacobian(self) -> pd.DataFrame:
        gradients = [gradient.estimate() for gradient in self._gradient_estimators]
        return pd.concat(gradients, axis=1, keys=self._target.index).T

    def suggest_next_position(self) -> pd.Series:
        for estimator in self._gradient_estimators:
            suggestion = estimator.require_measurement()
            if suggestion is not None and not suggestion.empty:
                return suggestion

        if self._current_position is None:
            raise RuntimeError('NewtonSolver: Position not initialized.')

        # our jacobian is sufficiently accurate
        required_diff = self.target.desired - self._current_values

        step, *_ = np.linalg.lstsq(self.jacobian, required_diff)
        return self._current_position + step

    def update_after_step(self, position: pd.Series, values: pd.Series, variances: pd.Series):
        for estimator, value, variance in zip(self._gradient_estimators, values, variances):
            estimator.update(position, value, variance, is_new_position=True)
        self._current_position = position[self._current_position.index]
        self._current_values = values[self._current_values.index]

    def to_hdf5(self):
        return dict(target=self.target,
                    gradient_estimators=self._gradient_estimators,
                    current_position=self._current_position,
                    current_values=self._current_values)


class NelderMeadSolver(Solver):
    def __init__(self,
                 target: pd.Series,
                 simplex: Sequence[Tuple[pd.Series, pd.Series]],
                 weights: pd.Series,
                 current_position: pd.Series):
        self.target = target
        self.simplex = list(simplex)
        self.weights = weights
        self.current_position = current_position

    def suggest_next_position(self) -> pd.Series:
        if len(self.simplex) < len(self.current_position) + 1:
            pass

        def intercept(requested_point: np.ndarray):
            for volts, params in self.simplex:
                if all((requested_point - volts).abs() < 1e-10):
                    return (params * self.weights).norm()**2
            return 1e9

        simplex = np.array([volts for volts, params in self.simplex])
        options = dict(initial_simplex=simplex, maxfev=1)

        results = scipy.optimize.minimize(intercept, x0=simplex[0], method='Nelder-Mead', options=options)

        raise NotImplementedError('TODO')

    def to_hdf5(self):
        raise NotImplementedError()


class ForwardingSolver(Solver):
    """Solves by forwarding the values of the given values and renaming them to a voltage vector which updates the
    given position"""
    def __init__(self,
                 target: pd.DataFrame,
                 values_to_position: pd.Series,
                 current_position: pd.Series,
                 next_position: pd.Series=None):
        """

        :param values_to_position: A series of strings
        :param next_position:
        """
        self._target = target
        self._values_to_position = values_to_position
        self._current_position = current_position
        if next_position is None:
            next_position = self._current_position.copy()
        else:
            next_position = next_position[self._current_position]
        self._next_position = next_position

    @property
    def target(self) -> pd.DataFrame:
        return self._target

    def suggest_next_position(self) -> pd.Series:
        return self._next_position

    def update_after_step(self, position: pd.Series, values: pd.Series, variances: pd.Series):
        self._current_position[position.index] = position
        self._next_position[position.index] = position

        new_position_names = self._values_to_position[values.index]
        self._next_position[new_position_names] = values

    def to_hdf5(self):
        return dict(target=self._target,
                    values_to_position=self._values_to_position,
                    current_position=self._current_position,
                    next_position=self._next_position)
