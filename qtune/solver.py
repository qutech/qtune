from typing import Tuple, Sequence, Optional

import numpy as np
import pandas as pd

import scipy.optimize

from qtune.gradient import GradientEstimator
from qtune.storage import HDF5Serializable


class Solver(metaclass=HDF5Serializable):
    """
    The solver class implements an algorithm to minimise the difference of the parameters to the target values.
    """
    @property
    def target(self) -> pd.Series:
        raise NotImplementedError()

    @target.setter
    def target(self, val):
        raise NotImplementedError()

    def suggest_next_step(self) -> pd.Series:
        raise NotImplementedError()

    def update_after_step(self, voltages: pd.Series, parameters: pd.Series):
        raise NotImplementedError()

    def to_hdf5(self):
        raise NotImplementedError()


class NewtonSolver(Solver):
    def __init__(self, target: pd.Series,
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
    def target(self) -> pd.Series:
        return self._target

    @target.setter
    def target(self, val):
        self._target = val[self._target.index]

    @property
    def jacobian(self) -> pd.DataFrame:
        gradients = [gradient.estimate() for gradient in self._gradient_estimators]
        return pd.DataFrame(gradients, columns=self._target.index)

    def suggest_next_voltage(self) -> pd.Series:
        for estimator in self._gradient_estimators:
            suggestion = estimator.require_measurement()
            if suggestion:
                return suggestion

        if self._current_position is None:
            raise RuntimeError('NewtonSolver: Position not initialized.')

        # our jacobian is sufficiently accurate
        required_diff = self.target - self._current_position

        step, *_ = np.linalg.lstsq(self.jacobian, required_diff)
        return self._current_position + step

    def update_after_step(self, voltages: pd.Series, parameters: pd.Series, variances: pd.Series):
        for estimator, value, variance in zip(self._gradient_estimators, parameters, variances):
            estimator.update(voltages, value, variance, is_new_position=True)
        self._current_position = parameters[self._current_position.index]
        self._current_values = parameters[self._current_values.index]

    def to_hdf5(self):
        return dict(target=self.target,
                    gradient_estimators=self._gradient_estimators,
                    current_position=self._current_position,
                    current_values=self._current_values)


class NelderMeadSolver(Solver):
    def __init__(self, target: pd.Series, simplex: Sequence[Tuple[pd.Series, pd.Series]], weights: pd.Series,
                 current_voltages: pd.Series):
        self.target = target
        self.simplex = list(simplex)
        self.weights = weights
        self.current_voltages = current_voltages

    def suggest_next_step(self) -> pd.Series:
        if len(self.simplex) < len(self.current_voltages) + 1:
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

