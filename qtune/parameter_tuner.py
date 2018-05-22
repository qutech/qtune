from typing import Sequence, List, Tuple

import numpy as np
import pandas as pd

from qtune.evaluator import Evaluator
from qtune.solver import Solver
from qtune.storage import HDF5Serializable


class ParameterTuner(metaclass=HDF5Serializable):
    """This class tunes a specific set of parameters which are defined by the given evaluators."""

    def __init__(self,
                 evaluators: Sequence[Evaluator],
                 solver: Solver,
                 tuned_voltages=None,
                 last_voltage=None,
                 last_parameter_values=None):
        self._tuned_voltages = tuned_voltages or []

        self._last_voltage = last_voltage
        if last_parameter_values is None:
            self._last_parameter_values = pd.Series(index=solver.target.index)
        else:
            self._last_parameter_values = last_parameter_values

        self._evaluators = tuple(evaluators)

        parameters = sorted(parameter
                            for evaluator in self._evaluators
                            for parameter in evaluator.parameters)
        if len(parameters) != len(set(parameters)):
            raise ValueError('Parameter duplicates: ', {p for p in parameters if parameters.count(p) > 1})

        self._solver = solver

        assert set(self.solver.target.index) == set(parameters)

    @property
    def solver(self) -> Solver:
        return self._solver

    @property
    def target(self) -> pd.DataFrame:
        return self.solver.target

    @property
    def parameters(self) -> Sequence[str]:
        """Alphabetically sorted parameters"""
        return self.solver.target.index

    @property
    def tuned_voltages(self) -> List[pd.Series]:
        """A list of the positions where the parameter set was successfully tuned."""
        return self._tuned_voltages

    def evaluate(self) -> Tuple[pd.Series, pd.Series]:
        #  no list comprehension for easier debugging
        parameters = []
        variances = []
        for evaluator in self._evaluators:
            parameter, variance = evaluator.evaluate()
            parameters.append(parameter)
            variances.append(variance)
        return pd.concat(parameters)[self.parameters], pd.concat(variances)[self.parameters]

    def is_tuned(self, voltages: pd.Series) -> bool:
        """Tell the tuner, the voltages have changed and that he might have to re-tune.

        :param voltages: Return if tuning condition is met
        :return:
        """
        raise NotImplementedError()

    def get_next_voltages(self) -> pd.Series:
        """The next voltage in absolute values.

        :return:
        """
        raise NotImplementedError()

    def to_hdf5(self) -> dict:
        return dict(evaluators=self._evaluators,
                    solver=self._solver,
                    tuned_voltages=self._tuned_voltages,
                    last_voltage=self._last_voltage,
                    last_parameter_values=self._last_parameter_values)


class SubsetTuner(ParameterTuner):
    """This tuner uses only a subset of gates to tune the parameters"""

    def __init__(self, evaluators: Sequence[Evaluator], gates: Sequence[str],
                 **kwargs):
        """
        :param evaluators:
        :param gates: Gates which are used tu tune the parameters
        :param tuned_voltages:
        :param last_voltage:
        :param last_parameter_values:
        """
        super().__init__(evaluators, **kwargs)

        self._gates = sorted(gates)

    def is_tuned(self, voltages: pd.Series):
        current_parameters, current_variances = self.evaluate()

        solver_voltages = voltages[self._gates]

        self._solver.update_after_step(solver_voltages, current_parameters, current_variances)

        self._last_voltage = voltages
        self._last_parameter_values = current_parameters[self._last_parameter_values.index]
        if ((self.target.desired - current_parameters).abs() < self.target['tolerance']).all():
            self._tuned_voltages.append(voltages)
            return True
        else:
            return False

    def get_next_voltages(self):
        solver_voltage = self._solver.suggest_next_position()
        result = pd.Series(self._last_voltage)

        result[solver_voltage.index] = solver_voltage

        return result

    def to_hdf5(self):
        parent_dict = super().to_hdf5()
        return dict(parent_dict, gates=self._gates)


class SensingDotTuner(ParameterTuner):
    """
    This tuner directly tunes to voltage points of interest. The evaluators return positions
    """

    def __init__(self, cheap_evaluators: Sequence[Evaluator], expensive_evaluators: Sequence[Evaluator],
                 gates: Sequence[str], min_threshold, cost_threshold, **kwargs):
        """

        :param cheap_evaluators:
        :param expensive_evaluators:
        :param gates:
        :param min_threshhold: If the parameters are below this threshold, the experiment is not tuned. This doesnt
        regard the optimal signal found but only the current one.
        :param cost_threshhold: If the parameters are below this threshold, the expensive parameters will be used.
        :param kwargs:
        """
        super().__init__(cheap_evaluators, **kwargs)
        self._gates = sorted(gates)
        self._cheap_evaluators = cheap_evaluators
        self._expensive_evaluators = expensive_evaluators
        self._min_threshold = min_threshold
        self._cost_threshold = cost_threshold

    def is_tuned(self, voltages: pd.Series):
        current_parameter, variances = self.evaluate(cheap=True)
        solver_voltages = voltages[self._gates]
        self._last_voltage = voltages
        for key in self._last_parameter_values.index:
            if key in current_parameter.index:
                self._last_parameter_values[key] = current_parameter[key]

        if current_parameter.le(self._min_threshold).any():
            if current_parameter.le(self._cost_threshold).any:
                current_parameter, variances = self.evaluate(cheap=False)

            self.solver.update_after_step(solver_voltages, current_parameter, variances)
            return False
        else:
            self._tuned_voltages.append(voltages)
            return True

    def get_next_voltages(self):
        solver_step = self._solver.suggest_next_position()

        return self._last_voltage.add(solver_step, fill_value=0)

    def evaluate(self, **kwargs) -> (pd.Series, pd.Series):
        #  no list comprehension for easier debugging
        cheap = kwargs["cheap"]
        parameters = []
        variances = []
        if cheap:
            for evaluator in self._cheap_evaluators:
                parameter, variance = evaluator.evaluate()
                parameters.append(parameter)
                variances.append(variance)
        else:
            for evaluator in self._expensive_evaluators:
                parameter, variance = evaluator.evaluate()
                parameters.append(parameter)
                variances.append(variance)
        return pd.concat(parameters).sort_index(), pd.concat(variances).sort_index()

    def to_hdf5(self):
        return dict(cheap_evaluators=self._cheap_evaluators,
                    expensive_evaluators=self._expensive_evaluators,
                    gates=self._gates,
                    min_threshold=self._min_threshold,
                    cost_threshold=self._cost_threshold,
                    solver=self.solver,
                    last_parameter_values=self._last_parameter_values)
