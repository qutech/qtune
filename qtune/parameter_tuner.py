from typing import Sequence

import numpy as np
import pandas as pd

from qtune.evaluator import Evaluator
from qtune.solver import Solver


class ParameterTuner:
    """This class tunes a specific set of parameters which are defined by the given evaluators."""

    def __init__(self, evaluators: Sequence[Evaluator],
                 desired_values: pd.Series,
                 tolerances: pd.Series,
                 solver: Solver,
                 tuned_positions=None, last_voltage=None, last_parameter_values=None):
        self._tuned_positions = tuned_positions or []

        self._last_voltage = last_voltage
        self._last_parameter_values = last_parameter_values

        self._evaluators = tuple(evaluators)

        parameters = sorted(parameter
                            for evaluator in self._evaluators
                            for parameter in evaluator.parameters)
        if len(parameters) != len(set(parameters)):
            raise ValueError('Parameter duplicates: ', {p for p in parameters if parameters.count(p) > 1})

        if not isinstance(desired_values, pd.Series):
            desired_values = pd.Series(desired_values, index=parameters)
        self._desired_values = desired_values[parameters]

        if not isinstance(tolerances, pd.Series):
            tolerances = pd.Series(tolerances, index=parameters)
        self._tolerances = tolerances[parameters]

        self._solver = solver

        self._solver.target = self._desired_values

    @property
    def solver(self) -> Solver:
        return self._solver

    @property
    def desired_values(self) -> pd.Series:
        return self._desired_values

    @desired_values.setter
    def desired_values(self, val: pd.Series):
        if isinstance(val, pd.Series):
            assert set(val.index) == set(self.parameters)
        self._desired_values[:] = val
        self._solver.target = self._desired_values

    @property
    def tolerances(self) -> pd.Series:
        return self._tolerances

    @tolerances.setter
    def tolerances(self, val):
        if isinstance(val, pd.Series):
            assert set(val.index) == set(self.parameters)
        self._tolerances[:] = val

    @property
    def parameters(self) -> Sequence[str]:
        """Alphabetically sorted parameters"""
        return self.desired_values.index

    @property
    def tuned_positions(self) -> List[pd.Series]:
        """A list of the positions where the parameter set was successfully tuned."""
        return self._tuned_positions

    def evaluate(self) -> pd.Series:
        #  no list comprehension for easier debugging
        values = []
        for evaluator in self._evaluators:
            values.append(evaluator.evaluate())
        return pd.concat(values).sort_index()

    def is_tuned(self, voltages: pd.Series) -> bool:
        """Tell the tuner, the voltages have changed and that he might have to re-tune.

        :param voltages: Return if tuning condition is met
        :return:
        """
        raise NotImplementedError()

    def get_next_voltage(self) -> pd.Series:
        """The next voltage in absolute values.

        :return:
        """
        raise NotImplementedError()


class SubsetTuner(ParameterTuner):
    """This tuner uses only a subset of gates to tune the parameters"""

    def __init__(self, evaluators: Sequence[Evaluator], gates: Sequence[str],
                 **kwargs):
        """
        :param evaluators:
        :param gates: Gates which are used tu tune the parameters
        :param tuned_positions:
        :param last_voltage:
        :param last_parameter_values:
        """
        super().__init__(evaluators, **kwargs)

        self._gates = sorted(gates)

    def is_tuned(self, voltages: pd.Series):
        current_values = self.evaluate()

        solver_voltages = voltages[self._gates]

        self._solver.update_after_step(solver_voltages, current_values)

        self._last_voltage = voltages

        if ((self.desired_values - current_values).abs() < self.tolerances).all():
            self._tuned_positions.append(voltages)
            return True
        else:
            return False

    def get_next_voltages(self):
        solver_step = self._solver.suggest_next_step()

        return self._last_voltage.add(solver_step, fill_value=0)
