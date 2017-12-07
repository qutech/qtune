import pandas as pd
import numpy as np
from typing import Tuple
from qtune.experiment import Experiment
from qtune.Evaluator import Evaluator
from qtune.Solver import Solver


class Autotuner:
    def __init__(self, experiment: Experiment, solver: Solver = None, evaluators: Tuple(Evaluator, ...) = (),
                 desired_values: pd.Series = pd.Series()):
        self.parameters = pd.Series()
        self.solver = solver
        self.experiment = experiment
        for e in evaluators:
            self.add_evaluator(e)
        self.desired_values = desired_values;

    @property
    def evaluators(self):
        return self.evaluators

    @evaluators.setter
    def evaluators(self, evaluators: Tuple(Evaluator, ...)):
        self.evaluators = evaluators

    @property
    def desired_values(self):
        return self.desired_values

    @desired_values.setter
    def desired_values(self, desired_values: pd.Series):
        assert(desired_values.index.tolist() == self.parameters.index.tolist())
        self.desired_values = desired_values

    def add_evaluator(self, new_evaluator: Evaluator):
        new_parameters = new_evaluator.parameters
        for i in new_parameters.index.tolist():
            if i in self.parameters:
                print('This Evaluator determines a parameter which is already being evaluated by another Evaluator')
                return
            self.parameters.append(i, verify_integrity=True)
        self.evaluators += (new_evaluator,)

    def evaluate_gradient(self, delta_u=4e-3, n_repetitions=3) -> pd.DataFrame:
        gradient = pd.DataFrame()
        for gate in self.experiment.gate_voltage_names:
            positive_detune_parameter = pd.Series()
            negative_detune_parameter = pd.Series()

            for i in self.parameters.index.tolist():
                positive_detune_parameter[i] = np.zeros((n_repetitions,), dtype=float)
                negative_detune_parameter[i] = np.zeros((n_repetitions,), dtype=float)

            current_gate_positions = self.experiment.read_gate_voltages()
            detuning = pd.Series(delta_u, gate)

            for i in range(1,n_repetitions):
                new_gate_positions = current_gate_positions.add(detuning, fill_value=0)
                self.experiment.set_gate_voltages(new_gate_positions)

                for e in self.evaluators:
                    evaluation_result = e.evaluate()

                    if evaluation_result['failed']:
                        evaluation_result = evaluation_result.drop(['failed'])
                        for r in evaluation_result.index.tolist():
                            evaluation_result[r] = np.nan
                    else:
                        evaluation_result = evaluation_result.drop(['failed'])

                    for r in evaluation_result.index.tolist():
                        positive_detune_parameter[r][i] = evaluation_result[r]

                new_gate_positions = current_gate_positions.add(-1.*detuning, fill_value=0)
                self.experiment.set_gate_voltages(new_gate_positions)

                for e in self.evaluators:
                    evaluation_result = e.evaluate()

                    if evaluation_result['failed']:
                        evaluation_result = evaluation_result.drop(['failed'])
                        for r in evaluation_result.index.tolist():
                            evaluation_result[r] = np.nan
                    else:
                        evaluation_result = evaluation_result.drop(['failed'])

                    for r in evaluation_result.index.tolist():
                        negative_detune_parameter[r][i] = evaluation_result[r]

            for i in self.parameters.index.tolist():
                positive_detune_parameter[i] = np.nanmean(positive_detune_parameter[i])
                negative_detune_parameter[i] = np.nanmean(negative_detune_parameter[i])
            gradient_column = (positive_detune_parameter.add(-1.*negative_detune_parameter))/2./delta_u
            gradient = gradient_column.join(gradient)
        return gradient

    def set_desired_values_manually(self):
        raise NotImplementedError

    def autotune(self):
        raise NotImplementedError


class ChargeDiagramAutotuner(Autotuner):
    def __init__(self, experiment: Experiment, solver: Solver = None, evaluators: Tuple(Evaluator, ...) = (),
                 desired_values: pd.Series = pd.Series()):
        super().__init__(experiment=experiment, solver=solver, evaluators=evaluators, desired_values=desired_values)
