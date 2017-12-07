import numpy as np
import pandas as pd
from typing import Tuple
from qtune.GradKalman import GradKalmanFilter
from qtune.Evaluator import Evaluator


class Solver:
    def __init__(self, gradient, desired_values: pd.Series, evaluators: Tuple(Evaluator, ...), gate_names):
        self.gradient = gradient
        self.gate_names = gate_names
        self.need_new_gradient = False
        self.evaluators = evaluators
        self.desired_values = desired_values.sort_index()
        self.parameter = pd.Series()
        for e in evaluators:
            self.parameter.append(e.parameters, verify_integrity=True)
        self.parameter.sort_index()
        assert(desired_values.index.tolist() == self.parameter.index.tolist())

    def suggest_next_step(self):
        raise NotImplementedError()

    def update_after_step(self, d_voltages_series: pd.Series):
        raise NotImplementedError()


class KalmanNewtonSolver(Solver):
    def __init__(self, evaluators: Tuple(Evaluator, ...), gradient, desired_values: pd.Series, gate_names, covariance=None,
                 noise=None, load_cov_noise=False, filename=None):
        if load_cov_noise:
            raise NotImplementedError
        super().__init__(gradient=gradient, desired_values=desired_values, evaluators=evaluators, gate_names=gate_names)
        self.evaluators = evaluators
        n_parameter, n_gates = gradient.shape()
        assert(len(evaluators) == n_parameter)
        assert(n_parameter == desired_values.size())
        self.grad_kalman = GradKalmanFilter(n_gates, n_parameter, initF=None, initX=gradient, initP=covariance,
                                            initR=noise, initQ=None)
        self.need_new_gradient = False

    def suggest_next_step(self):
        d_parameter_series = self.desired_values.add(-1.*self.parameter)
        d_parameter_series = d_parameter_series.sort_index()
        d_parameter_vector = np.asarray(d_parameter_series.values).T
        d_voltages_vector = np.linalg.solve(self.grad_kalman.grad, d_parameter_vector)
        d_voltages_series = pd.Series(d_voltages_vector, self.gate_names)
        return d_voltages_series

    def update_after_step(self, d_voltages_series: pd.Series) -> bool:
        current_parameter = self.parameter
        for e in self.evaluators:
            evaluation_result = e.evaluate
            if evaluation_result['failed']:
                return False
            evaluation_result.drop(['failed'])
            evaluated_parameters = evaluation_result.index.tolist()
            for i in evaluated_parameters:
                self.parameter[i] = evaluation_result[i]

        d_parameter_series = self.parameter.add(-1.*current_parameter)
        d_parameter_series = d_parameter_series.sort_index()
        d_parameter_vector = np.asarray(d_parameter_series.values)
        d_parameter_vector = d_parameter_vector.T
        d_voltages_series = d_voltages_series.sort_index()
        d_voltages_vector = np.asarray(d_voltages_series.values)
        d_voltages_vector = d_voltages_vector.T
        self.grad_kalman.update(d_voltages_vector, d_parameter_vector, hack=False)
        self.gradient = self.grad_kalman.grad







