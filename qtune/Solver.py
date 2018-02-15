import numpy as np
import pandas as pd
from typing import Tuple
from qtune.GradKalman import GradKalmanFilter
from qtune.Evaluator import Evaluator


class Solver:
    """
    The solver class implements an algorithm to minimise the difference of the parameters to the target values.
    """
    def __init__(self, gate_names=None, gradient=None, evaluators: Tuple[Evaluator, ...] = (),
                 desired_values: pd.Series = pd.Series()):
        self.gradient = gradient
        self.gate_names = gate_names
        self.need_new_gradient = False
        self.evaluators = ()
        self.desired_values = desired_values.sort_index()
        self.parameter = pd.Series()
        for e in evaluators:
            self.add_evaluator(e)
        self.parameter.sort_index()

    def check_dimensionality(self) -> bool:
        if self.gradient.shape != (len(self.parameter.index.tolist()), len(self.gate_names)):
            print('The gradients shape', self.gradient.shape, 'doesnt match the number of parameters',
                  self.parameter.index.tolist(), 'and gate names', len(self.gate_names), '!')
            return False
        elif self.desired_values.index.tolist() != self.parameter.index.tolist():
            print('The desired values', self.desired_values.index.tolist(), 'do not match the parameters',
                  self.parameter.index.tolist(), '!')
            return False
        else:
            return True

    def add_evaluator(self, evaluator: Evaluator):
        try:
            self.parameter.append(evaluator.parameters, verify_integrity=True)
        except ValueError:
            print('A parameter of evaluator', evaluator, 'is already evaluated by another evaluator!')
            return
        self.evaluators += (evaluator, )

    def suggest_next_step(self):
        raise NotImplementedError()

    def update_after_step(self, d_voltages_series: pd.Series):
        raise NotImplementedError()


class KalmanSolver(Solver):
    """
    Solver using the Kalman filter to update the gradient.
    """
    def __init__(self, gate_names=None, gradient=None, evaluators: Tuple[Evaluator, ...] = (),
                 desired_values: pd.Series = pd.Series(), covariance=None, noise=None, alpha=1.02):
        super().__init__(gate_names=gate_names, gradient=gradient, desired_values=desired_values, evaluators=evaluators)
        self.covariance = covariance
        self.noise = noise
        if gradient is not None:
            n_parameter, n_gates = gradient.shape()
            self.grad_kalman = GradKalmanFilter(nGates=n_gates, nParams=n_parameter, initF=None, initX=gradient,
                                                initP=covariance, initR=noise, initQ=None, alpha=alpha)

    def initialize_kalman(self, gradient=None, covariance=None, noise=None, alpha=1.02):
        if gradient is None:
            gradient = self.gradient
        n_parameters, n_gates = gradient.shape
        self.grad_kalman = GradKalmanFilter(nGates=n_gates, nParams=n_parameters, initX=gradient, initP=covariance,
                                            initR=noise, alpha=alpha)
        self.gradient = gradient

    def update_after_step(self, d_voltages_series: pd.Series, d_parameter_series: pd.Series=None):
        if d_parameter_series is None:
            current_parameter = self.parameter
            for e in self.evaluators:
                evaluation_result = e.evaluate()
                if evaluation_result['failed']:
                    return self.grad_kalman.grad, self.grad_kalman.cov, True
                evaluation_result.drop(['failed'])
                evaluated_parameters = evaluation_result.index.tolist()
                for i in evaluated_parameters:
                    self.parameter[i] = evaluation_result[i]
                self.parameter = self.parameter.sort_index()

            d_parameter_series = self.parameter.add(-1.*current_parameter)
        d_parameter_series = d_parameter_series.sort_index()
        d_parameter_vector = np.asarray(d_parameter_series.values)
        d_parameter_vector = d_parameter_vector.T
        d_voltages_series = d_voltages_series.sort_index()
        d_voltages_vector = np.asarray(d_voltages_series.values)
        d_voltages_vector = d_voltages_vector.T
        self.grad_kalman.update(d_voltages_vector, d_parameter_vector, hack=False)
        self.gradient = self.grad_kalman.grad
        return self.grad_kalman.grad, self.grad_kalman.cov, False


class KalmanNewtonSolver(KalmanSolver):
    """
    Uses the Newton algorithm to compute the voltage steps and the Kalman filter to update the gradient.
    """
    def __init__(self, gate_names=None, gradient=None, evaluators: Tuple[Evaluator, ...] = (),
                 desired_values: pd.Series = pd.Series(), covariance=None, noise=None, alpha=1.02):
        super().__init__(gate_names=gate_names, gradient=gradient, evaluators=evaluators, desired_values=desired_values,
                         covariance=covariance, noise=noise, alpha=alpha)

    def suggest_next_step(self) -> pd.Series:
        if not self.check_dimensionality():
            print('The internal dimensionality is not consistent! Cant predict next Step! Abort mission!')
            return
        d_parameter_series = self.desired_values.add(-1.*self.parameter)
        d_parameter_series = d_parameter_series.sort_index()
        d_parameter_vector = np.asarray(d_parameter_series.values).T
        d_voltages_vector = np.linalg.lstsq(self.grad_kalman.grad, d_parameter_vector)[0]
        d_voltages_series = pd.Series(d_voltages_vector, self.gate_names)
        return d_voltages_series







