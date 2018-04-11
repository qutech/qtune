import numpy as np
import pandas as pd
from typing import Tuple, Optional
from qtune.kalman_gradient import KalmanGradient
from qtune.evaluator import Evaluator


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

    def suggest_next_step(self) -> pd.Series:
        raise NotImplementedError()

    def update_after_step(self, voltages: pd.Series, measured_parameters: Optional[pd.Series]):
        raise NotImplementedError()


class KalmanSolver(Solver):
    """
    Solver using the Kalman filter to update the gradient.
    """
    def __init__(self, gate_names=None, gradient=None, evaluators: Tuple[Evaluator, ...] = (),
                 desired_values: pd.Series = pd.Series(), covariance=None, noise=None, alpha=1.02,
                 shifting_uncertainty=None):
        super().__init__(gate_names=gate_names, gradient=gradient, desired_values=desired_values, evaluators=evaluators)
        self.covariance = covariance
        self.noise = noise
        self.shifting_uncertainty = shifting_uncertainty
        if gradient is not None:
            n_parameter, n_gates = gradient.shape()
            self.grad_kalman = KalmanGradient(n_gates=n_gates,
                                              n_params=n_parameter,
                                              initial_gradient=gradient,
                                              initial_covariance_matrix=covariance,
                                              measurement_covariance_matrix=noise,
                                              alpha=alpha)

    def initialize_kalman(self, gradient=None, covariance=None, noise=None, shifting_uncertainty=None, alpha=1.02):
        if gradient is None:
            gradient = self.gradient
        n_parameters, n_gates = gradient.shape
        self.grad_kalman = KalmanGradient(n_gates=n_gates,
                                          n_params=n_parameters,
                                          initial_gradient=gradient,
                                          initial_covariance_matrix=covariance,
                                          measurement_covariance_matrix=noise,
                                          process_noise=shifting_uncertainty,
                                          alpha=alpha)
        self.gradient = gradient

    def update_after_step(self, d_voltages_series: pd.Series, d_parameter_series: pd.Series = None,
                          residuals_series: pd.Series = None):
        if d_parameter_series is None:
            current_parameter = self.parameter
            for e in self.evaluators:
                evaluation_result = e.evaluate()
                if evaluation_result['failed']:
                    return self.grad_kalman.grad, self.grad_kalman.cov, True
                evaluation_result.drop(['failed'])
                evaluated_parameters = evaluation_result.index.tolist()
                for i in evaluated_parameters:
                    if i != "residual":
                        self.parameter[i] = evaluation_result[i]
                self.parameter = self.parameter.sort_index()

            d_parameter_series = self.parameter.add(-1.*current_parameter)
        d_parameter_series = d_parameter_series.sort_index()
        d_parameter_vector = np.asarray(d_parameter_series.values)
        d_parameter_vector = d_parameter_vector.T
        d_voltages_series = d_voltages_series.sort_index()
        d_voltages_vector = np.asarray(d_voltages_series.values)
        d_voltages_vector = d_voltages_vector.T

        r = np.copy(self.grad_kalman.filter.R)
        residuals = residuals_series.sort_index()
        for j in range(self.parameter.size):
            r[j, j] = 0.1 * r[j, j] + residuals[j]

        self.grad_kalman.update(d_voltages_vector, d_parameter_vector, measurement_covariance=r, hack=False)
        self.gradient = self.grad_kalman.grad
        return self.grad_kalman.grad, self.grad_kalman.cov, False


class KalmanNewtonSolver(KalmanSolver):
    """
    Uses the Newton algorithm to compute the voltage steps and the Kalman filter to update the gradient.
    """
    def __init__(self, gate_names=None, gradient=None, evaluators: Tuple[Evaluator, ...] = (),
                 desired_values: pd.Series = pd.Series(), covariance=None, noise=None, alpha=1.02,
                 shifting_uncertainty=None):
        super().__init__(gate_names=gate_names, gradient=gradient, evaluators=evaluators, desired_values=desired_values,
                         covariance=covariance, noise=noise, alpha=alpha, shifting_uncertainty=shifting_uncertainty)

    def suggest_next_step(self) -> pd.Series:
        if not self.check_dimensionality():
            print('The internal dimensionality is not consistent! Cant predict next Step! Abort mission!')
            return pd.Series()
        d_parameter_series = self.desired_values.add(-1.*self.parameter)
        d_parameter_series = d_parameter_series.sort_index()
        d_parameter_vector = np.asarray(d_parameter_series.values).T
        d_voltages_vector = np.linalg.lstsq(self.grad_kalman.grad, d_parameter_vector)[0]
        d_voltages_series = pd.Series(d_voltages_vector, self.gate_names)
        return d_voltages_series
