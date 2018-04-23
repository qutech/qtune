from typing import Optional

import numpy as np
import pandas as pd

from qtune.kalman_gradient import KalmanGradient
from qtune.storage import HDF5Serializable
from qtune.util import get_orthogonal_vector, calculate_gradient_non_orthogonal

__all__ = ["GradientEstimator", "FiniteDifferencesGradientEstimator", "KalmanGradientEstimator"]


class GradientEstimator(metaclass=HDF5Serializable):
    """Estimate the gradient of a scalar function"""

    def change_position(self, new_position: pd.Series):
        raise NotImplementedError()

    def estimate(self) -> pd.Series:
        raise NotImplementedError()

    def require_measurement(self) -> Optional[pd.Series]:
        raise NotImplementedError()

    def update(self, voltages: pd.Series, value: float, covariance: float, is_new_position=False):
        raise NotImplementedError()

    def to_hdf5(self):
        raise NotImplementedError()


class FiniteDifferencesGradientEstimator(GradientEstimator):
    def __init__(self,
                 current_position: pd.Series,
                 epsilon: pd.Series, symmetric: bool=False,
                 current_estimate=None,
                 variance=None,
                 stored_measurements=None,
                 requested_measurements=None):
        self._current_position = current_position.sort_index()

        self._epsilon = epsilon.sort_index()

        self._current_estimate = current_estimate
        self._variance = variance

        self._stored_measurements = stored_measurements or []

        # only for debugging purposes
        self._requested_measurements = requested_measurements or []

        self._symmetric_calculation = symmetric

    def change_position(self, new_position: pd.Series):
        self._current_position[new_position.index] = new_position

    def estimate(self) -> pd.Series:
        return self._current_estimate

    def require_measurement(self) -> pd.Series:
        if self._current_estimate is None:
            if self._symmetric_calculation:
                measured_points = [v for v, *_ in self._stored_measurements]
                if len(measured_points) % 2 == 1:
                    self._requested_measurements.append(2*self._current_position - measured_points[-1][0])
                elif measured_points:
                    measured_points = [v2 - v1 for v2, v1 in zip(measured_points[1::2],
                                                                 measured_points[::2])]
                    self._requested_measurements.append(self._current_position + get_orthogonal_vector(measured_points) * self._epsilon)
            else:
                measured_points = [v for v, *_ in self._stored_measurements]
                if len(measured_points) == 0:
                    ov = get_orthogonal_vector([np.zeros_like(self._current_position.index, dtype=float)])
                    self._requested_measurements.append(self._current_position + ov)
                elif len(measured_points) < self._current_position.size:
                    ov = get_orthogonal_vector(measured_points)
                    self._requested_measurements.append(self._current_position + ov)
                else:
                    self._requested_measurements.append(self._current_position)
            return self._requested_measurements[-1]

    def update(self, voltages: pd.Series, value: float, covariance: pd.Series, is_new_position=False):
        if self._current_estimate is None:
            self._stored_measurements.append((voltages, value, covariance))

            positions, values, variances = zip(*self._stored_measurements)

            if ((self._symmetric_calculation and len(self._stored_measurements) == 2*self._current_position.size) or
                not (self._symmetric_calculation and len(self._stored_measurements) == self._current_position.size + 1)):
                    self._current_estimate, self._variance = calculate_gradient_non_orthogonal(positions=positions,
                                                                                               values=values,
                                                                                               variances=variances)

    def to_hdf5(self):
        return dict(current_position=self._current_position,
                    epsilon=self._epsilon,
                    current_estimate=self._current_estimate,
                    variance=self._variance,
                    stored_measurements=self._stored_measurements,
                    symmetric=self._symmetric_calculation,
                    requested_measurements=self._requested_measurements)


class KalmanGradientEstimator(GradientEstimator):
    def __init__(self, kalman_gradient: KalmanGradient, current_position: pd.Series, current_value: float,
                 maximum_covariance: float):
        self._kalman_gradient = kalman_gradient
        self._current_position = pd.Series(current_position)
        self._current_value = current_value

        self._maximum_covariance = maximum_covariance

    def to_hdf5(self):
        return dict(kalman_gradient=self._kalman_gradient,
                    current_position=self._current_position,
                    current_value=self._current_value,
                    maximum_covariance=self._maximum_covariance)

    def change_position(self, new_position: pd.Series):
        """
        :param new_position:
        :return:
        """
        self._current_position = new_position[self._current_position.index]

    def estimate(self) -> pd.Series:
        return pd.Series(self._kalman_gradient.grad, index=self._current_position.index)

    def require_measurement(self):
        if self._maximum_covariance:
            if np.any(np.linalg.eigvals(self._kalman_gradient.cov) > self._maximum_covariance):
                return self._current_position + self._kalman_gradient.sugg_diff_volts

    def update(self, voltages: pd.Series,
               value: float,
               covariance: float,
               is_new_position=False):
        diff_volts = (voltages - self._current_position)
        diff_param = (value - self._current_value)

        self._kalman_gradient.update(diff_volts=diff_volts,
                                     diff_params=[diff_param],
                                     measurement_covariance=covariance)

        if is_new_position:
            self._current_value = value
            self._current_position[:] = voltages
