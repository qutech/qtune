from typing import Optional

import numpy as np
import pandas as pd

from qtune.kalman_gradient import KalmanGradient
from qtune.storage import HDF5Serializable

__all__ = ["GradientEstimator", "FiniteDifferencesGradientEstimator", "KalmanGradientEstimator"]


class GradientEstimator(metaclass=HDF5Serializable):
    """Estimate the gradient of a scalar function"""

    def change_position(self, new_position: pd.Series):
        raise NotImplementedError()

    def estimate(self) -> pd.Series:
        raise NotImplementedError()

    def require_measurement(self) -> Optional[pd.Series]:
        raise NotImplementedError()

    def update(self, voltages: pd.Series, value: float, covariance: pd.Series, is_new_position=False):
        raise NotImplementedError()

    def to_hdf5(self):
        raise NotImplementedError()


class FiniteDifferencesGradientEstimator(GradientEstimator):
    def __init__(self, current_position: pd.Series, parameters: pd.Series, epsilon: pd.Series, symmetric=False):
        self._current_position = current_position.sort_index()
        self._current_parameters = parameters.sort_index()

        self._valid_distance = pd.Series(0, index=self._current_position.index)
        self._epsilon = epsilon.sort_index()

        self._current_estimate = None

        self._required_measurements = None

        self._symmetric_calculation = symmetric

    def change_position(self, new_position: pd.Series):
        if self._current_position and ((self._current_position - new_position).abs() < self._valid_distance).all():
            pass
        else:
            self._current_position[:] = new_position
            self._required_measurements = pd.DataFrame()

    def estimate(self):
        return self._current_estimate

    def require_measurement(self) -> pd.Series:
        if not self._required_measurements and self._current_estimate is None:
            self._required_measurements = []
            if self._symmetric_calculation:
                for position in self._current_position:
                    self._required_measurements.append(
                        self._current_position.add(pd.Series(data=[self._epsilon], index=[position]), fill_value=0.))
                    self._required_measurements.append(
                        self._current_position.add(pd.Series(data=[-1. * self._epsilon], index=[position]),
                                                   fill_value=0.))
            else:
                for position in self._current_position:
                    self._required_measurements.append(
                        self._current_position.add(pd.Series(data=[self._epsilon], index=[position]), fill_value=0.))
                self._required_measurements.append(self._current_position)
        if self._required_measurements:
            return self._required_measurements.pop()

    def update(self, voltages: pd.Series, value: float, covariance: pd.Series, is_new_position=False):
        """Is this even possible?"""
        raise NotImplementedError()

    def to_hdf5(self):
        raise NotImplementedError()


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
