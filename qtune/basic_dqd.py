# qtune: Automated fine tuning and optimization
#
#   Copyright (C) 2019  Julian D. Teske and Simon S. Humpohl
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation version 3 of the License.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
################################################################################

# @email: julian.teske@rwth-aachen.de

import pandas as pd
import numpy as np
from typing import Tuple, Sequence
from qtune.experiment import Experiment, Measurement


class BasicDQD(Experiment):
    """
    The BasicDQD class implements the characteristics of a double quantum dot experiment. It saves the default scans
    which are useful for fine tuning any double quantum dot.
    """
    default_line_scan = Measurement('line_scan',
                                    center=0., range=3e-3, gate='RFA', N_points=1280, ramptime=.0005,
                                    N_average=3, AWGorDecaDAC='DecaDAC')
    default_detune_scan = Measurement('detune_scan',
                                      center=0., range=2e-3, N_points=100, ramptime=.02,
                                      N_average=10, AWGorDecaDAC='AWG')
    default_lead_scan = Measurement('lead_scan', gate='B', AWGorDecaDAC='DecaDAC')
    default_load_scan = Measurement("load_scan")

    def __init__(self):
        self.signal_strength = 0.

    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        return self.default_line_scan, self.default_detune_scan, self.default_lead_scan, self.default_load_scan


class Simulator:
    def __init__(self, simulation_function, **options):
        """
        This function simulates measurements for the TestExperiment class.
        :param parameters: List of names of parameters, which will be extracted by the evaluator
        :param simulation_function: function handle which will compute the Data
        """
        self._options = options
        self._simulation_function = simulation_function

    def simulate_measurement(self, gate_voltages, measurement):
        return self._simulation_function(gate_voltages, measurement, self._options)


class TestExperiment(Experiment):
    """
    Experiment designed for integration tests. Uses the Simulator class to simulate generic experiments.
    """
    def __init__(self, initial_voltages: pd.Series, measurements: Sequence[Measurement],
                 simulator_dict: dict):
        """

        :param initial_voltages: Voltages set before the tuning started.
        :param measurements: Measurements known to the Experiment.
        :param simulator_dict: Maps Measurements by their id to Simulators.
        """
        self._gate_voltages = initial_voltages

        self._measurements = measurements
        self._simulator_dict = simulator_dict
        for measurement in measurements:
            if id(measurement) not in simulator_dict.keys():
                print("There is no simulation function implemented for the measurement " + measurement.name)
                raise RuntimeError

    @property
    def simulator_dict(self):
        return self._simulator_dict

    def measurements(self):
        return self._measurements

    def gate_voltage_names(self):
        return self._gate_voltages.index

    def read_gate_voltages(self):
        return self._gate_voltages.sort_index()

    def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:
        for gate in new_gate_voltages.index:
            self._gate_voltages[gate] = new_gate_voltages[gate]
        return new_gate_voltages

    def measure(self, measurement: Measurement):
        simulator = self.simulator_dict[id(measurement)]
        return simulator.simulate_measurement(self.read_gate_voltages(), measurement)


def load_simulation(gate_voltages, measurement_options, simulation_options):
    load_time_noise = .0
    over_all_noise = 0.
    simulated_curvature = 15. + np.exp(- gate_voltages[simulation_options["gate1"]] - gate_voltages[
        simulation_options["gate2"]]) + load_time_noise * (np.random.rand(1)[0] - 0.5)
    ydata = np.exp(np.arange(0., 500, 5) * -1. / simulated_curvature) + over_all_noise * (
                np.random.rand(100) - .5)
    xdata = np.arange(0., 500, 5)
    return np.reshape(np.concatenate((ydata, xdata)), (2, 100))


def ss1d_simulation(gate_voltages, measurement: Measurement, simulation_options):
    points = np.arange(-4e-3, 4e-3, 8e-3 / 1280.)
    factor = 1 + gate_voltages['SDB1'] - gate_voltages['N']
    return (1 + 1e3 * gate_voltages['SDB1']) * factor * np.exp(-.5 * (points + gate_voltages['SDB2']) ** 2 / 2e-6)


def ss2d_simulation(gate_voltages, measurement: Measurement, simulation_options):
    gate1 = simulation_options["gate1"]
    gate2 = simulation_options["gate2"]
    x = np.linspace(start=-5. + 1e3 * gate_voltages[gate2], stop=5. + 1e3 * gate_voltages[gate2], num=104)
    y = np.linspace(start=-5. + 1e3 * gate_voltages[gate1], stop=5. + 1e3 * gate_voltages[gate1], num=20)
    xx, yy = np.meshgrid(x, y, sparse=True)
    return (1 + yy) * (1 - .1 * abs(xx - 1e3 * gate_voltages[gate2]) - .1 * abs(yy - 1e3 * gate_voltages[gate1])) * np.sin(xx + yy)


def detune_simulation(gate_voltages, measurement: Measurement, simulation_options):
    central_upper_gate = simulation_options["central_upper_gate"]
    central_lower_gate = simulation_options["central_lower_gate"]
    left_gate = simulation_options["left_gate"]
    right_gate = simulation_options["right_gate"]
    parameters = measurement.options.copy()
    parameters['file_name'] = "detune_scan_" + measurement.get_file_name()
    parameters['N_points'] = float(parameters['N_points'])
    parameters['N_average'] = float(parameters['N_average'])

    x = np.linspace(parameters["center"] - parameters["range"], parameters["center"] + parameters["range"],
                    parameters["N_points"])
    simulated_width = np.exp(gate_voltages[central_upper_gate] - gate_voltages[right_gate]) + np.exp(
        gate_voltages[central_lower_gate] - gate_voltages[left_gate]) + 190. + 0. * (
                              np.random.rand(1)[0] - 0.5)
    y = np.tanh(x / (simulated_width * 1e-6))
    return y


def transition_simulation(gate_voltages, measurement: Measurement, simulation_options):

    gate_lead = simulation_options["gate_lead"]
    gate_opposite = simulation_options["gate_opposite"]
    simulated_center = (gate_voltages[gate_lead] - gate_voltages[gate_opposite]) * 1e-3
    x = np.linspace(measurement.options["center"] - measurement.options["range"],
                    measurement.options["center"] + measurement.options["range"],
                    measurement.options['N_points'])
    simulated_center = simulated_center + 0. * (np.random.rand(1)[0] - 0.5)
    x = x - simulated_center
    return np.tanh(x / measurement.options["range"] * 5.) + 0.0 * np.random.rand(x.shape[0])
