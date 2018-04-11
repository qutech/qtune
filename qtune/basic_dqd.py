import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
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

    def tune_qpc(self, qpc_position=None, tuning_range=4e-3, gate='SDB2'):
        raise NotImplementedError()

    def tune_qpc_2d(self, tuning_range):
        raise NotImplementedError

    def read_qpc_voltage(self) -> pd.Series:
        raise NotImplementedError()


class BasicQQD(Experiment):

    def __init__(self):
        self.left_sensing_gates = ["LT", "LB"]
        self.right_sensing_gates = ["RT", "RB"]
        self.primarily_left_sensing_gate = "LT"
        self.primarily_right_sensing_gate = "RT"
        self.centralizing_gates = ["PA", "PB", "PC", "PD"]
        self.left_signal_strength = 0.
        self.right_signal_strength = 0.
        self.tunable_gates = ["SA", "NAB", "NBC", "NCD", "SD", "TAB", "TBC", "TCD"]
        self.tunable_gates.sort()

    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        return NotImplementedError

    def tune_sensing_dot_1d(self, prior_position, tuning_range, gate):
        raise NotImplementedError

    def tune_tune_sensing_dot_2d(self, side, tuning_range):
        raise NotImplementedError

    def read_sensing_dot_voltages(self):
        raise NotImplementedError


class TestDQD(BasicDQD):
    """
    A test version for dry runs and debugging. Experimental data is simulated.
    """
    def __init__(self, initial_voltages=None, gate_names=None):
        if initial_voltages is None:
            initial_voltages = [1., 1., 1., 1., 1., 1., 0., 0.]
        if gate_names is None:
            gate_names = ["SB", "BB", "T", "N", "SA", "BA", "RFA", "RFB"]
        self.gate_voltages = pd.Series(initial_voltages, gate_names)
        self.gate_voltages = self.gate_voltages.sort_index()
        self._qpc_tuned = False

    def gate_voltage_names(self):
        return self.gate_voltages.index.tolist()

    def read_gate_voltages(self):
        return self.gate_voltages.sort_index()

    def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:
        for gate in new_gate_voltages.index:
            self.gate_voltages[gate] = new_gate_voltages[gate]
        return new_gate_voltages

    def tune_qpc(self, qpc_position=None, tuning_range=4e-3):
        self._qpc_tuned = True
        return [], pd.Series(data=[[self.gate_voltages["SB"]]], index=["qpc"])

    def read_qpc_voltage(self):
        return pd.Series(data=[[self.gate_voltages["SB"]]], index=["qpc"])

    def measure(self,
                measurement: Measurement) -> np.ndarray:
        if not self._qpc_tuned:
            self.tune_qpc()

        if measurement == 'line_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "line_scan" + measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            if parameters["gate"] == "RFA":
                simulated_center = 0.75 * (self.gate_voltages["SA"] - self.gate_voltages["BA"]) + 500. * (
                    self.gate_voltages["SA"] - self.gate_voltages["BA"]) ** 2 + 2. * (
                    self.gate_voltages["T"] - 1.) - 0.5 * (self.gate_voltages["N"] - 1.) + 0.2 * (
                    self.gate_voltages["SB"] - self.gate_voltages["BB"])

            elif parameters["gate"] == "RFB":
                simulated_center = self.gate_voltages["SB"] - self.gate_voltages["BB"] + 500. * (
                    self.gate_voltages["SB"] - self.gate_voltages["BB"]) ** 2 + 2. * (
                    self.gate_voltages["T"] - 1.) - 0.5 * (self.gate_voltages["N"] - 1.) + 0.2 * (
                    self.gate_voltages["SA"] - self.gate_voltages["BA"])
            else:
                raise ValueError("The gate in the measurement must be RFA or RFB!")
            x = np.linspace(parameters["center"] - parameters["range"], parameters["center"] + parameters["range"],
                            parameters["N_points"])
            simulated_center = simulated_center + 0.0002 * (np.random.rand(1)[0] - 0.5)
            x = x - simulated_center
            x = x
            y = np.tanh(x/parameters["range"]*5.) + 0.05 * np.random.rand(x.shape[0])

            return y
        elif measurement == 'detune_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "detune_scan_" + measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            x = np.linspace(parameters["center"] - parameters["range"], parameters["center"] + parameters["range"],
                            parameters["N_points"])
            simulated_width = 1./20.*(self.gate_voltages["T"] + self.gate_voltages["N"] - 2.) + 190e-6 + 70e-6 * (
                np.random.rand(1)[0] - 0.5)
            y = np.tanh(x/simulated_width)

            return y
        elif measurement == 'lead_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "lead_scan" + measurement.get_file_name()
            simulated_rise_time = .2 * (self.gate_voltages["SA"] - 1.) + 0.075 * (
                self.gate_voltages["T"] - 1.) + 0.2 + 0.00005 * (np.random.rand(1)[0] - 0.5)
            simulated_fall_time = .15 * (self.gate_voltages["SA"] - 1.) + .05 * (
                self.gate_voltages["T"] - 1.) + 0.2 + 0.00005 * (np.random.rand(1)[0] - 0.5)
            x = np.asarray(range(400)) / 100.
            y = np.zeros((400, ))
            for i in range(200):
                y[i] = (np.cosh(2. / 2. / simulated_rise_time) - np.exp(
                    (2. - 2. * x[i]) / 2. / simulated_rise_time)) / np.sinh(2. / 2. / simulated_rise_time)
            for i in range(200, 400):
                y[i] = -1.*(np.cosh(2. / 2. / simulated_fall_time) - np.exp(
                    (6. - 2. * x[i]) / 2. / simulated_fall_time)) / np.sinh(2. / 2. / simulated_fall_time)
            y0 = np.zeros(shape=y.shape)
            return np.concatenate((y0, y), axis=0)
        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))