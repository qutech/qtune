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

# TODO QQB or general QDArray? Move to other file
class BasicQQD(Experiment):

    def __init__(self):
        # Do wen want this hardcoded list here? Its not really used
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
            initial_voltages = [2., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        if gate_names is None:
            gate_names = ["BA", "BB", "N", "RFA", "RFB", "SA", "SB", "SDB1", "SDB2", "T"]
        self._gate_voltages = pd.Series(initial_voltages, gate_names)
        self._gate_voltages = self._gate_voltages.sort_index()

    def gate_voltage_names(self):
        return self._gate_voltages.index.tolist()

    def read_gate_voltages(self):
        return self._gate_voltages.sort_index()

    def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:
        for gate in new_gate_voltages.index:
            self._gate_voltages[gate] = new_gate_voltages[gate]
        print("new Voltages:")
        print(self._gate_voltages)
        return new_gate_voltages

    def measure(self, measurement: Measurement) -> np.ndarray:

        if measurement == 'line_scan':
            if measurement.parameter["gate"] == "SDB2":
                points = np.arange(-4e-3, 4e-3, 8e-3 / 1280.)
                return np.exp(-.5 * points**2 / 2e-6)
            elif measurement.parameter["gate"] == "RFA" or measurement.parameter["gate"] == "RFB":
                parameters = measurement.parameter.copy()
                parameters['file_name'] = "line_scan" + measurement.get_file_name()
                parameters['N_points'] = float(parameters['N_points'])
                parameters['N_average'] = float(parameters['N_average'])
                if parameters["gate"] == "RFA":
                    simulated_center = (self._gate_voltages["SA"] - self._gate_voltages["BA"]) * 1e3
                elif parameters["gate"] == "RFB":
                    simulated_center = (self._gate_voltages["SB"] - self._gate_voltages["BB"]) * 1e3
                else:
                    raise ValueError("The gate in the measurement must be RFA or RFB!")
                x = np.linspace(parameters["center"] - parameters["range"], parameters["center"] + parameters["range"],
                                parameters["N_points"])
                simulated_center = simulated_center + 0. * (np.random.rand(1)[0] - 0.5)
                x = x - simulated_center
                return np.tanh(x/parameters["range"]*5.) + 0.0 * np.random.rand(x.shape[0])

        elif measurement == 'detune_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "detune_scan_" + measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            x = np.linspace(parameters["center"] - parameters["range"], parameters["center"] + parameters["range"],
                            parameters["N_points"])
            simulated_width = np.exp(self._gate_voltages["T"] - self._gate_voltages["SA"]) + np.exp(
                self._gate_voltages["N"] - self._gate_voltages["SB"]) + 190. + 0. * (
                np.random.rand(1)[0] - 0.5)
            y = np.tanh(x/simulated_width)

            return y
        elif measurement == 'lead_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "lead_scan" + measurement.get_file_name()
            simulated_rise_time = .2 * (self._gate_voltages["SA"] - 1.) + 0.075 * (
                self._gate_voltages["T"] - 1.) + 0.2 + 0.00005 * (np.random.rand(1)[0] - 0.5)
            simulated_fall_time = .15 * (self._gate_voltages["SA"] - 1.) + .05 * (
                self._gate_voltages["T"] - 1.) + 0.2 + 0.00005 * (np.random.rand(1)[0] - 0.5)
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
        elif measurement == "load_scan":
            simulated_curvature = 15. + np.exp(- self._gate_voltages["T"] - self._gate_voltages["SA"]) + 0. * (
                        np.random.rand(1)[0] - 0.5)
            ydata = np.exp(np.arange(0., 500, 5) * -1. / simulated_curvature)
            xdata = np.arange(0., 500, 5)
            return np.reshape(np.concatenate((ydata, xdata)), (2, 100))
        elif measurement == "2d_scan":
            x = np.linspace(start=-5., stop=5., num=104)
            y = np.linspace(start=-5., stop=5., num=20)
            xx, yy = np.meshgrid(x, y, sparse=True)
            return np.sin(xx + yy + self._gate_voltages["SDB1"])
        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))

