"""special-measure backend
"""
import warnings
import io
import functools
import sys
from typing import Tuple, Sequence

import matlab.engine
import pd as pd
import numpy as np

from qtune.experiment import *
from qtune.util import time_string

def redirect_output(func):
    return functools.partial(func, stdout=sys.stdout, stderr=sys.stderr)


def to_malab(obj):
    if isinstance(obj, np.ndarray):
        return matlab.double(obj.tolist())


class SpecialMeasureMatlab:
    def __init__(self, connect=None, gui=None, special_measure_setup_script=None):
        if not connect:
            # start new instance
            gui = True if gui is None else gui

            if gui:
                self._engine = matlab.engine.start_matlab('-desktop')
            else:
                self._engine = matlab.engine.start_matlab('-nodesktop')

        else:
            if gui is not None:
                warnings.warn('gui switch was set but a connection to already existing matlab session was requested',
                              UserWarning)

            if connect is True:
                self._engine = matlab.engine.connect_matlab()
            else:
                self._engine = matlab.engine.connect_matlab(connect)

        if not self.engine.exist('smdata'):
            if special_measure_setup_script:
                getattr(self._engine, special_measure_setup_script)()

                if not self.engine.exist('smdata'):
                    raise RuntimeError('Special measure setup script did not create smdata')

    @property
    def engine(self):
        return self._engine

    @property
    def workspace(self):
        return self.engine.workspace

    def get_variable(self, var_name):
        return self.engine.util.py.get_from_workspace(var_name)


class BasicDQD(Experiment):
    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        self._matlab = matlab_instance

        self.default_line_scan = Measurement('line_scan',
                                             center=0, range=3e-3, gate='RFA', N_points=1280, ramptime=.0005,
                                             N_average=3, AWGorDecaDAC='AWG', file_name=time_string())
        self.default_charge_scan = Measurement('charge_scan',
                                               range_x=(-4., 4.), range_y=(-4., 4.), resolution=(50, 50))

    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        return (self.default_line_scan, )

    @property
    def gate_voltage_names(self) -> Tuple[GateIdentifier, ...]:
        return 'SB', 'BB', 'T', 'N', 'SA', 'BA'

    def measure(self,
                measurement: Measurement) -> pd.Series:

        if measurement == 'line_scan':
            return pd.Series()

        elif measurement == 'charge_scan':
            return pd.Series()

        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))


class LegacyDQD(Experiment):
    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        self._matlab = matlab_instance

        self.default_line_scan = Measurement('line_scan',
                                             center=0, range=3e-3, gate='RFA', N_points=1280, ramptime=.0005, N_average=3, AWGorDecaDAC='AWG', file_name=time_string())

        self.default_charge_scan = Measurement('charge_scan',
                                               range_x=(-4., 4.), range_y=(-4., 4.), resolution=(50, 50))

    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        return (self.default_line_scan,)

    @property
    def gate_voltage_names(self) -> Tuple:
        return tuple(sorted(self._matlab.engine.atune.read_gate_voltages().keys()))

    def read_gate_voltages(self):
        return pd.Series(self._matlab.engine.atune.read_gate_voltages()).sort_index()

    def set_gate_voltages(self, new_gate_voltages: pd.Series):
        self._matlab.engine.atune.set_gates_v_pretuned(dict(new_gate_voltages))

    def measure(self,
                measurement: Measurement) -> pd.Series:

        if measurement == 'line_scan':
            return self._matlab.engine.atune.PythonChargeLineScan(measurement.parameter)

        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))


class ChargeDiagram:
    charge_line_scan_lead_A = Measurement('line_scan', center=0, range=3e-3,
                                          gate='RFA', N_points=1280,
                                          ramptime=.0005,
                                          N_average=3,
                                          AWGorDecaDAC='AWG',
                                          file_name=time_string())

    charge_line_scan_lead_B = Measurement('line_scan', center=0, range=3e-3,
                                          gate='RFB', N_points=1280,
                                          ramptime=.0005,
                                          N_average=3,
                                          AWGorDecaDAC='AWG',
                                          file_name=time_string())

    def __init__(self, exp: Experiment,
                 charge_line_scan_lead_A: Measurement,
                 charge_line_scan_lead_B: Measurement,
                 matlab_engine: SpecialMeasureMatlab):
        self.experiment = exp
        self.matlab = matlab_engine

        self.position_lead_A = 0
        self.position_lead_B = 0
        self.gradient = np.asarray([[0, 0],
                                    [0, 0]])

        if charge_line_scan_lead_A is not None:
            self.charge_line_scan_lead_A = charge_line_scan_lead_A

        if charge_line_scan_lead_B is not None:
            self.charge_line_scan_lead_B = charge_line_scan_lead_B

    def measure_positions(self):
        data_A = self.experiment.measure(self.charge_line_scan_lead_A)
        self.position_lead_A = self.matlab.engine.qtune.at_find_lead_trans(data_A,
                                                                           self.charge_line_scan_lead_A.parameter[
                                                                               "center"],
                                                                           self.charge_line_scan_lead_A.parameter[
                                                                               "range"],
                                                                           self.charge_line_scan_lead_A.parameter[
                                                                               "N_points"])

        data_B = self.exp.measure(self.charge_line_scan_lead_B)
        self.position_lead_B = self.matlab.engine.qtune.at_find_lead_trans(data_B,
                                                                           self.charge_line_scan_lead_B.parameter[
                                                                               "center"],
                                                                           self.charge_line_scan_lead_B.parameter[
                                                                               "range"],
                                                                           self.charge_line_scan_lead_B.parameter[
                                                                               "N_points"])

    def calculate_gradient(self):
        current_gate_voltages = self.experiment.gate_voltages
        BA_inc = current_gate_voltages
        BA_inc["BA"] = BA_inc["BA"] + 1e-3
        BA_red = current_gate_voltages
        BA_red["BA"] = BA_inc["BA"] - 1e-3
        BB_inc = current_gate_voltages
        BB_inc["BB"] = BA_inc["BB"] + 1e-3
        BB_red = current_gate_voltages
        BB_red["BB"] = BA_inc["BB"] - 1e-3

        self.experiment.set_gate(BA_inc)
        self.measure_positions()
        pos_A_BA_inc = self.position_lead_A
        pos_B_BA_inc = self.position_lead_B

        self.experiment.set_gate(BA_red)
        self.measure_positions
        pos_A_BA_red = self.position_lead_A
        pos_B_BA_red = self.position_lead_B

        self.experiment.set_gate(BB_inc)
        self.measure_positions
        pos_A_BB_inc = self.position_lead_A
        pos_B_BB_inc = self.position_lead_B

        self.experiment.set_gate(BB_red)
        self.measure_positions
        pos_A_BB_red = self.position_lead_A
        pos_B_BB_red = self.position_lead_B

        self.gradient[0, 0] = (pos_A_BA_inc - pos_A_BA_red) / 2e-3
        self.gradient[0, 1] = (pos_A_BB_inc - pos_A_BB_red) / 2e-3
        self.gradient[1, 0] = (pos_B_BA_inc - pos_B_BA_red) / 2e-3
        self.gradient[1, 1] = (pos_B_BB_inc - pos_B_BB_red) / 2e-3

        self.experiment.set_gate_voltages(current_gate_voltages)

    def center_diagram(self):
        self.measure_positions()
        while (abs(self.position_lead_A) > 0.2e-3) or (abs(self.position_lead_B) > 0.2e-3):
            dpos = np.array([[self.position_lead_A], [self.position_lead_B]])
            du = np.linalg.solve(self.gradient, dpos)
            current_gate_voltages = self.experiment.gate_voltages
            new_gate_voltages = current_gate_voltages
            new_gate_voltages['BA'] = new_gate_voltages['BA'] + du[0, 0]
            new_gate_voltages['BB'] = new_gate_voltages['BB'] + du[1, 0]
            self.experiment.set_gate_voltages(new_gate_voltages)
            self.measure_positions()
