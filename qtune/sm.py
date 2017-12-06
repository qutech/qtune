"""special-measure backend
"""
import warnings
import io
import functools
import sys
from typing import Tuple, Sequence, Union
import os.path
import weakref

import matlab.engine
import pandas as pd
import numpy as np

from qtune.experiment import *
from qtune.util import time_string
from qtune.GradKalman import GradKalmanFilter
from qtune.Basic_DQD import BasicDQD
from qtune.chrg_diag import ChargeDiagram


def redirect_output(func):
    return functools.partial(func, stdout=sys.stdout, stderr=sys.stderr)


def matlab_files_path():
    return os.path.join(os.path.dirname(__file__), 'MATLAB')


class SpecialMeasureMatlab:
    connected_engines = weakref.WeakValueDictionary()
    """Keeps track of all connected engines as matlab.engine does not allow to connect to the same engine twice."""

    def __init__(self, connect=None, gui=None, special_measure_setup_script=None, silently_overwrite_path=None):
        if not connect:
            # start new instance
            gui = True if gui is None else gui
            self._engine = self._start_engine(gui)

        else:
            if gui is not None:
                warnings.warn('gui switch was set but a connection to already existing matlab session was requested',
                              UserWarning)
            self._engine = self._connect_to_engine(connect)

        self._init_special_measure(special_measure_setup_script)
        self._add_qtune_to_path(silently_overwrite_path)

    @classmethod
    def _start_engine(cls, gui: bool) -> matlab.engine.MatlabEngine:
        args = '-desktop' if gui else '-nodesktop'
        engine = matlab.engine.start_matlab(args)
        cls.connected_engines[int(engine.feature('getpid'))] = engine
        return engine

    @classmethod
    def _connect_to_engine(cls, connect: Union[bool, str]) -> matlab.engine.MatlabEngine:
        if connect is False:
            raise ValueError('False is not a valid argument')

        elif connect is True:
            try:
                return next(cls.connected_engines.values())
            except StopIteration:
                engine = matlab.engine.connect_matlab()
        else:
            try:
                return cls.connected_engines[connect]
            except KeyError:
                engine = matlab.engine.connect_matlab(connect)
        cls.connected_engines[engine.matlab.engine.engineName()] = engine
        return engine

    def _init_special_measure(self, special_measure_setup_script):
        if not self.engine.exist('smdata'):
            if special_measure_setup_script:
                getattr(self._engine, special_measure_setup_script)()

                if not self.engine.exist('smdata'):
                    raise RuntimeError('Special measure setup script did not create smdata')

    def _add_qtune_to_path(self, silently_overwrite_path):
        try:
            with io.StringIO() as devnull:
                qtune_path = self.engine.qtune.find_qtune(stderr=devnull, stdout=devnull)

            if not os.path.samefile(qtune_path, matlab_files_path()):
                if silently_overwrite_path is False:
                    return

                if silently_overwrite_path is None:
                    warn_text = 'Found other qtune on MATLAB path: {}\n Will override with {}'.format(qtune_path,
                                                                                                      matlab_files_path())
                    warnings.warn(warn_text, UserWarning)

                self.engine.addpath(matlab_files_path())

        except matlab.engine.MatlabExecutionError:
            #  not on path
            self.engine.addpath(matlab_files_path())

        #  ensure everything worked
        try:
            with io.StringIO() as devnull:
                self.engine.qtune.find_qtune(stderr=devnull, stdout=devnull)
        except matlab.engine.MatlabExecutionError as e:
            raise RuntimeError('Could not add +qtune to MATLAB path') from e

    def to_matlab(self, obj):
        if isinstance(obj, np.ndarray):
            raw = bytes(obj)
            obj_type_str = str(obj.dtype)
            conversions = {'float64': 'double',
                           'float32': 'single',
                           'bool': 'logical',
                           'int32': 'int32',
                           'uint64': 'uint64'}
            if obj_type_str in conversions:
                casted = self.engine.typecast(raw, conversions[obj_type_str])
            else:
                raise NotImplementedError('{} to MATLAB conversion'.format(obj.dtype))

            shape = tuple(reversed(obj.shape))
            if len(shape) == 1:
                shape = shape + (1,)

            casted = self.engine.reshape(casted, *shape)

            return self.engine.transpose(casted)
        else:
            raise NotImplementedError('To MATLAB conversion', obj)

    @property
    def engine(self):
        return self._engine

    @property
    def workspace(self):
        return self.engine.workspace

    def get_variable(self, var_name):
        return self.engine.util.py.get_from_workspace(var_name)


class LegacyDQD(BasicDQD):
    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        super().__init__()
        self._matlab = matlab_instance

    @property
    def gate_voltage_names(self) -> Tuple:
        return tuple(sorted(self._matlab.engine.atune.read_gate_voltages().keys()))

    def read_gate_voltages(self):
        return pd.Series(self._matlab.engine.qtune.read_gate_voltages()).sort_index()

    def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:
        new_gate_voltages = dict(new_gate_voltages)
        for key in new_gate_voltages:
            new_gate_voltages[key] = new_gate_voltages[key].item()
        return pd.Series(self._matlab.engine.qtune.set_gates_v_pretuned(new_gate_voltages))

    def read_qpc_voltage(self) -> pd.Series:
        return pd.Series(self._matlab.engine.qtune.readout_qpc())

    def tune_qpc(self, qpc_position=None, tuning_range=4e-3):
        if qpc_position is None:
            qpc_position = dict(self.read_qpc_voltage())['qpc'][0]
        qpc_tune_input={"tuning_range": tuning_range, "qpc_position": qpc_position, "file_name": time_string()}
        return self._matlab.engine.qtune.retune_qpc(qpc_tune_input)

    def measure(self,
                measurement: Measurement) -> pd.Series:
        self.tune_qpc()

        if measurement == 'line_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            return pd.Series(self._matlab.engine.qtune.PythonChargeLineScan(parameters))
        elif measurement == 'detune_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            return pd.Series(self._matlab.engine.qtune.PythonLineScan(parameters))
        elif measurement == 'lead_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = measurement.get_file_name()
            return pd.Series(self._matlab.engine.qtune.LeadScan(parameters))

        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))


class LegacyChargeDiagram(ChargeDiagram):
    charge_line_scan_lead_A = Measurement('line_scan', center=0., range=3e-3,
                                          gate='RFA', N_points=1280,
                                          ramptime=.0005,
                                          N_average=3,
                                          AWGorDecaDAC='DecaDAC')

    charge_line_scan_lead_B = Measurement('line_scan', center=0., range=3e-3,
                                          gate='RFB', N_points=1280,
                                          ramptime=.0005,
                                          N_average=3,
                                          AWGorDecaDAC='DecaDAC')

    def __init__(self, dqd: LegacyDQD,
                 matlab_engine: SpecialMeasureMatlab,
                 charge_line_scan_lead_A: Measurement = None,
                 charge_line_scan_lead_B: Measurement = None):
        self.dqd = dqd
        self.matlab = matlab_engine

        self.position_lead_A = 0.
        self.position_lead_B = 0.
        self.grad_kalman=GradKalmanFilter(2, 2, initX=np.zeros((2, 2), dtype=float))

        if charge_line_scan_lead_A is not None:
            self.charge_line_scan_lead_A = charge_line_scan_lead_A

        if charge_line_scan_lead_B is not None:
            self.charge_line_scan_lead_B = charge_line_scan_lead_B

    def measure_positions(self) -> Tuple[float, float]:
        current_gate_voltages = self.dqd.read_gate_voltages()
        RFA_eps = pd.Series(1e-3, ['RFA'])
        RFB_eps = pd.Series(1e-3, ['RFB'])
        voltages_for_pos_a = current_gate_voltages.add(-4*RFB_eps, fill_value=0)
        self.dqd.set_gate_voltages(voltages_for_pos_a)
        data_A = self.dqd.measure(self.charge_line_scan_lead_A)
        self.position_lead_A = self.matlab.engine.qtune.at_find_lead_trans(data_A.values.item(),
                                                                           float(self.charge_line_scan_lead_A.parameter[
                                                                                     "center"]),
                                                                           float(self.charge_line_scan_lead_A.parameter[
                                                                                     "range"]),
                                                                           float(self.charge_line_scan_lead_A.parameter[
                                                                                     "N_points"]))

        voltages_for_pos_b = current_gate_voltages.add(-4*RFA_eps, fill_value=0)
        self.dqd.set_gate_voltages(voltages_for_pos_b)
        data_B = self.dqd.measure(self.charge_line_scan_lead_B)
        self.position_lead_B = self.matlab.engine.qtune.at_find_lead_trans(data_B.values.item(),
                                                                           float(self.charge_line_scan_lead_B.parameter[
                                                                                     "center"]),
                                                                           float(self.charge_line_scan_lead_B.parameter[
                                                                                     "range"]),
                                                                           float(self.charge_line_scan_lead_B.parameter[
                                                                                     "N_points"]))
        self.dqd.set_gate_voltages(current_gate_voltages)
        return self.position_lead_A, self.position_lead_B

    def calculate_gradient(self):
        current_gate_voltages = self.dqd.read_gate_voltages()

        BA_eps = pd.Series(1e-3, ['BA'])
        print('current gate voltages should still be at pretuned point')
        print(current_gate_voltages)
        BB_eps = pd.Series(1e-3, ['BB'])

        BA_inc = current_gate_voltages.add(BA_eps, fill_value=0)
        BA_dec = current_gate_voltages.add(-BA_eps, fill_value=0)

        BB_inc = current_gate_voltages.add(BB_eps, fill_value=0)
        BB_dec = current_gate_voltages.add(-BB_eps, fill_value=0)

        self.dqd.set_gate_voltages(BA_inc)
        pos_A_BA_inc, pos_B_BA_inc = self.measure_positions()

        self.dqd.set_gate_voltages(BA_dec)
        pos_A_BA_dec, pos_B_BA_dec = self.measure_positions()

        self.dqd.set_gate_voltages(BB_inc)
        pos_A_BB_inc, pos_B_BB_inc = self.measure_positions()

        self.dqd.set_gate_voltages(BB_dec)
        pos_A_BB_dec, pos_B_BB_dec = self.measure_positions()

        gradient = np.zeros((2, 2), dtype=float)
        gradient[0, 0] = (pos_A_BA_inc - pos_A_BA_dec) / 2e-3
        gradient[0, 1] = (pos_A_BB_inc - pos_A_BB_dec) / 2e-3
        gradient[1, 0] = (pos_B_BA_inc - pos_B_BA_dec) / 2e-3
        gradient[1, 1] = (pos_B_BB_inc - pos_B_BB_dec) / 2e-3

        self.dqd.set_gate_voltages(current_gate_voltages)

        return gradient.copy()

    def initialize_kalman(self, initX=None, initP=None, initR=None, alpha=1.02):
        if initX is None:
            initX = self.calculate_gradient()
        self.grad_kalman = GradKalmanFilter(2, 2, initX=initX, initP=initP, initR=initR, alpha=alpha)

    def center_diagram(self):
        positions=self.measure_positions()
        while np.linalg.norm(positions) > 0.2e-3:
            current_position = (self.position_lead_A, self.position_lead_B)
            du = np.linalg.solve(self.grad_kalman.grad, current_position)
            if np.linalg.norm(du) > 2e-3:
                du = du*2e-3/np.linalg.norm(du)

            diff = pd.Series(-1*du, ['BA', 'BB'])
            new_gate_voltages = self.dqd.read_gate_voltages().add(diff, fill_value=0)
            self.dqd.set_gate_voltages(new_gate_voltages)

            positions = self.measure_positions()
            dpos = (positions[0] - current_position[0], positions[1] - current_position[1])
            self.grad_kalman.update(-1*du, dpos, hack=False)
