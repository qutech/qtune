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
from qtune.Solver import Evaluator


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
                measurement: Measurement) -> np.ndarray:
        self.tune_qpc()

        if measurement == 'line_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            return np.asarray(self._matlab.engine.qtune.PythonChargeLineScan(parameters))
        elif measurement == 'detune_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            return np.asarray(self._matlab.engine.qtune.PythonLineScan(parameters))
        elif measurement == 'lead_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = measurement.get_file_name()
            return np.asarray(self._matlab.engine.qtune.LeadScan(parameters))

        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))


class LegacyChargeDiagram(ChargeDiagram):
    def __init__(self, dqd: LegacyDQD,
                 matlab_engine: SpecialMeasureMatlab,
                 charge_line_scan_lead_A: Measurement = None,
                 charge_line_scan_lead_B: Measurement = None):
        super().__init__(dqd=dqd, charge_line_scan_lead_A=charge_line_scan_lead_A,
                         charge_line_scan_lead_B=charge_line_scan_lead_B)
        self.matlab = matlab_engine

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


class SMInterDotTCByLineScan(Evaluator):
    def __init__(self, dqd: BasicDQD, matlab_instance: SpecialMeasureMatlab,
                 parameters: pd.Series() = pd.Series((np.nan, ), ('tc', )), line_scan: Measurement=None):
        if line_scan is None:
            line_scan = dqd.measurements[1]
        super().__init__(dqd, line_scan, parameters)
        self.matlab = matlab_instance

    def evaluate(self) -> pd.Series:
        ydata = self.experiment.measure(self.measurements)
        center = self.measurements.parameter['center']
        scan_range = self.measurements.parameter['range']
        npoints = self. measurements.parameter['N_points']
        xdata = np.linspace(center - scan_range, center + scan_range, npoints)
        fitresult = self.matlab.engine.qtune.at_line_fit(self.matlab.to_matlab(xdata), self.matlab.to_matlab(ydata))
        tc = fitresult['tc']
        failed = bool(fitresult['failed'])
        self.parameters['tc'] = tc
        return pd.Series((tc, failed), ('tc', 'failed'))


class SMLeadTunnelTimeByLeadScan(Evaluator):
    def __init__(self, dqd: BasicDQD, matlab_instance: SpecialMeasureMatlab,
                 parameters: pd.Series() = pd.Series([np.nan, np.nan], ['t_rise', 't_fall']),
                 lead_scan: Measurement = None):
        if lead_scan is None:
            lead_scan = dqd.measurements[2]
        super().__init__(dqd, lead_scan, parameters)
        self.matlab = matlab_instance

    def evaluate(self) -> pd.Series:
        data = self.experiment.measure(self.measurements)
        fitresult = self.matlab.engine.qtune.lead_fit(self.matlab.to_matlab(data))
        t_rise = fitresult['t_rise']
        t_fall = fitresult['t_fall']
        failed = fitresult['failed']
        self.parameters['t_rise'] = t_rise
        self.parameters['t_fall'] = t_fall
        return pd.Series([t_rise, t_fall, failed], ['t_rise', 't_fall', 'failed'])
