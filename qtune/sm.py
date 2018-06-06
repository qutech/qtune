"""special-measure backend
"""
import warnings
import io
import functools
import sys
from typing import Tuple, Sequence, Union
import os.path
import weakref
import h5py

import matlab.engine
import pandas as pd
import numpy as np
import scipy.ndimage

from qtune.experiment import *
import qtune.experiment
from qtune.util import time_string
import qtune.util
from qtune.basic_dqd import BasicDQD
from qtune.basic_dqd import BasicQQD
from qtune.basic_dqd import BasicDQDRefactored
from qtune.chrg_diag import ChargeDiagram
from qtune.evaluator import Evaluator


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


class LegacyDQDRefactored(BasicDQDRefactored):
    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        super().__init__()
        self._matlab = matlab_instance

    @property
    def gate_voltage_names(self) -> Tuple:
        return tuple(sorted(self._matlab.engine.qtune.read_gate_voltages().keys()))

    def read_gate_voltages(self) -> pd.Series:
        return pd.Series(self._matlab.engine.qtune.read_gate_voltages()).sort_index()

    def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:
        current_gate_voltages = self.read_gate_voltages()
        for key in current_gate_voltages.index.tolist():
            if key not in new_gate_voltages.index.tolist():
                new_gate_voltages[key] = current_gate_voltages[key]
        new_gate_voltages = dict(new_gate_voltages)
        for key in new_gate_voltages:
            new_gate_voltages[key] = new_gate_voltages[key].item()
        return pd.Series(self._matlab.engine.qtune.set_gates_v_pretuned(new_gate_voltages))

    def measure(self, measurement: Measurement) -> np.ndarray:

        if measurement.name == 'line_scan':
            parameters = measurement.options.copy()
            parameters['file_name'] = "line_scan" + measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            return np.asarray(self._matlab.engine.qtune.PythonChargeLineScan(parameters))
        elif measurement.name == 'detune_scan':
            parameters = measurement.options.copy()
            parameters['file_name'] = "detune_scan_" + measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            return np.asarray(self._matlab.engine.qtune.PythonLineScan(parameters))
        elif measurement.name == 'lead_scan':
            parameters = measurement.options.copy()
            parameters['file_name'] = "lead_scan" + measurement.get_file_name()
            return np.asarray(self._matlab.engine.qtune.LeadScan(parameters))
        elif measurement.name == "load_scan":
            parameters = measurement.options.copy()
            parameters["file_name"] = "load_scan" + measurement.get_file_name()
            return np.asarray(self._matlab.engine.qtune.LoadScan(parameters))
        elif measurement.name == "2d_scan":
            qpc_2d_tune_input = {"range": measurement.options["scan_range"], "file_name": time_string()}
            return np.asarray(self._matlab.engine.qtune.PythonQPCScan2D(qpc_2d_tune_input))
        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))


class LegacyDQD(BasicDQD):
    """
    Implementation of the means to control a dqd aided by matlab functions. Necessary due to the Matlab legacy and to be
    replaced by Q-codes.
    """
    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        super().__init__()
        self._matlab = matlab_instance

    @property
    def gate_voltage_names(self) -> Tuple:
        return tuple(sorted(self._matlab.engine.qtune.read_gate_voltages().keys()))

    def read_gate_voltages(self) -> pd.Series:
        return pd.Series(self._matlab.engine.qtune.read_gate_voltages()).sort_index()

    def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:
        current_gate_voltages = self.read_gate_voltages()
        for key in current_gate_voltages.index.tolist():
            if key not in new_gate_voltages.index.tolist():
                new_gate_voltages[key] = current_gate_voltages[key]
        new_gate_voltages = dict(new_gate_voltages)
        for key in new_gate_voltages:
            new_gate_voltages[key] = new_gate_voltages[key].item()
        return pd.Series(self._matlab.engine.qtune.set_gates_v_pretuned(new_gate_voltages))

    def read_qpc_voltage(self) -> pd.Series:
        return pd.Series(self._matlab.engine.qtune.readout_qpc())

    def tune_qpc(self, qpc_position=None, tuning_range=4e-3, gate='SDB2'):
        if qpc_position is None:
            qpc_position = dict(self.read_qpc_voltage())['qpc'][0]
        qpc_tune_input = {"tuning_range": tuning_range, "qpc_position": qpc_position, "file_name": time_string(),
                          "gate": gate}
        tuning_output = self._matlab.engine.qtune.retune_qpc(qpc_tune_input)
        self._qpc_tuned = True
        self.signal_strength = tuning_output["signal_strength"]
        return tuning_output, self.read_qpc_voltage()

    def tune_qpc_2d(self, tuning_range=15e-3):
        qpc_2d_tune_input = {"range": tuning_range, "file_name": time_string()}
        data = np.asarray(self._matlab.engine.qtune.PythonQPCScan2D(qpc_2d_tune_input))

        n_lines = data.shape[0]
        n_points = data.shape[1]
        data_filterd = scipy.ndimage.filters.gaussian_filter1d(input=data, sigma=.5, axis=0, order=0, mode="nearest",
                                                               truncate=4.)
        data_diff = np.zeros((n_lines, n_points - 1))
        for j in range(n_lines):
            for i in range(n_points - 1):
                data_diff[j, i] = data_filterd[j, i + 1] - data_filterd[j, i]

        mins_in_lines = data_diff.min(1)
        min_line = np.argmin(mins_in_lines)
        min_point = np.argmin(data_diff[min_line])
        gate1_pos = float(min_line) / 20. * 2 * tuning_range - tuning_range
        gate2_pos = float(min_point) / 104. * 2 * tuning_range - tuning_range
        gate_change = {"gate1": gate1_pos, "gate2": gate2_pos}
        empty_output = self._matlab.engine.qtune.change_sensing_dot_gates(gate_change)
#        self.tune_qpc(gate='SDB1')
        self.tune_qpc(gate='SDB2')

    def measure(self, measurement: Measurement) -> np.ndarray:
        if not self._qpc_tuned:
            self.tune_qpc()

        if measurement.name == 'line_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "line_scan" + measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            return np.asarray(self._matlab.engine.qtune.PythonChargeLineScan(parameters))
        elif measurement.name == 'detune_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "detune_scan_" + measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            return np.asarray(self._matlab.engine.qtune.PythonLineScan(parameters))
        elif measurement.name == 'lead_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "lead_scan" + measurement.get_file_name()
            return np.asarray(self._matlab.engine.qtune.LeadScan(parameters))
        elif measurement.name == "load_scan":
            parameters = measurement.parameter.copy()
            parameters["file_name"] = "load_scan" + measurement.get_file_name()
            return np.asarray(self._matlab.engine.qtune.LoadScan(parameters))
        elif measurement.name == "2d_scan":
            qpc_2d_tune_input = {"range": measurement.parameter["scan_range"], "file_name": time_string()}
            return np.asarray(self._matlab.engine.qtune.PythonQPCScan2D(qpc_2d_tune_input))
        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))


class LegacyQQD(BasicQQD):

    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        super().__init__()
        self._matlab = matlab_instance
        self._left_sensing_dot_tuned = False
        self._right_sensing_dot_tuned = False

    def gate_voltage_names(self) -> Tuple:
        return tuple(sorted(self._matlab.engine.qtune.read_qqd_gate_voltages().keys()))

    def read_gate_voltages(self) -> pd.Series:
        return pd.Series(self._matlab.engine.qtune.read_qqd_gate_voltages()).sort_index()

    def _read_sensing_dot_voltages(self) -> pd.Series:
        return pd.Series(self._matlab.engine.qtune.read_qqd_sensing_dot_voltages()).sort_index()

    def tune_sensing_dot_1d(self, gate: str, prior_position: float=None, tuning_range: float=4e-3, n_points: int=1280,
                            scan_range: float=5e-3):
        if prior_position is None:
            prior_position = self._read_sensing_dot_voltages()[gate]
        sensing_dot_measurement = qtune.experiment.Measurement('line_scan',
                                    center=prior_position, range=scan_range, gate=gate, N_points=n_points, ramptime=.0005,
                                    N_average=1, AWGorDecaDAC='DecaDAC')
        data = self.measure(sensing_dot_measurement)
        detuning = qtune.util.find_stepes_point_sensing_dot(data, scan_range=scan_range, npoints=n_points)
        self._set_sensing_dot_voltages(pd.Series({gate: prior_position + detuning}))

    def tune_sensing_dot_2d(self, side: str, tuning_range=15e-3, n_lines=20, n_points=104):
        if side == "r" or side == "R" or side == "right":
            gate_T = "RT"
            gate_B = "RB"
        elif side == "l" or side == "L" or side == "left":
            gate_T = "LT"
            gate_B = "LB"
        else:
            print("Specify which sensing dot you would like to tune!")
            raise ValueError

        positions = self._read_sensing_dot_voltages()
        T_position = positions[gate_T]
        B_position = positions[gate_B]

        sensing_dot_measurement = qtune.experiment.Measurement('2d_scan', center=[T_position, B_position],
                                                               range=tuning_range, gate1=gate_T, gate2=gate_B,
                                                               N_points=1280, ramptime=.0005, n_lines=n_lines,
                                                               n_points=n_points, N_average=1, AWGorDecaDAC='DecaDAC')
        data = self.measure(sensing_dot_measurement)

        data_filterd = scipy.ndimage.filters.gaussian_filter1d(input=data, sigma=.5, axis=0, order=0,
                                                               mode="nearest",
                                                               truncate=4.)
        data_diff = np.zeros((n_lines, n_points - 1))
        for j in range(n_lines):
            for i in range(n_points - 1):
                data_diff[j, i] = data_filterd[j, i + 1] - data_filterd[j, i]

        mins_in_lines = data_diff.min(1)
        min_line = np.argmin(mins_in_lines)
        min_point = np.argmin(data_diff[min_line])
        gate_T_pos = float(min_line) / float(n_lines) * 2 * tuning_range - tuning_range
        gate_B_pos = float(min_point) / float(n_points) * 2 * tuning_range - tuning_range
        new_positions = {gate_T: gate_T_pos + T_position, gate_B: gate_B_pos + B_position}
        self._set_sensing_dot_voltages(pd.Series(new_positions))
        self.tune_sensing_dot_1d(gate=gate_T)

    def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:
        self._left_sensing_dot_tuned = False
        self._right_sensing_dot_tuned = False

        current_gate_voltages = self.read_gate_voltages()
        for key in current_gate_voltages.index.tolist():
            if key not in new_gate_voltages.index.tolist():
                new_gate_voltages[key] = current_gate_voltages[key]
        new_gate_voltages = dict(new_gate_voltages)
        for key in new_gate_voltages:
            new_gate_voltages[key] = new_gate_voltages[key].item()
        return pd.Series(self._matlab.engine.qtune.set_qqd_gate_voltages(new_gate_voltages))

    def _set_sensing_dot_voltages(self, new_sensing_dot_voltage: pd.Series):
        current_sensing_dot_voltages = self._read_sensing_dot_voltages()
        for key in current_sensing_dot_voltages.index.tolist():
            if key not in new_sensing_dot_voltage.index.tolist():
                new_sensing_dot_voltage[key] = current_sensing_dot_voltages[key]
        new_sensing_dot_voltage = dict(new_sensing_dot_voltage)
        for key in new_sensing_dot_voltage:
            new_sensing_dot_voltage[key] = new_sensing_dot_voltage[key].item()
        self._matlab.engine.qtune.set_sensing_dot_gate_voltages()

    def measure(self, measurement: Measurement) -> np.ndarray:
        if not self._left_sensing_dot_tuned:
            self.tune_sensing_dot_1d("RT")
        if not self._right_sensing_dot_tuned:
            self.tune_sensing_dot_1d("LT")

        if measurement == 'line_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "line_scan" + measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            return np.asarray(self._matlab.engine.qtune.PythonChargeLineScan(parameters))
        elif measurement == 'detune_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "detune_scan_" + measurement.get_file_name()
            parameters['N_points'] = float(parameters['N_points'])
            parameters['N_average'] = float(parameters['N_average'])
            return np.asarray(self._matlab.engine.qtune.PythonLineScan(parameters))
        elif measurement == 'lead_scan':
            parameters = measurement.parameter.copy()
            parameters['file_name'] = "lead_scan" + measurement.get_file_name()
            return np.asarray(self._matlab.engine.qtune.LeadScan(parameters))
        elif measurement == "load_scan":
            parameters = measurement.parameter.copy()
            parameters["file_name"] = "load_scan" + measurement.get_file_name()
            return np.asarray(self._matlab.engine.qtune.LoadScan(parameters))
        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))


class LegacyChargeDiagram(ChargeDiagram):
    """
    Charge diagram class using Matlab functions to detect lead transitions. Has already been replaced by the python
    version.
    """
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
        self.position_lead_A = self.matlab.engine.qtune.at_find_lead_trans(self.matlab.to_matlab(data_A),
                                                                           float(self.charge_line_scan_lead_A.parameter[
                                                                                     "center"]),
                                                                           float(self.charge_line_scan_lead_A.parameter[
                                                                                     "range"]),
                                                                           float(self.charge_line_scan_lead_A.parameter[
                                                                                     "N_points"]))

        voltages_for_pos_b = current_gate_voltages.add(-4*RFA_eps, fill_value=0)
        self.dqd.set_gate_voltages(voltages_for_pos_b)
        data_B = self.dqd.measure(self.charge_line_scan_lead_B)
        self.position_lead_B = self.matlab.engine.qtune.at_find_lead_trans(self.matlab.to_matlab(data_B),
                                                                           float(self.charge_line_scan_lead_B.parameter[
                                                                                     "center"]),
                                                                           float(self.charge_line_scan_lead_B.parameter[
                                                                                     "range"]),
                                                                           float(self.charge_line_scan_lead_B.parameter[
                                                                                     "N_points"]))
        self.dqd.set_gate_voltages(current_gate_voltages)
        return self.position_lead_A, self.position_lead_B


class SMInterDotTCByLineScan(Evaluator):
    """
    Adiabaticly sweeps the detune over the transition between the (2,0) and the (1,1) region. An Scurve is fitted and
    the width calculated as parameter for the inter dot coupling. Fitted with Matlab functions. Already replaced by a
    python version.
    """
    def __init__(self, dqd: BasicDQD, matlab_instance: SpecialMeasureMatlab,
                 parameters: pd.Series() = pd.Series((np.nan, ), ('parameter_tunnel_coupling', )), line_scan: Measurement=None):
        if line_scan is None:
            line_scan = dqd.measurements[1]
        super().__init__(dqd, line_scan, parameters)
        self.matlab = matlab_instance

    def evaluate(self, storing_group: h5py.Group) -> pd.Series:
        ydata = self.experiment.measure(self.measurements)
        center = self.measurements.parameter['center']
        scan_range = self.measurements.parameter['range']
        npoints = self. measurements.parameter['N_points']
        xdata = np.linspace(center - scan_range, center + scan_range, npoints)
        fitresult = self.matlab.engine.qtune.at_line_fit(self.matlab.to_matlab(xdata), self.matlab.to_matlab(ydata))
        tc = fitresult['tc']
        failed = bool(fitresult['failed'])
        self.parameters['parameter_tunnel_coupling'] = tc
        if storing_group is not None:
            storing_dataset = storing_group.create_dataset("evaluator_SMInterDotTCByLineScan", data=ydata)
            storing_dataset.attrs["center"] = center
            storing_dataset.attrs["scan_range"] = scan_range
            storing_dataset.attrs["npoints"] = npoints
            storing_dataset.attrs["parameter_tunnel_coupling"] = tc
            if failed:
                storing_dataset.attrs["parameter_tunnel_coupling"] = np.nan
        return pd.Series((tc, failed), ('parameter_tunnel_coupling', 'failed'))


class SMLeadTunnelTimeByLeadScan(Evaluator):
    """
    RF gates pulse over the transition between the (2,0) and the (1,0) region. Then exponential functions are fitted
    to calculate the time required for an electron to tunnel through the lead. Fitted with Matlab functions. Already
    replaced by a python version.
    """
    def __init__(self, dqd: BasicDQD, matlab_instance: SpecialMeasureMatlab,
                 parameters: pd.Series() = pd.Series([np.nan, np.nan], ['parameter_time_rise', 'parameter_time_fall']),
                 lead_scan: Measurement = None):
        if lead_scan is None:
            lead_scan = dqd.measurements[2]
        super().__init__(dqd, lead_scan, parameters)
        self.matlab = matlab_instance

    def evaluate(self, storing_group: h5py.Group) -> pd.Series:
        data = self.experiment.measure(self.measurements)
        fitresult = self.matlab.engine.qtune.lead_fit(self.matlab.to_matlab(data))
        t_rise = fitresult['t_rise']
        t_fall = fitresult['t_fall']
        failed = fitresult['failed']
        self.parameters['parameter_time_rise'] = t_rise
        self.parameters['parameter_time_fall'] = t_fall
        if storing_group is not None:
            storing_dataset = storing_group.create_dataset("evaluator_SMLeadTunnelTimeByLeadScan", data=data)
            storing_dataset.attrs["parameter_time_rise"] = t_rise
            storing_dataset.attrs["parameter_time_fall"] = t_fall
            if failed:
                storing_dataset.attrs["parameter_time_rise"] = np.nan
                storing_dataset.attrs["parameter_time_fall"] = np.nan
        return pd.Series([t_rise, t_fall, failed], ['parameter_time_rise', 'parameter_time_fall', 'failed'])


class SMLoadTime(Evaluator):
    """
    Measures the time required to reload a (2,0) singlet state. Fits an exponential function. Fitted with Matlab
    functions. Already replaced by a python version.
    """
    def __init__(self, dqd: BasicDQD, matlab_instance: SpecialMeasureMatlab,
                 parameters: pd.Series() = pd.Series([np.nan], ['parameter_time_load']),
                 load_scan: Measurement = None):
        if load_scan is None:
            load_scan = dqd.measurements[3]
        super().__init__(dqd, load_scan, parameters)
        self.matlab = matlab_instance

    def evaluate(self, storing_group: h5py.Group) -> pd.Series:
        data = self.experiment.measure(self.measurements)
        fitresult = self.matlab.engine.qtune.load_fit(self.matlab.to_matlab(data))
        parameter_time_load = fitresult['parameter_time_load']
        failed = fitresult['failed']
        self.parameters['parameter_time_load'] = parameter_time_load
        if storing_group is not None:
            storing_dataset = storing_group.create_dataset("evaluator_SMLoadTime", data=data)
            storing_dataset.attrs["parameter_time_load"] = parameter_time_load
            if failed:
                storing_dataset.attrs["parameter_time_load"] = np.nan
        return pd.Series([parameter_time_load, failed], ['parameter_time_load', 'failed'])




























































