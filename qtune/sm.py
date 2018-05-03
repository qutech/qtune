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

from qtune.util import time_string

from qtune.basic_dqd import BasicDQD
from qtune.evaluator import Evaluator
from qtune.experiment import Experiment, Measurement


def redirect_output(func):
    return functools.partial(func, stdout=sys.stdout, stderr=sys.stderr)


def matlab_files_path():
    return os.path.join(os.path.dirname(__file__), 'MATLAB')


class SpecialMeasureMatlab:
    """
    Keeps track of all connected engines as matlab.engine does not allow to connect to the same engine twice.
    """
    connected_engines = weakref.WeakValueDictionary()

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


#  class LegacyDQD(BasicDQD):
#     """
#     Implementation of the means to control a dqd aided by matlab functions. Necessary due to the Matlab legacy and to be
#     replaced by Q-codes.
#     """
#     def __init__(self, matlab_instance: SpecialMeasureMatlab):
#         super().__init__()
#         self._matlab = matlab_instance
#
#     @property
#     def gate_voltage_names(self) -> Tuple:
#         return tuple(sorted(self._matlab.engine.qtune.read_gate_voltages().keys()))
#
#     def read_gate_voltages(self) -> pd.Series:
#         return pd.Series(self._matlab.engine.qtune.read_gate_voltages()).sort_index()
#
#     def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:
#         current_gate_voltages = self.read_gate_voltages()
#         for key in current_gate_voltages.index.tolist():
#             if key not in new_gate_voltages.index.tolist():
#                 new_gate_voltages[key] = current_gate_voltages[key]
#         new_gate_voltages = dict(new_gate_voltages)
#         for key in new_gate_voltages:
#             new_gate_voltages[key] = new_gate_voltages[key].item()
#         return pd.Series(self._matlab.engine.qtune.set_gates_v_pretuned(new_gate_voltages))
#
#     def read_qpc_voltage(self) -> pd.Series:
#         return pd.Series(self._matlab.engine.qtune.readout_qpc())
#
#     def tune_qpc(self, qpc_position=None, tuning_range=4e-3, gate='SDB2'):
#         if qpc_position is None:
#             qpc_position = dict(self.read_qpc_voltage())['qpc'][0]
#         qpc_tune_input = {"tuning_range": tuning_range, "qpc_position": qpc_position, "file_name": time_string(),
#                           "gate": gate}
#         tuning_output = self._matlab.engine.qtune.retune_qpc(qpc_tune_input)
#         self._qpc_tuned = True
#         self.signal_strength = tuning_output["signal_strength"]
#         return tuning_output, self.read_qpc_voltage()
#
#     def tune_qpc_2d(self, tuning_range=15e-3):
#         qpc_2d_tune_input = {"range": tuning_range, "file_name": time_string()}
#         data = np.asarray(self._matlab.engine.qtune.PythonQPCScan2D(qpc_2d_tune_input))
#
#         n_lines = data.shape[0]
#         n_points = data.shape[1]
#         data_filterd = scipy.ndimage.filters.gaussian_filter1d(input=data, sigma=.5, axis=0, order=0, mode="nearest",
#                                                                truncate=4.)
#         data_diff = np.zeros((n_lines, n_points - 1))
#         for j in range(n_lines):
#             for i in range(n_points - 1):
#                 data_diff[j, i] = data_filterd[j, i + 1] - data_filterd[j, i]
#
#         mins_in_lines = data_diff.min(1)
#         min_line = np.argmin(mins_in_lines)
#         min_point = np.argmin(data_diff[min_line])
#         gate1_pos = float(min_line) / 20. * 2 * tuning_range - tuning_range
#         gate2_pos = float(min_point) / 104. * 2 * tuning_range - tuning_range
#         gate_change = {"gate1": gate1_pos, "gate2": gate2_pos}
#         empty_output = self._matlab.engine.qtune.change_sensing_dot_gates(gate_change)
# #        self.tune_qpc(gate='SDB1')
#         self.tune_qpc(gate='SDB2')
#
#     def measure(self, measurement: Measurement) -> np.ndarray:
#         if not self._qpc_tuned:
#             self.tune_qpc()
#
#         if measurement.name == 'line_scan':
#             parameters = measurement.parameter.copy()
#             parameters['file_name'] = "line_scan" + measurement.get_file_name()
#             parameters['N_points'] = float(parameters['N_points'])
#             parameters['N_average'] = float(parameters['N_average'])
#             return np.asarray(self._matlab.engine.qtune.PythonChargeLineScan(parameters))
#         elif measurement.name == 'detune_scan':
#             parameters = measurement.parameter.copy()
#             parameters['file_name'] = "detune_scan_" + measurement.get_file_name()
#             parameters['N_points'] = float(parameters['N_points'])
#             parameters['N_average'] = float(parameters['N_average'])
#             return np.asarray(self._matlab.engine.qtune.PythonLineScan(parameters))
#         elif measurement.name == 'lead_scan':
#             parameters = measurement.parameter.copy()
#             parameters['file_name'] = "lead_scan" + measurement.get_file_name()
#             return np.asarray(self._matlab.engine.qtune.LeadScan(parameters))
#         elif measurement.name == "load_scan":
#             parameters = measurement.parameter.copy()
#             parameters["file_name"] = "load_scan" + measurement.get_file_name()
#             return np.asarray(self._matlab.engine.qtune.LoadScan(parameters))
#         elif measurement.name == "2d_scan":
#             qpc_2d_tune_input = {"range": measurement.parameter["scan_range"], "file_name": time_string()}
#             return np.asarray(self._matlab.engine.qtune.PythonQPCScan2D(qpc_2d_tune_input))
#         else:
#             raise ValueError('Unknown measurement: {}'.format(measurement))

























































