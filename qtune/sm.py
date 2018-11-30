"""special-measure backend
"""
import warnings
import io
import functools
import sys
from typing import Tuple, Union, List, Dict, Set
import os.path
import weakref

import matlab.engine
import pandas as pd
import numpy as np

from qtune.experiment import *
from qtune.util import time_string
from qtune.basic_dqd import BasicDQD
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
        no_conversion_required = {float, int}
        if type(obj) in no_conversion_required:
            return obj
        elif isinstance(obj, pd.Series):
            return self.to_matlab(dict(obj))
        elif isinstance(obj, np.ndarray):
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
        elif isinstance(obj, np.generic):
            return float(np.asscalar(obj))
        elif isinstance(obj, List):
            return [self.to_matlab(x) for x in obj]
        elif isinstance(obj, Tuple):
            return (self.to_matlab(x) for x in obj)
        elif isinstance(obj, Dict):
            return {x: self.to_matlab(obj[x]) for x in obj}
        elif isinstance(obj, Set):
            return {self.to_matlab(x) for x in obj}
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


class LegacyDQDRefactored(BasicDQD):
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
        current_gate_voltages[new_gate_voltages.index] = new_gate_voltages[new_gate_voltages.index]
        current_gate_voltages = self._matlab.to_matlab(current_gate_voltages)
        return pd.Series(self._matlab.engine.qtune.set_gates_v_pretuned(current_gate_voltages))

    def measure(self, measurement: Measurement) -> np.ndarray:

        if measurement.name == 'line_scan':
            parameters = measurement.options.copy()
            parameters['file_name'] = "line_scan" + measurement.get_file_name()
            for name in ['N_points', 'N_average', 'center', 'range', 'ramptime']:
                if name in parameters:
                    parameters[name] = self._matlab.to_matlab(parameters[name])
            return np.asarray(self._matlab.engine.qtune.PythonChargeLineScan(parameters))
        elif measurement.name == 'detune_scan':
            parameters = measurement.options.copy()
            parameters['file_name'] = "detune_scan_" + measurement.get_file_name()
            for name in ['N_points', 'N_average', 'center', 'range', 'ramptime']:
                if name in parameters:
                    parameters[name] = self._matlab.to_matlab(parameters[name])
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
            qpc_2d_tune_input = {"range": self._matlab.to_matlab(measurement.options["range"]),
                                 "file_name": time_string(),
                                 'n_points': self._matlab.to_matlab(measurement.options['n_points'])}

            return np.asarray(self._matlab.engine.qtune.PythonQPCScan2D(qpc_2d_tune_input))
        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))


class MatlabEvaluator(Evaluator):
    def __init__(self, experiment: qtune.sm.LegacyDQDRefactored, **kwargs):
        super().__init__(experiment, **kwargs)

    def evaluate(self):
        raise NotImplementedError
        # return self.experiment._matlab.engine.

    def process_raw_data(self, raw_data):
        raise NotImplementedError

    def to_hdf5(self):
        return super().to_hdf5()