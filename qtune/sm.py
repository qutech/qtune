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
                    warn_text = 'Found other qtune on MATLAB path: {}\n Will override with {}'.format(qtune_path, matlab_files_path())
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


class BasicDQD(Experiment):
    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        self._matlab = matlab_instance

        self.default_line_scan = Measurement('line_scan',
                                             start=None, stop=None, N_points=50, N_average=10, time_per_point=.1)
        self.default_charge_scan = Measurement('charge_scan',
                                               range_x=(-4., 4.), range_y=(-4., 4.), resolution=(50, 50))

    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        return self.default_line_scan, self.default_charge_scan

    @property
    def gate_voltages(self) -> Tuple[GateIdentifier, ...]:
        return 'T', 'N', 'bla'

    def measure(self,
                gate_voltages: pd.Series,
                measurement: Measurement) -> pd.Series:

        if measurement == 'line_scan':
            return pd.Series()

        elif measurement == 'charge_scan':
            return pd.Series()

        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))


class LegacyDQD(BasicDQD):
    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        super().__init__(matlab_instance)

        self.default_lead_scan = Measurement('lead_scan')

    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        return (*super().measurements, self.default_lead_scan)

    def measure(self,
                gate_voltages: pd.Series,
                measurement: Measurement):
        if measurement == 'lead_scan':
            return pd.Series()
        else:
            return super().measure(gate_voltages, measurement)
        
        
class PrototypeDQD(Experiment):
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
                gate_voltages: pd.Series,
                measurement: Measurement) -> pd.Series:

        if measurement == 'line_scan':
            return self._matlab.engine.atune.PythonChargeLineScan(measurement.parameter)

        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))
