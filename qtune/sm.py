"""special-measure backend
"""
import warnings
import io
import functools
import sys
from typing import Tuple, Sequence

import matlab.engine
import pandas

from qtune.experiment import *


def redirect_output(func):
    return functools.partial(func, stdout=sys.stdout, stderr=sys.stderr)


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

        if 'smdata' not in self.engine.workspace:
            if special_measure_setup_script:
                getattr(self._engine, special_measure_setup_script)()

                if 'smdata' not in self.engine.workspace:
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
                gate_voltages: pandas.Series,
                measurement: Measurement) -> pandas.Series:

        if measurement == 'line_scan':
            return pandas.Series()

        elif measurement == 'charge_scan':
            return pandas.Series()

        else:
            raise ValueError('Unknown measurement: {}'.format(measurement))


class LegacyDQD(BasicDQD):
    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        super().__init__(matlab_instance)

        self.default_lead_scan = Measurement('lead_scan')

    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        return *super().measurements, self.default_lead_scan

    def measure(self,
                gate_voltages: pandas.Series,
                measurement: Measurement):
        if measurement == 'lead_scan':
            return pandas.Series()
        else:
            return super().measure(gate_voltages, measurement)
