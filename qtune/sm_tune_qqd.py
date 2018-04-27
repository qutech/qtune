import pandas as pd
import numpy as np
from typing import Tuple, Sequence
from numbers import Number

from qtune.experiment import Experiment, Measurement
from qtune.evaluator import Evaluator
from qtune.sm import SpecialMeasureMatlab
from qtune.storage import HDF5Serializable

# This file bundles everything connected to the QQD
# Is this the file structure we want?

# TODO Add Tests
# TODO QQB or general QDArray?
class SMTuneQQD(Experiment):
    """
    QQD implementation using the MATLAB backend and the tune.m script on the Trition 200 Setup
    """
    # TODO wrap special functions connected to tune runs and tunedata!
    # TODO Either allow for better control of scan parameters or provide interface to tunedata?

    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        super().__init__()
        self._matlab = matlab_instance
        # TODO Load tunedata to Python or interface it?

        # TODO List all possible arguments for Measurements here!
        self._measurements = {'sensor_2d': Measurement('sensor_2d'),
                              'sensor': Measurement('sensor'),
                              'chrg': None,
                              'resp': None,
                              'line': None,
                              'lead': None,
                              'jac': None,  # Probably part of the autotuner
                              'measp': None}

        self._n_dqds = 3
        self._n_sensors = 2
        self._sensor_gates = [{'T': "LT", 'P': "LP", 'B': "LB"},
                         {'T': "RT", 'P': "RP", 'B': "RB"}]

    def measurements(self) -> Tuple[Measurement, ...]:
        return tuple(self._measurements.values())

    def gate_voltage_names(self) -> Tuple:
        return tuple(sorted(self._matlab.engine.qtune.read_qqd_gate_voltages().keys())) # Why call MATLAB engine here?

    def read_gate_voltages(self) -> pd.Series:
        return pd.Series(self._matlab.engine.qtune.read_qqd_gate_voltages()).sort_index()

    def _read_sensing_dot_voltages(self) -> pd.Series:
        # TODO: Maybe allow for getting only one sensor at a time?
        return pd.Series(self._matlab.engine.qtune.read_qqd_sensing_dot_voltages()).sort_index()

    # Deprecated
    def tune_sensing_dot_1d(self, sensor: int, stepping_gate=None):
        """Provide functionality for 1d sensor dot tuning"""

        if sensor <= 0 or sensor >= len(self._sensors):
            raise ValueError("Sensor index out of range!")

        sensing_dot_measurement = self._measurements['sensor']
        sensing_dot_measurement.parameters.index = sensor

        # check stepping gate
        if stepping_gate is not None:
            if stepping_gate not in self._sensors[sensor].values():
                raise ValueError("Stepping gate clashes with selected sensor!")
            else:
                sensing_dot_measurement.parameters.stepping_gate=stepping_gate

        # Execute measurement and store data
        # TODO Pass data for further analysis by autotuner?
        data = self.measure(sensing_dot_measurement)

        # Set gates here or in MATLAB script?
        # detuning = qtune.util.find_stepes_point_sensing_dot(data, scan_range=scan_range, npoints=n_points)
        # self._set_sensing_dot_voltages(pd.Series({gate: prior_position + detuning}))

    # Deprecated
    def tune_sensing_dot_2d(self, sensor: int):
        """Provide functionality for 2d sensor dot tuning"""
        # TODO Should these functions allow for passing of kwargs to measurement?
        # TODO decide how much is done in MATLAB script. The autotuner could be completely agnostic of the sensor gates!
        # check if sensor index is in range
        if sensor <= 0 or sensor >= len(self._sensors):
            raise ValueError("Sensor index out of range!")

        # Read values of top and bottom gate -> also automatically done by the MATLAB script,
        # no need to go back and forth
        positions = self._read_sensing_dot_voltages()
        T_position = positions[self._sensors[sensor]['T']]
        B_position = positions[self._sensors[sensor]['B']]

        # Standard arguments are provided on a MATLAB level by tune script
        sensing_dot_measurement = self._measurements['sensor_2d']
        sensing_dot_measurement.parameters.index = sensor

        # Start Measurement
        # Data processing and further adjustments are done on the MATLAB level for now
        # -> pass data for further analysis if needed for the tuning process
        data = self.measure(sensing_dot_measurement)

        # 1d tuning at the end is done by the MATLAB script,
        # we could of course disable it and call it here if needed for the tuning stack
        # self.tune_sensing_dot_1d(gate=gate_T)

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
        # currently handled in MATLAB
        current_sensing_dot_voltages = self._read_sensing_dot_voltages()
        for key in current_sensing_dot_voltages.index.tolist():
            if key not in new_sensing_dot_voltage.index.tolist():
                new_sensing_dot_voltage[key] = current_sensing_dot_voltages[key]
        new_sensing_dot_voltage = dict(new_sensing_dot_voltage)
        for key in new_sensing_dot_voltage:
            new_sensing_dot_voltage[key] = new_sensing_dot_voltage[key].item()
        self._matlab.engine.qtune.set_sensing_dot_gate_voltages()

    def measure(self, measurement: Measurement) -> np.ndarray:
        """This function basically wraps the tune.m script on the Trition 200 setup"""
        if measurement not in self._measurements:
            raise ValueError('Unknown measurement: {}'.format(measurement))

        # Make sure ints are converted to float
        parameters = measurement.parameter.copy()
        for parameter, value in parameters.items():
            if isinstance(value, Number):
                parameters[parameter] = float(value)

        # check data structure of returned values -> Most likely MATLAB struct
        data = self._matlab.engine.tune.tune(measurement, parameters['index'], parameters)
        return data

class SMQQDLineScan(Evaluator):
    """
    Adiabaticly sweeps the detune over the transition between the (2,0) and the (1,1) region for ith DQD. An Scurve is fitted and
    the width calculated as parameter for the inter dot coupling. Fitted with Matlab functions. Can be replaced by python  code
    """
    def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                 parameters: pd.Series() = pd.Series({'tunnel_coupling': np.nan})):

        # This seems weired since the parameter is to be returned ^^^^^^^^^^^^^^^^
        if measurements is None:
            measurements = experiment._measurements['line'] # can we prevent hardcoding indices or accessing private vars here?
        super().__init__(experiment,measurements, parameters)


    def evaluate(self) -> pd.Series:
        data = pd.Series()
        failed = True
        tc = np.nan

        for measurement in self.measurements:
            data['measurement'] = self.experiment.measure(measurement)

            # TODO Process data
            # TODO Append to Series -> How do we connect measurements and parameters?


        return pd.Series((tc, failed), ('parameter_tunnel_coupling', 'failed'))

class SMQQDLeadScan(Evaluator):
    """
    foo
    """

    def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                 parameters: pd.Series() = pd.Series({'lead time': np.nan})):

        # This seems weired since the parameter is to be returned ^^^^^^^^^^^^^^^^
        if measurements is None:
           measurements = experiment._measurements[
                    'lead']  # can we pevent hardcoding indices or accessing private vars here?
        super().__init__(experiment, measurements, parameters)

    def evaluate(self) -> pd.Series:
        data = pd.Series()
        failed = True
        tc = np.nan
        for measurement in self.measurements:
                data['measurement'] = self.experiment.measure(measurement)

        # TODO Process data
        # TODO Append to Series -> How do we connect measurements and parameters?

        return pd.Series((tc, failed), ('parameter_tunnel_coupling', 'failed'))

class SMQQDSensor2d(Evaluator):
        """
        foo
        """

        def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                     parameters: pd.Series() = pd.Series({'sensor_position_2d': np.full(2,np.nan)})):

            if measurements is None:
                measurements = experiment._measurements['sensor']

            super().__init__(experiment, measurements, parameters)

        def evaluate(self) -> pd.Series:
            data = pd.Series()
            failed = True
            tc = np.nan

            for measurement in self.measurements:
                data['measurement'] = self.experiment.measure(measurement)

                # TODO Process data
                # TODO Append to Series -> How do we connect measurements and parameters?

            return pd.Series((tc, failed), ('sensor_position', 'failed'))

class SMQQDSensor(Evaluator):
            """
            foo
            """

            def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                         parameters: pd.Series() = pd.Series({'sensor_position': np.nan})):


                if measurements is None:
                    measurements = experiment._measurements['sensor']  # can we pevent hardcoding indices or accessing private vars here?
                super().__init__(experiment, measurements, parameters)

            def evaluate(self) -> pd.Series:
                data = pd.Series()
                failed = True
                tc = np.nan

                for measurement in self.measurements:
                    data['measurement'] = self.experiment.measure(measurement)

                    # TODO Process data
                    # TODO Append to Series -> How do we connect measurements and parameters?

                return pd.Series((tc, failed), ('sensor_position_2d', 'failed'))

class SMQQDJacobian(Evaluator):

            def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                         parameters: pd.Series() = pd.Series({'jacobian': np.nan})):

                if measurements is None:
                    measurements = experiment._measurements['sensor']  # can we pevent hardcoding indices or accessing private vars here?
                super().__init__(experiment, measurements, parameters)

            def evaluate(self) -> pd.Series:
                data = pd.Series()
                failed = True
                tc = np.nan

                for measurement in self.measurements:
                    data['measurement'] = self.experiment.measure(measurement)

            # TODO Process data
            # TODO Append to Series -> How do we connect measurements and parameters?

                return pd.Series((tc, failed), ('sensor_position_2d', 'failed'))

class SMQQDMeasuremetPoint(Evaluator):

    def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                 parameters: pd.Series() = pd.Series({'measp': np.nan})):

        if measurements is None:
            measurements = experiment._measurements[
                'sensor']  # can we pevent hardcoding indices or accessing private vars here?
        super().__init__(experiment, measurements, parameters)

    def evaluate(self) -> pd.Series:
        data = pd.Series()
        failed = True
        tc = np.nan

        for measurement in self.measurements:
            data['measurement'] = self.experiment.measure(measurement)

        # TODO Process data
        # TODO Append to Series -> How do we connect measurements and parameters?

        return pd.Series((tc, failed), ('sensor_position_2d', 'failed'))