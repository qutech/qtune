import pandas as pd
import numpy as np
from typing import Tuple
from numbers import Number

from qtune.experiment import Experiment, Measurement
from qtune.evaluator import Evaluator
from qtune.sm import SpecialMeasureMatlab
from qtune import mat2py

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
                              'chrg': Measurement('chrg'),
                              'resp': Measurement('resp'),
                              'line': Measurement('line'),
                              'lead': Measurement('lead'),
                              'load': Measurement('load'),
                              'load pos': Measurement('load pos'),
                              'chrg rnd': Measurement('chrg rnd'),
                              'chrg s': Measurement('chrg s'),
                              'tlp': Measurement('tlp')}

        self._n_dqds = 3
        self._n_sensors = 2
        self._sensor_gates = [{'T': "LT", 'P': "LP", 'B': "LB"},
                              {'T': "RT", 'P': "RP", 'B': "RB"}]

    def measurements(self) -> Tuple[Measurement, ...]:
        return tuple(self._measurements.values())

    def gate_voltage_names(self) -> Tuple:
        # TODO change MATLAB gate names or put them in the python part
        return tuple(sorted(self._matlab.engine.qtune.read_qqd_gate_voltages().keys()))  # Why call MATLAB engine here?

    def read_gate_voltages(self) -> pd.Series:
        return pd.Series(self._matlab.engine.qtune.read_qqd_gate_voltages()).sort_index()

    def _read_sensing_dot_voltages(self) -> pd.Series:
        # TODO: Maybe allow for getting only one sensor at a time?
        # TODO change MATLAB gate names or put them in the python part
        return pd.Series(self._matlab.engine.qtune.read_qqd_sensing_dot_voltages()).sort_index()

    def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:
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

    def tune(self, measurement_name, index: np.int, **kwargs) -> pd.Series:
        # Tune wrapper using the MATLAB syntax

        # data ---  args        <<<< contains struct arrays!
        #       |-  data        <<<< Important RAW data
        #       |-  ana         <<<< Result of analysis (if run!)
        #       |-  successful  <<<< duh!
        # Tune usage                         Operation string , INDEX (int)        , name value pair parameters

        options = kwargs
        tune_view = mat2py.MATLABFunctionView(self._matlab.engine, 'tune.tune')

        if options:
            for parameter, value in options.items():
                if isinstance(value, Number):
                    options[parameter] = float(value)

          # kwargs 2 name value pairs
            keys = list(options.keys())
            vals = list(options.values())
            name_value_pairs = [x for t in zip(keys, vals) for x in t]

            data_view = tune_view(measurement_name, *([np.float(index)] + name_value_pairs))
        else:
            data_view = tune_view(measurement_name, index)

        return pd.Series({'data': data_view})

    def pytune(self, measurement) -> pd.Series:
        # Tune wrapper using the autotune syntax
        options = dict(measurement.options)
        index = options['index']
        del options['index']

        result = self.tune(measurement_name=measurement._name, index=index, **options)

        return result

    def measure_legacy(self, measurement: Measurement) -> pd.Series:
        """This function basically wraps the tune.m script on the Trition 200 setup"""
        if measurement._name not in self._measurements.keys():
            raise ValueError('Unknown measurement: {}'.format(measurement))

        result = self.pytune(measurement)

        return result

    def measure(self, measurement: Measurement) -> pd.Series:
        """This function basically wraps the tune.m script on the Trition 200 setup"""
        if measurement._name not in self._measurements.keys():
            raise ValueError('Unknown measurement: {}'.format(measurement))

        result = self.pytune(measurement)

        if measurement._name == 'line':
            result = result['data'].ana.width
        elif measurement._name == 'lead':
            pass
        elif measurement._name == 'load':
            pass
        elif measurement._name == 'load pos':
            pass
        elif measurement._name == 'sensor':
            pass
        elif measurement._name == 'sensor 2d':
            pass
        elif measurement._name == 'chrg':
            pass
        elif measurement._name == 'chrg rnd':
            pass
        elif measurement._name == 'chrg s':
            pass
        elif measurement._name == 'tlp':
            pass

        return result


class SMQQDPassThru(Evaluator):
    """
    Pass thru Evaluator
    """
    def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                 parameters: pd.Series() = pd.Series({'pass_thru_parameter': np.nan})):

        if measurements is None:
            self._measurements = experiment.measurements

        super().__init__(experiment, measurements, parameters)

        if not parameters:
            self._parameters = pd.Series()

    def evaluate(self) -> pd.Series:

        for measurement in self.measurements:  # should just be one here, nasty hack since measurements is a tuple
            self._parameters[measurement._name] = self.experiment.measure(measurement)

        return self._parameters

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    measurements=self.measurements,
                    parameters=self.parameters)


class SMQQDLineScan(Evaluator):
    """
    Adiabaticly sweeps the detune over the transition between the (2,0) and the (1,1) region for ith DQD. An Scurve is
    fitted and the width calculated as parameter for the inter dot coupling. Fitted with Matlab functions.
    Can be replaced by python  code
    """
    def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                 parameters: pd.Series() = pd.Series({'tunnel_coupling': np.nan})):

        # This seems weired since the parameter is to be returned ^^^^^^^^^^^^^^^^
        if measurements is None:
            self._measurements = experiment._measurements['line']
            # can we prevent hardcoding indices or accessing private vars here?
        super().__init__(experiment, measurements, parameters)

    def evaluate(self) -> pd.Series:
        tunnel_coupling = tuple()
        failed = tuple()

        for measurement in self.measurements:  # should just be one here, nasty hack since measurements is a tuple
            data = self.experiment.measure_legacy(measurement)
            tunnel_coupling += (data['data'].ana.width,)
            failed += (bool(data['data'].successful),)

        return pd.Series({'tunnel_coupling': tunnel_coupling, 'failed': failed})

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    measurements=self.measurements,
                    parameters=self.parameters)


class SMQQDLeadScan(Evaluator):
    """
    foo
    """

    def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                 parameters: pd.Series() = pd.Series({'lead_time': np.nan})):

        # This seems weired since the parameter is to be returned ^^^^^^^^^^^^^^^^
        if measurements is None:
            measurements = experiment._measurements[
                    'lead']  # can we pevent hardcoding indices or accessing private vars here?
        super().__init__(experiment, measurements, parameters)

    def evaluate(self) -> pd.Series:
        data = pd.Series()
        failed = True
        lead_time = np.nan
        for measurement in self.measurements:
                data['measurement'] = self.experiment.measure(measurement)

        # TODO Process data

                return pd.Series({'lead_time': lead_time, 'failed': failed})

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    measurements=self.measurements,
                    parameters=self.parameters)


class SMQQDSensor2d(Evaluator):
        """
        foo
        """

        def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                     parameters: pd.Series() = pd.Series({'sensor_position_2d': np.full(2, np.nan)})):

            if measurements is None:
                measurements = experiment._measurements['sensor']

            super().__init__(experiment, measurements, parameters)

        def evaluate(self) -> pd.Series:
            data = pd.Series()
            failed = True
            position = np.full(2, np.nan)

            for measurement in self.measurements:
                data['measurement'] = self.experiment.measure(measurement)

                # TODO Process data

            return pd.Series({'sensor_position_2d': position, 'failed': failed})

        def to_hdf5(self):
            return dict(experiment=self.experiment,
                        measurements=self.measurements,
                        parameters=self.parameters)


class SMQQDSensor(Evaluator):
            """
            foo
            """

            def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                         parameters: pd.Series() = pd.Series({'sensor_position': np.nan})):

                if measurements is None:
                    measurements = experiment._measurements['sensor']
                    # can we pevent hardcoding indices or accessing private vars here?
                super().__init__(experiment, measurements, parameters)

            def evaluate(self) -> pd.Series:
                data = pd.Series()
                failed = True
                position = np.nan

                for measurement in self.measurements:
                    data['measurement'] = self.experiment.measure(measurement)

                    # TODO Process data

                    return pd.Series({'sensor_position': position, 'failed': failed})

            def to_hdf5(self):
                return dict(experiment=self.experiment,
                            measurements=self.measurements,
                            parameters=self.parameters)


class SMQQDJacobian(Evaluator):

            def __init__(self, experiment: SMTuneQQD, measurements: Tuple[Measurement],
                         parameters: pd.Series() = pd.Series({'jacobian': None})):

                if measurements is None:
                    measurements = experiment._measurements['sensor']
                    # can we pevent hardcoding indices or accessing private vars here?
                super().__init__(experiment, measurements, parameters)

            def evaluate(self) -> pd.Series:
                data = pd.Series()
                failed = True
                jacobian = pd.DataFrame()

                for measurement in self.measurements:
                    data['measurement'] = self.experiment.measure(measurement)

            # TODO Process data

                return pd.Series({'jacobian': jacobian, 'failed': failed})

            def to_hdf5(self):
                return dict(experiment=self.experiment,
                            measurements=self.measurements,
                            parameters=self.parameters)


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
        position = np.nan

        for measurement in self.measurements:
            data['measurement'] = self.experiment.measure(measurement)

        # TODO Process data

            return pd.Series({'measp': position, 'failed': failed})

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    measurements=self.measurements,
                    parameters=self.parameters)
