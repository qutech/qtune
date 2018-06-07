import pandas as pd
import numpy as np
from typing import Tuple, List, Sequence
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

    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        super().__init__()
        self._matlab = matlab_instance
        # TODO Add some more comments

        self._measurements = {'sensor 2d': Measurement('sensor 2d'),
                              'sensor': Measurement('sensor'),
                              'chrg': Measurement('chrg'),
                              'resp': Measurement('resp'),
                              'line': Measurement('line'),
                              'lead': Measurement('lead'),
                              'load': Measurement('load'),
                              'load pos': Measurement('load pos'),
                              'chrg rnd': Measurement('chrg rnd'),
                              'chrg s': Measurement('chrg s'),
                              'stp': Measurement('stp'),
                              'tl': Measurement('tl')}

        self._n_dqds = 3                                            # TODO property and use this from tunedata
        self._n_sensors = 2                                         # TODO property and use this from tunedata
        self._sensor_gates = [{'T': "LT", 'P': "LP", 'B': "LB"},    # TODO property and use this from tunedata
                              {'T': "RT", 'P': "RP", 'B': "RB"}]

    def measurements(self) -> Tuple[Measurement, ...]:
        return tuple(self._measurements.values())

    def gate_voltage_names(self) -> Tuple:
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

    def measure(self, measurement: Measurement) -> np.ndarray:
        """This function basically wraps the tune.m script on the Trition 200 setup"""
        if measurement._name not in self._measurements.keys():
            raise ValueError('Unknown measurement: {}'.format(measurement))

        result = self.pytune(measurement)

        if measurement._name == 'line':
            result = np.array(result['data'].ana.width)
        elif measurement._name == 'lead':
            result = np.array(result['data'].ana.fitParams[1][3])
        elif measurement._name == 'load':
            pass
        elif measurement._name == 'load pos':
            pass
        elif measurement._name == 'chrg rnd':
            pass
        elif measurement._name == 'chrg s':
            pass
        elif measurement._name == 'sensor':
            result = np.array(result['data'].ana.xVal, result['data'].ana.minGradData)
        elif measurement._name == 'sensor 2d':
            result = np.array([result['data'].ana.xVal, result['data'].ana.yVal])
        elif measurement._name == 'chrg':
            result = np.squeeze(result['data'].ana.O)
        elif measurement._name == 'stp':
            result = np.array([result['data'].ana.STp_x, result['data'].ana.STp_y])
        elif measurement._name == 'tl':
            result = np.array([result['data'].ana.Tp_x, result['data'].ana.Tp_y])

        return result


class SMQQDPassThru(Evaluator):
    """
    Pass thru Evaluator
    """
    def __init__(self, experiment: SMTuneQQD, measurements: List[Measurement],
                 parameters: List[str]):

        super().__init__(experiment, measurements, parameters)


    def evaluate(self) -> Tuple[pd.Series, pd.Series]:

        result = pd.Series(index=self._parameters)
        return_values = np.array([])

        # get all measurement results first
        for measurement in self.measurements:
            return_values = np.append(return_values, self.experiment.measure(measurement))

        # deal meas results to parameters
        # one value for each parameter since they have already been evaluated
        for parameter, value in zip(self.parameters, return_values):
             result[parameter]=value

        return result, pd.Series()

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    measurements=self.measurements,
                    parameters=self.parameters)





