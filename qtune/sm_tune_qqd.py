import pandas as pd
import numpy as np
from typing import Tuple, List, Sequence
from numbers import Number
from itertools import count

from qtune.experiment import Experiment, Measurement
from qtune.evaluator import Evaluator
from qtune.sm import SpecialMeasureMatlab
from qtune import mat2py

# This file bundles everything connected to the QQD
# Is this the file structure we want?


# TODO Add Tests
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
        # are _n_dqds and _n_sensors used anywhere?
        # The specification of sensor gates is meant to be done implicitly by the use of the parameter tuner. Julian

    def measurements(self) -> Tuple[Measurement, ...]:
        return tuple(self._measurements.values())

    def gate_voltage_names(self) -> Tuple:
        return tuple(sorted(self._matlab.engine.qtune.read_qqd_gate_voltages().keys()))  # Why call MATLAB engine here?
    # This way the gate names are stored in only one place, which is a matlab file. Julian

    def read_gate_voltages(self) -> pd.Series:
        return pd.Series(self._matlab.engine.qtune.read_qqd_gate_voltages()).sort_index()


    def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:

        current_gate_voltages = self.read_gate_voltages()
        #current_gate_voltages[new_gate_voltages.index] = new_gate_voltages[new_gate_voltages.index]
        gate_voltages_to_matlab= new_gate_voltages.to_dict()
        for key, value in gate_voltages_to_matlab.items():
            gate_voltages_to_matlab[key] = float(value)
        self._matlab.engine.qtune.set_qqd_gate_voltages(gate_voltages_to_matlab)
        
        return self.read_gate_voltages() # set_qqd_gate_voltages


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
            data_view = tune_view(measurement_name, np.float(index))

        return pd.Series({'data': data_view})

    def pytune(self, measurement) -> pd.Series:
        # Tune wrapper using the autotune syntax
        # where do your Measurements get the option 'index'?
        options = dict(measurement.options)
        index = options['index']
        del options['index']

        result = self.tune(measurement_name=measurement.name, index=index, **options)

        return result

    def measure_legacy(self, measurement: Measurement) -> pd.Series:
        """This function basically wraps the tune.m script on the Trition 200 setup"""
        if measurement.name not in self._measurements.keys():
            raise ValueError('Unknown measurement: {}'.format(measurement))

        result = self.pytune(measurement)

        return result

    def measure(self, measurement: Measurement) -> np.ndarray:
        """This function basically wraps the tune.m script on the Trition 200 setup"""
        # TODO allow this to create the measurement on the fly when arguments cntrl, index, options are passed
        if measurement.name not in self._measurements.keys():
            raise ValueError('Unknown measurement: {}'.format(measurement))

        result = self.pytune(measurement)

        if measurement.name == 'line':
            ret = np.array(result['data'].ana.width)
        elif measurement.name == 'lead':
            ret = np.array(result['data'].ana.fitParams[1][3])
        elif measurement.name == 'load':
            ret = result
        elif measurement.name == 'load pos':
            ret = result
        elif measurement.name == 'chrg rnd':
            ret = result
        elif measurement.name == 'chrg s':
            ret = result
        elif measurement.name == 'sensor':
            ret = np.array([result['data'].ana.xVal, np.abs(result['data'].ana.minGradData)])
        elif measurement.name == 'sensor 2d':
            ret = np.array([result['data'].ana.xVal, result['data'].ana.yVal, np.abs(result['data'].ana.minGradData)])
        elif measurement.name == 'chrg':
            ret = np.squeeze(result['data'].ana.O)
        elif measurement.name == 'stp':
            ret = np.array([result['data'].ana.STp_x, result['data'].ana.STp_y])
        elif measurement.name == 'tl':
            ret = np.array([result['data'].ana.Tp_x, result['data'].ana.Tp_y])
        elif measurement.name == 'resp':
            n = len(result['data'].ana)
            ret = np.full(n, np.nan)
            for i in range(n):
                ret[i] = result['data'].ana[i].position

        return ret


class SMQQDPassThru(Evaluator):
    """
    Pass thru Evaluator
    """
    def __init__(self, experiment: SMTuneQQD, measurements: List[Measurement],
                 parameters: List[str]):

        super().__init__(experiment, measurements, parameters, tuple(), tuple(), 'PassThruEvaluator')
        self._count = count(0)

    def evaluate(self) -> Tuple[pd.Series, pd.Series]:

        result = pd.Series(index=self._parameters)
        error = pd.Series(index=self._parameters)

        return_values = np.array([])

        # get all measurement results first
        for measurement in self.measurements:
            return_values = np.append(return_values, self.experiment.measure(measurement))

        # deal meas results to parameters
        # one value for each parameter since they have already been evaluated
        for parameter, value in zip(self.parameters, return_values):
            result[parameter] = value
            error[parameter] = np.abs((value/100)**2)   # Estimate the error as 10% of the value


        self._raw_x_data = (next(self._count),)
        self._raw_y_data = tuple(result)
        return result, error

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    measurements=self.measurements,
                    parameters=self.parameters)
