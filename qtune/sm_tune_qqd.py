# qtune: Automated fine tuning and optimization
#
#   Copyright (C) 2019  Julian D. Teske and Simon S. Humpohl
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation version 3 of the License.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
################################################################################

# @email: julian.teske@rwth-aachen.de

import pandas as pd
import numpy as np
from typing import Tuple, List, Sequence
from numbers import Number
from itertools import count

from qtune.experiment import Experiment, Measurement
from qtune.evaluator import Evaluator
from qtune.sm import SpecialMeasureMatlab
from qtune import mat2py
from qtune.history import History


# This file bundles everything connected to the QQD
# Is this the file structure we want?

class QQDMeasurement(Measurement):
    """
    This class saves all necessary information for a measurement.
    """

    def __init__(self, name, **kwargs):
        self.data = None

    def to_hdf5(self):
        return dict(self.options,
                    name=self.name)


# TODO Add Tests
class SMTuneQQD(Experiment):
    """
    QQD implementation using the MATLAB backend and the tune.m script on the Trition 200 Setup
    """

    def __init__(self, matlab_instance: SpecialMeasureMatlab):
        super().__init__()
        self._matlab = matlab_instance
        # TODO Add some more comments

        self._last_file_name = None

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

        self._n_dqds = 3  # TODO property and use this from tunedata
        self._n_sensors = 2  # TODO property and use this from tunedata
        self._sensor_gates = [{'T': "LT", 'P': "LP", 'B': "LB"},  # TODO property and use this from tunedata
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
        current_gate_voltages[new_gate_voltages.index] = new_gate_voltages[new_gate_voltages.index]
        gate_voltages_to_matlab = current_gate_voltages.to_dict()
        for key, value in gate_voltages_to_matlab.items():
            gate_voltages_to_matlab[key] = float(value)
        self._matlab.engine.qtune.set_qqd_gate_voltages(gate_voltages_to_matlab)

        return self.read_gate_voltages()  # set_qqd_gate_voltages

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
            raise ValueError(f'Unknown measurement: {measurement}')

        for key in measurement.options:
            if isinstance(measurement.options[key], np.generic):
                measurement.options[key] = self._matlab.to_matlab(measurement.options[key])

        result = self.pytune(measurement)

        if measurement.name in ['line', 'lead', 'load', 'load pos', 'chrg rnd', 'chrg s', 'sensor', 'sensor 2d', 'chrg',
                                'resp']:
            ret = result
        elif measurement.name == 'stp':
            ret = np.array([result['data'].ana.STp_x, result['data'].ana.STp_y])
        elif measurement.name == 'tl':
            ret = np.array([result['data'].ana.Tp_x, result['data'].ana.Tp_y])
        else:
            raise ValueError(f'Measurement {measurement} not implemented')

        return ret

    def __repr__(self):
        return "{type}({data})".format(type=type(self).__name__,
                                       data=self._matlab.engine.matlab.engine.engineName())


class SMQQDPassThru(Evaluator):
    """
    Pass thru Evaluator
    """

    def __init__(self, experiment: SMTuneQQD, measurements: List[Measurement],
                 parameters: List[str], name: str, raw_x_data=tuple(), raw_y_data=tuple(), last_file_names=None,
                 error=None, reference_residual_sum=None):

        super().__init__(experiment, measurements, parameters, raw_x_data, raw_y_data, name=name)
        self._count = count(0)
        self._error = error
        if reference_residual_sum is None:
            self._reference_residual_sum = pd.Series(index=parameters, data=np.nan)
        elif not isinstance(reference_residual_sum, pd.Series):
            self._reference_residual_sum = pd.Series(index=parameters, data=reference_residual_sum)
        else:
            self._reference_residual_sum = reference_residual_sum
        self._n_error_estimate = 8
        self._last_file_names = last_file_names

    def evaluate_error(self):
        self.logger.info(f'Evaluating {self.parameters} {self._n_error_estimate} times to estimate error.')
        values = []

        self._error = pd.Series(index=self.parameters)
        for i in range(self._n_error_estimate):
            tmp, _ = self.evaluate()
            values.append(tmp)

        df = pd.DataFrame(values)
        self._error = df.var(0)

    def evaluate(self) -> Tuple[pd.Series, pd.Series]:
        self.logger.info(f'Evaluating {self.parameters}.')
        result = pd.Series(index=self._parameters)
        error = pd.Series(index=self._parameters)

        if self._error is None: self.evaluate_error()

        return_values = np.array([])
        gof = np.array([])
        sum_residuals = np.array([])
        variances = np.array([])
        self._last_file_names = []

        # get all measurement results first
        for measurement in self.measurements:
            measurement_result = self.experiment.measure(measurement)
            self._last_file_names.append(self.experiment.get_last_file_name())
            if measurement.name in ['lead']:
                return_values = np.append(return_values, measurement_result[0:-1])
                # gof = np.append(gof, np.full(len(return_values),measurement_result[-1]))
                sum_residuals = np.append(sum_residuals, np.full(len(measurement_result) - 1, measurement_result[-1]))
            elif measurement.name == 'line':
                return_values = np.append(return_values, measurement_result[0])
                variances = np.append(variances, measurement_result[1])
                sum_residuals = np.append(sum_residuals, np.full(len(measurement_result), np.nan))
            elif measurement.name == 'resp':
                return_values = np.append(return_values, measurement_result[0:4])
                for i in range(4):
                    if np.isclose(measurement_result[4 + i], 0) or measurement_result[4 + i] != measurement_result[
                        4 + i]:
                        sum_residuals = np.append(sum_residuals, [np.nan])
                    else:
                        sum_residuals = np.append(sum_residuals, measurement_result[4 + i])
            else:
                return_values = np.append(return_values, measurement_result)
                # gof = np.append(gof, np.full(len(return_values),np.nan))
                sum_residuals = np.append(sum_residuals, np.full(len(measurement_result), np.nan))

        # deal meas results to parameters
        # one value for each parameter since they have already been evaluated
        if measurement.name == 'line':
            for parameter, value, variance in zip(self.parameters, return_values, variances):
                result[parameter] = value
                error[parameter] = variance
        else:
            for parameter, value, sum_res in zip(self.parameters, return_values, sum_residuals):
                result[parameter] = value
                # The r square error is written to the return values and used to scale the error
                if sum_res == sum_res:
                    if self._reference_residual_sum[parameter] == self._reference_residual_sum[parameter]:
                        error[parameter] = self._error[parameter] * sum_res / self._reference_residual_sum[parameter]
                    else:
                        error[parameter] = self._error[parameter]
                else:
                    error[parameter] = self._error[parameter]

        self._raw_x_data = (next(self._count),)
        self._raw_y_data = tuple(result)
        return result, error

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    last_file_names=self._last_file_names,
                    error=self._error,
                    reference_residual_sum=self._reference_residual_sum)


class QQDLine(Evaluator):
    def __init__(self, experiment: SMTuneQQD, measurements: List[Measurement],
                 parameters: List[str], name: str, raw_x_data=tuple(), raw_y_data=tuple(), last_file_names=None,
                 maximal_value=np.inf, minimal_value=-np.inf):
        super().__init__(experiment, measurements, parameters, raw_x_data, raw_y_data, name=name)
        self._count = count(0)
        self._last_file_names = last_file_names
        self._maximal_value = maximal_value
        self._minimal_value = minimal_value

    def evaluate(self):
        self.logger.info(f'Evaluating {self.parameters}.')
        measurement_result = self.experiment.measure(self.measurements[0])
        value, variance = self.process_raw_data(measurement_result)
        return pd.Series(data=value, index=self.parameters), pd.Series(data=variance, index=self.parameters)

    def process_raw_data(self, raw_data):
        """
        This method averages the evaluations and takes the Variances.
        :param raw_data: return of measurement (Matlab struct)
        :return: parameter estimate and its estimated variance
        """
        width = np.full(len(raw_data['data'].ana), np.nan)
        for i in range(len(raw_data['data'].ana)):
            width[i] = raw_data['data'].ana[i].width
        value = np.mean(width)
        variance = np.var(width) / len(raw_data['data'].ana)
        self._last_file_names = raw_data['data'].args.fullFile
        if value < self._minimal_value or value > self._maximal_value:
            value = np.nan
        return value, variance

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    last_file_names=self._last_file_names,
                    maximal_value=self._maximal_value,
                    minimal_value=self._minimal_value)


class QQDLead(Evaluator):
    def __init__(self, experiment: SMTuneQQD, measurements: List[Measurement],
                 parameters: List[str], name: str, raw_x_data=tuple(), raw_y_data=tuple(), last_file_names=None,
                 error: float = 1, reference_residual_sum: float = 1., maximal_value=np.inf, minimal_value=-np.inf):
        super().__init__(experiment, measurements, parameters, raw_x_data, raw_y_data, name=name)
        self._error = error
        self._reference_residual_sum = reference_residual_sum
        self._last_file_names = last_file_names
        self._maximal_value = maximal_value
        self._minimal_value = minimal_value

    def evaluate(self) -> Tuple[pd.Series, pd.Series]:
        self.logger.info(f'Evaluating {self.parameters}.')
        measurement_result = self.experiment.measure(self.measurements[0])
        value, variance = self.process_raw_data(measurement_result)
        return pd.Series(data=value, index=self.parameters), pd.Series(data=variance, index=self.parameters)

    def process_raw_data(self, raw_data):
        value = raw_data['data'].ana.fitParams[1][3]
        sum_residuals = raw_data['data'].ana.sumResiduals
        self._last_file_names = raw_data['data'].args.fullFile
        if value < self._minimal_value or value > self._maximal_value:
            value = np.nan
        return value, self._error * sum_residuals / self._reference_residual_sum

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    last_file_names=self._last_file_names,
                    error=self._error,
                    reference_residual_sum=self._reference_residual_sum,
                    maximal_value=self._maximal_value,
                    minimal_value=self._minimal_value)


class QQDSensor1D(Evaluator):
    def __init__(self, experiment: SMTuneQQD, measurements: List[Measurement],
                 parameters: List[str], name: str, raw_x_data=tuple(), raw_y_data=tuple(), last_file_names=None):
        super().__init__(experiment, measurements, parameters, raw_x_data, raw_y_data, name=name)
        self._last_file_names = last_file_names

    def evaluate(self) -> Tuple[pd.Series, pd.Series]:
        self.logger.info(f'Evaluating {self.parameters}.')
        measurement_result = self.experiment.measure(self.measurements[0])
        position, slope = self.process_raw_data(measurement_result)
        return pd.Series(data=[position, slope], index=self.parameters), pd.Series(data=[np.nan, np.nan],
                                                                                   index=self.parameters)

    def process_raw_data(self, raw_data):
        position = raw_data['data'].ana.xVal
        slope = np.abs(raw_data['data'].ana.minGradData)
        self._last_file_names = raw_data['data'].args.fullFile
        return position, slope

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    last_file_names=self._last_file_names)


class QQDSensor2D(Evaluator):
    def __init__(self, experiment: SMTuneQQD, measurements: List[Measurement],
                 parameters: List[str], name: str, raw_x_data=tuple(), raw_y_data=tuple(), last_file_names=None):
        super().__init__(experiment, measurements, parameters, raw_x_data, raw_y_data, name=name)
        self._last_file_names = last_file_names

    def evaluate(self) -> Tuple[pd.Series, pd.Series]:
        self.logger.info(f'Evaluating {self.parameters}.')
        measurement_result = self.experiment.measure(self.measurements[0])
        position_x, position_y, slope = self.process_raw_data(measurement_result)
        return pd.Series(data=[position_x, position_y, slope], index=self.parameters), \
               pd.Series(data=[np.nan, np.nan, np.nan], index=self.parameters)

    def process_raw_data(self, raw_data):
        position_x = raw_data['data'].ana.xVal
        position_y = raw_data['data'].ana.yVal
        slope = np.abs(raw_data['data'].ana.minGradData)
        self._last_file_names = raw_data['data'].args.fullFile
        return position_x, position_y, slope

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    last_file_names=self._last_file_names)


class QQDCharge(Evaluator):
    def __init__(self, experiment: SMTuneQQD, measurements: List[Measurement],
                 parameters: List[str], name: str, raw_x_data=tuple(), raw_y_data=tuple(), last_file_names=None,
                 error=1):
        super().__init__(experiment, measurements, parameters, raw_x_data, raw_y_data, name=name)
        self._last_file_names = last_file_names
        self._error = error

    def evaluate(self) -> Tuple[pd.Series, pd.Series]:
        self.logger.info(f'Evaluating {self.parameters}.')
        measurement_result = self.experiment.measure(self.measurements[0])
        positions, errors = self.process_raw_data(measurement_result)
        return pd.Series(data=positions, index=self.parameters), \
               pd.Series(data=errors, index=self.parameters)

    def process_raw_data(self, raw_data):
        positions = np.squeeze(raw_data['data'].ana.O)
        errors = np.full(shape=positions.shape, fill_value=self._error)
        self._last_file_names = raw_data['data'].args.fullFile
        return positions, errors

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    last_file_names=self._last_file_names,
                    error=self._error)


class QQDDistTP(Evaluator):
    def __init__(self, experiment: SMTuneQQD, measurements: List[Measurement],
                 parameters: List[str], name: str, raw_x_data=tuple(), raw_y_data=tuple(), last_file_names=None,
                 error=1):
        super().__init__(experiment, measurements, parameters, raw_x_data, raw_y_data, name=name)
        self._last_file_names = last_file_names
        self._error = error

    def evaluate(self) -> Tuple[pd.Series, pd.Series]:
        self.logger.info(f'Evaluating {self.parameters}.')
        measurement_result = self.experiment.measure(self.measurements[0])
        distance, errors = self.process_raw_data(measurement_result)
        return pd.Series(data=distance, index=self.parameters), \
               pd.Series(data=errors, index=self.parameters)

    def process_raw_data(self, raw_data):
        distance = np.linalg.norm(
            np.array(raw_data['data'].ana.triplePointRight) - np.array(raw_data['data'].ana.triplePointLeft))
        self._last_file_names = raw_data['data'].args.fullFile
        return distance, self._error

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    last_file_names=self._last_file_names,
                    error=self._error)


class QQDResp(Evaluator):
    def __init__(self, experiment: SMTuneQQD, measurements: List[Measurement],
                 parameters: List[str], name: str, raw_x_data=tuple(), raw_y_data=tuple(), last_file_names=None,
                 error=1, min_transition_height=-np.inf, min_position=None, max_position=None):
        super().__init__(experiment, measurements, parameters, raw_x_data, raw_y_data, name=name)
        self._last_file_names = last_file_names
        self._error = pd.Series(data=error, index=self.parameters)
        self._min_transition_height = min_transition_height
        if min_position is None:
            self._min_position = pd.Series(index=self.parameters, data=-np.inf)
        elif isinstance(min_position, pd.Series):
            self._min_position = min_position
        else:
            self._min_position = pd.Series(index=self.parameters, data=min_position)
        if max_position is None:
            self._max_position = pd.Series(index=self.parameters, data=-np.inf)
        elif isinstance(max_position, pd.Series):
            self._max_position = max_position
        else:
            self._max_position = pd.Series(index=self.parameters, data=max_position)

    def evaluate(self) -> Tuple[pd.Series, pd.Series]:
        self.logger.info(f'Evaluating {self.parameters}.')
        measurement_result = self.experiment.measure(self.measurements[0])
        positions, errors = self.process_raw_data(measurement_result)
        for i in range(len(positions)):
            if positions[i] < self._min_position.iloc[i] or positions[i] > self._max_position.iloc[i]:
                positions[i] = np.nan
        return pd.Series(data=positions, index=self.parameters), \
               pd.Series(data=errors, index=self.parameters)

    def process_raw_data(self, raw_data):
        n = 2
        # len(raw_data['data'].ana)
        positions = np.full(n, np.nan)
        for i in range(n):
            positions[i] = raw_data['data'].ana[i].position
            if raw_data['data'].ana[i].visiblity < self._min_transition_height:
                positions[i] = np.nan
        self._last_file_names = raw_data['data'].args.fullFile
        return positions, self._error

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    last_file_names=self._last_file_names,
                    error=self._error,
                    min_transition_height=self._min_transition_height,
                    min_position=self._min_position,
                    max_position=self._max_position)


def reevaluate_data_with_matlab(experiment, history: History, evaluator_name, indices, hold=True):
    if isinstance(indices, int):
        indices = [indices, ]
    results = []
    for i in indices:
        evaluator_data = history._evaluator_data[evaluator_name].iloc[i]
        measurement = evaluator_data.measurements[0]
        measurement.options['loadFile'] = evaluator_data._last_file_names
        results.append(experiment.measure(measurement))
        if hold:
            _ = input('Type anything to continue.')
    return results
