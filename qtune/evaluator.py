from qtune.experiment import Experiment, Measurement
from typing import Sequence, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import optimize
import qtune.util
import scipy.ndimage
import logging
from qtune.storage import HDF5Serializable


class Evaluator(metaclass=HDF5Serializable):
    """
    The evaluator classes conduct measurements, calculate the parameters with the scan data and save the results in the
    HDF5 library.
    """
    def __init__(self,
                 experiment: Experiment,
                 measurements: Sequence[Measurement],
                 parameters: Sequence[str],
                 raw_x_data: Tuple[Optional[np.ndarray]],
                 raw_y_data: Tuple[Optional[np.ndarray]],
                 name: str):
        self._experiment = experiment
        self._measurements = tuple(measurements)  # Is this the behaviour that was intended?
        self._parameters = tuple(parameters)
        self._raw_x_data = raw_x_data
        self._raw_y_data = raw_y_data
        self._name = name

    @property
    def logger(self):
        return logging.getLogger(name="qtune")

    @property
    def name(self):
        return self._name

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @property
    def measurements(self) -> Sequence[Measurement]:
        return self._measurements

    @property
    def parameters(self) -> Sequence[str]:
        return self._parameters

    @property
    def raw_data(self):
        return self._raw_x_data, self._raw_y_data

    def evaluate(self) -> (pd.Series, pd.Series):
        """ This function is necessary. It is designed to conduct the measurements on the experiment. """
        raise NotImplementedError()

    def process_raw_data(self, raw_data):
        """ This function is optional. It can be used to separate the scan (typically applying a pulse to the
        experiment) from the calculation on the raw data (e.g. fitting). This sepration facilitates the recalculation
        of parameters from stored data."""
        raise NotImplementedError()

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    measurements=self.measurements,
                    parameters=self.parameters,
                    raw_x_data=self._raw_x_data,
                    raw_y_data=self._raw_y_data,
                    name=self.name)

    def __repr__(self):
        return "{type}({data})".format(type=type(self).__name__, data=self.to_hdf5())


class FittingEvaluator(Evaluator):
    """ This evaluator is designed for the evaluation of data by fits. """
    def __init__(self,
                 experiment: Experiment,
                 measurements: Sequence[Measurement],
                 parameters: Sequence[str],
                 raw_x_data: Tuple[Optional[np.ndarray]],
                 raw_y_data: Tuple[Optional[np.ndarray]],
                 fit_results: Optional[pd.Series],
                 initial_fit_arguments: Optional,
                 name: str):

        self._fit_results = fit_results
        self._initial_fit_arguments = initial_fit_arguments
        super().__init__(experiment=experiment, measurements=measurements, parameters=parameters,
                         raw_x_data=raw_x_data, raw_y_data=raw_y_data, name=name)

    @property
    def fit_results(self) -> Optional[pd.Series]:
        return self._fit_results

    @property
    def initial_fit_arguments(self):
        return self._initial_fit_arguments

    def evaluate(self) -> (pd.Series, pd.Series):
        raise NotImplementedError()

    def process_raw_data(self, raw_data):
        raise NotImplementedError()

    def to_hdf5(self):
        parent_dict = super().to_hdf5()
        return dict(parent_dict,
                    fit_results=self.fit_results,
                    initial_fit_arguments=self.initial_fit_arguments)


class LeadTunnelTimePrefitVersion(FittingEvaluator):
    def __init__(self, experiment: Experiment,
                 parameters: Sequence[str]=('parameter_time_rise', 'parameter_time_fall'),
                 measurements: Tuple[Measurement] = None,
                 sample_rate: float=1e8,
                 raw_x_data: Tuple[Optional[np.ndarray]]=None,
                 raw_y_data: Tuple[Optional[np.ndarray]]=None,
                 fit_results: Optional[pd.Series]=None,
                 evaluation_arguments=None,
                 initial_fit_args=None,
                 name='LeadTunnelTimeByLeadScan'):
        if measurements is None:
            measurements = (Measurement('lead_scan', gate='B', AWGorDecaDAC='DecaDAC'), )
        if evaluation_arguments is None:
            evaluation_arguments = {'t_fall': 50e-9, 't_rise': 50e-9, 'begin_rise': 70e-9, 'begin_fall': 2070e-9}

        self.evaluation_arguments = evaluation_arguments
        self.sample_rate = sample_rate
        super().__init__(experiment=experiment, measurements=measurements, parameters=parameters,
                         raw_x_data=raw_x_data, raw_y_data=raw_y_data, fit_results=fit_results,
                         initial_fit_arguments=initial_fit_args, name=name)

    def evaluate(self) -> (pd.Series, pd.Series):
        self._raw_y_data = []
        self._raw_x_data = []
        raw_data = self.experiment.measure(self.measurements[0])
        self._fit_results, residual = self.process_raw_data(raw_data)
        t_rise = self._fit_results['t_rise']
        t_fall = self._fit_results['t_fall']
        error_t_rise = t_rise / 5.
        error_t_fall = t_fall / 5.
        return pd.Series([t_rise, t_fall], ['parameter_time_rise', 'parameter_time_fall']), pd.Series(
            [error_t_rise, error_t_fall], ['parameter_time_rise', 'parameter_time_fall'])

    def process_raw_data(self, raw_data=None):
        if raw_data is None:
            raw_data = self.raw_data[0]
        y_data = raw_data[1, :] - raw_data[0, :]
        self._raw_y_data = y_data
        n_points = len(y_data)
        x_data = np.arange(start=0, stop=n_points) / self.sample_rate
        self._raw_x_data = x_data
        p0 = [y_data[round(.25 * n_points)] - y_data[round(.75 * n_points)],
              self.evaluation_arguments['t_fall'], self.evaluation_arguments['t_rise'],
              self.evaluation_arguments['begin_rise'], self.evaluation_arguments['begin_fall'], np.mean(y_data)]
        bounds = ([-np.inf, 0., 0., -np.inf, -np.inf, -np.inf],
                  [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        # prefit
        self._initial_fit_arguments = pd.Series(data=p0, index=['height', 't_fall', 't_rise', 'begin_rise',
                                                                'begin_fall', 'offset'])
        popt, pcov = optimize.curve_fit(f=func_lead_times_v1, p0=p0, bounds=bounds, xdata=x_data, ydata=y_data)

        # weighted fit
        sigma = np.ones(n_points)
        significant_lenght = int(.35 * n_points)
        start_rise = int(popt[3] / 4e-6 + 3)
        start_fall = int(popt[4] / 4e-6 + 3)
        sigma[start_rise:start_rise + significant_lenght] = .1
        sigma[start_fall:start_fall + significant_lenght] = .1
        popt, pcov = optimize.curve_fit(f=func_lead_times_v1, p0=popt, sigma=sigma, bounds=bounds, xdata=x_data,
                                        ydata=y_data)
        self._fit_results = pd.Series(data=popt, index=['height', 't_fall', 't_rise', 'begin_rise', 'begin_fall',
                                                        'offset'])
        self.evaluation_arguments = self._fit_results
        residuals = y_data - func_lead_times_v1(x_data, **dict(self._fit_results))
        scaled_residual = np.nanmean(residuals) / self._fit_results['height']
        return self._fit_results, scaled_residual

    def to_hdf5(self):
        parent_dict = super().to_hdf5()
        return dict(parent_dict,
                    evaluation_arguments=self.evaluation_arguments)


def func_lead_times_v1(x, height: float, t_fall: float, t_rise: float, begin_rise: float, begin_fall: float,
                       offset: float):
    half_time = 2e-6
    x = np.squeeze(x)
    n_points = len(x)
    y = np.zeros((n_points, ))
    c_rise = np.cosh(.5 * half_time / t_rise)
    s_rise = np.sinh(.5 * half_time / t_rise)
    c_fall = np.cosh(.5 * half_time / t_fall)
    s_fall = np.sinh(.5 * half_time / t_fall)
    for i in range(n_points):
        if x[i] < begin_rise:
            e = np.exp(.5*(3 * half_time - 2. * (x[i] + x[n_points - 1] - (begin_fall - half_time))) / t_fall)
            signed_height = -1. * height
            y[i] = offset + .5 * signed_height * (c_fall - e) / s_fall
        elif begin_rise <= x[i] < begin_fall:
            e = np.exp(.5*(half_time - 2. * (x[i] - begin_rise)) / t_rise)
            signed_height = height
            y[i] = offset + .5 * signed_height * (c_rise - e) / s_rise
        else:
            e = np.exp(.5*(3 * half_time - 2. * (x[i] - (begin_fall - half_time))) / t_fall)
            signed_height = -1. * height
            y[i] = offset + .5 * signed_height * (c_fall - e) / s_fall
    return y


class LeadTunnelTimeByLeadScan(FittingEvaluator):
    """
    RF gates pulse over the transition between the (2,0) and the (1,0) region. Then exponential functions are fitted
    to calculate the time required for an electron to tunnel through the lead.
    """
    def __init__(self, experiment: Experiment,
                 parameters: Sequence[str]=('parameter_time_rise', 'parameter_time_fall'),
                 measurements: Tuple[Measurement] = None,
                 sample_rate: float=1e8,
                 raw_x_data: Tuple[Optional[np.ndarray]]=None,
                 raw_y_data: Tuple[Optional[np.ndarray]]=None,
                 fit_results: Optional[pd.Series]=None,
                 evaluation_arguments=None,
                 initial_fit_args=None,
                 name='LeadTunnelTimeByLeadScan'):
        if measurements is None:
            measurements = (Measurement('lead_scan', gate='B', AWGorDecaDAC='DecaDAC'), )
        if evaluation_arguments is None:
            evaluation_arguments = {'t_fall': 50e-9, 't_rise': 50e-9, 'begin_rise': 70e-9, 'begin_fall': 2070e-9}

        self.evaluation_arguments = evaluation_arguments
        self.sample_rate = sample_rate
        super().__init__(experiment=experiment, measurements=measurements, parameters=parameters,
                         raw_x_data=raw_x_data, raw_y_data=raw_y_data, fit_results=fit_results,
                         initial_fit_arguments=initial_fit_args, name=name)

    def evaluate(self) -> (pd.Series, pd.Series):
        raw_data = self.experiment.measure(self.measurements[0])
        fitresult, residual = self.process_raw_data(raw_data)
        self._fit_results = fitresult
        t_rise = self._fit_results['t_rise']
        t_fall = self._fit_results['t_fall']
        error_t_rise = t_rise / 5.
        error_t_fall = t_fall / 5.
        return pd.Series([t_rise, t_fall], ['parameter_time_rise', 'parameter_time_fall']), pd.Series(
            [error_t_rise, error_t_fall], ['parameter_time_rise', 'parameter_time_fall'])

    def process_raw_data(self, raw_data=None):
        if raw_data is None:
            raw_data = self.raw_data
        y_data = raw_data[1, :] - raw_data[0, :]
        self._raw_y_data = y_data
        x_data = np.arange(start=0, stop=len(y_data)) / self.sample_rate
        self._raw_x_data = x_data
        n_points = len(y_data)
        p0 = [y_data[round(.25 * n_points)] - y_data[round(.75 * n_points)],
              self.evaluation_arguments['t_fall'], self.evaluation_arguments['t_rise'],
              self.evaluation_arguments['begin_rise'], self.evaluation_arguments['begin_fall'], np.mean(y_data)]
        begin_lin = int(round(p0[4] / 10e-9))
        end_lin = begin_lin + 2
        slope = (y_data[end_lin] - y_data[begin_lin]) / (x_data[end_lin] - x_data[begin_lin])
        linear_offset = y_data[begin_lin] - x_data[begin_lin] * slope
        p0 += [slope, linear_offset, x_data[end_lin] - x_data[begin_lin]]
        begin_lin_1 = int(round(p0[3] / 10e-9))
        end_lin_1 = begin_lin_1 + 2
        slope_1 = (y_data[end_lin_1] - y_data[begin_lin_1]) / (x_data[end_lin_1] - x_data[begin_lin_1])
        linear_offset_1 = y_data[begin_lin_1] - x_data[begin_lin_1] * slope_1
        p0 += [slope_1, linear_offset_1, x_data[end_lin_1] - x_data[begin_lin_1]]
        bounds = ([-np.inf, 0., 0., -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0., -np.inf, -np.inf, 0.],
                  [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                   float(n_points / 161. / self.sample_rate), np.inf, np.inf,
                   float(n_points / 161. / self.sample_rate)])
        self._initial_fit_arguments = pd.Series(data=p0, index=['height', 't_fall', 't_rise', 'begin_rise',
                                                                'begin_fall',
                                                                'offset', 'slope',
                                                                'offset_linear', 'length_lin_fall', 'slope_1',
                                                                'offset_linear_1',
                                                                'length_lin_rise'])

        popt, pcov = optimize.curve_fit(f=func_lead_times_v2, p0=p0, bounds=bounds, xdata=x_data, ydata=y_data)
        fitresult = pd.Series(data=popt,
                              index=['height', 't_fall', 't_rise', 'begin_rise', 'begin_fall', 'offset', 'slope',
                                     'offset_linear', 'length_lin_fall', 'slope_1', 'offset_linear_1',
                                     'length_lin_rise'])
        residuals = y_data - func_lead_times_v2(x_data, **dict(fitresult))
        scaled_residual = np.nanmean(residuals) / fitresult['height']
        qtune.util.plot_raw_data_fit(y_data=y_data, x_data=x_data)
        # TODO: properly scale and use residual
        return fitresult, scaled_residual

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    sample_rate=self.sample_rate,
                    evaluation_arguments=self.evaluation_arguments)


def func_lead_times_v2(x, height: float, t_fall: float, t_rise: float, begin_rise: float, begin_fall: float,
                       offset: float, slope: float, offset_linear: float, length_lin_fall: float, slope_1: float,
                       offset_linear_1: float, length_lin_rise: float):
    exp_begin_rise = begin_rise + length_lin_rise
    exp_begin_fall = begin_fall + length_lin_fall
    half_time = 2e-6
    x = np.squeeze(x)
    n_points = len(x)
    y = np.zeros((n_points, ))
    c = np.cosh(.5 * half_time / t_rise)
    s = np.sinh(.5 * half_time / t_rise)
    for i in range(n_points):
        if exp_begin_rise <= x[i] <= begin_fall:
            e = np.exp(.5*(begin_fall - 2. * x[i]) / t_rise)
            signed_height = height
            y[i] = offset + .5 * signed_height * (c - e) / s

        elif x[i] <= begin_rise:
            e = np.exp(.5*(1. * begin_fall - 2. * (x[i] + x[n_points - 1])) / t_fall)
            signed_height = -1. * height
            y[i] = offset + .5 * signed_height * (c - e) / s
        elif begin_fall < x[i] < exp_begin_fall:
            y[i] = offset_linear + x[i] * slope
        elif begin_rise < x[i] < exp_begin_rise:
            y[i] = offset_linear_1 + x[i] * slope_1
        else:
            e = np.exp(.5*(half_time + 2. * begin_fall - 2. * x[i]) / t_fall)
            signed_height = -1. * height
            y[i] = offset + .5 * signed_height * (c - e) / s
    return y


class InterDotTCByLineScan(FittingEvaluator):
    """ Measurement of an inter dot tunnel coupling by the broadening of the transition line. The width of the
    transition is extracted by fitting an s-curve (hyperbolic tangent) to the signal from the avoided crossing. """
    def __init__(self, experiment: Experiment, parameters: Sequence[str]=('parameter_tunnel_coupling', ),
                 measurements: Tuple[Measurement] = None, raw_x_data: Tuple[Optional[np.ndarray]]=None,
                 raw_y_data: Tuple[Optional[np.ndarray]]=None, fit_results: Optional[pd.Series]=None,
                 initial_fit_arguments=None, intermediate_fit_arguments=None, name='InterDotTCByLineScan'):
        if measurements is None:
            measurements = (Measurement('detune_scan', center=0., range=2e-3, N_points=100, ramptime=.02, N_average=10,
                                        AWGorDecaDAC='AWG'), )

        if initial_fit_arguments is None:
            self._initial_fit_arguments = pd.Series(index=['offset', 'slope', 'height', 'position', 'width'])
        else:
            self._initial_fit_arguments = initial_fit_arguments

        if intermediate_fit_arguments is None:
            self.intermediate_fit_arguments = pd.Series(index=['offset', 'slope', 'height', 'position', 'width'])
        else:
            self.intermediate_fit_arguments = intermediate_fit_arguments

        super().__init__(experiment=experiment, measurements=measurements, parameters=parameters,
                         raw_x_data=raw_x_data, raw_y_data=raw_y_data, fit_results=fit_results,
                         initial_fit_arguments=initial_fit_arguments, name=name)

    def evaluate(self):
        ydata = np.squeeze(self.experiment.measure(self.measurements[0]))
        self._raw_y_data = ydata
        if len(ydata.shape) == 2:
            n_points = ydata.shape[1]
        elif len(ydata.shape) > 2:
            self._fit_results = pd.Series(data=np.nan, index=['offset', 'slope', 'height', 'position', 'width'])
            self.logger.error('The Evaluator' + str(self) + 'received measurement data of an invalid dimension.')
            return pd.Series([np.nan], ["parameter_tunnel_coupling"]), pd.Series([np.nan],
                                                                                 ["parameter_tunnel_coupling"])
        else:
            n_points = ydata.size
        self._raw_x_data = np.linspace(self.measurements[0].options['center'] - self.measurements[0].options['range'],
                                       self.measurements[0].options['center'] + self.measurements[0].options['range'],
                                       n_points)
        try:
            fitresult, residual = self.process_raw_data((self._raw_x_data, self._raw_y_data))
        except RuntimeError:
            self.logger.error('The following evaluator could not evaluate its data: ' + str(self))
            self._fit_results = pd.Series(data=np.nan, index=['offset', 'slope', 'height', 'position', 'width'])
            return pd.Series([np.nan], ["parameter_tunnel_coupling"]), pd.Series([np.nan],
                                                                                 ["parameter_tunnel_coupling"])
        self._fit_results = fitresult
        tc_in_mus = fitresult['width'] * 1e6
        return pd.Series([tc_in_mus], ["parameter_tunnel_coupling"]), \
            pd.Series([residual], ["parameter_tunnel_coupling"])

    def process_raw_data(self, raw_data):
        x_data = raw_data[0]
        y_data = raw_data[1]
        if len(y_data.shape) == 2:
            y_data = np.nanmean(y_data, 0)
        n_points = x_data.size
        poly_fit_result_start = np.polyfit(x=x_data[:round(.25 * n_points)], y=y_data[:round(.25 * n_points)], deg=1)
        poly_fit_result_end = np.polyfit(x=x_data[-round(.25 * n_points):], y=y_data[-round(.25 * n_points):], deg=1)
        height = (poly_fit_result_end[1] + poly_fit_result_end[0] * x_data[-1]) - (
                poly_fit_result_start[1] + poly_fit_result_start[0] * x_data[-1])
        predicted_transition_position = \
            x_data[qtune.util.new_find_lead_transition_index(data=y_data - poly_fit_result_end[1] - poly_fit_result_end[
                0] * x_data, width_in_index_points=n_points // 12)]
        initial_parameters = [poly_fit_result_start[1], poly_fit_result_start[0], height, predicted_transition_position,
                              np.ptp(x_data) / 8.]
        self._initial_fit_arguments = dict(pd.Series(data=initial_parameters,
                                                     index=['offset', 'slope', 'height', 'position', 'width']))
        popt, pcov = optimize.curve_fit(f=func_inter_dot_coupling, p0=initial_parameters, xdata=x_data, ydata=y_data)
        # second fit which emphasises the range around the transition
        position_point = int((popt[3] + np.ptp(x_data)) / 2. / np.ptp(x_data) * n_points)
        weights = np.ones(n_points)
        heavy_range = round(2 * popt[4] / np.ptp(x_data) * n_points)
        weights[max(0, int(position_point - heavy_range)):min(n_points - 1, int(position_point + heavy_range))] = .1
        self.intermediate_fit_arguments.iloc[:] = popt
        popt, pcov = optimize.curve_fit(f=func_inter_dot_coupling, p0=popt, sigma=weights, xdata=x_data, ydata=y_data)
        residuals = y_data[int(0.3 * n_points):n_points] - \
            func_inter_dot_coupling(x_data[int(0.3 * n_points):n_points], popt[0], popt[1], popt[2], popt[3],
                                    popt[4])
        residual = np.nanmean(np.square(residuals)) / (popt[2] * popt[2]) * 2e4
        fitresult = pd.Series(data=popt, index=['offset', 'slope', 'height', 'position', 'width'])
        return fitresult, residual

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    intermediate_fit_arguments=self.intermediate_fit_arguments)


def func_inter_dot_coupling(xdata, offset: float, slope: float, height: float, position: float, width: float):
    return offset + slope * xdata + .5 * height * (1 + np.tanh((xdata - position) / width))


class NewLoadTime(FittingEvaluator):
    """
    Measures the time required to reload a (2,0) singlet state, by pulsing close to the (2,0)-(1,0) transition. The
    (2,0) triplet decays via the (1,0) state into a (2,0) singlet state. The wait time is varied at the decay point and
    the resulting occupation probability is fit to an exponential decay. The life time of the decay is returned as
    singlet reload time.
    """
    def __init__(self, experiment: Experiment, parameters: Sequence[str]=('parameter_time_load',),
                 measurements: Measurement = None, raw_x_data: Tuple[Optional[np.ndarray]]=None,
                 raw_y_data: Tuple[Optional[np.ndarray]]=None, fit_results: Optional[pd.Series]=None,
                 initial_fit_arguments=None, initial_curvature=10, name='LoadTime'):
        if measurements is None:
            measurements = (Measurement("load_scan"), )
        if initial_fit_arguments is None:
            initial_fit_arguments = pd.Series(index=['offset', 'height', 'curvature'], data=[np.nan, np.nan, 10])
        self.initial_curvature = initial_curvature
        super().__init__(experiment=experiment, measurements=measurements, parameters=parameters,
                         raw_x_data=raw_x_data, raw_y_data=raw_y_data, fit_results=fit_results,
                         initial_fit_arguments=initial_fit_arguments, name=name)

    def evaluate(self) -> (pd.Series, pd.Series):
        data = self.experiment.measure(self.measurements[0])
        try:
            fitresult, residual = self.process_raw_data(data)
        except RuntimeError:
            self.logger.error('The following evaluator could not fit its data:' + str(self))
            fitresult = pd.Series(data=np.nan, index=['offset', 'height', 'curvature'])
            residual = np.nan
        self._fit_results = fitresult
        self._raw_y_data = data[0, :]
        self._raw_x_data = data[1, :]
        return pd.Series([fitresult['curvature']], ['parameter_time_load']),\
            pd.Series([residual], ["parameter_time_load"])

    def process_raw_data(self, raw_data):
        x_data = raw_data[1, :]
        y_data = raw_data[0, :]
        initial_curvature = self.initial_curvature
        p0 = [np.nanmin(y_data), np.nanmax(y_data) - np.nanmin(y_data), initial_curvature]
        bounds = ([-np.inf, -np.inf, 2.],
                  [np.inf, np.inf, 300.])
        popt, pcov = optimize.curve_fit(f=func_load_time, p0=p0, bounds=bounds, xdata=x_data, ydata=y_data)
        if popt[2] < 0.:
            p0[2] = self.initial_fit_arguments['curvature'] * 20
            popt, pcov = optimize.curve_fit(f=func_load_time, p0=p0, bounds=bounds, xdata=x_data, ydata=y_data)
        self._initial_fit_arguments.iloc[:] = p0
        residuals = y_data - func_load_time(x_data, popt[0], popt[1], popt[2])
        residual = np.nanmean(np.square(residuals)) / (popt[1] * popt[1]) * 1500.
        return pd.Series(data=popt, index=['offset', 'height', 'curvature']), residual

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    initial_curvature=self.initial_curvature)


def func_load_time(xdata, offset: float, height: float, curvature: float):
    return offset + height * np.exp(-1. * xdata / curvature)


class LeadTransition(Evaluator):
    """
    Finds the transition on the edge of the charge diagram. The transition is detected as point of largest slope.
    """

    def __init__(self,
                 experiment: Experiment,
                 shifting_gates: Optional[Sequence[str]]=("RFB", "RFA"),
                 sweeping_gates: Optional[Sequence[str]]=('RFA', 'RFB'),
                 charge_diagram_width: float=4e-3,
                 measurements=None,
                 parameters: Sequence[str]=("position_RFA", "position_RFB"),
                 raw_x_data: Tuple[Optional[np.ndarray]]=None,
                 raw_y_data: Tuple[Optional[np.ndarray]]=None,
                 transition_positions=pd.Series(),
                 transition_width=.2e-3,
                 name='LeadTransition'):
        if measurements is None:
            measurements = []
            for gate in sweeping_gates:
                line_scan = Measurement('line_scan', center=0., range=4e-3, gate=gate, N_points=320, ramptime=.001,
                                        N_average=7, AWGorDecaDAC='DecaDAC')
                measurements.append(line_scan)
        self._shifting_gates = shifting_gates
        self._charge_diagram_width = charge_diagram_width
        self.transition_width = transition_width
        self.transition_positions = transition_positions
        super().__init__(experiment, measurements, parameters, raw_x_data=raw_x_data,
                         raw_y_data=raw_y_data, name=name)

    def evaluate(self):
        current_gate_voltages = self.experiment.read_gate_voltages()
        raw_data = []
        for measurement, shift_gate in zip(self.measurements, self._shifting_gates):
            shift = pd.Series(-1. * self._charge_diagram_width, [shift_gate])
            self.experiment.set_gate_voltages(current_gate_voltages.add(shift, fill_value=0.))
            raw_data.append(self.experiment.measure(measurement))
            self.experiment.set_gate_voltages(current_gate_voltages)
        transition_positions, error = self.process_raw_data(raw_data=raw_data)

        # TODO: this should be obsolete
        self.experiment.set_gate_voltages(current_gate_voltages)

        return transition_positions, error

    def process_raw_data(self, raw_data):
        self.transition_positions = pd.Series()
        error = pd.Series()
        self._raw_x_data = []
        self._raw_y_data = []
        for data, gate in zip(raw_data, self._shifting_gates):
            if len(data.shape) == 2:
                data = data.mean(0)
            transition_pos = \
                qtune.util.new_find_lead_transition_index(data, int(.2e-3 / self.measurements[0].options["range"] *
                                                          self.measurements[0].options["N_points"]))
            x_data = np.linspace(start=self._measurements[0].options['center'] - self.measurements[0].options['range'],
                                 stop=self._measurements[0].options['center'] + self.measurements[0].options['range'],
                                 num=self.measurements[0].options['N_points'])
            self._raw_x_data.append(x_data)
            self._raw_y_data.append(data)
            self.transition_positions["position_" + gate] = x_data[transition_pos]
            error["position_" + gate] = .01e-3**2
        return self.transition_positions, error

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    transition_positions=self.transition_positions,
                    shifting_gates=self._shifting_gates,
                    charge_diagram_width=self._charge_diagram_width,
                    transition_width=.2e-3)


class SensingDot1D(Evaluator):
    """
    Sweep one gate of the sensing dot to find the point of steepest slope on the current coulomb peak. The slope is
    returned to quantify the signal strength.
    """

    def __init__(self,
                 experiment: Experiment,
                 sweeping_gate: str="SDB2",
                 parameters: Sequence[str]=("position_SDB2", "current_signal", "optimal_signal"),
                 measurements: Measurement=None,
                 raw_x_data: Tuple[Optional[np.ndarray]]=None,
                 raw_y_data: Tuple[Optional[np.ndarray]] = None,
                 current_signal=None,
                 optimal_signal=None,
                 optimal_position=None,
                 name='SensingDot1D'):
        self._sweeping_gate = sweeping_gate
        self.current_signal = current_signal
        self.optimal_signal = optimal_signal
        self.optimal_position = optimal_position
        if measurements is None:
            measurements = (Measurement('line_scan',
                                        center=None, range=4e-3, gate="SDB2",
                                        N_points=1280, ramptime=.0005,
                                        N_average=1, AWGorDecaDAC='DecaDAC'),)
        super().__init__(experiment, measurements=measurements, parameters=parameters,
                         raw_x_data=raw_x_data, raw_y_data=raw_y_data, name=name)

    def evaluate(self):
        sensing_dot_measurement = self.measurements[0]
        values = pd.Series()
        error = pd.Series()
        gate = self._sweeping_gate
        sensing_dot_measurement.options["gate"] = gate
        sensing_dot_measurement.options["center"] = self.experiment.read_gate_voltages()[gate]

        data = self.experiment.measure(sensing_dot_measurement)
        current_signal, optimal_signal, optimal_position = self.process_raw_data(raw_data=data)

        self._raw_y_data = data
        self._raw_x_data = np.linspace(
            start=self._measurements[0].options['center'] - self.measurements[0].options['range'],
            stop=self._measurements[0].options['center'] + self.measurements[0].options['range'],
            num=self.measurements[0].options['N_points'])

        self.optimal_position = sensing_dot_measurement.options["center"] + optimal_position
        values["position_" + gate] = sensing_dot_measurement.options["center"] + optimal_position
        error["position_" + gate] = 0.1e-3
        values["current_signal"] = current_signal
        error["current_signal"] = current_signal / 5
        values["optimal_signal"] = optimal_signal
        error["optimal_signal"] = optimal_signal / 5
        return values, error

    def process_raw_data(self, raw_data):
        raw_data *= -1
        raw_data = np.squeeze(raw_data)
        # invert the sign to get rising flank
        data_filterd = scipy.ndimage.filters.gaussian_filter1d(input=raw_data, sigma=raw_data.size/64, axis=0, order=0,
                                                               mode="nearest", truncate=.2)
        data_filtered_diff = np.diff(data_filterd, n=1)
        # data_filtered_diff_smoothed = scipy.ndimage.filters.gaussian_filter1d(input=data_filtered_diff,
        #                                                                      sigma=raw_data.size/100,
        #                                                                      axis=0, order=0,
        #                                                                      mode="nearest",
        #                                                                      truncate=4.)
        # data_filtered_diff_smoothed = np.squeeze(data_filtered_diff_smoothed)
        # self.current_signal = abs(data_filtered_diff_smoothed[int(self.measurements[0].options["N_points"] / 2)])
        self.current_signal = 0.5 * abs(
            data_filtered_diff[int(.5 * self.measurements[0].options["N_points"])] +
            data_filtered_diff[int(.5 * self.measurements[0].options["N_points"]) + 1])
        self.optimal_signal = abs(data_filtered_diff.min())
        self.optimal_position = data_filtered_diff.argmin()
        self.optimal_position = float(self.optimal_position) / float(self.measurements[0].options["N_points"]) * \
            2 * self.measurements[0].options["range"] - self.measurements[0].options["range"]
        return self.current_signal, self.optimal_signal, self.optimal_position

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    sweeping_gate=self._sweeping_gate,
                    current_signal=self.current_signal,
                    optimal_signal=self.optimal_signal,
                    optimal_position=self.optimal_position)


class SensingDot2D(Evaluator):
    """
    Two dimensional sensing dot scan. Returns the point of higest slope. The slope is returned to quantify the signal
    strength. The coulomb peak might be changed.
    """

    def __init__(self, experiment: Experiment,
                 sweeping_gates: Sequence[str]=("SDB1", "SDB2"),
                 scan_range: float=10e-3,
                 parameters: Sequence[str]=("position_SDB1", "position_SDB2"),
                 measurements: Measurement=None,
                 raw_x_data: Tuple[Optional[np.ndarray]]=None,
                 raw_y_data: Tuple[Optional[np.ndarray]] = None,
                 name='SensingDot2D',
                 new_voltages=pd.Series()):
        self._sweeping_gates = sweeping_gates
        self._scan_range = scan_range
        self._new_voltages = new_voltages
        if measurements is None:
            measurements = (Measurement('2d_scan', center=[None, None],
                                        range=scan_range,
                                        gate1=sweeping_gates[0],
                                        gate2=sweeping_gates[1], ramptime=.0005, n_lines=20,
                                        n_points=104, N_average=1, AWGorDecaDAC='DecaDAC'),)
        super().__init__(experiment, measurements, parameters=parameters, raw_x_data=raw_x_data,
                         raw_y_data=raw_y_data, name=name)

    def evaluate(self):
        self.measurements[0].options["center"] = [
            self.experiment.read_gate_voltages()[self.measurements[0].options["gate1"]],
            self.experiment.read_gate_voltages()[self.measurements[0].options["gate2"]]]
        data = self.experiment.measure(self.measurements[0])
        self._raw_y_data = data
        self._raw_x_data = []
        for i in range(2):
            self._raw_x_data.append(np.linspace(self.measurements[0].options['center'][i] -
                                                self.measurements[0].options['range'],
                                                self.measurements[0].options['center'][i] +
                                                self.measurements[0].options['range'],
                                                data.shape[i]))
        self._new_voltages, error = self.process_raw_data(data)
        self._new_voltages["position_" + self.measurements[0].options["gate1"]] += \
            self.measurements[0].options['center'][0]
        self._new_voltages["position_" + self.measurements[0].options["gate2"]] += \
            self.measurements[0].options['center'][1]
        return self._new_voltages, error

    def process_raw_data(self, raw_data):
        raw_data *= -1
        n_lines, n_points = raw_data.shape

        # invert sign to get rising flank
        # data_filtered = scipy.ndimage.filters.gaussian_filter1d(input=raw_data, sigma=.5, axis=0, order=0,
        #                                                        mode="nearest",
        #                                                        truncate=4.)
        # data_diff = np.diff(data_filtered, n=1)
        # change like in the 1 d case
        data_filtered = scipy.ndimage.filters.gaussian_filter1d(input=raw_data,
                                                                sigma=20, axis=0,
                                                                order=0,
                                                                mode="nearest",
                                                                truncate=.1)
        data_diff = np.diff(data_filtered[:, ::2], n=1)
        mins_in_lines = data_diff.min(1)
        min_line = np.argmin(mins_in_lines)
        min_point = np.argmin(data_diff[min_line])
        gate_1_pos = float(min_line) / n_lines * 2 * \
            self.measurements[0].options["range"] - self.measurements[0].options["range"]
        gate_2_pos = float(min_point) / (n_points / 2) * 2 * \
            self.measurements[0].options["range"] - self.measurements[0].options["range"]
        new_voltages = pd.Series([gate_1_pos, gate_2_pos], ["position_" + self.measurements[0].options["gate1"],
                                                            "position_" + self.measurements[0].options["gate2"]])
        error = pd.Series([.1e-3, .1e-3], ["position_" + self.measurements[0].options["gate1"],
                                           "position_" + self.measurements[0].options["gate2"]])
        return new_voltages, error

    def to_hdf5(self):
        return dict(super().to_hdf5(),
                    sweeping_gates=self._sweeping_gates,
                    scan_range=self._scan_range,
                    new_voltages=self._new_voltages)


class AveragingEvaluator(Evaluator):
    def __init__(self, evaluator: Evaluator, n_measurement_repetitions: int, raw_x_data=None, raw_y_data=None,
                 name=None):
        self._evaluator = evaluator
        self.n_measurement_repetitions = n_measurement_repetitions
        if raw_x_data is None:
            raw_x_data = []
        if raw_y_data is None:
            raw_y_data = []
        if name is None:
            name = 'Averaging' + self._evaluator.name
        super().__init__(experiment=self._evaluator.experiment, measurements=self._evaluator.measurements,
                         parameters=self._evaluator.parameters, raw_x_data=raw_x_data, raw_y_data=raw_y_data,
                         name=name)

    def evaluate(self):
        self._raw_x_data = []
        self._raw_y_data = []
        parameter_list = []
        for i in range(self.n_measurement_repetitions):
            parameters, errors = self._evaluator.evaluate()
            parameter_list.append(parameters)
            self._raw_x_data.append(self._evaluator._raw_x_data)
            self._raw_y_data.append(self._evaluator._raw_y_data)

        averaged_parameters = pd.Series()
        outer_error = pd.Series()
        for parameter in self.parameters:
            averaged_parameters[parameter] = np.nanmean([el[parameter] for el in parameter_list])
            outer_error[parameter] = np.nanvar([el[parameter] for el in parameter_list])

        return averaged_parameters, outer_error

    def process_raw_data(self, raw_data):
        raise NotImplementedError

    def to_hdf5(self):
        return dict(evaluator=self._evaluator,
                    n_measurement_repetitions=self.n_measurement_repetitions,
                    raw_x_data=self._raw_x_data,
                    raw_y_data=self._raw_y_data,
                    name=self.name)
