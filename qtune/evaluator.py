from qtune.experiment import Experiment, Measurement
from qtune.basic_dqd import BasicDQD
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
                 last_x_data: Optional[np.ndarray],
                 last_y_data: Optional[np.ndarray],
                 last_fit_results: Optional[pd.Series]=None,
                 fit_function: Optional=None):
        self._experiment = experiment
        self._measurements = tuple(measurements)  # Is this the behaviour that was intended?
        self._parameters = tuple(parameters)
        self._last_x_data = last_x_data
        self._last_y_data = last_y_data
        self._last_fit_results = last_fit_results
        self._fit_function = fit_function

    @property
    def logger(self):
        return logging.getLogger(name="qtune")

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
    def last_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._last_x_data, self._last_y_data

    @property
    def last_fit_results(self) -> Optional[pd.Series]:
        return self._last_fit_results

    @property
    def fit_function(self):
        return self._fit_function

    def evaluate(self) -> (pd.Series, pd.Series):
        raise NotImplementedError()

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    measurements=self.measurements,
                    parameters=self.parameters,
                    last_x_data=self.last_data[0],
                    last_y_data=self.last_data[1],
                    last_fit_results=self.last_fit_results)

    def __repr__(self):
        return "{type}({data})".format(type=type(self), data=self.to_hdf5())


class NewLeadTunnelTimeByLeadScan(Evaluator):
    """
    RF gates pulse over the transition between the (2,0) and the (1,0) region. Then exponential functions are fitted
    to calculate the time required for an electron to tunnel through the lead.
    """
    def __init__(self, experiment: Experiment,
                 parameters: Sequence[str]=('parameter_time_rise', 'parameter_time_fall'),
                 lead_scan: Measurement = None, sample_rate: float=1e8, last_x_data: Optional[np.ndarray]=None,
                 last_y_data: Optional[np.ndarray]=None, last_fit_results: Optional[pd.Series]=None,
                 fit_function: Optional=None):
        if lead_scan is None:
            lead_scan = Measurement('lead_scan', gate='B', AWGorDecaDAC='DecaDAC')
        if fit_function is None:
            fit_function = func_lead_times_v2
        self.sample_rate = sample_rate
        super().__init__(experiment=experiment, measurements=(lead_scan, ), parameters=parameters,
                         last_x_data=last_x_data, last_y_data=last_y_data, last_fit_results=last_fit_results,
                         fit_function=fit_function)

    def evaluate(self) -> (pd.Series, pd.Series):
        data = self.experiment.measure(self.measurements[0])
        self._last_y_data = data[1, :] - data[0, :]
        self._last_x_data = np.arange(start=0, stop=self._last_y_data) / self.sample_rate
        fitresult, residual = new_fit_lead_times(x_data=self._last_x_data, y_data=self._last_y_data)
        # TODO: properly scale and use residual
        self._last_fit_results = fitresult
        t_rise = fitresult['t_rise']
        t_fall = fitresult['t_fall']
        error_t_rise = t_rise / 5.
        error_t_fall = t_fall / 5.
        return pd.Series([t_rise, t_fall], ['parameter_time_rise', 'parameter_time_fall']), pd.Series(
            [error_t_rise, error_t_fall], ['parameter_time_rise', 'parameter_time_fall'])

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    parameters=self.parameters,
                    sample_rate=self.sample_rate,
                    lead_scan=self.measurements[0],
                    last_x_data=self.last_data[0],
                    last_y_data=self.last_data[1],
                    last_fit_results=self.last_fit_results)


def new_fit_lead_times(x_data: np.ndarray, y_data: np.ndarray):
    samprate = 1e8
    n_points = len(y_data)
    p0 = [y_data[round(.25 * n_points)] - y_data[round(.75 * n_points)],
          50e-9, 50e-9, 70e-9, 2070e-9, np.mean(y_data)]
    begin_lin = int(round(p0[4] / 10e-9))
    end_lin = begin_lin + 5
    slope = (y_data[end_lin] - y_data[begin_lin]) / (x_data[end_lin] - x_data[begin_lin])
    linear_offset = y_data[begin_lin] - x_data[begin_lin] * slope
    p0 += [slope, linear_offset, x_data[end_lin] - x_data[begin_lin]]
    begin_lin_1 = int(round(p0[3] / 10e-9))
    end_lin_1 = begin_lin_1 + 5
    slope_1 = (y_data[end_lin_1] - y_data[begin_lin_1]) / (x_data[end_lin_1] - x_data[begin_lin_1])
    linear_offset_1 = y_data[begin_lin_1] - x_data[begin_lin_1] * slope_1
    p0 += [slope_1, linear_offset_1, x_data[end_lin_1] - x_data[begin_lin_1]]
    bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0., -np.inf, -np.inf, 0.],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, float(n_points / 60. * samprate),
               np.inf,
               np.inf, float(n_points / 60. * samprate)])
    popt, pcov = optimize.curve_fit(f=func_lead_times_v2, p0=p0, bounds=bounds, xdata=x_data, ydata=y_data)
    fitresult = pd.Series(data=popt, index=['height', 't_fall', 't_rise', 'begin_rise', 'begin_fall', 'offset', 'slope',
                                            'offset_linear', 'lenght_lin_fall', 'slope_1', 'offset_linear_1',
                                            'length_lin_rise'])
    residuals = y_data - func_lead_times_v2(x_data, **dict(fitresult))
    scaled_residual = np.nanmean(residuals) / fitresult['height']
    return fitresult, scaled_residual


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
            e = np.exp(.5*(half_time - 2. * x[i]) / t_rise)
            signed_hight = height
            y[i] = offset + .5 * signed_hight * (c - e) / s

        elif x[i] <= begin_rise:
            e = np.exp(.5*(1. * half_time - 2. * (x[i] + x[n_points - 1])) / t_fall)
            signed_hight = -1. * height
            y[i] = offset + .5 * signed_hight * (c - e) / s
        elif begin_fall < x[i] < exp_begin_fall:
            y[i] = offset_linear + x[i] * slope
        elif begin_rise < x[i] < exp_begin_rise:
            y[i] = offset_linear_1 + x[i] * slope_1
        else:
            e = np.exp(.5*(3. * half_time - 2. * x[i]) / t_fall)
            signed_hight = -1. * height
            y[i] = offset + .5 * signed_hight * (c - e) / s
    return y


class NewInterDotTCByLineScan(Evaluator):
    def __init__(self, experiment: Experiment, parameters: Sequence[str]=('parameter_tunnel_coupling', ),
                 line_scan: Measurement = None, last_x_data: Optional[np.ndarray]=None,
                 last_y_data: Optional[np.ndarray]=None, last_fit_results: Optional[pd.Series]=None,
                 fit_function: Optional=None):
        if line_scan is None:
            line_scan = Measurement('detune_scan', center=0., range=2e-3, N_points=100, ramptime=.02, N_average=10,
                                    AWGorDecaDAC='AWG')
        if fit_function is None:
            fit_function = func_inter_dot_coupling

        super().__init__(experiment=experiment, measurements=(line_scan, ), parameters=parameters,
                         last_x_data=last_x_data, last_y_data=last_y_data, last_fit_results=last_fit_results,
                         fit_function=fit_function)

    def evaluate(self):
        ydata = np.squeeze(self.experiment.measure(self.measurements[0]))
        self._last_y_data = ydata
        if len(ydata.shape) == 2:
            ydata = np.nanmean(ydata)
        elif len(ydata.shape) > 2:
            self._last_fit_results = pd.Series(data=np.nan, index=['offset', 'slope', 'height', 'position', 'width'])
            self.logger.error('The Evaluator' + str(self) + 'received measurement data of an invalid dimension.')
            return pd.Series([np.nan], ["parameter_tunnel_coupling"]), pd.Series([np.nan],
                                                                                 ["parameter_tunnel_coupling"])
        self._last_x_data = np.linspace(self.measurements[0].options['center'] - self.measurements[0].options['range'],
                                        self.measurements[0].options['center'] + self.measurements[0].options['range'],
                                        ydata.size)
        try:
            fitresult, residual = new_fit_inter_dot_coupling(self._last_x_data, self._last_y_data)
        except RuntimeError:
            self.logger.error('The following evaluator could not evaluate its data: ' + str(self))
            self._last_fit_results = pd.Series(data=np.nan, index=['offset', 'slope', 'height', 'position', 'width'])
            return pd.Series([np.nan], ["parameter_tunnel_coupling"]), pd.Series([np.nan],
                                                                                 ["parameter_tunnel_coupling"])
        self._last_fit_results = fitresult
        tc_in_mus = fitresult['width'] * 1e6
        return pd.Series([tc_in_mus], ["parameter_tunnel_coupling"]), \
            pd.Series([residual], ["parameter_tunnel_coupling"])

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    parameters=self.parameters,
                    line_scan=self.measurements[0],
                    last_x_data=self.last_data[0],
                    last_y_data=self.last_data[1],
                    last_fit_results=self.last_fit_results)


def new_fit_inter_dot_coupling(x_data: np.ndarray, y_data: np.ndarray):
    n_points = x_data.size
    poly_fit_result_start = np.polyfit(x=x_data[:round(.25 * n_points)], y=y_data[:round(.25 * n_points)], deg=1)
    poly_fit_result_end = np.polyfit(x=x_data[-round(.25 * n_points):], y=y_data[-round(.25 * n_points):], deg=1)
    height = (poly_fit_result_end[1] + poly_fit_result_end[0] * x_data[-1]) - (
                poly_fit_result_start[1] + poly_fit_result_start[0] * x_data[-1])
    predicted_transition_position = \
        x_data[qtune.util.new_find_lead_transition_index(data=y_data, width_in_index_points=n_points // 12)]
    initial_parameters = [poly_fit_result_start[1], poly_fit_result_start[0], height, predicted_transition_position,
                          np.ptp(x_data) / 8.]
    popt, pcov = optimize.curve_fit(f=func_inter_dot_coupling, p0=initial_parameters, xdata=x_data, ydata=y_data)
    # second fit which emphasises the range around the transition
    position_point = int((popt[3] + np.ptp(x_data)) / 2. / np.ptp(x_data) * n_points)
    weights = np.ones(n_points)
    heavy_range = round(2 * popt[4] / np.ptp(x_data) * n_points)
    weights[max(0, int(position_point - heavy_range)):min(n_points - 1, int(position_point + heavy_range))] = .1
    popt, pcov = optimize.curve_fit(f=func_inter_dot_coupling, p0=popt, sigma=weights, xdata=x_data, ydata=y_data)
    residuals = y_data[int(0.3 * n_points):n_points] - \
        func_inter_dot_coupling(x_data[int(0.3 * n_points):n_points], popt[0], popt[1], popt[2], popt[3], popt[4])
    residual = np.nanmean(np.square(residuals)) / (popt[2] * popt[2]) * 2e4
    fitresult = pd.Series(data=popt, index=['offset', 'slope', 'height', 'position', 'width'])
    return fitresult, residual


def func_inter_dot_coupling(xdata, offset: float, slope: float, height: float, position: float, width: float):
    return offset + slope * xdata + .5 * height * (1 + np.tanh((xdata - position) / width))


class NewLoadTime(Evaluator):
    """
    Measures the time required to reload a (2,0) singlet state. Fits an exponential function.
    """
    def __init__(self, experiment: Experiment, parameters: Sequence[str]=('parameter_time_load',),
                 load_scan: Measurement = None, last_x_data: Optional[np.ndarray]=None,
                 last_y_data: Optional[np.ndarray]=None, last_fit_results: Optional[pd.Series]=None,
                 fit_function: Optional=None):
        if load_scan is None:
            load_scan = Measurement("load_scan")
        if fit_function is None:
            fit_function = func_load_time
        super().__init__(experiment, (load_scan, ), parameters, last_x_data=last_x_data, last_y_data=last_y_data,
                         last_fit_results=last_fit_results, fit_function=fit_function)

    def evaluate(self) -> (pd.Series, pd.Series):
        data = self.experiment.measure(self.measurements[0])
        try:
            fitresult, residual = new_fit_load_time(x_data=data[1, :], y_data=data[0, :])
        except RuntimeError:
            self.logger.error('The following evaluator could not fit its data:' + str(self))
            fitresult = pd.Series(data=np.nan, index=['offset', 'height', 'curvature'])
            residual = np.nan
        self._last_fit_results = fitresult
        self._last_y_data = data[0, :]
        self._last_x_data = data[1, :]
        return pd.Series([fitresult['curvature']], ['parameter_time_load']),\
            pd.Series([residual], ["parameter_time_load"])

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    parameters=self.parameters,
                    load_scan=self.measurements[0],
                    last_x_data=self.last_data[0],
                    last_y_data=self.last_data[1],
                    last_fit_results=self.last_fit_results)


def new_fit_load_time(x_data, y_data, initial_curvature=(10, 200)):
    p0 = [np.nanmin(y_data), np.nanmax(y_data) - np.nanmin(y_data), initial_curvature[0]]
    bounds = ([-np.inf, -np.inf, 2.],
              [np.inf, np.inf, 300.])
    popt, pcov = optimize.curve_fit(f=func_load_time, p0=p0, bounds=bounds, xdata=x_data, ydata=y_data)
    if popt[2] < 0.:
        p0[2] = initial_curvature[1]
        popt, pcov = optimize.curve_fit(f=func_load_time, p0=p0, bounds=bounds, xdata=x_data, ydata=y_data)
    residuals = y_data - func_load_time(x_data, popt[0], popt[1], popt[2])
    residual = np.nanmean(np.square(residuals)) / (popt[1] * popt[1]) * 1500.
    return pd.Series(data=popt, index=['offset', 'height', 'curvature']), residual


def func_load_time(xdata, offset: float, height: float, curvature: float):
    return offset + height * np.exp(-1. * xdata / curvature)


class LeadTransition(Evaluator):
    """
    Finds the transition on the edge of the charge diagram
    """

    def __init__(self,
                 experiment: Experiment,
                 shifting_gates: Sequence[str]=("RFA", "RFB"),
                 charge_diagram_width: float=4e-3,
                 parameters: Sequence[str]=("position_RFA", "position_RFB"),
                 line_scan: Measurement=None,
                 last_x_data: Optional[np.ndarray]=None,
                 last_y_data: Optional[np.ndarray]=None):
        if line_scan is None:
            default_line_scan_a = Measurement('line_scan', center=0., range=4e-3,
                                              gate='RFA', N_points=320,
                                              ramptime=.001,
                                              N_average=7,
                                              AWGorDecaDAC='DecaDAC')
        else:
            default_line_scan_a = line_scan
        self._shifting_gates = shifting_gates
        self._charge_diagram_width = charge_diagram_width
        super().__init__(experiment, (default_line_scan_a, ), parameters, last_x_data=last_x_data,
                         last_y_data=last_y_data)

    def evaluate(self):
        transition_position = pd.Series()
        error = pd.Series()
        current_gate_voltages = self.experiment.read_gate_voltages()
        for gate in self._shifting_gates:
            shift = pd.Series(-1. * self._charge_diagram_width, [gate])
            self.experiment.set_gate_voltages(current_gate_voltages.add(shift, fill_value=0.))
            self.measurements[0].options["gate"] = gate
            data = self.experiment.measure(self.measurements[0])
            transition_position["position_" + gate] =\
                qtune.util.find_lead_transition(data, float(self.measurements[0].options["center"]),
                                                float(self.measurements[0].options["range"]),
                                                int(self.measurements[0].options["N_points"]))
            error["position_" + gate] = .1e-3
        self.experiment.set_gate_voltages(current_gate_voltages)

        return transition_position, error

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    parameters=self.parameters,
                    line_scan=self.measurements[0],
                    shifting_gates=self._shifting_gates,
                    charge_diagram_width=self._charge_diagram_width,
                    last_x_data=self.last_data[0],
                    last_y_data=self.last_data[1])


class SensingDot1D(Evaluator):
    """
    Sweep one gate of the sensing dot to find the point of steepest slope on the current
    """

    def __init__(self,
                 experiment: Experiment,
                 sweeping_gate: str="SDB2",
                 parameters: Sequence[str]=("position_SDB2", "current_signal", "optimal_signal"),
                 sensing_dot_measurement: Measurement=None,
                 last_x_data: Optional[np.ndarray]=None,
                 last_y_data: Optional[np.ndarray] = None):
        self._sweeping_gate = sweeping_gate
        if sensing_dot_measurement is None:
            sensing_dot_measurement = Measurement('line_scan',
                                                  center=None, range=4e-3, gate="SDB2",
                                                  N_points=1280, ramptime=.0005,
                                                  N_average=1, AWGorDecaDAC='DecaDAC')
        super().__init__(experiment, measurements=(sensing_dot_measurement,), parameters=parameters,
                         last_x_data=last_x_data, last_y_data=last_y_data)

    def evaluate(self):
        sensing_dot_measurement = self.measurements[0]
        values = pd.Series()
        error = pd.Series()
        gate = self._sweeping_gate
        sensing_dot_measurement.options["gate"] = gate
        sensing_dot_measurement.options["center"] = self.experiment.read_gate_voltages()[gate]

        data = self.experiment.measure(sensing_dot_measurement)
        data_filterd = scipy.ndimage.filters.gaussian_filter1d(input=data, sigma=.5, axis=0, order=0,
                                                               mode="nearest",
                                                               truncate=4.)
        data_filtered_diff = np.diff(data_filterd, n=1)
        data_filtered_diff_smoothed = scipy.ndimage.filters.gaussian_filter1d(input=data_filtered_diff, sigma=.5,
                                                                              axis=0, order=0,
                                                                              mode="nearest",
                                                                              truncate=4.)
        data_filtered_diff_smoothed = np.squeeze(data_filtered_diff_smoothed)
        current_signal = abs(data_filtered_diff_smoothed[int(self.measurements[0].options["N_points"] / 2)])
        optimal_signal = abs(data_filtered_diff_smoothed.min())
        optimal_position = data_filtered_diff_smoothed.argmin()
        optimal_position = float(optimal_position) / float(self.measurements[0].options["N_points"]) * 2 * \
            self.measurements[0].options["range"] - self.measurements[0].options["range"]

        values["position_" + gate] = sensing_dot_measurement.options["center"] + optimal_position
        error["position_" + gate] = 0.1e-3
        values["current_signal"] = current_signal
        error["current_signal"] = current_signal / 5
        values["optimal_signal"] = optimal_signal
        error["optimal_signal"] = optimal_signal / 5
        return values, error

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    parameters=self.parameters,
                    sensing_dot_measurement=self.measurements[0],
                    sweeping_gate=self._sweeping_gate,
                    last_x_data=self.last_data[0],
                    last_y_data=self.last_data[1])


class SensingDot2D(Evaluator):
    """
    Two dimensional sensing dot scan. Coulomb peak might be changed
    """

    def __init__(self, experiment: Experiment,
                 sweeping_gates: Sequence[str]=("SDB1", "SDB2"),
                 scan_range: float=15e-3,
                 parameters: Sequence[str]=("position_SDB1", "position_SDB2"),
                 sensing_dot_measurement: Measurement=None,
                 last_x_data: Optional[np.ndarray]=None,
                 last_y_data: Optional[np.ndarray] = None):
        self._sweeping_gates = sweeping_gates
        self._scan_range = scan_range
        if sensing_dot_measurement is None:
            sensing_dot_measurement = Measurement('2d_scan', center=[None, None],
                                                  range=scan_range,
                                                  gate1=sweeping_gates[0],
                                                  gate2=sweeping_gates[1],
                                                  N_points=1280, ramptime=.0005, n_lines=20,
                                                  n_points=104, N_average=1, AWGorDecaDAC='DecaDAC')
        super().__init__(experiment, (sensing_dot_measurement,), parameters=parameters, last_x_data=last_x_data,
                         last_y_data=last_y_data)

    def evaluate(self):
        self.measurements[0].options["center"] = [
            self.experiment.read_gate_voltages()[self.measurements[0].options["gate1"]],
            self.experiment.read_gate_voltages()[self.measurements[0].options["gate2"]]]
        data = self.experiment.measure(self.measurements[0])
        data_filterd = scipy.ndimage.filters.gaussian_filter1d(input=data, sigma=.5, axis=0, order=0,
                                                               mode="nearest",
                                                               truncate=4.)
        data_diff = np.diff(data_filterd, n=1)
        mins_in_lines = data_diff.min(1)
        min_line = np.argmin(mins_in_lines)
        min_point = np.argmin(data_diff[min_line])
        gate_1_pos = float(min_line) / float(self.measurements[0].options["n_lines"]) * 2 * \
            self.measurements[0].options["range"] - self.measurements[0].options["range"]
        gate_2_pos = float(min_point) / float(self.measurements[0].options["N_points"]) * 2 * \
            self.measurements[0].options["range"] - self.measurements[0].options["range"]
        new_voltages = pd.Series([gate_1_pos, gate_2_pos], ["position_" + self.measurements[0].options["gate1"],
                                                            "position_" + self.measurements[0].options["gate2"]])
        error = pd.Series([.1e-3, .1e-3], ["position_" + self.measurements[0].options["gate1"],
                                           "position_" + self.measurements[0].options["gate2"]])
        return new_voltages, error

    def to_hdf5(self):
        return dict(experiment=self.experiment,
                    parameters=self.parameters,
                    sensing_dot_measurement=self.measurements[0],
                    sweeping_gates=self._sweeping_gates,
                    scan_range=self._scan_range,
                    last_x_data=self.last_data[0],
                    last_y_data=self.last_data[1])
    