from qtune.experiment import Experiment, Measurement
from qtune.basic_dqd import TestDQD, BasicDQD
import qtune.chrg_diag
from typing import Tuple
import pandas as pd
import numpy as np
import h5py
from scipy import optimize
import matplotlib.pyplot as plt
import qtune.util
import scipy.ndimage


class Evaluator:
    """
    The evaluator classes conduct measurements, calculate the parameters with the scan data and save the results in the HDF5 library.
    """
    def __init__(self, experiment: Experiment, measurements: Tuple[Measurement, ...], parameters: pd.Series,
                 name: str):
        self.experiment = experiment
        self.measurements = measurements
        self.parameters = parameters
        self.name = name

    def evaluate(self, storing_group: h5py.Group) -> pd.Series:
        raise NotImplementedError


class TestEvaluator(Evaluator):
    def __init__(self, experiment: TestDQD, measurements=None,
                 parameters=pd.Series((np.nan, np.nan), ('linsine', 'quadratic')), ):
        super().__init__(experiment=experiment, measurements=measurements, parameters=parameters)

    def evaluate(self, storing_group: h5py.Group) -> pd.Series:
        test_voltages = self.experiment.read_gate_voltages()
        test_voltages = test_voltages.sort_index()
        linsine = test_voltages[0] * np.sin(test_voltages[1])
        quadratic = test_voltages[1] * test_voltages[1]
        return pd.Series([linsine, quadratic, False], ['linsine', 'quadratic', 'failed'])


class LeadTunnelTimeByLeadScan(Evaluator):
    """
    RF gates pulse over the transition between the (2,0) and the (1,0) region. Then exponential functions are fitted
    to calculate the time required for an electron to tunnel through the lead.
    """
    def __init__(self, dqd: BasicDQD,
                 parameters: pd.Series() = pd.Series([np.nan, np.nan], ['parameter_time_rise', 'parameter_time_fall']),
                 lead_scan: Measurement = None, name: str=""):
        if lead_scan is None:
            lead_scan = dqd.measurements[2]
        super().__init__(dqd, lead_scan, parameters, name)

    def evaluate(self) -> pd.Series:
        data = self.experiment.measure(self.measurements)
        plt.ion()
        plt.figure(50)
        plt.clf()
        fitresult = fit_lead_times(data)
        plt.show()
#        plt.pause(0.05)
        t_rise = fitresult['t_rise']
        t_fall = fitresult['t_fall']
        failed = fitresult['failed']
        self.parameters['parameter_time_rise'] = t_rise
        self.parameters['parameter_time_fall'] = t_fall

        return pd.Series([t_rise, t_fall], ['parameter_time_rise', 'parameter_time_fall'])


class InterDotTCByLineScan(Evaluator):
    """
    Adiabaticly sweeps the detune over the transition between the (2,0) and the (1,1) region. An Scurve is fitted and
    the width calculated as parameter for the inter dot coupling.
    """
    def __init__(self, dqd: BasicDQD, parameters: pd.Series() = pd.Series((np.nan,), ('parameter_tunnel_coupling',)),
                 line_scan: Measurement = None, name: str=""):
        if line_scan is None:
            line_scan = dqd.measurements[1]
        super().__init__(dqd, line_scan, parameters, name)

    def evaluate(self) -> (pd.Series, pd.Series):
        ydata = self.experiment.measure(self.measurements)
        center = self.measurements.parameter['center']
        scan_range = self.measurements.parameter['range']
#        npoints = self.measurements.parameter['N_points']
        npoints = ydata.shape[1]
        plt.ion()
        plt.figure(51)
        plt.clf()
        try:
            fitresult = fit_inter_dot_coupling(data=ydata.copy(), center=center, scan_range=scan_range, npoints=npoints)
        except:
            fitresult = pd.Series(data=[True, np.nan, np.nan], index=['failed', 'parameter_tunnel_coupling', "residual"])
#        plt.pause(0.05)
        tc = fitresult['tc']
        failed = bool(fitresult['failed'])
        residual = fitresult["residual"]
        self.parameters['parameter_tunnel_coupling'] = tc
        return pd.Series([tc], ["parameter_tunnel_coupling"]), pd.Series([residual], ["residual"])


class LoadTime(Evaluator):
    """
    Measures the time required to reload a (2,0) singlet state. Fits an exponential function.
    """
    def __init__(self, dqd: BasicDQD, parameters: pd.Series() = pd.Series((np.nan,), ('parameter_time_load',)),
                 load_scan: Measurement = None, name: str=""):
        if load_scan is None:
            load_scan = dqd.measurements[3]
        super().__init__(dqd, (load_scan, ), parameters, name)

    def evaluate(self) -> (pd.Series, pd.Series):
        data = self.experiment.measure(self.measurements[0])
        n_points = data.shape[1]
        plt.ion()
        plt.figure(81)
        plt.clf()
        plt.figure(81)
        try:
            fitresult = fit_load_time(data=data, n_points=n_points, )
        except:
            fitresult = pd.Series(data=[True, np.nan, np.nan], index=['failed', 'parameter_time_load', "residual"])
        plt.pause(0.05)
        parameter_time_load = fitresult['parameter_time_load']
        failed = fitresult['failed']
        residual = fitresult["residual"]
        self.parameters['parameter_time_load'] = parameter_time_load
        return pd.Series([parameter_time_load], ['parameter_time_load']), pd.Series([residual], ["residual"])


class LeadTransition(Evaluator):
    """
    Finds the transition on the edge of the charge diagram
    """

    def __init__(self, experiment: Experiment, name, parameters: pd.Series() = pd.Series([4e-3, ["RFA", "RFB"]],
                                                                                         ["charge_diagram_width",
                                                                                          "shifting_gates"])):
        default_line_scan_a = Measurement('line_scan', center=0., range=4e-3,
                                          gate='RFA', N_points=320,
                                          ramptime=.001,
                                          N_average=7,
                                          AWGorDecaDAC='DecaDAC')
        self.shifting_gates = parameters["shifting_gates"]
        self.charge_diagram_width = parameters["charge_diagram_width"]
        super().__init__(experiment, (default_line_scan_a, ), pd.Series(), name=name)

    def evaluate(self, storing_group: h5py.Group):
        current_position = pd.Series()
        error = pd.Series()
        current_gate_voltages = self.experiment.read_gate_voltages()
        for gate in self.shifting_gates:
            shift = pd.Series(-1. * self.charge_diagram_width, [gate])
            self.experiment.set_gate_voltages(current_gate_voltages.add(shift))
            self.measurements[0].parameter["gate"] = gate
            data = self.experiment.measure(self.measurements[0])
            current_position["position_" + gate] = qtune.util.find_lead_transition(data,
                                                                     float(
                                                                         self.measurements[0].parameter["center"]),
                                                                     float(
                                                                         self.measurements[0].parameter["range"]),
                                                                     int(self.measurements[0].parameter["N_points"]))
            error["position_" + gate] = .1e-3
        self.experiment.set_gate_voltages(current_gate_voltages)

        return current_position, error


class SensingDot1D(Evaluator):
    """
    Sweep one gate of the sensing dot to find the point of steepest slope on the current
    """

    def __init__(self,
                 experiment: Experiment,
                 name,
                 parameters: pd.Series() = pd.Series([["SDB2"]],
                                                     ["sweeping_gates"])):
        self.sweeping_gates = parameters["sweeping_gates"]
        sensing_dot_measurement = Measurement('line_scan',
                                              center=None, range=4e-3, gate="SDB2",
                                              N_points=1280, ramptime=.0005,
                                              N_average=1, AWGorDecaDAC='DecaDAC')
        super().__init__(experiment, (sensing_dot_measurement,), pd.Series(), name=name)

    def evaluate(self, storing_group: h5py.Group):
        sensing_dot_measurement = self.measurements[0]
        new_gate_voltage = pd.Series()
        error = pd.Series()
        for gate in self.sweeping_gates:
            sensing_dot_measurement.parameter["gate"] = gate
            sensing_dot_measurement.parameter["center"] = self.experiment.read_gate_voltages()[gate]
            data = self.experiment.measure(sensing_dot_measurement)
            detuning = qtune.util.find_stepes_point_sensing_dot(data=data,
                                                                scan_range=sensing_dot_measurement.parameter[
                                                                    "scan_range"],
                                                                npoints=sensing_dot_measurement.parameter["N_points"])
            new_gate_voltage["position_" + gate] = sensing_dot_measurement.parameter["center"] + detuning
            error["position_" + gate] = 0.1e-3

        return new_gate_voltage, error


class SensingDot2D(Evaluator):
    """
    Two dimensional sensing dot scan. Coulomb peak might be changed
    """

    def __init__(self, experiment: Experiment, name,
                 parameters: pd.Series() = pd.Series([15e-3, ["SDB2", "SDB1"]], ["scan_range", "gates"])):
        sensing_dot_measurement = Measurement('2d_scan', center=[None, None],
                                              range=parameters["scan_range"],
                                              gate1=parameters["gates"][0],
                                              gate2=parameters["gates"][1],
                                              N_points=1280, ramptime=.0005, n_lines=20,
                                              n_points=104, N_average=1, AWGorDecaDAC='DecaDAC')
        super().__init__(experiment, (sensing_dot_measurement,), pd.Series(), name=name)

    def evaluate(self, storing_group: h5py.Group):
        self.measurements[0].parameter["center"] = [
            self.experiment.read_gate_voltages()[self.measurements[0].parameter["gate1"]],
            self.experiment.read_gate_voltages()[self.measurements[0].parameter["gate2"]]]
        data = self.experiment.measure(self.measurements[0])
        data_filterd = scipy.ndimage.filters.gaussian_filter1d(input=data, sigma=.5, axis=0, order=0,
                                                               mode="nearest",
                                                               truncate=4.)
        data_diff = np.diff(data_filterd, n=1)
        mins_in_lines = data_diff.min(1)
        min_line = np.argmin(mins_in_lines)
        min_point = np.argmin(data_diff[min_line])
        gate_1_pos = float(min_line) / float(self.measurements[0].parameter["n_lines"]) * 2 * \
            self.measurements[0].parameter["range"] - self.measurements[0].parameter["range"]
        gate_2_pos = float(min_point) / float(self.measurements[0].parameter["N_points"]) * 2 * \
            self.measurements[0].parameter["range"] - self.measurements[0].parameter["range"]
        new_voltages = pd.Series([gate_1_pos, gate_2_pos], ["position_" + self.measurements[0].parameter["gate1"],
                                                            "position_" + self.measurements[0].parameter["gate1"]])
        error = pd.Series([.1e-3, .1e-3], ["position_" + self.measurements[0].parameter["gate1"],
                                           "position_" + self.measurements[0].parameter["gate1"]])
        return new_voltages, error


def fit_lead_times(data: np.ndarray, **kwargs):
    ydata = data[1, :] - data[0, :]
    samprate = 1e8
    n_points = len(ydata)
    xdata = np.asarray([i for i in range(n_points)]) / samprate
    p0 = [ydata[round(1. / 4. * n_points)] - ydata[round(3. / 4. * n_points)],
          50e-9, 50e-9, 70e-9, 2070e-9, np.mean(ydata)]
    begin_lin = int(round(p0[4] / 10e-9))
    end_lin = begin_lin + 5
    slope = (ydata[end_lin] - ydata[begin_lin]) / (xdata[end_lin] - xdata[begin_lin])
    linear_offset = ydata[begin_lin] - xdata[begin_lin] * slope
    p0 += [slope, linear_offset, xdata[end_lin] - xdata[begin_lin]]
    begin_lin_1 = int(round(p0[3] / 10e-9))
    end_lin_1 = begin_lin_1 + 5
    slope_1 = (ydata[end_lin_1] - ydata[begin_lin_1]) / (xdata[end_lin_1] - xdata[begin_lin_1])
    linear_offset_1 = ydata[begin_lin_1] - xdata[begin_lin_1] * slope_1
    p0 += [slope_1, linear_offset_1, xdata[end_lin_1] - xdata[begin_lin_1]]
    plt.plot(xdata * 1e6, ydata, "b.")
    plt.plot(xdata * 1e6,
             func_lead_times_v2(xdata, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7], p0[8], p0[9], p0[10],
                                p0[11]), "k--")
    bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0., -np.inf, -np.inf, 0.],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, float(n_points / 60. * samprate), np.inf, np.inf,
               float(n_points / 60. * samprate)])
    popt, pcov = optimize.curve_fit(f=func_lead_times_v2, p0=p0, bounds=bounds, xdata=xdata, ydata=ydata)
    plt.plot(xdata * 1e6,
             func_lead_times_v2(xdata, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8],
                                popt[9], popt[10], popt[11]), "r")
    plt.draw()
#    plt.pause(0.05)
    failed = 0
    fitresult = pd.Series(data=[popt[1], popt[2], failed], index=["t_fall", "t_rise", "failed"])
    return fitresult


def func_lead_times_v2(x, hight: float, t_fall: float, t_rise: float, begin_rise: float, begin_fall: float,
                       offset: float, slope: float, offset_linear: float, lenght_lin_fall: float, slope_1: float,
                       offset_linear_1: float, length_lin_rise: float):
    exp_begin_rise = begin_rise + length_lin_rise
    exp_begin_fall = begin_fall + lenght_lin_fall
    half_time = 2e-6
    x = np.squeeze(x)
    n_points = len(x)
    y = np.zeros((n_points, ))
    for i in range(n_points):
        if x[i] >= exp_begin_rise and x[i] <= begin_fall:
            c = np.cosh(.5*half_time/t_rise)
            s = np.sinh(.5*half_time/t_rise)
            e = np.exp(.5*(half_time - 2. * x[i]) / t_rise)
            signed_hight = hight
            y[i] = offset + .5 * signed_hight * (c - e) / s

        elif x[i] <= begin_rise:
            c = np.cosh(.5*half_time/t_fall)
            s = np.sinh(.5*half_time/t_fall)
            e = np.exp(.5*(1. * half_time - 2. * (x[i] + x[n_points - 1])) / t_fall)
            signed_hight = -1. * hight
            y[i] = offset + .5 * signed_hight * (c - e) / s
        elif x[i] > begin_fall and x[i] < exp_begin_fall:
            y[i] = offset_linear + x[i] * slope
        elif x[i] > begin_rise and x[i] < exp_begin_rise:
            y[i] = offset_linear_1 + x[i] * slope_1
        else:
            c = np.cosh(.5*half_time/t_fall)
            s = np.sinh(.5*half_time/t_fall)
            e = np.exp(.5*(3. * half_time - 2. * x[i]) / t_fall)
            signed_hight = -1. * hight
            y[i] = offset + .5 * signed_hight * (c - e) / s
    return y


def fit_inter_dot_coupling(data, plot_fit=True, cut_fist_fifth=True, **kwargs):
    failed = 0
    center = kwargs["center"]
    scan_range = kwargs["scan_range"]
    npoints = kwargs["npoints"]
    xdata = np.linspace(center - scan_range, center + scan_range, npoints)
    if len(data.shape) == 1:
        ydata = np.squeeze(data)
    else:
        ydata = np.squeeze(np.mean(data, 0))
    if cut_fist_fifth:
        cut_points = int(0.15 * float(npoints))
        ydata = ydata[cut_points - 1:npoints-1]
        xdata = xdata[cut_points - 1:npoints-1]
        npoints = npoints - cut_points
    m_last_part, b_last_part = np.polyfit(xdata[int(round(0.75*npoints)):npoints-1], ydata[int(round(0.75*npoints)):npoints-1], 1)
    m_first_part, b_first_part = np.polyfit(xdata[0:int(round(0.25*npoints))], ydata[0:int(round(0.25*npoints))], 1)
    height = (b_last_part + m_last_part * xdata[npoints - 1]) - (b_first_part + m_first_part * xdata[npoints - 1])
    position = qtune.chrg_diag.find_lead_transition(data=ydata - xdata * m_first_part, center=center,
                                                    scan_range=scan_range, npoints=npoints, width=scan_range / 12.)
    p0 = [b_first_part, m_first_part, height, position, scan_range / 8.]
    if plot_fit:
        plt.plot(1e3 * xdata, ydata, "b.", label="Data")
    popt, pcov = optimize.curve_fit(f=func_inter_dot_coupling, p0=p0, xdata=xdata, ydata=ydata)

    weights = np.ones(npoints)
    position_point = int((popt[3] + scan_range) / 2. / scan_range * npoints)
    heavy_range = 0.25
    if position_point < heavy_range * npoints:
        begin_weight = 0
    else:
        begin_weight = position_point - int(heavy_range * npoints)
    if (npoints - position_point) < heavy_range * npoints:
        end_weight = npoints
    else:
        end_weight = position_point + int(heavy_range * npoints)
    weights[begin_weight:end_weight] = .1

    popt, pcov = optimize.curve_fit(f=func_inter_dot_coupling, p0=popt, sigma=weights, xdata=xdata, ydata=ydata)

    if plot_fit:
        plt.plot(1e3 * xdata, func_inter_dot_coupling(xdata, popt[0], popt[1], popt[2], popt[3], popt[4]), "r",
                 label="Fit")
        plt.xlabel("Detuning $\epsilon$ [mV]", fontsize=22)
        plt.ylabel("Signal [a.u.]", fontsize=22)
        plt.gca().tick_params("x", labelsize=22)
        plt.gca().tick_params("y", labelsize=0)
        plt.legend(fontsize=16)
        fig = plt.gcf()
        fig.set_size_inches(8.5, 8)
        plt.show()

    residuals = ydata[int(0.3 * npoints):npoints] - func_inter_dot_coupling(xdata[int(0.3 * npoints):npoints], popt[0], popt[1], popt[2], popt[3], popt[4])
    residual = np.nanmean(np.square(residuals)) / (popt[2] * popt[2]) * 2e4
    width_in_mus = popt[4] * 1e6
    fit_result = pd.Series(data=[width_in_mus, failed, residual], index=["tc", "failed", "residual"])
    return fit_result


def func_inter_dot_coupling_2_slopes(xdata, offset: float, slope_left: float, slope_right: float, height: float, position: float,
                            width: float):
    n_points = xdata.shape[0]
    i_position = int(n_points / 2)
    for i in range(n_points):
        if xdata[i] > position:
            i_position = i
            break
    ydata = np.ones(xdata.shape)
    ydata[0:i_position] = offset + slope_left * xdata[0:i_position] + .5 * height * (
        1 + np.tanh((xdata[0:i_position] - position) / width))
    ydata[i_position:n_points] = offset + slope_right * xdata[i_position:n_points] + .5 * height * (
        1 + np.tanh((xdata[i_position:n_points] - position) / width))
    return ydata


def func_inter_dot_coupling_parabola(xdata, offset: float, slope: float, curvature: float, height: float, position: float, width: float):
    return offset + slope * xdata + curvature * (xdata - xdata[0]) * (xdata - xdata[0]) + .5 * height * (1 + np.tanh((xdata - position) / width))


def func_inter_dot_coupling(xdata, offset: float, slope: float, height: float, position: float, width: float):
    return offset + slope * xdata + .5 * height * (1 + np.tanh((xdata - position) / width))


def fit_load_time(data, plot_fit=True, **kwargs):
    failed = 0
    n_points = data.shape[1]
    ydata = data[0, 1:n_points]
    xdata = data[1, 1:n_points]
    min = np.nanmin(ydata)
    max = np.nanmax(ydata)
    initial_curvature = 10.
    p0 = [min, max - min, initial_curvature]
    bounds = ([-np.inf, -np.inf, 2.],
              [np.inf, np.inf, 500.])
#    plt.plot(xdata, func_load_time(xdata, p0[0], p0[1], p0[2]), "k--", label="Fit Starting Values")
#    popt, pcov = optimize.curve_fit(f=func_load_time, p0=p0, bounds=bounds, xdata=xdata, ydata=ydata)
    popt, pcov = optimize.curve_fit(f=func_load_time, p0=p0, xdata=xdata, ydata=ydata)
    if popt[2] < 0.:
        initial_curvature = 200
        p0 = [min, max - min, initial_curvature]
        if plot_fit:
            plt.plot(xdata, func_load_time(xdata, p0[0], p0[1], p0[2]), "k--")
        popt, pcov = optimize.curve_fit(f=func_load_time, p0=p0, xdata=xdata, ydata=ydata)
    if plot_fit:
        plt.plot(xdata, ydata, "b.", label="Data")
        plt.plot(xdata, func_load_time(xdata, popt[0], popt[1], popt[2]), "r", label="Fit")
        plt.xlabel("Reload time [ns]", fontsize=22)
        plt.gca().tick_params("x", labelsize=22)
        plt.gca().tick_params("y", labelsize=0)
        plt.ylabel("Signal [a.u.]", fontsize=22)
        plt.legend(fontsize=16)
        fig = plt.gcf()
        fig.set_size_inches(8.5, 8)
        plt.show()
    residual = ydata - func_load_time(xdata, popt[0], popt[1], popt[2])
    #residual = np.nanmean(np.square(residual)) / np.ptp(ydata)
    residual = np.nanmean(np.square(residual)) / (popt[1] * popt[1]) * 1500.
    fit_result = pd.Series(data=[popt[2], failed, residual], index=["parameter_time_load", "failed", "residual"])
    return fit_result


def func_load_time(xdata, offset: float, height: float, curvature: float):
    return offset + height * np.exp(-1. * xdata / curvature)
