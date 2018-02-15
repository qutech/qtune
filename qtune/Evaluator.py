from qtune.experiment import Experiment, Measurement
from qtune.Basic_DQD import TestDQD, BasicDQD
import qtune.chrg_diag
from typing import Tuple
import pandas as pd
import numpy as np
import h5py
from scipy import optimize
import matplotlib.pyplot as plt


class Evaluator:
    """
    The evaluator classes conduct measurements, calculate the parameters with the scan data and save the results in the HDF5 library.
    """
    def __init__(self, experiment: Experiment, measurements: Tuple[Measurement, ...], parameters: pd.Series):
        self.experiment = experiment
        self.measurements = measurements
        self.parameters = parameters

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
                 lead_scan: Measurement = None):
        if lead_scan is None:
            lead_scan = dqd.measurements[2]
        super().__init__(dqd, lead_scan, parameters)

    def evaluate(self, storing_group: h5py.Group) -> pd.Series:
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
        if storing_group is not None:
            storing_dataset = storing_group.create_dataset("evaluator_SMLeadTunnelTimeByLeadScan", data=data)
            storing_dataset.attrs["parameter_time_rise"] = t_rise
            storing_dataset.attrs["parameter_time_fall"] = t_fall
            if failed:
                storing_dataset.attrs["parameter_time_rise"] = np.nan
                storing_dataset.attrs["parameter_time_fall"] = np.nan
        return pd.Series([t_rise, t_fall, failed], ['parameter_time_rise', 'parameter_time_fall', 'failed'])


class InterDotTCByLineScan(Evaluator):
    """
    Adiabaticly sweeps the detune over the transition between the (2,0) and the (1,1) region. An Scurve is fitted and
    the width calculated as parameter for the inter dot coupling.
    """
    def __init__(self, dqd: BasicDQD, parameters: pd.Series() = pd.Series((np.nan,), ('parameter_tunnel_coupling',)),
                 line_scan: Measurement = None):
        if line_scan is None:
            line_scan = dqd.measurements[1]
        super().__init__(dqd, line_scan, parameters)

    def evaluate(self, storing_group: h5py.Group) -> pd.Series:
        ydata = self.experiment.measure(self.measurements)
        center = self.measurements.parameter['center']
        scan_range = self.measurements.parameter['range']
        npoints = self. measurements.parameter['N_points']
        plt.ion()
        plt.figure(51)
        plt.clf()
        fitresult = fit_inter_dot_coupling(data=ydata, center=center, scan_range=scan_range, npoints=npoints)
#        plt.pause(0.05)
        tc = fitresult['tc']
        failed = bool(fitresult['failed'])
        self.parameters['parameter_tunnel_coupling'] = tc
        if storing_group is not None:
            storing_dataset = storing_group.create_dataset("evaluator_SMInterDotTCByLineScan", data=ydata)
            storing_dataset.attrs["center"] = center
            storing_dataset.attrs["scan_range"] = scan_range
            storing_dataset.attrs["npoints"] = npoints
            storing_dataset.attrs["parameter_tunnel_coupling"] = tc
            if failed:
                storing_dataset.attrs["parameter_tunnel_coupling"] = np.nan
        return pd.Series((tc, failed), ('parameter_tunnel_coupling', 'failed'))


class LoadTime(Evaluator):
    """
    Measures the time required to reload a (2,0) singlet state. Fits an exponential function.
    """
    def __init__(self, dqd: BasicDQD, parameters: pd.Series() = pd.Series((np.nan,), ('parameter_time_load',)),
                 load_scan: Measurement = None):
        if load_scan is None:
            load_scan = dqd.measurements[3]
        super().__init__(dqd, load_scan, parameters)

    def evaluate(self, storing_group: h5py.Group) -> pd.Series:
        data = self.experiment.measure(self.measurements)
        n_points = data.shape[1]
        plt.ion()
        plt.figure(81)
        plt.clf()
        plt.figure(81)
        fitresult = fit_load_time(data=data, n_points=n_points, )
        plt.pause(0.05)
        parameter_time_load = fitresult['parameter_time_load']
        failed = fitresult['failed']
        self.parameters['parameter_time_load'] = parameter_time_load
        if storing_group is not None:
            storing_dataset = storing_group.create_dataset("evaluator_LoadTime", data=data)
            storing_dataset.attrs["parameter_time_load"] = parameter_time_load
            if failed:
                storing_dataset.attrs["parameter_time_load"] = np.nan
        return pd.Series([parameter_time_load, failed], ['parameter_time_load', 'failed'])


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


def fit_inter_dot_coupling(data, **kwargs):
    failed = 0
    center = kwargs["center"]
    scan_range = kwargs["scan_range"]
    npoints = kwargs["npoints"]
    xdata = np.linspace(center - scan_range, center + scan_range, npoints)
    ydata = np.squeeze(data)
    m_last_part, b_last_part = np.polyfit(xdata[int(round(0.75*npoints)):npoints-1], ydata[int(round(0.75*npoints)):npoints-1], 1)
    m_first_part, b_first_part = np.polyfit(xdata[0:int(round(0.25*npoints))], ydata[0:int(round(0.25*npoints))], 1)
    height = (b_last_part + m_last_part * xdata[npoints - 1]) - (b_first_part + m_first_part * xdata[npoints - 1])
    position = qtune.chrg_diag.find_lead_transition(data=ydata - xdata * m_first_part, center=center, scan_range=scan_range, npoints=npoints,
                                    width=scan_range / 12.)
    p0 = [b_first_part, m_first_part, height, position, scan_range / 8.]
    plt.plot(xdata, ydata, "b.")
    plt.plot(xdata, func_inter_dot_coupling(xdata, p0[0], p0[1], p0[2], p0[3], p0[4]), "k--")
    popt, pcov = optimize.curve_fit(f=func_inter_dot_coupling, p0=p0, xdata=xdata, ydata=ydata)
    plt.plot(xdata, func_inter_dot_coupling(xdata, popt[0], popt[1], popt[2], popt[3], popt[4]), "r")
    plt.draw()
    width_in_mus = popt[4] * 1e6
    fit_result = pd.Series(data=[width_in_mus, failed], index=["tc", "failed"])
    return fit_result


def func_inter_dot_coupling(xdata, offset: float, slope: float, height: float, position: float, width: float):
    return offset + slope * xdata + .5 * height * (1 + np.tanh((xdata - position) / width))

def fit_load_time(data, **kwargs):
    failed = 0
    n_points = data.shape[1]
    ydata = data[0, 1:n_points]
    xdata = data[1, 1:n_points]
    min = np.nanmin(ydata)
    max = np.nanmin(ydata)
    initial_curvature = 10
    p0 = [min, max - min, initial_curvature]
    bounds = ([-np.inf, -np.inf, 2.],
              [np.inf, np.inf, 100.])
    popt, pcov = optimize.curve_fit(f=func_load_time, p0=p0, bounds=bounds, xdata=xdata, ydata=ydata)
    if popt[2] < 0.:
        initial_curvature = 200
        p0 = [min, max - min, initial_curvature]
        popt, pcov = optimize.curve_fit(f=func_load_time, p0=p0, xdata=xdata, ydata=ydata)
    plt.plot(xdata, ydata, "b.")
    plt.plot(xdata, func_load_time(xdata, p0[0], p0[1], p0[2]), "k--")
    plt.plot(xdata, func_load_time(xdata, popt[0], popt[1], popt[2]), "r")
    plt.draw()
    fit_result = pd.Series(data=[popt[2], failed], index=["parameter_time_load", "failed"])
    return fit_result

def func_load_time(xdata, offset: float, height: float, curvature: float):
    return offset + height * np.exp(-1. * xdata / curvature)
