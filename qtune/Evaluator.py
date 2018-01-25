from qtune.experiment import Experiment, Measurement
from qtune.Basic_DQD import TestDQD, BasicDQD
from typing import Tuple
import pandas as pd
import numpy as np
import h5py
from scipy import optimize
import matplotlib.pyplot as plt


class Evaluator:
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
        fitresult = fit_lead_times(data)
        plt.pause(0.05)
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


def fit_lead_times(ydata: np.ndarray):
    samprate = 1e8
    n_points = len(ydata)
    xdata = np.asarray([i for i in range(n_points)]) / samprate
    p0 = [ydata[round(1. / 4. * n_points)] - ydata[round(3. / 4. * n_points)],
          50e-9, 50e-9, 70e-9, 2070e-9, np.mean(ydata)]
    begin_lin = int(round(p0[4] / 10e-9))
    end_lin = begin_lin + 7
    slope = (ydata[end_lin] - ydata[begin_lin]) / (xdata[end_lin] - xdata[begin_lin])
    linear_offset = ydata[begin_lin] - xdata[begin_lin] * slope
    p0 += [slope, linear_offset, xdata[end_lin]]
    begin_lin_1 = int(round(p0[3] / 10e-9))
    end_lin_1 = begin_lin_1 + 7
    slope_1 = (ydata[end_lin_1] - ydata[begin_lin_1]) / (xdata[end_lin_1] - xdata[begin_lin_1])
    linear_offset_1 = ydata[begin_lin_1] - xdata[begin_lin_1] * slope_1
    p0 += [slope_1, linear_offset_1, xdata[end_lin_1]]
    plt.plot(xdata * 1e6, ydata, "b.")
    plt.plot(xdata * 1e6, func_lead_times_v2(xdata, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7], p0[8], p0[9], p0[10], p0[11]), "k--")
    popt, pcov = optimize.curve_fit(f=func_lead_times_v2, p0=p0, xdata=xdata, ydata=ydata)
    plt.plot(xdata * 1e6, func_lead_times_v2(xdata, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11]), "r")
    plt.pause(0.05)
    failed = 0
    fitresult = pd.Series(data=[popt[1], popt[2], failed], index=["t_fall", "t_rise", "failed"])
    return fitresult


def func_lead_times_v2(x, hight: float, t_fall: float, t_rise: float, begin_rise: float, begin_fall: float,
                       offset: float, slope: float, offset_linear: float, exp_begin_fall: float, slope_1: float,
                       offset_linear_1: float, exp_begin_rise: float):
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
