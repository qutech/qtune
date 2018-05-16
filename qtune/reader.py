import os
import h5py
import operator
import pandas as pd
import matplotlib.pyplot as plt
import qtune.storage
import qtune.autotuner
import qtune.solver
import qtune.gradient

from typing import Tuple

parameter_information = {
    "position_RFA": {
        "entity_unit": "Transition Position Lead A (V)",
        "name": "Transition Position Lead A"
    },
    "position_RFB": {
        "entity_unit": "Transition Position Lead B (V)",
        "name": "Transition Position Lead B"
    },
    "parameter_tunnel_coupling": {
        "entity_unit": "Tunnel Coupling (\mu V)",
        "name": "Inter-Dot Tunnel Coupling AB"
    },
    "current_signal": {
        "entity_unit": "Signal (a.u.)",
        "name": "Current SDot Signal"
    },
    "optimal_signal": {
        "entity_unit": "Signal (a.u.)",
        "name": "Optimal SDot Signal"
    },
    "position_SDB2": {
        "entity_unit": "Voltage (V)",
        "name": "Voltage on SDB2"
    },
    "parameter_time_load": {
        "entity_unit": "Time (ns)",
        "name": "Singlet Reload Time"
    }
}

def read_files(file_or_files, reserved=None):
    if isinstance(file_or_files, str):
        files = [file_or_files]
    else:
        files = list(file_or_files)

    data = []

    for file_name in files:
        with h5py.File(file_name, 'r') as root:
            entries = [key for key in root.keys() if not key.startswith('#')]

            for entry in entries:
                entry_data = qtune.storage.from_hdf5(root[entry], reserved)

                data.append((entry, entry_data))

    return data.sort(key=operator.itemgetter(0))


class Reader:
    def __init__(self, path):
        self._file_path = path
        self.tuner_list, self.index_list = self.load_data()

        self.voltage_list = []
        self.parameter_list = []
        self.gradient_list = []
        self.covariance_list = []
        for tuning_hierarchy in self.tuner_list:
            self.voltage_list.append(extract_voltages_from_hierarchy(tuning_hierarchy))
            self.parameter_list.append(extract_parameters_from_hierarchy(tuning_hierarchy))
            grad, cov = extract_gradients_from_hierarchy(tuning_hierarchy)
            self.gradient_list.append(grad)
            self.covariance_list.append(cov)

    @property
    def gate_names(self):
        return self.voltage_list[0].index

    @property
    def parameter_names(self):
        return self.parameter_list[0].index

    @property
    def gradient_controlled_parameter_names(self):
        return self.gradient_list[0].index

    def load_data(self):
        """
        writes lists with all relavant Data.
        :return: Voltages, Parameters, Gradients, Covariances
        """
        directory_content = os.listdir(self._file_path)

        tuner_list = []
        index_list = []
        for file in directory_content:
            hdf5_handle = h5py.File(self._file_path + r"\\" + file, mode="r")
            loaded_data = qtune.storage.from_hdf5(hdf5_handle, reserved={"experiment": None})
            autotuner = loaded_data["autotuner"]
            assert(isinstance(autotuner, qtune.autotuner.Autotuner))

            tuner_list.append(autotuner._tuning_hierarchy)
            index_list.append(autotuner._current_tuner_index)
        return tuner_list, index_list

    def plot_tuning(self, voltage_indices=None, parameter_names=None, mode=""):
        if "all_voltages" in mode:
            voltage_indices = self.gate_names
        if "all_parameters" in mode:
            parameter_names = self.parameter_names
        if "all_gradients" in mode:
            gradient_names = self.gradient_controlled_parameter_names
        if "with_duplicates" in mode:
            eliminate_duplicates = False
        else:
            eliminate_duplicates = True

        if voltage_indices is None:
            voltage_fig = None
            voltage_ax = None
        else:
            voltage_fig, voltage_ax = plot_voltages(self.voltage_list,
                                                    voltage_indices=voltage_indices,
                                                    eliminate_duplicates=eliminate_duplicates)

        if parameter_names is None:
            parameter_fig = None
            parameter_ax = None
        else:
            parameter_fig, parameter_ax = plot_parameters(self.parameter_list,
                                                          parameter_names=parameter_names,
                                                          eliminate_duplicates=eliminate_duplicates)

        plt.show()
        return voltage_fig, voltage_ax, parameter_fig, parameter_ax


def plot_voltages(voltage_list, voltage_indices: Tuple[str], eliminate_duplicates: bool=True):
    if eliminate_duplicates:
        voltage_list = [v for i, v in enumerate(voltage_list) if i == 0 or v.equals(voltage_list[i - 1])]

    voltage_dataframe = pd.concat(voltage_list, axis=1)
    voltage_fig, voltage_ax = plt.subplots()
    if voltage_indices is None:
        voltage_indices = voltage_dataframe.index
    voltage_ax.plot(voltage_dataframe.T[voltage_indices])
    voltage_ax.legend(voltage_indices)
    voltage_ax.set_title("Gate Voltages")
    voltage_ax.set_xlabel("Measurement Number")
    voltage_ax.set_ylabel("Voltage (V)")
    return voltage_fig, voltage_ax


def plot_parameters(parameter_list, parameter_names: Tuple[str], eliminate_duplicates: bool=True):
    if eliminate_duplicates:
        parameter_list = [v for i, v in enumerate(parameter_list) if i == 0 or v.equals(parameter_list[i - 1])]
    parameter_dataframe = pd.concat(parameter_list, axis=1).T

    parameter_fig, parameter_ax = plt.subplots(nrows=len(parameter_names))
    if len(parameter_names) == 1:
        parameter_ax = [parameter_ax, ]
    for i, parameter in enumerate(parameter_names):
        parameter_ax[i].plot(parameter_dataframe[parameter])
        parameter_ax[i].set_ylabel(parameter_information[parameter]["entity_unit"])
        parameter_ax[i].set_title(parameter_information[parameter]["name"])
    parameter_ax[len(parameter_names) - 1].set_xlabel("Measurement Number")
    return parameter_fig, parameter_ax


def plot_gradients(gradient_list, covariance_list, parameter_names: Tuple[str], eliminate_duplicates: bool=True):
    if eliminate_duplicates:
        covariance_list = [covariance_list[i] for i, v in enumerate(gradient_list) if
                           i == 0 or v.equals(gradient_list[i - 1])]
        gradient_list = [v for i, v in enumerate(gradient_list) if i == 0 or v.equals(gradient_list[i - 1])]

    gradient_list = [pd.Series(x) for x in gradient_list]



def extract_voltages_from_hierarchy(tuning_hierarchy):
    voltages = pd.Series()
    for par_tuner in tuning_hierarchy:
        for gate in pd.Series(par_tuner._last_voltage).index:
            if gate not in voltages.index:
                voltages[gate] = par_tuner._last_voltage[gate]
    return voltages


def extract_parameters_from_hierarchy(tuning_hierarchy):
    parameters = pd.Series()
    for par_tuner in tuning_hierarchy:
        parameters = parameters.append(par_tuner._last_parameter_values)
    return parameters


def extract_gradients_from_hierarchy(tuning_hierarchy):
    gradients = dict()
    covariances = dict()
    for tuner in tuning_hierarchy:
        if isinstance(tuner.solver, qtune.solver.NewtonSolver):
            for i, grad_est in enumerate(tuner.solver._gradient_estimators):
                if isinstance(grad_est, qtune.gradient.KalmanGradientEstimator):
                    gradients[tuner.solver.target.index[i]] = grad_est.estimate()
                    covariances[tuner.solver.target.index[i]] = grad_est._kalman_gradient.cov
                elif isinstance(grad_est, qtune.gradient.FiniteDifferencesGradientEstimator):
                    gradients[tuner.solver.target.index[i]] = grad_est.estimate()
                    covariances[tuner.solver.target.index[i]] = grad_est._covariance
    return pd.DataFrame(gradients), covariances
