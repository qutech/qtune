import os
import h5py
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import qtune.storage
import qtune.autotuner
import qtune.solver
import qtune.gradient
import qtune.parameter_tuner
import os.path

from typing import Tuple, Optional, Set, Dict

parameter_information = {
    "position_RFA": {
        "entity_unit": "Transition Position Lead A (V)",
        "name": "Transition Position Lead A",
        "gradient_unit": "Gradient Transition Lead A (V/V)"
    },
    "position_RFB": {
        "entity_unit": "Transition Position Lead B (V)",
        "name": "Transition Position Lead B",
        "gradient_unit": "Gradient Transition Lead B (V/V)"
    },
    "parameter_tunnel_coupling": {
        "entity_unit": "Tunnel Coupling (\mu V)",
        "name": "Inter-Dot Tunnel Coupling AB",
        "gradient_unit": "Gradient Tunnel Coupling (\mu V / V)"
    },
    "current_signal": {
        "entity_unit": "Signal (a.u.)",
        "name": "Current SDot Signal",
        "gradient_unit": "Gradient Current SDot Signal (a.u.)"
    },
    "optimal_signal": {
        "entity_unit": "Signal (a.u.)",
        "name": "Optimal SDot Signal",
        "gradient_unit": "Gradient Optimal SDot Signal (a.u.)"
    },
    "position_SDB2": {
        "entity_unit": "Voltage (V)",
        "name": "Voltage on SDB2",
        "gradient_unit": "Gradient Voltage on SDB2 (V/V)"
    },
    "position_SDB1": {
        "entity_unit": "Voltage (V)",
        "name": "Voltage on SDB1",
        "gradient_unit": "Gradient Voltage on SDB1 (V/V)"
    },
    "parameter_time_load": {
        "entity_unit": "Time (ns)",
        "name": "Singlet Reload Time",
        "gradient_unit": "Gradient Singlet Reload Time(ns/V)"
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


class History:
    def __init__(self, directory_or_file: Optional[str]):
        self._data_frame = pd.DataFrame()
        self._gate_names = None
        self._parameter_names = None
        self._gradient_controlled_parameters = None
        if directory_or_file is None:
            pass
        elif os.path.isdir(directory_or_file):
            self.load_directory(directory_or_file)
        elif os.path.isfile(directory_or_file):
            self.load_file(directory_or_file)
        else:
            raise RuntimeWarning("The directory of file used to instantiate the qtune.history.History object could not"
                                 "be identified as such. An empty History object is instantiated.")

    @property
    def gate_names(self) -> Set[str]:
        return self._gate_names

    @property
    def parameter_names(self) -> Set[str]:
        return self._parameter_names

    @property
    def gradient_controlled_parameter_names(self) -> Set[str]:
        return self._gradient_controlled_parameters

    @property
    def mode_options(self):
        return ["all_voltages", "all_parameters", "all_gradients", "with_duplicates", "with_grad_covariances",
                "with_par_covariances"]

    def get_parameter_values(self, parameter_name):
        return self._data_frame[parameter_name]

    def get_parameter_std(self, parameter_name):
        return self._data_frame[parameter_name + '#var'].apply(np.sqrt)

    def get_gate_values(self, gate_name):
        return self._data_frame[gate_name]

    def get_gradients(self, parameter_name: str):
        elems = [name for name in self._data_frame.columns
                 if name.startswith(parameter_name + '#') and name.endswith('#grad')]
        return self._data_frame[elems].rename(lambda name: name.split('#')[1])

    def read_autotuner_to_data_frame(self, autotuner: qtune.autotuner.Autotuner, start: int = 0,
                                     end: Optional[int] = None) -> pd.DataFrame:
        if end is None:
            end = len(autotuner.tuning_hierarchy)
        elif end == 0:
            voltages = extract_voltages_from_hierarchy(autotuner.tuning_hierarchy)
            return pd.DataFrame(dict(voltages), index=[0, ])
        relevant_hierarchy = autotuner.tuning_hierarchy[start:end]
        voltages = extract_voltages_from_hierarchy(relevant_hierarchy)
        parameters, variances = extract_parameters_from_hierarchy(relevant_hierarchy)
        gradients, grad_covariances = extract_gradients_from_hierarchy(relevant_hierarchy)

        if self._data_frame.empty:
            self._gate_names = set(voltages.index)
            self._parameter_names = set(parameters.index)
            self._gradient_controlled_parameters = list(gradients.keys())

        voltages = dict(voltages)
        parameters = dict(parameters)
        variances = {create_name_parameter_variance(par_name): variances[par_name] for par_name in variances.index}
        gradients = {create_gradient_name(par_name, gate_name): gradients[par_name][gate_name]
                     for par_name in gradients.keys()
                     for gate_name in gradients[par_name].index}
        grad_covariances = {parameter_name: extract_diagonal_from_data_frame(grad_covariances[parameter_name])
                            for parameter_name in grad_covariances.keys()}
        grad_covariances = {create_gradient_covariance_name(par_name, gate_name): grad_covariances[par_name][gate_name]
                            for par_name in grad_covariances.keys()
                            for gate_name in grad_covariances[par_name].index}
        tuner_index = {"tuner_index": autotuner.current_tuner_index}

        return pd.DataFrame({**voltages, **parameters, **variances, **gradients, **grad_covariances,
                             **tuner_index}, index=[0, ], dtype=float)

    def append_autotuner(self, autotuner: qtune.autotuner.Autotuner):
        if self._data_frame.empty:
            self._data_frame = self.read_autotuner_to_data_frame(autotuner)
            return

        voltages = extract_voltages_from_hierarchy(autotuner.tuning_hierarchy).sort_index()
        evaluated_tuner_index = autotuner.current_tuner_index
        if autotuner.voltages_to_set is not None or autotuner.current_tuner_status:
            evaluated_tuner_index += 1
        new_information = self.read_autotuner_to_data_frame(autotuner, end=evaluated_tuner_index)
        if voltages.equals(self._data_frame[sorted(self.gate_names)].iloc[-1]):
            # stay in the row
            self._data_frame.loc[self._data_frame.index[-1], new_information.columns] = new_information.iloc[0]
        else:
            self._data_frame = self._data_frame.append(new_information, ignore_index=True)

    def load_directory(self, path):
        directory_content = sorted(os.listdir(path))
        for file in directory_content:
            self.load_file(path=os.path.join(path, file))

    def load_file(self, path):
        hdf5_handle = h5py.File(path, mode="r")
        loaded_data = qtune.storage.from_hdf5(hdf5_handle, reserved={"experiment": None})
        autotuner = loaded_data["autotuner"]
        self.append_autotuner(autotuner=autotuner)

    def plot_tuning(self, voltage_indices=None, parameter_names=None, gradient_names=None, mode=""):
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
        if "with_grad_covariances" in mode:
            with_grad_covariances = True
        else:
            with_grad_covariances = False
        if "with_par_covariances" in mode:
            with_par_covariances = True
        else:
            with_par_covariances = False

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
                                                          par_covariance_list=self.par_covariance_list,
                                                          parameter_names=parameter_names,
                                                          eliminate_duplicates=eliminate_duplicates,
                                                          with_covariances=with_par_covariances)

        if gradient_names is None:
            gradient_fig = None
            gradient_ax = None
        else:
            gradient_fig, gradient_ax = plot_gradients(gradient_list=self.gradient_list,
                                                       covariance_list=self.covariance_list,
                                                       parameter_names=gradient_names,
                                                       eliminate_duplicates=eliminate_duplicates,
                                                       with_covariances=with_grad_covariances)

        return [voltage_fig, parameter_fig, gradient_fig], [voltage_ax, parameter_ax, gradient_ax]


def load_data(file_path):
    """
    writes lists with all relavant Data.
    :return: Voltages, Parameters, Gradients, Covariances
    """
    directory_content = os.listdir(file_path)

    tuner_list = []
    index_list = []
    for file in directory_content:
        hdf5_handle = h5py.File(os.path.join(file_path, file), mode="r")
        loaded_data = qtune.storage.from_hdf5(hdf5_handle, reserved={"experiment": None})
        autotuner = loaded_data["autotuner"]
        assert (isinstance(autotuner, qtune.autotuner.Autotuner))

        tuner_list.append(autotuner._tuning_hierarchy)
        index_list.append(autotuner._current_tuner_index)
    return tuner_list, index_list


def create_name_parameter_variance(parameter_name: str) -> str:
    return parameter_name + "#var"


def create_gradient_name(parameter_name: str, gate_name: str) -> str:
    return parameter_name + "#" + gate_name + "#grad"


def create_gradient_covariance_name(parameter_name: str, gate_name: str) -> str:
    return parameter_name + "#" + gate_name + "#cov"


def plot_voltages(voltage_list, voltage_indices: Tuple[str], eliminate_duplicates: bool = True):
    if eliminate_duplicates:
        voltage_list = [v for i, v in enumerate(voltage_list) if i == 0 or not v.equals(voltage_list[i - 1])]

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


def plot_parameters(parameter_list, par_covariance_list, parameter_names: Tuple[str], eliminate_duplicates: bool,
                    with_covariances: bool):
    if eliminate_duplicates:
        par_covariance_list = [par_covariance_list[i] for i, v in enumerate(parameter_list) if
                               i == 0 or not v.equals(parameter_list[i - 1])]
        parameter_list = [v for i, v in enumerate(parameter_list) if i == 0 or not v.equals(parameter_list[i - 1])]
    parameter_dataframe = pd.concat(parameter_list, axis=1).T

    parameter_fig, parameter_ax = plt.subplots(nrows=len(parameter_names))
    if len(parameter_names) == 1:
        parameter_ax = [parameter_ax, ]
    for i, parameter in enumerate(parameter_names):
        if with_covariances:
            cov_data_frame = pd.concat(par_covariance_list, axis=1).fillna(0.)
            parameter_ax[i].errorbar(x=np.arange(parameter_dataframe.shape[0]), y=parameter_dataframe[parameter],
                                     yerr=cov_data_frame.applymap(np.sqrt).T[parameter])
        else:
            parameter_ax[i].plot(parameter_dataframe[parameter])
        parameter_ax[i].set_ylabel(parameter_information[parameter]["entity_unit"])
        parameter_ax[i].set_title(parameter_information[parameter]["name"])
    parameter_ax[len(parameter_names) - 1].set_xlabel("Measurement Number")
    return parameter_fig, parameter_ax


def plot_gradients(gradient_list, covariance_list, parameter_names: Tuple[str], eliminate_duplicates: bool,
                   with_covariances: bool):
    if eliminate_duplicates:
        covariance_list = [covariance_list[i] for i, v in enumerate(gradient_list) if
                           i == 0 or not v.equals(gradient_list[i - 1])]
        gradient_list = [v for i, v in enumerate(gradient_list) if i == 0 or not v.equals(gradient_list[i - 1])]

    grad_fig, grad_ax = plt.subplots(nrows=len(parameter_names))
    for i, parameter in enumerate(parameter_names):
        grad_dataframe = pd.concat([pd.Series(gradients[parameter]) for gradients in gradient_list], axis=1,
                                   ignore_index=True)
        # drop not existing columns
        grad_dataframe = grad_dataframe.T.dropna().T
        if with_covariances:
            cov_series_list = [extract_diagonal_from_data_frame(covariances[parameter]) for covariances in
                               covariance_list]
            cov_data_frame = pd.concat(cov_series_list, axis=1).T.dropna().T
            for gate in grad_dataframe.index:
                grad_ax[i].errorbar(x=np.arange(grad_dataframe.shape[1]), y=grad_dataframe.T[gate],
                                    yerr=cov_data_frame.applymap(np.sqrt).T[gate])
        else:
            grad_ax[i].plot(grad_dataframe.T)
        grad_ax[i].set_ylabel(parameter_information[parameter]["gradient_unit"])
        grad_ax[i].legend(gradient_list[0].index)
    grad_ax[0].set_title("Response Matrix")
    grad_ax[len(parameter_names) - 1].set_xlabel("Measurement Number")
    return grad_fig, grad_ax


def extract_voltages_from_hierarchy(tuning_hierarchy) -> pd.Series:
    return tuning_hierarchy[0].last_voltages


def extract_parameters_from_hierarchy(tuning_hierarchy) -> (pd.Series, pd.Series):
    parameters = pd.Series()
    variances = pd.Series()
    for par_tuner in tuning_hierarchy:
        parameter, variance = par_tuner.last_parameters_and_variances
        if isinstance(par_tuner, qtune.parameter_tuner.SubsetTuner):
            relevant_parameters = par_tuner.solver.target.desired.index[
                ~par_tuner.solver.target.desired.apply(np.isnan)]
            parameters = parameters.append(parameter[relevant_parameters])
            variances = variances.append(variance[relevant_parameters])
        else:
            parameters = parameters.append(parameter)
            variances = variances.append(variance)
    return parameters, variances


def extract_gradients_from_hierarchy(tuning_hierarchy) -> (Dict[str, pd.Series], Dict[str, pd.DataFrame]):
    gradients = dict()
    covariances = dict()
    for tuner in tuning_hierarchy:
        if isinstance(tuner.solver, qtune.solver.NewtonSolver):
            for i, grad_est in enumerate(tuner.solver.gradient_estimators):
                gradients[tuner.solver.target.index[i]] = grad_est.estimate()
                covariances[tuner.solver.target.index[i]] = grad_est.covariance()
    return gradients, covariances


def extract_diagonal_from_data_frame(df: pd.DataFrame) -> pd.Series:
    if not set(df.index) == set(df.columns):
        raise ValueError("The index must match the columns to extract diagonal elements!")
    return pd.Series(data=[df[i][i] for i in df.index], index=df.index)
