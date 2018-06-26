import os
import operator
import re
from typing import Optional, Set, Dict

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import qtune.storage
import qtune.autotuner
import qtune.solver
import qtune.gradient
import qtune.parameter_tuner


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
    _parameter_variance_name = '{parameter_name}#var'
    _gradient_name = '{parameter_name}#{gate_name}'
    _gradient_covariance_name = '{parameter_name}#{gate_name_1}#{gate_name_2}#cov'
    _gradient_variance_name = '{parameter_name}#{gate_name}#{gate_name}#cov'

    def __init__(self, directory_or_file: Optional[str]):
        self._data_frame = pd.DataFrame()
        self._gate_names = set()
        self._parameter_names = set()
        self._gradient_controlled_parameters = dict()
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
    def gradient_controlled_parameter_names(self) -> Dict[str, Set[str]]:
        return self._gradient_controlled_parameters

    @property
    def mode_options(self):
        return ["all_voltages", "all_parameters", "all_gradients", "with_grad_covariances", "with_par_variances"]

    def get_parameter_values(self, parameter_name) -> pd.Series:
        return self._data_frame[parameter_name]

    def get_parameter_std(self, parameter_name) -> pd.Series:
        return self._data_frame[self._parameter_variance_name.format(parameter_name=parameter_name)].apply(np.sqrt)

    def get_gate_values(self, gate_name) -> pd.Series:
        return self._data_frame[gate_name]

    def get_gradients(self, parameter_name: str) -> pd.DataFrame:
        regex = re.compile(self._gradient_name.format(parameter_name=parameter_name,
                                                      gate_name='([\w\s\d]+)'))
        return self._data_frame.filter(regex=regex).rename(lambda name: regex.findall(name)[0], axis='columns')

    def get_gradient_covariances(self, parameter_name) -> pd.DataFrame:
        regex = re.compile('%s#([\w\s\d]+)#([\w\s\d]+)#cov' % parameter_name)
        df = self._data_frame.filter(regex=regex)

        columns = sorted(df.columns, key=regex.findall)

        return pd.DataFrame(df[columns], columns=pd.MultiIndex.from_tuples(map(regex.findall, df.columns)))

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
            self._gradient_controlled_parameters = {parameter_name: set(gradient.index)
                                                    for parameter_name, gradient in gradients.items()}

        voltages = dict(voltages)
        parameters = dict(parameters)
        variances = {self._parameter_variance_name.format(parameter_name=par_name): variance
                     for par_name, variance in variances.items()}
        gradients = {self._gradient_name.format(parameter_name=par_name, gate_name=gate_name): grad_entry
                     for par_name, gradient in gradients.items()
                     for gate_name, grad_entry in gradient.items()}
        grad_covariances = {k: v
                            for parameter_name, gradient_covariance_matrix in grad_covariances
                            for k, v in self._unravel_gradient_covariance_matrix(parameter_name,
                                                                                 gradient_covariance_matrix).items()}
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

    def plot_tuning(self, voltage_indices=None, parameter_names=None, gradient_parameter_names=None, mode=""):
        if "all_voltages" in mode:
            voltage_indices = sorted(self.gate_names)
        if "all_parameters" in mode:
            parameter_names = sorted(self.parameter_names)
        if "all_gradients" in mode:
            gradient_parameter_names = sorted(self.gradient_controlled_parameter_names)
        if "with_grad_covariances" in mode:
            with_grad_covariances = True
        else:
            with_grad_covariances = False
        if "with_par_variances" in mode:
            with_par_variances = True
        else:
            with_par_variances = False

        if voltage_indices is None:
            voltage_fig = None
            voltage_ax = None
        else:
            voltage_fig, voltage_ax = plot_voltages(self._data_frame[voltage_indices])

        if parameter_names is None:
            parameter_fig = None
            parameter_ax = None
        else:
            if with_par_variances:
                var_names = [create_name_parameter_variance(par_name) for par_name in parameter_names]
            else:
                var_names = []
            parameter_fig, parameter_ax = plot_parameters(self._data_frame[parameter_names],
                                                          self._data_frame[var_names])

        if gradient_parameter_names is None:
            gradient_fig = None
            gradient_ax = None
        else:
            gradient_indices = [create_gradient_name(par_name, gate_name)
                                for par_name in gradient_parameter_names
                                for gate_name in self.gradient_controlled_parameter_names[par_name]]
            if with_grad_covariances:
                covariance_indices = [create_gradient_covariance_name(name.split('#')[0], name.split('#')[1])
                                      for name in gradient_indices
                                      if with_grad_covariances]
            else:
                covariance_indices = []
            gradient_fig, gradient_ax = plot_gradients(self._data_frame[gradient_indices],
                                                       self._data_frame[covariance_indices])

        return [voltage_fig, parameter_fig, gradient_fig], [voltage_ax, parameter_ax, gradient_ax]

    @classmethod
    def _unravel_gradient_covariance_matrix(cls, parameter_name, covariance_matrix: pd.DataFrame):
        return {
            cls._gradient_covariance_name.format(parameter_name=parameter_name,
                                                 gate_name_1=gate_1,
                                                 gate_name_2=gate_2): cov_entry
            for gate_1, cov_column in covariance_matrix.items()
            for gate_2, cov_entry in cov_column.items()
        }


def create_name_parameter_variance(parameter_name: str) -> str:
    return parameter_name + "#var"


def create_gradient_name(parameter_name: str, gate_name: str) -> str:
    return parameter_name + "#" + gate_name + "#grad"


def create_gradient_covariance_name(parameter_name: str, gate_name: str) -> str:
    return parameter_name + "#" + gate_name + "#cov"


def plot_voltages(voltage_data_frame: pd.DataFrame):
    voltage_fig, voltage_ax = plt.subplots()
    voltage_ax.plot(voltage_data_frame)
    voltage_ax.legend(voltage_data_frame.columns)
    voltage_ax.set_title("Gate Voltages")
    voltage_ax.set_xlabel("Measurement Number")
    voltage_ax.set_ylabel("Voltage (V)")
    return voltage_fig, voltage_ax


def plot_parameters(parameter_data_frame: pd.DataFrame, parameter_variance_data_frame: Optional[pd.DataFrame]):
    if parameter_variance_data_frame is None:
        parameter_variance_data_frame = pd.DataFrame()
    parameter_fig, parameter_ax = plt.subplots(nrows=len(parameter_data_frame.columns))
    for i, parameter in enumerate(parameter_data_frame.columns):
        if create_name_parameter_variance(parameter) in parameter_variance_data_frame.columns:
            parameter_ax[i].errorbar(x=parameter_data_frame.index, y=parameter_data_frame[parameter],
                                     yerr=parameter_variance_data_frame[
                                         create_name_parameter_variance(parameter)].map(np.sqrt))
        else:
            parameter_ax[i].plot(parameter_data_frame[parameter])
        parameter_ax[i].set_ylabel(parameter_information[parameter]["entity_unit"])
        parameter_ax[i].set_title(parameter_information[parameter]["name"])
    parameter_ax[len(parameter_data_frame.columns) - 1].set_xlabel("Measurement Number")
    return parameter_fig, parameter_ax


def plot_gradients(gradient_data_frame: pd.DataFrame, diagonal_covariance_data_frame: Optional[pd.DataFrame]):
    if diagonal_covariance_data_frame is None:
        diagonal_covariance_data_frame = pd.DataFrame()
    parameter_names = sorted({name.split("#")[0] for name in gradient_data_frame.columns})
    parameter_gate_names = {par_name: [] for par_name in parameter_names}
    for name in gradient_data_frame.columns:
        parameter_gate_names[name.split('#')[0]].append(name.split('#')[1])
    grad_fig, grad_ax = plt.subplots(nrows=len(parameter_names))
    for i, parameter in enumerate(parameter_names):
        for gate_name in parameter_gate_names[parameter]:
            if create_gradient_covariance_name(parameter, gate_name) in diagonal_covariance_data_frame.columns:
                grad_ax[i].errorbar(x=gradient_data_frame.index,
                                    y=gradient_data_frame[create_gradient_name(parameter, gate_name)],
                                    yerr=diagonal_covariance_data_frame[
                                        create_gradient_covariance_name(parameter, gate_name)].map(np.sqrt))
            else:
                grad_ax[i].plot(gradient_data_frame[create_gradient_name(parameter, gate_name)])
        grad_ax[i].set_ylabel(parameter_information[parameter]["gradient_unit"])
        grad_ax[i].legend(parameter_gate_names[parameter])
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
