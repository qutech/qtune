import os
import h5py
import operator
import pandas as pd
import qtune.storage
import qtune.autotuner
import qtune.solver
import qtune.gradient


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
            with tuner.solver._gradient_estimators as grad_estimators:
                for i in range(len(grad_estimators)):
                    if isinstance(grad_estimators[i], qtune.gradient.KalmanGradientEstimator):
                        gradients[tuner.solver.target.index[i]] = grad_estimators[i]._kalman_gradient.grad
                        covariances[tuner.solver.target.index[i]] = grad_estimators[i]._kalman_gradient.cov
                    elif isinstance(grad_estimators[i], qtune.gradient.FiniteDifferencesGradientEstimator):
                        gradients[tuner.solver.target.index[i]] = grad_estimators[i]._current_estimate
                        covariances[tuner.solver.target.index[i]] = grad_estimators[i]._covariance
    return gradients, covariances
