import os
import h5py
import pandas as pd
import qtune.storage
import qtune.autotuner

class Reader:
    def __init__(self, path):
        self._file_path = path
        self.tuner_list, self.index_list = self.load_data()

        self.voltage_list = []
        self.parameter_list = []
        for tuning_hierarchy in self.tuner_list:
            self.voltage_list.append(extract_voltages_from_hierarchy(tuning_hierarchy))
            self.parameter_list.append(extract_parameters_from_hierarchy(tuning_hierarchy))


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
        parameters.append(pd.Series(par_tuner._last_parameter_values))
    return parameters

#def extract_gradients_from_hierarchy
