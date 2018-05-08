import os
import h5py
import pandas as pd
import qtune.storage
import qtune.autotuner

class Reader:
    def __init__(self, path):
        self._file_path = path
        self.tuner_list, self.index_list = self.load_data()

        voltage_list = []
        paramater_list = []
        for tuning_hierarchy in self.tuner_list:
            voltage_list.append(extract_voltages_from_hierarchy(tuning_hierarchy))
            paramater_list.append(extract_parameters_from_hierarchy(tuning_hierarchy))


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
            autotuner = qtune.storage.from_hdf5(hdf5_handle, reserved={"experiment": None})
            assert(isinstance(autotuner, qtune.autotuner.Autotuner))

            tuner_list.append(autotuner._tuning_hierarchy)
            index_list.append(autotuner._current_tuner_index)
        return tuner_list, index_list


def extract_voltages_from_hierarchy(tuning_hierarchy):
    voltages = pd.Series()
    for par_tuner in tuning_hierarchy:
        for gate in par_tuner._last_voltage.index():
            if gate not in voltages.index():
                voltages[gate] = par_tuner._last_voltage[gate]
    return voltages


def extract_parameters_from_hierarchy(tuning_hierarchy):
    parameters = pd.Series()
    parameters.append(par_tuner._last_parameter_values for par_tuner in tuning_hierarchy)

#def extract_gradients_from_hierarchy
