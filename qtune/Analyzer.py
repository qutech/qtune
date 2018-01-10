import pandas as pd
import numpy as np
import h5py


class Analyzer:
    def __init__(self, filename: str=None):
        self.gate_names = None
        self.parameter_names = None
        if filename is not None:
            self.load_file(filename)
        else:
            self.root_group = None

    def load_file(self, filename: str):
        self.root_group = h5py.File(filename)
        gate_names = self.root_group["Gate_Names"]
        nan_array = np.empty(gate_names.shape)
        nan_array[:] = np.nan
        self.gate_names = pd.Series(nan_array, gate_names)

    def load_parameter_names(self, tune_run_number=1):
        tune_sequence_group = self.root_group["Tunerun_" + str(tune_run_number) + "/Tune_Sequence"]
        self.parameter_names = tune_sequence_group["Parameter_Names"][:]

    def load_cd_gradient(self, gradient_number=1, tune_run_number=0):
        tune_run_group = self.root_group["Tunerun_" + str(tune_run_number)]
        if not tune_run_group.__contains__("Charge_Diagram_" + str(gradient_number)):
            if tune_run_number > 0:
                print("There is no gradient saved for the charge diagram number " + str(
                    gradient_number) + " in tune run number" + str(
                    tune_run_number) + ". We try to find it in a previous run, since it might have been copied.")
                self.load_cd_gradient(gradient_number, tune_run_number-1)
            else:
                print("There is no gradient saved for the charge diagram number " + str(
                    gradient_number) + " in tune run number" + str(
                    tune_run_number) + ".")

        cd_group = tune_run_group["Charge_Diagram_" + str(gradient_number)]
        gradient, covariance, noise = load_gradient_from_group(cd_group)
        return gradient, covariance, noise

    def load_cd_gradient_as_pd(self, gradient_number=1, tune_run_number=0):
        gradient, covariance, noise = self.load_cd_gradient(gradient_number, tune_run_number)
        gradient = pd.DataFrame(gradient, ["PosA", "PosB"], ["BA", "BB"])
        return gradient

    def load_gradient(self, gradient_number=1, tune_run_number=0):
        tune_run_group = self.root_group["Tunerun_" + str(tune_run_number)]
        if not tune_run_group.__contains__("Gradient_Setup_" + str(gradient_number)):
            if tune_run_number > 0:
                print("There is no gradient saved for the gradient setup number " + str(
                    gradient_number) + " in tune run number" + str(
                    tune_run_number) + ". We try to find it in a previous run, since it might have been copied.")
                self.load_cd_gradient(gradient_number, tune_run_number - 1)
            else:
                print("There is no gradient saved for the gradient setup number " + str(
                    gradient_number) + " in tune run number" + str(
                    tune_run_number) + ".")

        cd_group = tune_run_group["Gradient_setup_" + str(gradient_number)]
        gradient, covariance, noise = load_gradient_from_group(cd_group)
        return gradient, covariance, noise

    def load_gradient_as_pd(self, gradient_number=1, tune_run_number=0):
        gradient, covariance, noise = self.load_gradient(gradient_number, tune_run_number)
        if self.parameter_names is None:
            print("You need to load the parameter names first!")
            return
        gradient = pd.DataFrame(gradient, self.parameter_names, self.gate_names)
        return gradient


def load_gradient_from_group(data_group: h5py.Group):
    gradient = data_group['gradient'][:]
    heuristic_covariance = data_group['heuristic_covariance'][:]
    heuristic_noise = data_group['heuristic_noise'][:]
    return gradient, heuristic_covariance, heuristic_noise


def print_group_content(data_group: h5py.Group):
    print("Subgroups:")
    for element in data_group:
        if isinstance(element, h5py.Group):
            print(element)
    print("Datasets:")
    for element in data_group:
        if isinstance(element, h5py.Dataset):
            print(element)
    print("Attributes:")
    for element in data_group.attrs:
        print(element)
        print(data_group.attrs[element])
        print("\n")

