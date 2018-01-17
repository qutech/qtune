import pandas as pd
import numpy as np
import h5py

known_evaluators = pd.Series([["parameter_tunnel_coupling"], ["parameter_time_rise", "parameter_time_fall"]],
                             ["evaluator_SMInterDotTCByLineScan", "evaluator_SMLeadTunnelTimeByLeadScan"])
known_evaluators = known_evaluators.sort_index()


class Analyzer:
    def __init__(self, filename: str=None):
        self.gate_names = None
        self.tunable_gate_names = None
        self.parameter_names = None
        if filename is not None:
            self.load_file(filename)
        else:
            self.root_group = None

    def load_file(self, filename: str):
        self.root_group = h5py.File(filename)
        gate_names = self.root_group["gate_names"][:]
        self.gate_names = gate_names
        self.tunable_gate_names = self.root_group["tunable_gate_names"]

    def load_parameter_names(self, tune_run_number=1):
        tune_sequence_group = self.root_group["tunerun_" + str(tune_run_number) + "/tune_sequence"]
        self.parameter_names = tune_sequence_group["parameter_names"][:]

    def load_cd_gradient(self, gradient_number=1, tune_run_number=0):
        tune_run_group = self.load_tune_run_group(tune_run_number)
        if not tune_run_group.__contains__("charge_diagram_" + str(gradient_number)):
            if tune_run_number > 0:
                print("There is no gradient saved for the charge diagram number " + str(
                    gradient_number) + " in tune run number" + str(
                    tune_run_number) + ". We try to find it in a previous run, since it might have been copied.")
                self.load_cd_gradient(gradient_number, tune_run_number-1)
            else:
                print("There is no gradient saved for the charge diagram number " + str(
                    gradient_number) + " in tune run number" + str(
                    tune_run_number) + ".")

        cd_group = tune_run_group["charge_diagram_" + str(gradient_number)]
        gradient, covariance, noise = load_gradient_from_group(cd_group)
        return gradient, covariance, noise

    def load_cd_gradient_as_pd(self, gradient_number=1, tune_run_number=0):
        gradient, covariance, noise = self.load_cd_gradient(gradient_number, tune_run_number)
        gradient = pd.DataFrame(gradient, ["PosA", "PosB"], ["BA", "BB"])
        return gradient

    def load_gradient_setup(self, gradient_number=1, tune_run_number=0):
        cd_group = self.load_gradient_group(gradient_number=gradient_number, tune_run_number=tune_run_number)
        gradient, covariance, noise = load_gradient_from_group(cd_group)
        return gradient, covariance, noise

    def load_gradient_tunerun(self, step_number: int=1, tune_run_number=1):
        tune_run_group = self.load_tune_run_group(tune_run_number)
        tune_sequence_group = tune_run_group["tune_sequence"]
        if tune_sequence_group.__contains__("step_" + str(step_number)):
            step_group = tune_sequence_group["step_" + str(step_number)]
        else:
            print("There is no group step " + str(step_number) + " in tune run number " + str(tune_run_number))
            return
        gradient, heuristic_covariance, heuristic_noise = load_gradient_from_group(step_group)
        return gradient, heuristic_covariance, heuristic_noise

    def load_gradient_pd(self, gradient_number=1, tune_run_number=0):
        gradient, covariance, noise = self.load_gradient_tunerun(gradient_number, tune_run_number)
        if self.parameter_names is None:
            print("You need to load the parameter names first!")
            return
        gradient = pd.DataFrame(gradient, self.parameter_names, self.tunable_gate_names)
        return gradient

    def load_gradient_sequence_pd(self, tune_run_number=1, start: int=0, end: int=None):
        self.load_parameter_names(tune_run_number=tune_run_number)
        tune_run_group = self.load_tune_run_group(tune_run_number)
        if end is None:
            end = count_steps_in_sequence(tune_run_group["tune_sequence"])
        gradient_sequence_pd = pd.DataFrame(None, index=self.parameter_names, columns=self.tunable_gate_names)
        for parameter in self.parameter_names:
            for gate in self.tunable_gate_names:
                gradient_sequence_pd[gate][parameter] = np.zeros((end + 1 - start, ))

        for counter in range(start, end):
            gradient_pd = self.load_gradient_pd(gradient_number=counter, tune_run_number=tune_run_number)
            for parameter in self.parameter_names:
                for gate in self.tunable_gate_names:
                    gradient_sequence_pd[gate][parameter][counter] = gradient_pd[gate][parameter]
        return gradient_sequence_pd

    def load_gate_voltages_and_parameters(self, data_group: h5py.Group) -> (pd.Series, pd.Series):
        if data_group.__contains__("gate_voltages"):
            gate_voltage_data_set = data_group["gate_voltages"]
            gate_voltages = gate_voltage_data_set[:]
            gate_voltages_pd = pd.Series(gate_voltages, self.gate_names)
        else:
            gate_voltages_pd = None

        parameter_name_list = []
        parameter_value_list = []
        for key in data_group.keys():
            if "evaluator_" in key:
                evaluator_data_set = data_group[key]
                for parameter_name in evaluator_data_set.attrs:
                    parameter_name_list += [parameter_name]
                    parameter_value_list += [evaluator_data_set.attrs[parameter_name]]
        parameters = pd.Series(parameter_value_list, parameter_name_list)
        parameters_pd = parameters.sort_index()
        return gate_voltages_pd, parameters_pd

    def load_gate_voltages_and_parameters_sequence(self, tune_run_number: int = 0, start: int = 0, end: int = None) -> (
            pd.Series, pd.Series):
        if start < 0:
            print("First step has number 0. There are no negative indices!")
            return
        tune_run_group = self.load_tune_run_group(tune_run_number)
        tune_sequence_group = tune_run_group["tune_sequence"]
        if end is None:
            end = count_steps_in_sequence(tune_sequence_group)
        gate_voltages_sequence_pd = pd.Series()
        for gate in self.gate_names:
            gate_voltages_sequence_pd[gate] = np.zeros((end - start, ))
        parameters_sequence_pd = pd.Series()
        run_parameters = tune_sequence_group["parameter_names"][:]
        for parameter in run_parameters:
            parameters_sequence_pd[parameter] = np.zeros((end - start, ))
        for counter in range(start, end):
            gate_voltages_pd, parameters_pd = self.load_gate_voltages_and_parameters(
                tune_sequence_group["step_" + str(counter)])
            for gate in self.gate_names:
                gate_voltages_sequence_pd[gate][counter] = gate_voltages_pd[gate]
            for parameter in self.parameter_names:
                parameters_sequence_pd[parameter][counter] = parameters_pd[parameter]
        return gate_voltages_sequence_pd, parameters_sequence_pd

    def load_desired_values(self, tune_run_number):
        tune_run_group = self.load_tune_run_group(tune_run_number)
        desired_values_pd = pd.Series(tune_run_group["desired_values"], self.parameter_names)
        return desired_values_pd

    def load_kalman_tune_run(self, tune_run_number=1, start: int=0, end: int=None):
        desired_values_pd = self.load_desired_values(tune_run_number)
        gate_voltages_sequence_pd, parameters_sequence_pd = self.load_gate_voltages_and_parameters_sequence(
            tune_run_number, start, end)
        gradient_sequence_pd = self.load_gradient_sequence_pd(tune_run_number=tune_run_number, start=start, end=end)
        return desired_values_pd, gate_voltages_sequence_pd, parameters_sequence_pd, gradient_sequence_pd

    def plot_kalman_tune_run(self, tune_run_number=1, start:int=0, end: int=None):
        desired_values_pd, gate_voltages_sequence_pd, parameters_sequence_pd, gradient_sequence_pd = \
            self.load_kalman_tune_run(tune_run_number, start, end)
        return

    def load_single_values_gradient_calculation(self, gradient_number: int = 1, tune_run_number: int = 0) -> (
            pd.Series, int, float):
        gradient_group = self.load_gradient_group(gradient_number=gradient_number, tune_run_number=tune_run_number)
        n_repetitions = gradient_group.attrs("n_repetitions")
        delta_u = gradient_group.attrs("delta_u")
        single_values_pd = pd.DataFrame()
        temp_parameter = pd.Series()
        for gate in self.tunable_gate_names:
            for parameter in self.parameter_names:
                single_values_pd[parameter, gate] = np.zeros((n_repetitions, 2))
            for i in range(n_repetitions):
                positive_run_group = gradient_group["positive_detune_run_" + gate + "_" + str(i)]
                for element in positive_run_group.keys():
                    if "evaluator_" in element:
                        for parameter_name in positive_run_group[element].attrs:
                            temp_parameter[parameter_name] = positive_run_group[element].attrs(parameter_name)
                temp_parameter = temp_parameter.sort_index()
                for parameter in self.parameter_names:
                    single_values_pd[parameter, gate][i, 0] = temp_parameter[parameter]

                negative_run_group = gradient_group["negative_detune_run_" + gate + "_" + str(i)]
                for element in negative_run_group.keys():
                    if "evaluator_" in element:
                        for parameter_name in negative_run_group[element].attrs:
                            temp_parameter[parameter_name] = negative_run_group[element].attrs(parameter_name)
                temp_parameter = temp_parameter.sort_index()
                for parameter in self.parameter_names:
                    single_values_pd[parameter, gate][i, 1] = temp_parameter[parameter]
        return single_values_pd, n_repetitions, delta_u

    def load_tune_run_group(self, tune_run_number) -> h5py.Group:
        if self.root_group.__contains__("tunerun_" + str(tune_run_number)):
            tune_run_group = self.root_group["tunerun_" + str(tune_run_number)]
        else:
            print("There is no tunerun number " + str(tune_run_number) + "in this file.")
            raise KeyError("Group does not exist.")
        return tune_run_group

    def load_gradient_group(self, gradient_number: int, tune_run_number: int) -> h5py.Group:
        tune_run_group = self.load_tune_run_group(tune_run_number)
        if not tune_run_group.__contains__("gradient_setup_" + str(gradient_number)):
            if tune_run_number > 0:
                print("There is no gradient saved for the gradient setup number " + str(
                    gradient_number) + " in tune run number" + str(
                    tune_run_number) + ". We  will try to find it in a previous run, since it might have been copied.")
                return self.load_gradient_group(gradient_number, tune_run_number - 1)
            else:
                print("There is no gradient saved for the gradient setup number " + str(
                    gradient_number) + " in tune run number" + str(
                    tune_run_number) + ".")
                raise KeyError("Group does not exist.")
        gradient_group = tune_run_group["gradient_setup_" + str(gradient_number)]
        return gradient_group

    def logout(self):
        self.root_group.close()


def count_steps_in_sequence(sequence_group: h5py.Group):
    counter = 0
    for key in sequence_group.keys():
        if "step_" in key:
            counter += 1
    return counter


def load_gradient_from_group(data_group: h5py.Group):
    gradient = data_group['gradient'][:]
    heuristic_covariance = data_group['heuristic_covariance'][:]
    if "heuristic_noise" in data_group:
        heuristic_noise = data_group['heuristic_noise'][:]
    else:
        heuristic_noise = None
    return gradient, heuristic_covariance, heuristic_noise


def print_group_content(data_group: h5py.Group):
    print("Subgroups:")
    for element in data_group:
        if isinstance(data_group[element], h5py.Group):
            print(element)
    print("Datasets:")
    for element in data_group:
        if isinstance(data_group[element], h5py.Dataset):
            print(element)
    print("Attributes:")
    for element in data_group.attrs:
        print(element)
        print(data_group.attrs[element])
        print("\n")


def load_single_evaluation_from_group(data_group: h5py.Group, evaluator_name: str=None, evaluator_number: int =-1):
    if evaluator_name is None:
        if evaluator_number == -1:
            print("Please choose an evaluator. This can be done by name or from the Series of known evaluators.")
            raise ValueError
        evaluator_name = known_evaluators.index.tolist()[evaluator_number]

    evaluator_data_set = data_group[evaluator_name]
    raw_data = evaluator_data_set[:]
    parameters_pd = pd.Series()
    information_pd = pd.Series()
    for attribute in evaluator_data_set.attrs.keys():
        if "parameter_" in attribute:
            parameters_pd[attribute] = evaluator_data_set.attrs(attribute)
        else:
            information_pd[attribute] = evaluator_data_set.attrs(attribute)
    return raw_data, parameters_pd, information_pd

