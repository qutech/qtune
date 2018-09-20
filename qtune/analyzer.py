import pandas as pd
import numpy as np
import h5py
from scipy import optimize
from qtune.util import find_lead_transition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import qtune.evaluator
from qtune.kalman_gradient import KalmanGradient

"""
known_evaluators = pd.Series([["parameter_tunnel_coupling"], ["parameter_time_rise", "parameter_time_fall"]],
                             ["evaluator_SMInterDotTCByLineScan", "evaluator_SMLeadTunnelTimeByLeadScan"])
known_evaluators = known_evaluators.sort_index()
"""

class Analyzer:
    def __init__(self, filename: str=None):
        self.gate_names = None
        self.tunable_gate_names = None
        self._parameter_names = None
        self.evaluator_names = None
        if filename is not None:
            self.load_file(filename)
        else:
            self.root_group = None

    @property
    def parameter_names(self):
        if self._parameter_names is None:
            print("You need to load the Parameter names!")
        return self._parameter_names

    @parameter_names.setter
    def parameter_names(self, parameter_names):
        self._parameter_names = parameter_names

    def load_file(self, filename: str):
        self.root_group = h5py.File(filename, "r")
        gate_names = self.root_group["gate_names"][:]
        self.gate_names = gate_names
        self.tunable_gate_names = self.root_group["tunable_gate_names"][:]

    def load_parameter_names(self, tune_run_number=1):
        if self.root_group.__contains__("tunerun_" + str(tune_run_number) + "/tune_sequence"):
            parameter_containing_group = self.root_group["tunerun_" + str(tune_run_number) + "/tune_sequence"]
        elif self.root_group.__contains__("tunerun_" + str(tune_run_number) + "/gradient_setup_1"):
            parameter_containing_group = self.root_group["tunerun_" + str(tune_run_number) + "/gradient_setup_1"]
        else:
            print("parameter_names could not be loaded!")
            return
        self.parameter_names = parameter_containing_group["parameter_names"][:]

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
        gradient_group = self.load_gradient_group(gradient_number=gradient_number, tune_run_number=tune_run_number)
        gradient, covariance, noise = load_gradient_from_group(gradient_group)
        return gradient, covariance, noise

    def load_gradient_minimization_covariance(self, gradient_number=1, tune_run_number=0):
        gradient_group = self.load_gradient_group(gradient_number=gradient_number, tune_run_number=tune_run_number)
        gradient, covariance, noise = load_gradient_from_group(gradient_group)
        gradient_kalman_group = gradient_group[gradient]
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
        if self.root_group.__contains__("tunerun_" + str(tune_run_number) + "/gradient_setup_" + str(gradient_number)):
            gradient, covariance, noise = self.load_gradient_setup(gradient_number=gradient_number, tune_run_number=tune_run_number)
        elif self.root_group.__contains__("tunerun_" + str(tune_run_number) + "/tune_sequence"):
            gradient, covariance, noise = self.load_gradient_tunerun(gradient_number, tune_run_number)
        else:
            gradient = None
            print("Gradient could not be loaded!")
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
                gradient_sequence_pd[gate][parameter] = np.zeros((end - start, ))

        for counter in range(start, end):
            gradient_pd = self.load_gradient_pd(gradient_number=counter, tune_run_number=tune_run_number)
            for parameter in self.parameter_names:
                for gate in self.tunable_gate_names:
                    gradient_sequence_pd[gate][parameter][counter-start] = gradient_pd[gate][parameter]
        return gradient_sequence_pd

    def load_noise_from_gradient_pd(self, tune_run_number=1, gradient_number: int=1):
        self.load_parameter_names(tune_run_number=tune_run_number)
        tune_run_group = self.load_tune_run_group(tune_run_number)
        gradient_group = tune_run_group["gradient_setup_" + str(gradient_number)]
        noise = gradient_group["heuristic_noise"][:]
        noise_pd = pd.DataFrame(data=noise, index=self.parameter_names, columns=self.parameter_names)
        return noise_pd

    def plot_gradient_sequence(self, tune_run_number=1, figure_number=60, start=0, end=None, with_covariance: bool=True):
        gradient_sequence_pd = self.load_gradient_sequence_pd(tune_run_number=tune_run_number, start=start, end=end)
        tune_run_group = self.load_tune_run_group(tune_run_number)
        if end is None:
            end = count_steps_in_sequence(tune_run_group["tune_sequence"])
        if with_covariance:
            covariance_sequence_pd = self.load_covariance_sequence_pd(tune_run_number=tune_run_number, start=start, end=end)
        for parameter in self.parameter_names:
            plt.figure(figure_number)
            figure_number += 1
            for gate in self.tunable_gate_names:
                if with_covariance:
                    plt.errorbar(x=list(range(start, end)), y=1e-3 * gradient_sequence_pd[gate][parameter],
                                 yerr=1e-3 * np.sqrt(covariance_sequence_pd[parameter.decode("ascii") + "_" + gate.decode("ascii")][
                                     parameter.decode("ascii") + "_" + gate.decode("ascii")]))
                    plt.ylabel(gradient_plot_ylabel(parameter.decode("ascii")), fontsize=22)
                    plt.xlabel("Iteration Number", fontsize=22)
                    plt.gca().tick_params("x", labelsize=22)
                    plt.gca().tick_params("y", labelsize=22)
                    fig = plt.gcf()
                    fig.set_size_inches(9.5, 8)
                else:
                    plt.plot(gradient_sequence_pd[gate][parameter])
            plt.legend([gate_name.decode("ascii") for gate_name in self.tunable_gate_names], fontsize=22)
            plt.title(
                "Gradient elements in the line " + parameter_plot_name(parameter.decode("ascii"),
                                                                       with_unit=False),
                fontsize=24)
            plt.show()

    def load_covariance_sequence_pd(self, tune_run_number=1, start: int=0, end: int=None):
        self.load_parameter_names(tune_run_number=tune_run_number)
        tune_run_group = self.load_tune_run_group(tune_run_number)
        if end is None:
            end = count_steps_in_sequence(tune_run_group["tune_sequence"])
        covariance_index = []
        for parameter in self.parameter_names:
            for gate in self.tunable_gate_names:
                covariance_index += [parameter.decode("ascii") + "_" + gate.decode("ascii")]
        covariance_sequence_pd = pd.DataFrame(index=covariance_index, columns=covariance_index)
        for index in covariance_index:
            for index2 in covariance_index:
                covariance_sequence_pd[index][index2] = np.zeros((end - start, ))
        for step in range(start, end):
            gradient, heuristic_covariance, heuristic_noise = self.load_gradient_tunerun(step_number=step,
                                                                                    tune_run_number=tune_run_number)
            for parameter in range(len(self.parameter_names)):
                for gate in range(len(self.tunable_gate_names)):
                    for parameter2 in range(len(self.parameter_names)):
                        for gate2 in range(len(self.tunable_gate_names)):
                            index_number = gate + parameter * len(self.tunable_gate_names)
                            index_number2 = gate2 + parameter2 * len(self.tunable_gate_names)
                            covariance_sequence_pd[covariance_index[index_number2]][covariance_index[index_number]][
                                step - start] = heuristic_covariance[index_number, index_number2]
        return covariance_sequence_pd

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
                    parameter_name_list += [str.encode(parameter_name)]
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
        run_parameters = tune_sequence_group["parameter_names"][:]
        parameters_sequence_pd = pd.Series()
        for parameter in run_parameters:
            parameters_sequence_pd[parameter] = np.zeros((end - start, ))
        for counter in range(start, end):
            gate_voltages_pd, parameters_pd = self.load_gate_voltages_and_parameters(
                tune_sequence_group["step_" + str(counter)])
            for gate in self.gate_names:
                gate_voltages_sequence_pd[gate][counter - start] = gate_voltages_pd[gate]
            for parameter in self.parameter_names:
                parameters_sequence_pd[parameter][counter - start] = parameters_pd[parameter]
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

    def plot_kalman_tune_run(self, tune_run_number=1, start: int=0, end: int=None, gradient_number: int=1,
                             gradient_tune_run: int=0):
        if end is None:
            tune_run_group = self.root_group["tunerun_" + str(tune_run_number)]
            end = count_steps_in_sequence(tune_run_group["tune_sequence"])
        desired_values_pd, gate_voltages_sequence_pd, parameters_sequence_pd, gradient_sequence_pd = \
            self.load_kalman_tune_run(tune_run_number, start, end)
        noise_pd = self.load_noise_from_gradient_pd(tune_run_number=gradient_tune_run, gradient_number=gradient_number)
        number_parameter = len(self.parameter_names)
        plt.figure(1)
        for i in range(number_parameter):
            plt.subplot(number_parameter, 1, i+1)
            plt.errorbar(x=list(range(start, end)), y=parameters_sequence_pd[self.parameter_names[i]],
                         yerr=np.full((end - start,),
                                      np.sqrt(noise_pd[self.parameter_names[i]][self.parameter_names[i]])))
#            plt.plot(parameters_sequence_pd[self.parameter_names[i]], "r")
            plt.axhline(desired_values_pd[self.parameter_names[i]])
            if i == 0:
                plt.title("Parameters", fontsize=24)
            plt.ylabel(parameter_plot_name(self.parameter_names[i].decode("ascii")), fontsize=22)
            plt.gca().tick_params("x", labelsize=22)
            plt.gca().tick_params("y", labelsize=22)
        plt.xlabel("Iteration Number", fontsize=22)
        fig = plt.gcf()
        fig.set_size_inches(8.5, 8)
        plt.figure(2)
        for gate in self.gate_names:
            plt.plot(gate_voltages_sequence_pd[gate])
        plt.legend([gate.decode("ascii") for gate in self.gate_names])
        plt.figure(3)
        for gate in self.gate_names:
            plt.plot(gate_voltages_sequence_pd[gate] - gate_voltages_sequence_pd[gate][0])
        plt.legend([gate.decode("ascii") for gate in self.gate_names])
        plt.title("Change in Gate Voltages")
        plt.show()
        return

    def calculate_kalman_gradient(self, gate_voltages_sequence_pd, parameters_sequence_pd, initial_grad, initial_cov,
                                  initial_noise, number_steps, alpha, with_relative_error=False, relative_error=0.0004,
                                  residuals=None, process_noise=None):

        if process_noise is None:
            process_noise = np.zeros((8, 8))


        load_uncertainty = 0.15
        inter_dot_uncertainty = 0.6
        for i in range(4):
            process_noise[i, i] = load_uncertainty * load_uncertainty * 1e6
        for i in range(4, 8):
            process_noise[i, i] = inter_dot_uncertainty * inter_dot_uncertainty * 1e6

        kalman = KalmanGradient(self.tunable_gate_names.size, self.parameter_names.size,
                                initial_gradient=initial_grad,
                                initial_covariance_matrix=initial_cov,
                                measurement_covariance_matrix=initial_noise,
                                alpha=alpha,
                                process_noise=process_noise)
        covariance_index = []
        for parameter in self.parameter_names:
            for gate in self.tunable_gate_names:
                covariance_index += [parameter.decode("ascii") + "_" + gate.decode("ascii")]
        gradient_sequence_pd = pd.DataFrame(index=self.parameter_names, columns=self.tunable_gate_names)
        covariance_sequence_pd = pd.DataFrame(index=covariance_index, columns=covariance_index)
        for gate in self.tunable_gate_names:
            for parameter in self. parameter_names:
                gradient_sequence_pd[gate][parameter] = np.zeros((number_steps + 1, ))

        for cov_index1 in covariance_index:
            for cov_index2 in covariance_index:
                covariance_sequence_pd[cov_index1][cov_index2] = np.zeros((number_steps + 1, ))
        d_voltage_pd = pd.Series()
        d_parameters_pd = pd.Series()
        for i in range(number_steps):

            gradient_pd_temp = pd.DataFrame(kalman.grad, index=self.parameter_names,
                                                     columns=self.tunable_gate_names)
            covariance_pd_temp = pd.DataFrame(kalman.cov, index=covariance_index, columns=covariance_index)

            for gate in self.tunable_gate_names:
                for parameter in self.parameter_names:
                    gradient_sequence_pd[gate][parameter][i] = gradient_pd_temp[gate][parameter]

            for cov_index1 in covariance_index:
                for cov_index2 in covariance_index:
                    covariance_sequence_pd[cov_index1][cov_index2][i] = covariance_pd_temp[cov_index1][
                        cov_index2]

            for gate in self.tunable_gate_names:
                d_voltage_pd[gate] = gate_voltages_sequence_pd[gate][i+1] - gate_voltages_sequence_pd[gate][i]
            for parameter in self.parameter_names:
                d_parameters_pd[parameter] = parameters_sequence_pd[parameter][i + 1] - \
                                             parameters_sequence_pd[parameter][i]
            d_voltage_pd = d_voltage_pd.sort_index()
            d_parameters_pd = d_parameters_pd.sort_index()

            if with_relative_error:
                r = np.copy(kalman.filter.R)
                expected_d_parameter = np.dot(kalman.grad, d_voltage_pd.as_matrix())
                for j in range(self.parameter_names.size):
                    r[j, j] += relative_error * parameters_sequence_pd[j][i + 1] * parameters_sequence_pd[j][i + 1] \
                               + .0025 * (d_parameters_pd[j] - expected_d_parameter[j]) * (
                    d_parameters_pd[j] - expected_d_parameter[j])
                kalman.update(d_voltage_pd.as_matrix(), d_parameters_pd.as_matrix(), measurement_covariance=r, hack=False)
            elif residuals is not None:
                r = np.copy(kalman.filter.R)
                residuals = residuals.sort_index()
                for j in range(self.parameter_names.size):
                    #r[j, j] += 2e7 * residuals[j][i + 1]
                    r[j, j] = 0.1 * r[j, j] + residuals[j][i] + residuals[j][i + 1]
                try:
                    kalman.update(d_voltage_pd.as_matrix(), d_parameters_pd.as_matrix(), measurement_covariance=r, hack=False)
                except:
                    print(r)


            else:
                kalman.update(d_voltage_pd.as_matrix(), d_parameters_pd.as_matrix())

        gradient_pd_temp = pd.DataFrame(kalman.grad, index=self.parameter_names,
                                                 columns=self.tunable_gate_names)
        covariance_pd_temp = pd.DataFrame(kalman.cov, index=covariance_index, columns=covariance_index)

        for gate in self.tunable_gate_names:
            for parameter in self.parameter_names:
                gradient_sequence_pd[gate][parameter][number_steps] = gradient_pd_temp[gate][parameter]

        for cov_index1 in covariance_index:
            for cov_index2 in covariance_index:
                covariance_sequence_pd[cov_index1][cov_index2][number_steps] = covariance_pd_temp[cov_index1][
                    cov_index2]

        return gradient_sequence_pd, covariance_sequence_pd

    def plot_concatenate_kalman_tune_run(self, tune_run_numbers, with_covariance=True, with_offset=True,
                                         recalculate_gradient=False, alpha=1.02, process_noise=None, recalculate_parameters=False,
                                         reduced_for_paper=False):
        if reduced_for_paper:
            standard_font_size = 14
            fig_size_inches_par = []
            fig_size_inches_grad = []
            fig_size_inches_volt = []
            x_limits = []
        else:
            standard_font_size = 9
            fig_size_inches_par = [7, 3.5]
            fig_size_inches_grad = [7, 2.5]
            fig_size_inches_volt = [7, 3]
            x_limits = [-2, 190]

        number_runs = len(tune_run_numbers)
        desired_values_pd_concatenated_temp, gate_voltages_sequence_pd_concatenated, parameters_sequence_pd_concatenated, \
        gradient_sequence_pd_concatenated = \
            self.load_kalman_tune_run(tune_run_number=tune_run_numbers[0])
        covariance_sequence_pd_concatenated = self.load_covariance_sequence_pd(tune_run_number=tune_run_numbers[0])
        desired_values_pd_concatenated = pd.Series()
        if recalculate_parameters:
            parameters_sequence_pd_concatenated_recalc, residuals_pd_concatenated = self.recalculate_parameters(
                tune_run_numbers[0])
        for parameter in self.parameter_names:
            desired_values_pd_concatenated[parameter] = [desired_values_pd_concatenated_temp[parameter]]
        number_steps = [
            count_steps_in_sequence(self.root_group["tunerun_" + str(tune_run_numbers[0]) + "/tune_sequence"]) - 1]
        for i in range(1, number_runs):
            desired_values_pd, gate_voltages_sequence_pd, parameters_sequence_pd, gradient_sequence_pd = \
                self.load_kalman_tune_run(tune_run_number=tune_run_numbers[i])
            covariance_sequence_pd = self.load_covariance_sequence_pd(tune_run_number=tune_run_numbers[i])
            if recalculate_parameters:
                parameters_sequence_pd_recalc, residuals_pd = self.recalculate_parameters(tune_run_numbers[i])
                for parameter in self.parameter_names:
                    residuals_pd_concatenated[parameter] = np.concatenate(
                        [residuals_pd_concatenated[parameter], residuals_pd[parameter]], 0)
                    parameters_sequence_pd_concatenated_recalc[parameter] = np.concatenate(
                        [parameters_sequence_pd_concatenated_recalc[parameter],
                         parameters_sequence_pd_recalc[parameter]], 0)
            for gate in self.gate_names:
                gate_voltages_sequence_pd_concatenated[gate] = np.concatenate(
                    [gate_voltages_sequence_pd_concatenated[gate], gate_voltages_sequence_pd[gate]], 0)

            number_steps = np.concatenate([number_steps, [count_steps_in_sequence(
                self.root_group["tunerun_" + str(tune_run_numbers[i]) + "/tune_sequence"])]], 0)
            for parameter in self.parameter_names:
                parameters_sequence_pd_concatenated[parameter] = np.concatenate(
                    [parameters_sequence_pd_concatenated[parameter], parameters_sequence_pd[parameter]], 0)

                desired_values_pd_concatenated[parameter] = np.concatenate(
                    [np.reshape(desired_values_pd_concatenated[parameter], i),
                     np.reshape(desired_values_pd[parameter], 1)], 0)
                for gate in self.tunable_gate_names:
                    gradient_sequence_pd_concatenated[gate][parameter] = np.concatenate(
                        [gradient_sequence_pd_concatenated[gate][parameter], gradient_sequence_pd[gate][parameter]],
                        0)
                    covariance_sequence_pd_concatenated[parameter.decode("ascii") + "_" + gate.decode("ascii")]\
                    [parameter.decode("ascii") + "_" + gate.decode("ascii")] = \
                        np.concatenate([covariance_sequence_pd_concatenated\
                        [parameter.decode("ascii") + "_" + gate.decode("ascii")]
                        [parameter.decode("ascii") + "_" + gate.decode("ascii")],
                        covariance_sequence_pd[parameter.decode("ascii") + "_" + gate.decode("ascii")]\
                        [parameter.decode("ascii") + "_" + gate.decode("ascii")]], 0)

        for i in range(1, number_runs):
            number_steps[i] = number_steps[i] + number_steps[i - 1]

        gate_colours = {"BA": 'b', "BB": 'g', "N": 'r', "SA": 'c', "SB": 'm', "T": 'k'}
        tunable_gate_colours = {"N": 'r', "SA": 'c', "SB": 'm', "T": 'k'}

        if recalculate_gradient:
            initial_gradient, initial_heuristic_covariance, initial_heuristic_noise = self.load_gradient_setup(
                gradient_number=1, tune_run_number=0)
#            initial_heuristic_covariance = 5. * initial_heuristic_covariance
            if recalculate_parameters:
                gradient_sequence_pd_recalculated, covariance_sequence_pd_recalculated = self.calculate_kalman_gradient(
                    gate_voltages_sequence_pd_concatenated, parameters_sequence_pd_concatenated_recalc,
                    initial_grad=initial_gradient,
                    initial_cov=initial_heuristic_covariance, initial_noise=initial_heuristic_noise,
                    number_steps=number_steps[number_runs - 1], alpha=alpha, residuals=residuals_pd_concatenated)
            else:
                gradient_sequence_pd_recalculated, covariance_sequence_pd_recalculated = self.calculate_kalman_gradient(
                    gate_voltages_sequence_pd_concatenated, parameters_sequence_pd_concatenated,
                    initial_grad=initial_gradient,
                    initial_cov=initial_heuristic_covariance, initial_noise=initial_heuristic_noise,
                    number_steps=number_steps[number_runs - 1], alpha=alpha,
                    process_noise=process_noise)

            figure_number = 30
            for parameter in self.parameter_names:
                plt.figure(figure_number, figsize=(5, 4))
                figure_number += 1
                for gate in self.tunable_gate_names:
                    plt.errorbar(x=list(range(number_steps[number_runs - 1] + 1)),
                                 y=1e-3 * gradient_sequence_pd_recalculated[gate][parameter],
                                 yerr=1e-3 * np.sqrt(
                                     covariance_sequence_pd_recalculated[
                                         parameter.decode("ascii") + "_" + gate.decode("ascii")][
                                         parameter.decode("ascii") + "_" + gate.decode("ascii")]),
                                 label=gate.decode("ascii"), color=tunable_gate_colours[gate.decode("ascii")])
                    plt.gca().tick_params("x", labelsize=standard_font_size)
                    plt.gca().tick_params("y", labelsize=standard_font_size)
                    plt.xlim(x_limits)
#                    plt.ylim(-20, 3)
                    # plt.title(
                    #    "Recalculated gradient row: " + parameter_plot_name(
                    #        parameter.decode("ascii"), with_unit=False) + r"; with $\alpha = $" + str(alpha),
                    #    fontsize=18)
                    #plt.title("Gradient row: " + parameter_plot_name(parameter.decode("ascii"), with_unit=False),
                    #          fontsize=18)
                    for j in range(number_runs - 1):
                        plt.axvline(x=number_steps[j])
                    plt.legend(fontsize=standard_font_size)
                    plt.ylabel(gradient_plot_ylabel(parameter.decode("ascii")), fontsize=standard_font_size)
                    plt.xlabel("Iteration Number", fontsize=standard_font_size)
                    fig = plt.gcf()
                    fig.set_size_inches(fig_size_inches_grad)

        figure_number = 1
        plt.figure(figure_number, figsize=(5, 4))

        number_parameter = len(self.parameter_names)
        for i in range(number_parameter):
            if recalculate_parameters:
                plt.figure(figure_number + 1, figsize=(5, 4))
                # plt.subplot(1, 1, 1)
                plt.subplot(number_parameter, 1, i + 1)
                #plt.plot(residuals_pd_concatenated[self.parameter_names[i]], "r")
                plt.errorbar(x=list(range(number_steps[number_runs - 1] + 1)),
                             y=parameters_sequence_pd_concatenated_recalc[self.parameter_names[i]],
                             yerr=np.sqrt(residuals_pd_concatenated[self.parameter_names[i]]), color="r")
#                plt.xlim(9.5, 22.5)
                # if i == 0:
                    # plt.title("Recalculated Parameters with Residuals", fontsize=18)
                    # plt.title("Parameters", fontsize=18)
                for j in range(number_runs - 1):
                    plt.axvline(x=number_steps[j])
                plt.ylabel(parameter_plot_name(self.parameter_names[i].decode("ascii")), fontsize=standard_font_size)
                if i == number_parameter - 1:
                    plt.xlabel("Iteration Number", fontsize=standard_font_size)
                y_des_val = [desired_values_pd_concatenated[self.parameter_names[i]][0],
                             desired_values_pd_concatenated[self.parameter_names[i]][0]]

                start_desired_val_range = 0.01
                x_des_val = [start_desired_val_range, number_steps[0]]
                start_desired_val_range = number_steps[0] + 0.01
                for run in range(1, number_runs):
                    y_des_val = np.concatenate([y_des_val,
                                                [desired_values_pd_concatenated[self.parameter_names[i]][run],
                                                 desired_values_pd_concatenated[self.parameter_names[i]][run]]], 0)
                    x_des_val = np.concatenate([x_des_val, [start_desired_val_range, number_steps[run]]], 0)
                    start_desired_val_range = number_steps[run] + 0.01
                plt.plot(x_des_val, y_des_val, "b")

            plt.figure(figure_number)
            plt.subplot(number_parameter, 1, i + 1)
            plt.plot(parameters_sequence_pd_concatenated[self.parameter_names[i]], "r")
            plt.plot(parameters_sequence_pd_concatenated[self.parameter_names[i]], "k.")
            plt.xlim(x_limits)
            # plt.locator_params(nbins=5)
            plt.gca().tick_params("x", labelsize=standard_font_size)
            plt.gca().tick_params("y", labelsize=standard_font_size)
            # if i == 0:
                # plt.title("Parameters", fontsize=18)
            start_desired_val_range = 0.01
            y_des_val = [desired_values_pd_concatenated[self.parameter_names[i]][0],
                         desired_values_pd_concatenated[self.parameter_names[i]][0]]
            x_des_val = [start_desired_val_range, number_steps[0]]
            start_desired_val_range = number_steps[0] + 0.01
            for run in range(1, number_runs):
                y_des_val = np.concatenate([y_des_val,
                                            [desired_values_pd_concatenated[self.parameter_names[i]][run],
                                             desired_values_pd_concatenated[self.parameter_names[i]][run]]], 0)
                x_des_val = np.concatenate([x_des_val, [start_desired_val_range, number_steps[run]]], 0)
                start_desired_val_range = number_steps[run] + 0.01
            plt.plot(x_des_val, y_des_val, "b")
            for j in range(number_runs - 1):
                plt.axvline(x=number_steps[j])

            plt.ylabel(parameter_plot_name(self.parameter_names[i].decode("ascii")), fontsize=standard_font_size)

        plt.xlabel("Iteration Number", fontsize=standard_font_size)
        fig = plt.gcf()
        fig.set_size_inches(fig_size_inches_par)
        plt.tight_layout()
        plt.draw()

        figure_number += 2

        for k, parameter in enumerate(self.parameter_names):
            if reduced_for_paper:
                plt.figure(figure_number, figsize=(5, 4))
            else:
                plt.figure(figure_number, figsize=(5, 4))
            figure_number += 1
            for gate in self.tunable_gate_names:
                if with_covariance:
                    plt.errorbar(x=list(range(number_steps[number_runs - 1] + 1)), y=1e-3 * gradient_sequence_pd_concatenated[gate][parameter],
                                 yerr=1e-3 * np.sqrt(
                                     covariance_sequence_pd_concatenated[parameter.decode("ascii") + "_" + gate.decode("ascii")][
                                         parameter.decode("ascii") + "_" + gate.decode("ascii")]),
                                label=gate.decode("ascii"), color=tunable_gate_colours[gate.decode("ascii")])
                    if reduced_for_paper:
                        plt.gca().tick_params("x", labelsize=standard_font_size)
                        plt.gca().tick_params("y", labelsize=standard_font_size)
                        plt.gca().set_xlim(-1, 105)
                    else:
                        plt.gca().tick_params("x", labelsize=standard_font_size)
                        plt.gca().tick_params("y", labelsize=standard_font_size)

                    plt.xlim(x_limits)
#                    plt.ylim(-20, 3)

                else:
                    plt.plot(gradient_sequence_pd_concatenated[gate][parameter], gate_colours[gate.decode("ascii")],
                             label=gate.decode("ascii"))
                    plt.gca().tick_params("x", labelsize=standard_font_size)
                    plt.gca().tick_params("y", labelsize=standard_font_size)
                    # plt.gca().set_xlim(-1, 105)
                for j in range(number_runs-1):
                    plt.axvline(x=number_steps[j])
            plt.legend(ncol=4)
            #plt.title(
            #    "Gradient elements in the matrix row: " + parameter_plot_name(parameter.decode("ascii"), with_unit=False),
            #    fontsize=18)
            plt.legend(ncol=4, fontsize=standard_font_size, loc='best', fancybox=True, framealpha=0.5)
            plt.ylabel(gradient_plot_ylabel(parameter.decode("ascii")), fontsize=standard_font_size)
            plt.xlabel("Iteration Number", fontsize=standard_font_size)
            fig = plt.gcf()
            fig.set_size_inches(fig_size_inches_grad)
            plt.gca().set_ylim([-12, -20][k], [4, 12][k])
            """
       
            if reduced_for_paper:
                fig.set_size_inches(6, 4) # for paper with fontsize 13
                
                plt.locator_params(axis='y', nbins=5)
            else:
                fig.set_size_inches(8.5, 6)
                # plt.gca().set_ylim([-13, -20][k], [2, 12][k])
     """
            plt.tight_layout()

        if reduced_for_paper:
            plt.figure(figure_number, figsize=(5, 4))
        else:
            plt.figure(figure_number, figsize=(8.5, 6))
        figure_number += 1
        offset = 0.
        for gate in self.gate_names:
            for j in range(number_runs-1):
                plt.axvline(x=number_steps[j])
            if with_offset:
                plt.plot(1000. * (
                gate_voltages_sequence_pd_concatenated[gate] - gate_voltages_sequence_pd_concatenated[gate][0] +
                offset), gate_colours[gate.decode("ascii")], label=gate.decode("ascii"))
                plt.gca().tick_params("x", labelsize=standard_font_size)
                plt.gca().tick_params("y", labelsize=standard_font_size)
                offset += 20e-3
            else:
                plt.plot(
                    1000. * (
                    gate_voltages_sequence_pd_concatenated[gate] - gate_voltages_sequence_pd_concatenated[gate][0]),
                    gate_colours[gate.decode("ascii")], label=gate.decode("ascii"))
                plt.gca().tick_params("x", labelsize=standard_font_size)
                plt.gca().tick_params("y", labelsize=standard_font_size)
        plt.legend(fontsize=standard_font_size, ncol=6)
        plt.xlabel("Iteration Number", fontsize=standard_font_size)
        plt.xlim(x_limits)
        plt.ylim(-45, 120)
        if with_offset:
            plt.ylabel("Voltages (mV) (Offset changed)", fontsize=standard_font_size)
        else:
            plt.ylabel("Voltage Difference (mV)", fontsize=standard_font_size)
        fig = plt.gcf()
        fig.set_size_inches(fig_size_inches_volt)
        plt.tight_layout()
        # plt.title("Changes in Gate Voltages", fontsize=18)

    def recalculate_parameters(self, tune_run_number, start=0, end=None):
        self.load_evaluator_names(tune_run_number=tune_run_number)
        self.load_parameter_names(tune_run_number=tune_run_number)
        tune_run_group = self.load_tune_run_group(tune_run_number=tune_run_number)
        tune_sequence_group = tune_run_group["tune_sequence"]
        if end is None:
            end = count_steps_in_sequence(tune_sequence_group)
        parameters_pd = pd.Series()
        residuals_pd = pd.Series()
        for parameter in self.parameter_names:
            parameters_pd[parameter] = np.zeros((end-start, ))
            residuals_pd[parameter] = np.zeros((end-start, ))
            
        for i in range(start, end):
            for evaluator in self.evaluator_names:
                raw_data = tune_sequence_group["step_" + str(i) + "/" + evaluator][:]
                if evaluator == "evaluator_SMInterDotTCByLineScan" or evaluator == "evaluator_InterDotTCByLineScan":
                    ydata = np.squeeze(raw_data)
                    scan_range = tune_sequence_group["step_" + str(i) + "/" + evaluator].attrs["scan_range"]
                    if len(ydata.shape) == 1:
                        npoints = len(ydata)
                    else:
                        npoints = ydata.shape[1]
                    center = 0.  # TODO: change for real center
                    fitresult = qtune.evaluator.fit_inter_dot_coupling(data=ydata, plot_fit=False, center=center, scan_range=scan_range,
                                                                       npoints=npoints)
                    parameters_pd[b'parameter_tunnel_coupling'][i] = fitresult["tc"]
                    residuals_pd[b'parameter_tunnel_coupling'][i] = fitresult["residual"]
                elif evaluator == "evaluator_SMLoadTime" or evaluator == "evaluator_LoadTime":
                    fitresult = qtune.evaluator.fit_load_time(data=raw_data, plot_fit=False)
                    parameters_pd[b'parameter_time_load'][i] = fitresult["parameter_time_load"]
                    residuals_pd[b'parameter_time_load'][i] = fitresult["residual"]
                else:
                    print("No plotting implemented for this evaluator.")

        return parameters_pd, residuals_pd

    def plot_concatenate_kalman_tune_run_long_run(self, tune_run_numbers):
        """"""

        number_runs = len(tune_run_numbers)
        desired_values_pd_concatenated_temp, gate_voltages_sequence_pd_concatenated, parameters_sequence_pd_concatenated, \
        gradient_sequence_pd_concatenated = \
            self.load_kalman_tune_run(tune_run_number=tune_run_numbers[0])

        desired_values_pd_concatenated = pd.Series()
        for parameter in self.parameter_names:
            desired_values_pd_concatenated[parameter] = [desired_values_pd_concatenated_temp[parameter]]
        number_steps = [
            count_steps_in_sequence(self.root_group["tunerun_" + str(tune_run_numbers[0]) + "/tune_sequence"]) - 1]
        for i in range(1, number_runs):
            desired_values_pd, gate_voltages_sequence_pd, parameters_sequence_pd, gradient_sequence_pd = \
                self.load_kalman_tune_run(tune_run_number=tune_run_numbers[i])

            for gate in self.gate_names:
                gate_voltages_sequence_pd_concatenated[gate] = np.concatenate(
                    [gate_voltages_sequence_pd_concatenated[gate], gate_voltages_sequence_pd[gate]], 0)

            number_steps = np.concatenate([number_steps, [count_steps_in_sequence(
                self.root_group["tunerun_" + str(tune_run_numbers[i]) + "/tune_sequence"])]], 0)
            for parameter in self.parameter_names:
                parameters_sequence_pd_concatenated[parameter] = np.concatenate(
                    [parameters_sequence_pd_concatenated[parameter], parameters_sequence_pd[parameter]], 0)

                desired_values_pd_concatenated[parameter] = np.concatenate(
                    [np.reshape(desired_values_pd_concatenated[parameter], i),
                     np.reshape(desired_values_pd[parameter], 1)], 0)
                for gate in self.tunable_gate_names:
                    gradient_sequence_pd_concatenated[gate][parameter] = np.concatenate(
                        [gradient_sequence_pd_concatenated[gate][parameter], gradient_sequence_pd[gate][parameter]],
                        0)
        plt.figure(1, figsize=(5, 4))
        for i in range(1, number_runs):
            number_steps[i] = number_steps[i] + number_steps[i - 1]

        number_parameter = len(self.parameter_names)
        for i in range(number_parameter):
            plt.subplot(number_parameter, 1, i + 1)
            plt.plot(parameters_sequence_pd_concatenated[self.parameter_names[i]], "r")
            # plt.title("Parameters")
            start_desired_val_range = 0.01
            y_des_val = [desired_values_pd_concatenated[self.parameter_names[i]][0],
                         desired_values_pd_concatenated[self.parameter_names[i]][0]]
            x_des_val = [start_desired_val_range, number_steps[0]]
            start_desired_val_range = number_steps[0] + 0.01
            for run in range(1, number_runs):
                y_des_val = np.concatenate([y_des_val,
                                            [desired_values_pd_concatenated[self.parameter_names[i]][run],
                                             desired_values_pd_concatenated[self.parameter_names[i]][run]]], 0)
                x_des_val = np.concatenate([x_des_val, [start_desired_val_range, number_steps[run]]], 0)
                start_desired_val_range = number_steps[run] + 0.01
            plt.plot(x_des_val, y_des_val, "b")
            #            plt.axhline(desired_values_pd[self.parameter_names[i]])
            for j in range(number_runs - 1):
                plt.axvline(x=number_steps[j])
            plt.ylabel(self.parameter_names[i].decode("ascii"))
            plt.xlabel("Iteration Number")
            plt.draw()
        figure_number = 2
        gate_colours = {"BA": 'b', "BB": 'g', "N": 'r', "SA": 'c', "SB": 'm', "T": 'k'}
        tunable_gate_colours = {"N": 'r', "SA": 'c', "SB": 'm', "T": 'y'}
        for parameter in self.parameter_names:
            plt.figure(figure_number, figsize=(5, 4))
            figure_number += 1
            for gate in self.tunable_gate_names:
                plt.plot(gradient_sequence_pd_concatenated[gate][parameter], gate_colours[gate.decode("ascii")],
                         label=gate.decode("ascii"))
                for j in range(number_runs):
                    if j != 1:
                        plt.axvline(x=number_steps[j])
                        #                plt.axvline(x=number_steps[j])
            plt.legend()
            # plt.title("Gradient elements in the line " + parameter.decode("ascii"))

        plt.figure(figure_number, figsize=(5, 4))
        figure_number += 1
        for gate in self.gate_names:
            for j in range(number_runs):
                if j != 1:
                    plt.axvline(x=number_steps[j])
                    #                plt.axvline(x=number_steps[j])
            plt.plot(gate_voltages_sequence_pd_concatenated[gate] - gate_voltages_sequence_pd_concatenated[gate][0],
                     gate_colours[gate.decode("ascii")], label=gate.decode("ascii"))
        plt.legend()
        # plt.title("Changes in gate voltages")

    def load_raw_measurement_pd(self, step_group):
        raw_measurement_pd = pd.Series()
        attribute_info_pd = pd.Series()
        for evaluator in self.evaluator_names:
            raw_measurement_pd[evaluator] = step_group[evaluator][:]
            attribute_info = {key: step_group[evaluator].attrs[key] for key in step_group[evaluator].attrs.keys()}
            attribute_info_pd[evaluator] = attribute_info
        return raw_measurement_pd, attribute_info_pd

    def load_raw_measurement_gradient_calculation(self, gradient_number: int = 1, tune_run_number: int = 0) -> (
            pd.DataFrame, pd.DataFrame, int, float):
        self.load_evaluator_names(tune_run_number=tune_run_number)
        gradient_group = self.load_gradient_group(gradient_number=gradient_number, tune_run_number=tune_run_number)
        delta_u = gradient_group.attrs["delta_u"]
        n_repetitions = gradient_group.attrs["n_repetitions"]
        raw_measurement_positive_detune_pd = pd.DataFrame(index=self.evaluator_names, columns=self.tunable_gate_names)
        raw_measurement_negative_detune_pd = pd.DataFrame(index=self.evaluator_names, columns=self.tunable_gate_names)
        parameter_positive_detune_pd = pd.DataFrame(index=self.evaluator_names, columns=self.tunable_gate_names)
        parameter_negative_detune_pd = pd.DataFrame(index=self.evaluator_names, columns=self.tunable_gate_names)
        for i in range(n_repetitions):
            for gate in self.tunable_gate_names:
                raw_measurement_pos_pd, attribute_info_pd_pos = self.load_raw_measurement_pd(
                    step_group=gradient_group["positive_detune_run_" + gate.decode("ascii") + "_" + str(i)])
                raw_measurement_neg_pd, attribute_info_pd_neg = self.load_raw_measurement_pd(
                    step_group=gradient_group["positive_detune_run_" + gate.decode("ascii") + "_" + str(i)])
                for evaluator in self.evaluator_names:
                    matrix_pos = raw_measurement_pos_pd[evaluator]
                    matrix_neg = raw_measurement_neg_pd[evaluator]
                    if i == 0:
                        raw_measurement_positive_detune_pd[gate][evaluator] = matrix_pos
                        raw_measurement_negative_detune_pd[gate][evaluator] = matrix_neg
                        parameter_positive_detune_pd[gate][evaluator] = attribute_info_pd_pos[evaluator]
                        parameter_negative_detune_pd[gate][evaluator] = attribute_info_pd_neg[evaluator]
                    else:
                        raw_measurement_positive_detune_pd[gate][evaluator] = np.concatenate(
                            (raw_measurement_positive_detune_pd[gate][evaluator], matrix_pos))
                        raw_measurement_negative_detune_pd[gate][evaluator] = np.concatenate(
                            (raw_measurement_negative_detune_pd[gate][evaluator], matrix_neg))
        return raw_measurement_positive_detune_pd, raw_measurement_negative_detune_pd, parameter_positive_detune_pd, \
            parameter_negative_detune_pd, n_repetitions, delta_u

    def plot_raw_measurement_gradient_calculation(self, gradient_number: int = 1, tune_run_number: int = 0):
        #TODO: use plot_raw_measurement function if the plotting is to be changed
        raw_measurement_positive_detune_pd, raw_measurement_negative_detune_pd, parameter_positive_detune_pd, \
        parameter_negative_detune_pd, n_repetitions, delta_u = \
            self.load_raw_measurement_gradient_calculation(gradient_number=gradient_number,
                                                           tune_run_number=tune_run_number)
        print("Please chose a gate by typing the number next to its name: ")
        num_tunable_gates = len(self.tunable_gate_names)
        for i in range(num_tunable_gates):
            print(self.tunable_gate_names[i].decode("ascii") + ": " + str(i))
        gate = input()
        gate = int(gate)
        gate = self.tunable_gate_names[gate]
        print("please chose an evaluator by typing the number next to its name: ")
        num_evaluators = len(self.evaluator_names)
        for i in range(num_evaluators):
            print(self.evaluator_names[i] + ": " + str(i))
        evaluator = input()
        evaluator = int(evaluator)
        evaluator = self.evaluator_names[evaluator]
        plt.ion()
        if evaluator == "evaluator_SMLeadTunnelTimeByLeadScan":
            for i in range(n_repetitions):
                plt.figure(1)
                plt.plot(raw_measurement_positive_detune_pd[gate][evaluator][2 * i, :])
                plt.plot(raw_measurement_positive_detune_pd[gate][evaluator][2 * i + 1, :])
                plt.legend(["Data", "Background"])
                plt.draw()
                plt.pause(0.05)
                plt.figure(2)
                diff = raw_measurement_positive_detune_pd[gate][evaluator][2 * i + 1, :] - \
                    raw_measurement_positive_detune_pd[gate][evaluator][2 * i, :]
                qtune.Evaluator.fit_lead_times(diff)
                plt.legend(["BG subtracted"])
                plt.pause(0.05)
                decision_continue = input("Type STOP to stop. Type anything else to continue.")
                if decision_continue == "STOP":
                    plt.close()
                    return
                else:
                    plt.close()
                    plt.figure(1)
                    plt.close()
            for i in range(n_repetitions):
                plt.figure(1)
                plt.plot(raw_measurement_negative_detune_pd[gate][evaluator][2 * i, :])
                plt.plot(raw_measurement_negative_detune_pd[gate][evaluator][2 * i + 1, :])
                plt.legend(["Data", "Background"])
                plt.draw()
                plt.pause(0.05)
                plt.figure(2)
                diff = raw_measurement_negative_detune_pd[gate][evaluator][2 * i + 1, :] - \
                       raw_measurement_negative_detune_pd[gate][evaluator][2 * i, :]
                #                plt.plot(diff)
                qtune.Evaluator.fit_lead_times(diff)
                #plt.legend(["BG subtracted"])
                plt.pause(0.05)
                decision_continue = input("Type STOP to stop. Type anything else to continue.")
                if decision_continue == "STOP":
                    plt.close()
                    return
                else:
                    plt.close()
                    plt.figure(1)
                    plt.close()
        elif evaluator == "evaluator_SMInterDotTCByLineScan":
            for i in range(n_repetitions):
                plt.figure(1)
                ydata = raw_measurement_positive_detune_pd[gate][evaluator][i, :]
                scan_range = parameter_positive_detune_pd[gate][evaluator]["scan_range"]
                npoints = len(ydata)
                center = 0. # TODO: change for real center
                qtune.Evaluator.fit_inter_dot_coupling(data=ydata, center=center, scan_range=scan_range, npoints=npoints)
                plt.pause(0.05)
                decision_continue = input("Type STOP to stop. Type anything else to continue.")
                if decision_continue == "STOP":
                    plt.close()
                    return
                else:
                    plt.close()
        else:
            print("No plotting implemented for this evaluator.")

    def load_evaluator_names(self, tune_run_number: int=1):
        tune_run_group = self.load_tune_run_group(tune_run_number=tune_run_number)
        if tune_run_group.__contains__("tune_sequence/step_0"):
            first_step_group = tune_run_group["tune_sequence/step_0"]
            list_evaluator_names = []
            for element in first_step_group:
                if isinstance(first_step_group[element], h5py.Dataset) and "evaluator_" in element:
                    list_evaluator_names += [element, ]
            self.evaluator_names = list_evaluator_names
        elif tune_run_group.__contains__("gradient_setup_1/gradient_kalman"):
            first_measurement_group = tune_run_group["gradient_setup_1/gradient_kalman/noise_estimation/measurement_0"]
            list_evaluator_names = []
            for element in first_measurement_group:
                if isinstance(first_measurement_group[element], h5py.Dataset) and "evaluator_" in element:
                    list_evaluator_names += [element, ]
            self.evaluator_names = list_evaluator_names
        else:
            first_gradient_group = self.load_gradient_group(gradient_number=1, tune_run_number=tune_run_number)
            run_group = first_gradient_group["negative_detune_run_" + self.tunable_gate_names[0].decode("ascii") + "_0"]
            list_evaluator_names = []
            for element in run_group:
                if isinstance(run_group[element], h5py.Dataset) and "evaluator_" in element:
                    list_evaluator_names += [element, ]
            self.evaluator_names = list_evaluator_names
        return list_evaluator_names

    def load_raw_measurement_sequence_pd(self, start: int=0, end: int=None, tune_run_number: int=1):
        self.load_evaluator_names(tune_run_number=tune_run_number)
        tune_run_group = self.load_tune_run_group(tune_run_number=tune_run_number)
        sequence_group = tune_run_group["tune_sequence"]
        if end is None:
            end = count_steps_in_sequence(sequence_group)
        raw_measurement_sequence_pd = pd.DataFrame(index=self.evaluator_names, columns=range(start, end))
        attribute_info_sequence_pd = pd.DataFrame(index=self.evaluator_names, columns=range(start, end))
        for i in range(start, end):
            raw_measurement_pd, attribute_info_pd = self.load_raw_measurement_pd(sequence_group["step_" + str(i)])
            for evaluator in self.evaluator_names:
                raw_measurement_sequence_pd[i][evaluator] = raw_measurement_pd[evaluator]
                attribute_info_sequence_pd[i][evaluator] = attribute_info_pd[evaluator]
        return raw_measurement_sequence_pd, attribute_info_sequence_pd

    def plot_raw_measurement_tune_run(self, tune_run_number: int=1, start: int=0, end: int=None, figure_numer: int=70):
        self.load_evaluator_names(tune_run_number=tune_run_number)
        tune_run_group = self.load_tune_run_group(tune_run_number=tune_run_number)
        tune_sequence_group = tune_run_group["tune_sequence"]
        if end is None:
            end = count_steps_in_sequence(tune_sequence_group)
        raw_measurement_sequence_pd, attribute_info_sequence_pd = \
            self.load_raw_measurement_sequence_pd(start=start, end=end, tune_run_number=tune_run_number)
        plt.ion()
        for i in range(start, end):
            for evaluator in self.evaluator_names:
                plt.ylabel(evaluator)
                plt.pause(0.05)
                print(i)
                if plot_raw_measurement(evaluator, raw_data=raw_measurement_sequence_pd[i][evaluator],
                                     attribute_info=attribute_info_sequence_pd[i][evaluator],
                                     figure_number=figure_numer) == "Stop":
                    return

    def load_noise_estimation(self, noise_estimation_group: h5py.Group, n_noise_measurements: int):
        gate_voltages = noise_estimation_group["gate_voltages"][:]
        gate_voltages_pd = pd.Series(data=gate_voltages, index = self.gate_names)
        parameter_sequence_pd = pd.Series()
        for parameter in self.parameter_names:
            parameter_sequence_pd[parameter.decode("ascii")] = np.zeros((n_noise_measurements, ))
        for i in range(n_noise_measurements):
            measurement_group = noise_estimation_group["measurement_" + str(i)]
            for evaluator in self.evaluator_names:
                evaluator_data_set = measurement_group[evaluator]
                for attribute in evaluator_data_set.attrs.keys():
                    if "parameter_" in attribute:
                        parameter_sequence_pd[attribute][i] = evaluator_data_set.attrs[attribute]
        return parameter_sequence_pd, gate_voltages_pd

    def load_covariance_minimization(self, min_cov_group: h5py.Group, n_minimization_measurements: int):
        gate_voltage_sequence_pd = pd.Series()
        for gate in self.gate_names:
            gate_voltage_sequence_pd[gate] = np.zeros((n_minimization_measurements,))
        for i in range(n_minimization_measurements):
            step_group = min_cov_group["step_" + str(i)]
            gate_voltages_pd = pd.Series(data=step_group["gate_voltages"][:], index=self.gate_names)
            for gate in self.gate_names:
                gate_voltage_sequence_pd[gate][i] = gate_voltages_pd[gate]
        parameter_sequence_pd = pd.Series()
        for parameter in self.parameter_names:
            parameter_sequence_pd[parameter.decode("ascii")] = np.zeros((n_minimization_measurements,))
        for i in range(n_minimization_measurements):
            step_group = min_cov_group["step_" + str(i)]
            for evaluator in self.evaluator_names:
                evaluator_data_set = step_group[evaluator]
                for attribute in evaluator_data_set.attrs.keys():
                    if "parameter_" in attribute:
                        parameter_sequence_pd[attribute][i] = evaluator_data_set.attrs[attribute]
        gradient_sequence_pd = pd.DataFrame(None, index=self.parameter_names, columns=self.tunable_gate_names)
        for parameter in self.parameter_names:
            for gate in self.tunable_gate_names:
                gradient_sequence_pd[gate][parameter] = np.zeros((n_minimization_measurements,))
        covariance_index = []
        for parameter in self.parameter_names:
            for gate in self.tunable_gate_names:
                covariance_index += [parameter.decode("ascii") + "_" + gate.decode("ascii")]
        covariance_sequence_pd = pd.DataFrame(index=covariance_index, columns=covariance_index)
        for index in covariance_index:
            for index2 in covariance_index:
                covariance_sequence_pd[index][index2] = np.zeros((n_minimization_measurements, ))
        for i in range(n_minimization_measurements):
            step_group = min_cov_group["step_" + str(i)]
            gradient_pd = pd.DataFrame(data=step_group["gradient"][:], index=self.parameter_names,
                                       columns=self.tunable_gate_names)
            covariance_pd = pd.DataFrame(data=step_group["heuristic_covariance"][:], index=covariance_index,
                                         columns=covariance_index)
            for parameter in self.parameter_names:
                for gate in self.tunable_gate_names:
                    gradient_sequence_pd[gate][parameter][i] = gradient_pd[gate][parameter]
            for index in covariance_index:
                for index2 in covariance_index:
                    covariance_sequence_pd[index][index2][i] = covariance_pd[index][index2]
        return gradient_sequence_pd, covariance_sequence_pd, parameter_sequence_pd, gate_voltage_sequence_pd

    def load_min_cov_calculation(self, gradient_number: int = 1, tune_run_number: int = 0):
        self.load_evaluator_names(tune_run_number=tune_run_number)
        gradient_group = self.load_gradient_group(gradient_number=gradient_number, tune_run_number=tune_run_number)
        gradient_kalman_group = gradient_group["gradient_kalman"]
        n_noise_measurements = gradient_kalman_group.attrs["n_noise_measurements"]
        noise_estimation_group = gradient_kalman_group["noise_estimation"]
        noise_parameter_sequence_pd, noise_gate_voltages_pd = self.load_noise_estimation(noise_estimation_group=noise_estimation_group,
                                                                 n_noise_measurements=n_noise_measurements)
        n_minimization_measurements = gradient_kalman_group.attrs["n_minimization_measurements"]
        min_cov_group = gradient_kalman_group["covariance_minimization"]
        gradient_sequence_pd, covariance_sequence_pd, parameter_sequence_pd, gate_voltage_sequence_pd = \
            self.load_covariance_minimization(min_cov_group=min_cov_group,
                                              n_minimization_measurements=n_minimization_measurements)
        return noise_parameter_sequence_pd, noise_gate_voltages_pd, gradient_sequence_pd, covariance_sequence_pd, \
               parameter_sequence_pd, gate_voltage_sequence_pd, n_noise_measurements, n_minimization_measurements

    def plot_min_cov_calculation(self, gradient_number: int = 1, tune_run_number: int = 0, figure_number: int=80):
        noise_parameter_sequence_pd, noise_gate_voltages_pd, gradient_sequence_pd, covariance_sequence_pd, \
        parameter_sequence_pd, gate_voltage_sequence_pd, n_noise_measurements, n_minimization_measurements = \
            self.load_min_cov_calculation(gradient_number=gradient_number, tune_run_number=tune_run_number)

        for parameter in noise_parameter_sequence_pd.index:
            plt.figure(figure_number)
            plt.close()
            plt.figure(figure_number)
            plt.plot(noise_parameter_sequence_pd[parameter])
            plt.title(parameter)
            figure_number += 1

        for parameter in self.parameter_names:
            plt.figure(figure_number)
            plt.close()
            plt.figure(figure_number)
            figure_number += 1
            for gate in self.tunable_gate_names:
                    plt.errorbar(x=list(range(n_minimization_measurements)), y=gradient_sequence_pd[gate][parameter],
                                 yerr=np.sqrt(covariance_sequence_pd[parameter.decode("ascii") + "_" + gate.decode("ascii")][
                                     parameter.decode("ascii") + "_" + gate.decode("ascii")]))
            plt.legend([gate_name.decode("ascii") for gate_name in self.tunable_gate_names])
            plt.title(parameter.decode("ascii"))
            plt.show()

    def load_single_values_gradient_calculation(self, gradient_number: int = 1, tune_run_number: int = 0) -> (
            pd.Series, int, float):
        gradient_group = self.load_gradient_group(gradient_number=gradient_number, tune_run_number=tune_run_number)
        n_repetitions = gradient_group.attrs["n_repetitions"]
#        n_repetitions = gradient_group["n_repetitions"].value
        delta_u = gradient_group.attrs["delta_u"]
#        delta_u = gradient_group["delta_u"].value
        single_values_pd = pd.DataFrame(None, index=self.parameter_names, columns=self.tunable_gate_names)
        temp_parameter = pd.Series()
        for gate in self.tunable_gate_names:
            for parameter in self.parameter_names:
                single_values_pd[gate][parameter] = np.zeros((n_repetitions, 2))
            for i in range(n_repetitions):
                positive_run_group = gradient_group["positive_detune_run_" + gate.decode("ascii") + "_" + str(i)]
                for element in positive_run_group.keys():
                    if "evaluator_" in element:
                        for parameter_name in positive_run_group[element].attrs.keys():
                            if "parameter_" in parameter_name:
                                temp_parameter[parameter_name] = positive_run_group[element].attrs[parameter_name]
                temp_parameter = temp_parameter.sort_index()

                for parameter in self.parameter_names:
                    single_values_pd[gate][parameter][i, 0] = temp_parameter[parameter.decode("ascii")]

                negative_run_group = gradient_group["negative_detune_run_" + gate.decode("ascii") + "_" + str(i)]
                for element in negative_run_group.keys():
                    if "evaluator_" in element:
                        for parameter_name in negative_run_group[element].attrs.keys():
                            if "parameter_" in parameter_name:
                                temp_parameter[parameter_name] = negative_run_group[element].attrs[parameter_name]
                temp_parameter = temp_parameter.sort_index()
                for parameter in self.parameter_names:
                    single_values_pd[gate][parameter][i, 1] = temp_parameter[parameter.decode("ascii")]
        return single_values_pd, n_repetitions, delta_u

    def load_tune_run_group(self, tune_run_number) -> h5py.Group:
        if self.root_group.__contains__("tunerun_" + str(tune_run_number)):
            tune_run_group = self.root_group["tunerun_" + str(tune_run_number)]
            self.load_parameter_names(tune_run_number=tune_run_number)
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
                    tune_run_number) + ".")
                decision = input("Would you like to load the gradient from the previous run? (Y/N)")
                if decision == "Y":
                    print("Gradient will be loaded from a previous run!")
                    return self.load_gradient_group(gradient_number, tune_run_number - 1)
                elif decision == "N":
                    print("No gradient could be loaded!")
                    raise KeyError("Group does not exist.")
                else:
                    print("This was a yes or no question. Answer with Y or N!")
                    return self.load_gradient_group(gradient_number=gradient_number, tune_run_number=tune_run_number)
            else:
                print("There is no gradient saved for the gradient setup number " + str(
                    gradient_number) + " in tune run number" + str(
                    tune_run_number) + ".")
                raise KeyError("Group does not exist.")
        gradient_group = tune_run_group["gradient_setup_" + str(gradient_number)]
        return gradient_group

    def logout(self):
        self.root_group.close()

    def compute_gradient_from_raw_data(self, fit_functions, evaluators, gradient_number: int = 1,
                                       tune_run_number: int = 0):
        raw_measurement_positive_detune_pd, raw_measurement_negative_detune_pd, n_repetitions, delta_u = \
            self.load_raw_measurement_gradient_calculation(gradient_number=gradient_number,
                                                           tune_run_number=tune_run_number)
        n_evaluators = len(evaluators)
        positive_detune = pd.DataFrame()
        negative_detune = pd.DataFrame()
        positive_detune_parameter = pd.Series()
        negative_detune_parameter = pd.Series()
        gradient_pd = pd.DataFrame(index=self.parameter_names, columns=self.tunable_gate_names)
        gradient_std_pd = pd.DataFrame(index=self.parameter_names, columns=self.tunable_gate_names)
        measurement_std_pd = pd.Series(index=self.parameter_names)
        for gate in self.tunable_gate_names:
            for parameter in self.parameter_names:
                gradient_pd[gate][parameter] = np.zeros((n_repetitions, ))
                for i in range(n_repetitions):
                    for evaluator_number in range(n_evaluators):
                        if len(np.shape(raw_measurement_positive_detune_pd[gate][parameter])) == 1:
                            n_lines_per_measurement = 1
                        else:
                            n_lines_per_measurement = np.shape(raw_measurement_positive_detune_pd[gate][parameter])[
                                                          0] / n_repetitions
                        gradient_pd[gate][parameter][i] = fit_functions[evaluator_number]

    def load_square_grid_evaluation_reconstruct_voltages(self, start=1, gate1="T", gate2="N", max_distance=15e-3,
                                                         step_size=3e-3, n_steps=11):

        #reconstruct_voltages:
        data = pd.Series()
        for gate1_index in range(int(n_steps / 2)):

            for gate2_index in range(n_steps):
                new_entry = pd.Series(data=[2 * gate1_index * step_size, gate2_index * step_size], index=[gate1, gate2])
                data[str(gate2_index + 2 * n_steps * gate1_index)] = new_entry

            for gate2_index in range(n_steps):
                new_entry = pd.Series(data=[(2 * gate1_index + 1) * step_size, (n_steps - 1 - gate2_index) * step_size],
                                      index=[gate1, gate2])
                data[str(n_steps + gate2_index + 2 * n_steps * gate1_index)] = new_entry

        if (n_steps % 2) == 1:
            for gate2_index in range(n_steps):
                new_entry = pd.Series(data=[(n_steps - 1) * step_size, gate2_index * step_size],
                                      index=[gate1, gate2])
                data[str(gate2_index + n_steps * (n_steps - 1))] = new_entry

        #load data
        for i in range(n_steps * n_steps):
            data[str(i)]["evaluator_LoadTime"] = self.root_group[
                                                     "parameter_evaluation/parameter_evaluation_" + str(
                                                         i + 1 + start) + "/evaluator_LoadTime"][:]
            data[str(i)]["evaluator_SMInterDotTCByLineScan"] = self.root_group[
                                                                   "parameter_evaluation/parameter_evaluation_" + str(
                                                                       i + 1 + start) + "/evaluator_SMInterDotTCByLineScan"][
                                                               :]


        # calculate parameters
        data_array_tc = np.zeros((n_steps, n_steps))
        data_array_load = np.zeros((n_steps, n_steps))
        gate1_voltage_values = np.zeros((n_steps, n_steps))
        gate2_voltage_values = np.zeros((n_steps, n_steps))

        for gate1_index in range(int(n_steps / 2)):

            for gate2_index in range(n_steps):
                fitresult = qtune.evaluator.fit_inter_dot_coupling(
                    data[str(gate2_index + 2 * n_steps * gate1_index)]["evaluator_SMInterDotTCByLineScan"], center=0.,
                    scan_range=2e-3, npoints=100, plot_fit=False)
                data_array_tc[2 * gate1_index, gate2_index] = fitresult["tc"]
                fitresult = qtune.evaluator.fit_load_time(
                    data[str(gate2_index + 2 * n_steps * gate1_index)]["evaluator_LoadTime"], plot_fit=False)
                data_array_load[2 * gate1_index, gate2_index] = fitresult["parameter_time_load"]
                gate1_voltage_values[2 * gate1_index, gate2_index] = data[str(gate2_index + 2 * n_steps * gate1_index)][
                    gate1]
                gate2_voltage_values[2 * gate1_index, gate2_index] = data[str(gate2_index + 2 * n_steps * gate1_index)][
                    gate2]

            for gate2_index in range(n_steps):
                fitresult = qtune.evaluator.fit_inter_dot_coupling(
                    data[str(n_steps + gate2_index + 2 * n_steps * gate1_index)]["evaluator_SMInterDotTCByLineScan"],
                    center=0., scan_range=2e-3, npoints=100, plot_fit=False)
                data_array_tc[2 * gate1_index + 1, n_steps - 1 - gate2_index] = fitresult["tc"]
                fitresult = qtune.evaluator.fit_load_time(
                    data[str(n_steps + gate2_index + 2 * n_steps * gate1_index)]["evaluator_LoadTime"], plot_fit=False)
                data_array_load[2 * gate1_index + 1, n_steps - 1 - gate2_index] = fitresult["parameter_time_load"]
                gate1_voltage_values[2 * gate1_index + 1, n_steps - 1 - gate2_index] = \
                data[str(n_steps + gate2_index + 2 * n_steps * gate1_index)][gate1]
                gate2_voltage_values[2 * gate1_index + 1, n_steps - 1 - gate2_index] = \
                data[str(n_steps + gate2_index + 2 * n_steps * gate1_index)][gate2]

        if (n_steps % 2) == 1:
            for gate2_index in range(n_steps):
                fitresult = qtune.evaluator.fit_inter_dot_coupling(
                    data[str(gate2_index + n_steps * (n_steps - 1))]["evaluator_SMInterDotTCByLineScan"], center=0.,
                    scan_range=2e-3, npoints=100, plot_fit=False)
                data_array_tc[n_steps - 1, gate2_index] = fitresult["tc"]
                fitresult = qtune.evaluator.fit_load_time(
                    data[str(gate2_index + n_steps * (n_steps - 1))]["evaluator_LoadTime"], plot_fit=False)
                data_array_load[n_steps - 1, gate2_index] = fitresult["parameter_time_load"]
                gate1_voltage_values[n_steps - 1, gate2_index] = data[str(gate2_index + n_steps * (n_steps - 1))][gate1]
                gate2_voltage_values[n_steps - 1, gate2_index] = data[str(gate2_index + n_steps * (n_steps - 1))][gate2]

        #plot

        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        Axes3D.plot_surface(ax, X=gate2_voltage_values, Y=gate1_voltage_values, Z=data_array_tc)
        plt.title('Tunnel Coupling')

        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        Axes3D.plot_surface(ax, X=gate2_voltage_values, Y=gate1_voltage_values, Z=data_array_load)
        plt.title('Load Time')


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
#        print("\n")


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


def fit_lead_times_old(ydata: np.ndarray):
    samprate = 1e8
    n_points = len(ydata)
    xdata = np.asarray([i for i in range(n_points)]) / samprate
    p0 = [ydata[round(1. / 4. * n_points)] - ydata[round(3. / 4. * n_points)],
          50e-9, 50e-9, 70e-9, 2070e-9, np.mean(ydata)]
    popt, pcov = optimize.curve_fit(f=func_lead_times, p0=p0, xdata=xdata, ydata=ydata)
    begin_lin = int(round(popt[4] / 10e-9))
    end_lin = begin_lin + 7
    slope = (ydata[end_lin] - ydata[begin_lin]) / (xdata[end_lin] - xdata[begin_lin])
    linear_offset = ydata[begin_lin] - xdata[begin_lin] * slope
    p0 += [slope, linear_offset, xdata[end_lin]]
    begin_lin_1 = int(round(popt[3] / 10e-9))
    end_lin_1 = begin_lin_1 + 7
    slope_1 = (ydata[end_lin_1] - ydata[begin_lin_1]) / (xdata[end_lin_1] - xdata[begin_lin_1])
    linear_offset_1 = ydata[begin_lin_1] - xdata[begin_lin_1] * slope_1
    p0 += [slope_1, linear_offset_1, xdata[end_lin_1]]
    plt.plot(xdata, ydata)
    plt.plot(xdata, func_lead_times_v2(xdata, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7], p0[8], p0[9], p0[10], p0[11]))
    popt, pcov = optimize.curve_fit(f=func_lead_times_v2, p0=p0, xdata=xdata, ydata=ydata)
    plt.figure(6)
    plt.plot(xdata, ydata)
    plt.plot(xdata, func_lead_times_v2(xdata, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11]))
    plt.pause(0.05)
    print("fit parameters")
    print(popt)
    return popt, pcov


def func_lead_times(x, hight: float, t_fall: float, t_rise: float, begin_rise: float, begin_fall: float,
                    offset: float):
    x = np.squeeze(x)
    n_points = len(x)
    y = np.zeros((n_points, ))
    for i in range(n_points):
        if x[i] >= begin_rise and x[i] <= begin_fall:
            c = np.cosh(.5*begin_fall/t_rise)
            s = np.sinh(.5*begin_fall/t_rise)
            e = np.exp(.5*(begin_fall - 2. * x[i]) / t_rise)
            signed_hight = hight
        elif x[i] < begin_rise:
            c = np.cosh(.5*begin_fall/t_fall)
            s = np.sinh(.5*begin_fall/t_fall)
            e = np.exp(.5*(1. * begin_fall - 2. * (x[i] + x[n_points - 1])) / t_fall)
            signed_hight = -1. * hight
        else:
            c = np.cosh(.5*begin_fall/t_fall)
            s = np.sinh(.5*begin_fall/t_fall)
            e = np.exp(.5*(3. * begin_fall - 2. * x[i]) / t_fall)
            signed_hight = -1. * hight
        y[i] = offset + .5 * signed_hight * (c - e) / s
    return y


def plot_raw_measurement(evaluator, raw_data, attribute_info, figure_number):
    if evaluator == "evaluator_SMLeadTunnelTimeByLeadScan" or evaluator == "evaluator_LeadTunnelTimeByLeadScan":
        plt.figure(figure_number)
        plt.plot(raw_data[0])
        plt.plot(raw_data[1])
        plt.legend(["Data", "Background"])
        plt.draw()
        plt.pause(0.05)
        plt.figure(figure_number + 1)
        plt.close()
        plt.figure(figure_number + 1)
        qtune.evaluator.fit_lead_times(raw_data)
        plt.pause(0.05)
    elif evaluator == "evaluator_SMInterDotTCByLineScan" or evaluator == "evaluator_InterDotTCByLineScan":
        plt.figure(figure_number)
        plt.close()
        plt.figure(figure_number)
        ydata = np.squeeze(raw_data)
        scan_range = attribute_info["scan_range"]
        if len(ydata.shape) == 1:
            npoints = len(ydata)
        else:
            npoints = ydata.shape[1]
        center = attribute_info["center"]  # TODO: change for real center
        qtune.evaluator.fit_inter_dot_coupling(data=ydata, center=center, scan_range=scan_range, npoints=npoints)
        plt.pause(0.05)
    elif evaluator == "evaluator_SMLoadTime" or evaluator == "evaluator_LoadTime":
        plt.figure(figure_number)
        plt.close()
        plt.figure(figure_number)
        qtune.evaluator.fit_load_time(data=raw_data)
        plt.pause(0.05)
    else:
        print("No plotting implemented for this evaluator.")
    decision_continue = input("Type STOP to stop. Type anything else to continue.")
    if decision_continue == "STOP":
#        plt.close(figure_number)
#        plt.close(figure_number + 1)
        return "Stop"


def parameter_plot_name(name: str, with_unit: bool=True):
    if with_unit:
        if name == "parameter_time_load":
            # return "Singlet Load Time $t_{sr}$ $(ns)$"
            return "$t_{sr}$ $(ns)$"
        if name == "parameter_tunnel_coupling":
            # return "Transition Width $w$ $(\mu V)$"
            return "$w$ $(\mu V)$"
    else:
        if name == "parameter_time_load":
            return "Singlet load time"
        if name == "parameter_tunnel_coupling":
            return "Tunnel coupling"


def gradient_plot_ylabel(parameter: str):
    if parameter == "parameter_time_load":
        # return r"Gradient Elements $\partial t_{sr}/ \partial V_i$ $(ns /m V)$"
        return "$ \partial t_{sr}/ \partial V_i$ $(ns /m V)$"
    if parameter == "parameter_tunnel_coupling":
        # return "Gradient Elements $\partial w / \partial V_i$ $(\mu V /m V)$"
        return "$\partial w / \partial V_i$ $(\mu V /m V)$"
