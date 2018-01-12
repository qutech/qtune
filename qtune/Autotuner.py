import pandas as pd
import numpy as np
import copy
import h5py
from typing import Tuple
from qtune.util import time_string
from qtune.experiment import Experiment
from qtune.Evaluator import Evaluator
from qtune.Solver import Solver, KalmanSolver
from qtune.Basic_DQD import BasicDQD
from qtune.chrg_diag import ChargeDiagram


class Autotuner:
    def __init__(self, experiment: Experiment, solver: Solver = None, evaluators: Tuple[Evaluator, ...] = (),
                 desired_values: pd.Series = pd.Series(), tuning_accuracy: pd.Series = pd.Series(),
                 data_directory: str = ''):
        self.parameters = pd.Series()
        self.solver = solver
        self.experiment = experiment
        self._evaluators = ()
        for e in evaluators:
            self.add_evaluator(e)
        self._desired_values = desired_values
        self.tuning_accuracy = tuning_accuracy
        self.tunable_gates = self.experiment.read_gate_voltages()
        self.gradient = None
        self.gradient_std = None
        self.evaluation_std = None
        self.tune_run_number = 0
        self.gradient_number = 0
        self.charge_diagram_number = 0
        self.hdf5file = h5py.File(data_directory + r'\Autotuner_' + time_string(), 'w')
        tunerun_group = self.hdf5file.create_group('tunerun_' + str(self.tune_run_number))
        self.current_tunerun_group = tunerun_group

    @property
    def evaluators(self):
        return self._evaluators

    @evaluators.setter
    def evaluators(self, evaluators: Tuple[Evaluator, ...]):
        self._evaluators = evaluators

    def add_evaluator(self, new_evaluator: Evaluator):
        new_parameters = new_evaluator.parameters
        for i in new_parameters.index.tolist():
            if i in self.parameters:
                print('This Evaluator determines a parameter which is already being evaluated by another Evaluator')
                return
            series_to_add = pd.Series((new_parameters[i],), (i,))
            self.parameters = self.parameters.append(series_to_add, verify_integrity=True)
        self._evaluators += (new_evaluator, )
        self.solver.add_evaluator(evaluator=new_evaluator)

    @property
    def desired_values(self):
        return self._desired_values

    @desired_values.setter
    def desired_values(self, desired_values: pd.Series):
        assert(desired_values.index.tolist() == self.parameters.index.tolist())
        self._desired_values = desired_values
        if self.solver is not None:
            self.solver.desired_values = desired_values

    def set_desired_values_manually(self):
        raise NotImplementedError

    def set_gate_voltages(self, new_voltages):
        self.experiment.set_gate_voltages(new_gate_voltages=new_voltages)

    def shift_gate_voltages(self, new_voltages: pd.Series, step_size=2.e-3):
        start_voltages = self.experiment.read_gate_voltages()
        d_voltages_series = new_voltages.add(-1. * start_voltages, fill_value=0.)
        d_voltage_abs = d_voltages_series.abs()
        if d_voltage_abs > step_size:
            voltage_step = d_voltages_series * d_voltage_abs / step_size
            self.set_gate_voltages(new_voltages=start_voltages + voltage_step)
            self.shift_gate_voltages(new_voltages=new_voltages, step_size=step_size)
        elif d_voltage_abs > step_size / 20.:
            self.set_gate_voltages(new_voltages=new_voltages)

    def evaluate_parameters(self, storing_group: h5py.Group=None) -> pd.Series:
        parameters = pd.Series()
        for e in self.evaluators:
            evaluation_result = e.evaluate(storing_group)

            if evaluation_result['failed']:
                evaluation_result = evaluation_result.drop(['failed'])
                for r in evaluation_result.index.tolist():
                    evaluation_result[r] = np.nan
            else:
                evaluation_result = evaluation_result.drop(['failed'])

            for r in evaluation_result.index.tolist():
                parameters[r] = evaluation_result[r]
        self.parameters = copy.deepcopy(parameters)
        return parameters

    def tuning_complete(self) -> bool:
        complete = True
        for parameter in self.parameters.index.tolist():
            if np.abs(self.parameters[parameter]-self.desired_values[parameter]) > self.tuning_accuracy[parameter]:
                complete = False
        return complete

    def ready_to_tune(self) -> bool:
        if self.solver is None:
            print('You need to setup a solver!')
            return False

    def evaluate_gradient_covariance_noise(self, delta_u=4e-3, n_repetitions=3) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series]:
        gradient_group = self.current_tunerun_group.create_group("gradient_setup_" + str(self.gradient_number))
        gradient_group["n_repetitions"] = n_repetitions
        gradient_group["delta_u"] = delta_u
        current_gate_positions = self.experiment.read_gate_voltages()
        positive_detune = pd.DataFrame()
        negative_detune = pd.DataFrame()
        positive_detune_parameter = pd.Series()
        negative_detune_parameter = pd.Series()
        for gate in self.tunable_gates.index.tolist():
            for parameter in self.parameters.index.tolist():
                positive_detune_parameter[parameter] = np.zeros((n_repetitions,), dtype=float)
                negative_detune_parameter[parameter] = np.zeros((n_repetitions,), dtype=float)

            detuning = pd.Series((delta_u, ), (gate, ))
            new_gate_positions = current_gate_positions.add(detuning, fill_value=0)
            self.shift_gate_voltages(new_gate_positions)

            for i in range(n_repetitions):
                run_subgroup = gradient_group.create_group("positive_detune_run_" + gate + "_" + str(i))
                save_gate_voltages(run_subgroup, new_gate_positions)
                evaluation_result = self.evaluate_parameters(run_subgroup)
                for r in evaluation_result.index.tolist():
                    (positive_detune_parameter[r])[i] = evaluation_result[r]

            new_gate_positions = current_gate_positions.add(detuning.multiply(-1.), fill_value=0)
            self.shift_gate_voltages(new_gate_positions)

            for i in range(n_repetitions):
                run_subgroup = gradient_group.create_group("negative_detune_run_" + gate + "_" + str(i))
                save_gate_voltages(run_subgroup, new_gate_positions)
                evaluation_result = self.evaluate_parameters(run_subgroup)
                for r in evaluation_result.index.tolist():
                    (negative_detune_parameter[r])[i] = evaluation_result[r]

            self.shift_gate_voltages(current_gate_positions)

            positive_detune_parameter_df = positive_detune_parameter.to_frame(gate)
            if positive_detune.empty:
                positive_detune = positive_detune_parameter_df
            else:
                positive_detune = positive_detune.join(positive_detune_parameter_df)
            negative_detune_parameter_df = negative_detune_parameter.to_frame(gate)
            if negative_detune.empty:
                negative_detune = negative_detune_parameter_df
            else:
                negative_detune = negative_detune.join(negative_detune_parameter_df)

        gradient = (positive_detune - negative_detune) / 2. / delta_u
        gradient_std = gradient.applymap(np.nanstd)
        gradient = gradient.applymap(np.nanmean)
        evaluation_std = positive_detune_parameter.apply(np.nanstd)

        self.gradient = gradient
        self.gradient_std = gradient_std
        self.evaluation_std = evaluation_std

        self.gradient_number += 1

        gradient_matrix, covariance, evaluation_noise = self.converte_gradient_heuristic_data(gradient, gradient_std,
                                                                                              evaluation_std)
        save_gradient_data(gradient_group, gradient_matrix, covariance, evaluation_noise)

        return gradient, gradient_std, evaluation_std

    def converte_gradient_heuristic_data(self, gradient: pd.DataFrame, gradient_std: pd.DataFrame,
                                         evaluation_std: pd.Series):
        gradient = gradient.sort_index(0)
        gradient = gradient.sort_index(1)
        gradient_matrix = gradient.as_matrix()
        if gradient_std is not None:
            gradient_std = gradient_std.sort_index(0)
            gradient_std = gradient_std.sort_index(1)
            gradient_std_matrix = gradient_std.as_matrix()
            n_parameters, n_gates = gradient.shape
            covariance = np.zeros((n_parameters * n_gates, n_parameters * n_gates))
            for n_p in range(n_parameters):
                for n_g in range(n_gates):
                    covariance[n_g + n_p * n_gates, n_g + n_p * n_gates] = gradient_std_matrix[n_p, n_g]
        else:
            covariance = None
        if evaluation_std is not None:
            evaluation_std = evaluation_std.sort_index()
            evaluation_std_matrix = evaluation_std.as_matrix()
            n_parameters, n_gates = gradient.shape
            evaluation_noise = np.zeros([n_parameters, n_parameters])
            for n_p in range(n_parameters):
                evaluation_noise[n_p, n_p] = evaluation_std_matrix[n_p] * evaluation_std_matrix[n_p]
        else:
            evaluation_noise = None
        return gradient_matrix, covariance, evaluation_noise

    def set_solver(self, solver: Solver):
        self.solver = solver

    def autotune(self) -> bool:
        raise NotImplementedError


class ChargeDiagramAutotuner(Autotuner):
    def __init__(self, dqd: BasicDQD, solver: Solver = None, evaluators: Tuple[Evaluator, ...] = (),
                 desired_values: pd.Series = pd.Series(), tuning_accuracy: pd.Series = pd.Series(),
                 charge_diagram_gradient=None, charge_diagram_covariance=None, charge_diagram_noise=None,
                 data_directory: str = r'Y:\GaAs\Autotune\Data\UsingPython\AutotuneData'):
        super().__init__(experiment=dqd, solver=solver, evaluators=evaluators, desired_values=desired_values,
                         tuning_accuracy=tuning_accuracy, data_directory=data_directory)
        self.charge_diagram = ChargeDiagram(dqd=dqd)
        gate_names = self.tunable_gates.drop(['RFA', 'RFB'])
        gate_names = gate_names.sort_index()
        gate_names = gate_names.index.tolist()
        gate_names = np.asarray(gate_names, dtype='S20')
        self.hdf5file.create_dataset("gate_names", data=gate_names)
        self.tunable_gates = self.tunable_gates.drop(['RFA', 'RFB', 'BA', 'BB'])
        self.tunable_gates = self.tunable_gates.sort_index()
        tunable_gate_names = self.tunable_gates.index.tolist()
        tunable_gate_names = np.asarray(tunable_gate_names, dtype="S20")
        self.hdf5file.create_dataset("tunable_gate_names", data=tunable_gate_names)
        if charge_diagram_gradient is not None or charge_diagram_covariance is not None or charge_diagram_noise is not None:
            self.initialize_charge_diagram_kalman(charge_diagram_gradient=charge_diagram_gradient,
                                                  charge_diagram_covariance=charge_diagram_covariance,
                                                  charge_diagram_noise=charge_diagram_noise,
                                                  heuristic_measurement=False, load_file=False)

    def initialize_charge_diagram_kalman(self, charge_diagram_gradient=None, charge_diagram_covariance=None,
                                         charge_diagram_noise=None, heuristic_measurement: bool = False, n_noise=15,
                                         n_cov=15, filename: str = None, filepath: str = "tunerun_0/charge_diagram_1",
                                         load_file: bool = False):
        if heuristic_measurement:
            charge_diagram_gradient, charge_diagram_covariance, charge_diagram_noise = self.measure_charge_diagram_histogram(
                n_noise=n_noise,
                n_cov=n_cov,)
        elif load_file:
            if filename is None:
                print('Please insert a file name!')
                return
            charge_diagram_gradient, charge_diagram_covariance, charge_diagram_noise = load_gradient_data(
                filename=filename,
                filepath=filepath)
            self.charge_diagram_number += 1
            save_group = self.hdf5file.create_group(
                'tunerun_' + str(self.tune_run_number) + r'\charge_diagram_' + str(self.charge_diagram_number))
            save_gradient_data(save_group, charge_diagram_gradient, charge_diagram_covariance,
                                    charge_diagram_noise)

        self.charge_diagram.initialize_kalman(initX=charge_diagram_gradient, initP=charge_diagram_covariance,
                                              initR=charge_diagram_noise)

    def set_gate_voltages(self, new_voltages: pd.Series):
        self.experiment.set_gate_voltages(new_gate_voltages=new_voltages)
        self.charge_diagram.center_diagram()

    def measure_charge_diagram_histogram(self, n_noise=10, n_cov=10) -> Tuple[
        np.array, np.array, np.array]:
        position_histo = np.zeros((n_noise, 2))
        grad_histo = np.zeros((n_cov, 2, 2))
        for i in range(0, n_noise):
            position_histo[i] = self.charge_diagram.measure_positions()
        for i in range(0, n_cov):
            grad_histo[i] = self.charge_diagram.calculate_gradient()

        gradient = np.nanmean(grad_histo, 0)
        std_position = np.std(position_histo, 0)
        std_grad = np.std(grad_histo, 0)
        heuristic_covariance = np.zeros((4, 4))
        heuristic_covariance[0, 0] = 2. * std_grad[0, 0]
        heuristic_covariance[1, 1] = 2. * std_grad[0, 1]
        heuristic_covariance[2, 2] = 2. * std_grad[1, 0]
        heuristic_covariance[3, 3] = 2. * std_grad[1, 1]
        heuristic_noise = np.zeros((2, 2))
        heuristic_noise[0, 0] = (2. * std_position[0]) * (2. * std_position[0])
        heuristic_noise[1, 1] = (2. * std_position[1]) * (2. * std_position[1])

        self.charge_diagram_number += 1
        save_group = self.hdf5file.create_group(
            'tunerun_' + str(self.tune_run_number) + r'\charge_diagram_' + str(self.charge_diagram_number))
        save_gradient_data(save_group, gradient, heuristic_covariance, heuristic_noise)
        return gradient, heuristic_covariance, heuristic_noise


class CDKalmanAutotuner(ChargeDiagramAutotuner):
    def set_solver(self, kalman_solver: KalmanSolver, gradient: pd.DataFrame = None,
                   evaluators: Tuple[Evaluator, ...] = (),
                   desired_values: pd.Series = pd.Series(), gradient_std: pd.DataFrame = None,
                   evaluation_std: pd.Series = None, alpha=1.02, load_data=False, filename: str = None,
                   filepath: str = "tunerun_0/gradient_setup_1"):
        if load_data:
            if filename is None:
                print('You need to insert a filename, if you want to load data!')
                return
            gradient_matrix, covariance, evaluation_noise = load_gradient_data(filename=filename,
                                                                                    filepath=filepath)
            self.gradient_number += 1
            gradient_group = self.current_tunerun_group.create_group("gradient_setup_" + str(self.gradient_number))
            save_gradient_data(gradient_group, gradient_matrix, covariance, evaluation_noise)

        elif gradient is None:
            print('You need to set or load a gradient!')
            return
        else:
            gradient_matrix, covariance, evaluation_noise = self.converte_gradient_heuristic_data(gradient,
                                                                                                  gradient_std,
                                                                                                  evaluation_std)

        self.solver = kalman_solver
        self.solver.gate_names = self.tunable_gates.index.tolist()
        if evaluators != ():
            for e in evaluators:
                self.solver.add_evaluator(e)
        if not desired_values.empty:
            self.solver.desired_values = desired_values
        self.solver.initialize_kalman(gradient=gradient_matrix, covariance=covariance, noise=evaluation_noise,
                                      alpha=alpha)

    def autotune(self, number_steps=1000, supervised: bool=False) -> bool:
        counter = 0
        if not self.ready_to_tune():
            print('The tuner setup is not complete!')
            return False
        tune_sequence_group = self.current_tunerun_group.create_group("tune_sequence")
        self.desired_values = self.desired_values.sort_index()
        self.current_tunerun_group.create_dataset("desired_values", data=self.desired_values.as_matrix())
        current_step_group = tune_sequence_group.create_group("step_" + str(counter))
        save_gate_voltages(current_step_group, self.experiment.read_gate_voltages())
        self.tune_run_number += 1
        parameters = self.evaluate_parameters(current_step_group)
        parameters = parameters.sort_index()
        parameter_names = parameters.index.tolist()
        parameter_names = np.asarray(parameter_names, dtype='S20')
        tune_sequence_group.create_dataset("parameter_names", data=parameter_names)
        counter += 1
        while counter < number_steps+1 and not self.tuning_complete():
            current_step_group = tune_sequence_group.create_group("step_" + str(self.tune_run_number))
            save_gate_voltages(current_step_group, self.experiment.read_gate_voltages())
            self.solver.parameter = parameters
            d_voltages = self.solver.suggest_next_step()
            current_voltages = self.experiment.read_gate_voltages()
            new_voltages = current_voltages.add(d_voltages, fill_value=0.)
            if supervised:
                try:
                    new_voltages = manual_check(new_voltages, current_voltages, d_voltages)
                except KeyboardInterrupt:
                    break
            try:
                self.shift_gate_voltages(new_voltages=new_voltages)
            except:
                print("The gates could not be shifted. Maybe the solver wants to go to extreme values!")
                return False
            new_parameters = self.evaluate_parameters(current_step_group)
            d_parameter = new_parameters - parameters
            new_gradient, new_covariance, failed = self.solver.update_after_step(d_voltages_series=d_voltages,
                                                                                 d_parameter_series=d_parameter)
            save_gradient_data(current_step_group, new_gradient, new_covariance, None)
            parameters = new_parameters
            counter += 1
        self.tune_run_number += 1
        return True


def manual_check(new_voltages: pd.Series, current_voltages: pd.Series, d_voltages: pd.Series):
    print("The Solver want to go from:")
    print(current_voltages)
    print("to:")
    print(new_voltages)
    print("which is a change by:")
    print(d_voltages)
    print("The absolute voltage change is:")
    print(d_voltages.abs())
    action = input("Would you prefer to accept (A) or change (C) the step or even stop (S) the tuning?")
    if action == "A":
        return new_voltages
    elif action == "C":
        decision = input("Would you like to multiply the step with a constant? (Y/N)")
        if decision == "Y":
            multiplicator = input("Please enter the multiplicator.")
            multiplicator = float(multiplicator)
            second_check = input("Are you sure, that you want to multiply with:" + str(multiplicator) + "? (Y/N)")
            if second_check == "Y":
                mult_d_voltages = d_voltages * multiplicator
                new_voltages = current_voltages.add(mult_d_voltages, fill_value=0.)
                return new_voltages
            elif second_check == "N":
                print("Restart the check.")
                return manual_check(new_voltages, current_voltages, d_voltages)
            else:
                print("Invalid input! Restart")
                return manual_check(new_voltages, current_voltages, d_voltages)
        elif decision == "N":
            print("No other possibilities have been implemented up to now. Restart!")
            return manual_check(new_voltages, current_voltages, d_voltages)
        else:
            print("Invalid input! Restart")
            return manual_check(new_voltages, current_voltages, d_voltages)
    elif action == "S":
        second_check = input(
            "Are you sure, you want to stop the tuning? In this case write STOP. Otherwise write cancel.")
        if second_check == "STOP":
            raise KeyboardInterrupt
        elif second_check == "cancel":
            print("OK, the check will be restarted.")
            return manual_check(new_voltages, current_voltages, d_voltages)
        else:
            print("Invalid input! Restart")
            return manual_check(new_voltages, current_voltages, d_voltages)
    else:
        print("Invalid input! Restart")
        return manual_check(new_voltages, current_voltages, d_voltages)


def load_gradient_data(filename: str, filepath: str=None):
    root_group = h5py.File(filename, 'r')
    if filepath is None:
        data_group = root_group
    else:
        data_group = root_group[filepath]
    gradient = data_group['gradient'][:]
    heuristic_covariance = data_group['heuristic_covariance'][:]
    heuristic_noise = data_group['heuristic_noise'][:]
    return gradient, heuristic_covariance, heuristic_noise


def save_gradient_data(save_group: h5py.Group, gradient, heuristic_covariance, heuristic_noise):
    save_group.attrs["time"] = time_string()
    save_group.create_dataset("gradient", data=gradient)
    if heuristic_covariance is not None:
        save_group.create_dataset("heuristic_covariance", data=heuristic_covariance)
    if heuristic_noise is not None:
        save_group.create_dataset("heuristic_noise", data=heuristic_noise)


def save_gate_voltages(save_group: h5py.Group, gate_voltages: pd.Series):
    gate_voltages = gate_voltages.sort_index()
    save_group.create_dataset("gate_voltages", data=gate_voltages.as_matrix())





















