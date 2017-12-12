import pandas as pd
import numpy as np
import pickle
from typing import Tuple
from qtune.experiment import Experiment
from qtune.Evaluator import Evaluator
from qtune.Solver import Solver
from qtune.Basic_DQD import BasicDQD
from qtune.chrg_diag import ChargeDiagram
from qtune.Kalman_heuristics import load_charge_diagram_gradient_covariance_noise_from_histogram, \
    save_charge_diagram_histogram_position_gradient


class Autotuner:
    def __init__(self, experiment: Experiment, solver: Solver = None, evaluators: Tuple[Evaluator, ...] = (),
                 desired_values: pd.Series = pd.Series()):
        self.parameters = pd.Series()
        self.solver = solver
        self.experiment = experiment
        self._evaluators = ()
        for e in evaluators:
            self.add_evaluator(e)
        self._desired_values = desired_values
        self.tunable_gates = self.experiment.read_gate_voltages()
        self.gradient = None
        self.gradient_std = None
        self.evaluation_std = None

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
        self._evaluators += (new_evaluator,)

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

    def evaluate_gradient_covariance_noise(self, delta_u=4e-3, n_repetitions=3, save_to_file: bool = False,
                                           filename: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
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
            self.set_gate_voltages(new_gate_positions)

            for i in range(n_repetitions):

                for e in self.evaluators:
                    evaluation_result = e.evaluate()

                    if evaluation_result['failed']:
                        evaluation_result = evaluation_result.drop(['failed'])
                        for r in evaluation_result.index.tolist():
                            evaluation_result[r] = np.nan
                    else:
                        evaluation_result = evaluation_result.drop(['failed'])

                    for r in evaluation_result.index.tolist():
                        positive_detune_parameter[r][i] = evaluation_result[r]

            self.set_gate_voltages(current_gate_positions)
            new_gate_positions = current_gate_positions.add(-1.*detuning, fill_value=0)
            self.set_gate_voltages(new_gate_positions)

            for i in range(n_repetitions):

                for e in self.evaluators:
                    evaluation_result = e.evaluate()

                    if evaluation_result['failed']:
                        evaluation_result = evaluation_result.drop(['failed'])
                        for r in evaluation_result.index.tolist():
                            evaluation_result[r] = np.nan
                    else:
                        evaluation_result = evaluation_result.drop(['failed'])

                    for r in evaluation_result.index.tolist():
                        negative_detune_parameter[r][i] = evaluation_result[r]

            self.set_gate_voltages(current_gate_positions)

            positive_detune_parameter_df = positive_detune_parameter.to_frame(gate)
            if positive_detune.empty:
                positive_detune = positive_detune_parameter_df
            else:
                positive_detune = positive_detune_parameter_df.join(positive_detune_parameter)
            negative_detune_parameter_df = negative_detune_parameter.to_frame(gate)
            if negative_detune.empty:
                negative_detune = negative_detune_parameter_df
            else:
                negative_detune = negative_detune_parameter_df.join(positive_detune_parameter)

        gradient = (positive_detune - negative_detune) / 2. / delta_u
        gradient_std = gradient.apply(np.nanstd)
        gradient = gradient.apply(np.nanmean)
        evaluation_std = positive_detune_parameter.apply(np.nanstd)

        if save_to_file:
            save_data = pd.Series([gradient, gradient_std, evaluation_std],
                                  ['gradient', 'gradient_std', 'evaluation_std'])
            with open(filename, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.gradient = gradient
        self.gradient_std = gradient_std
        self.evaluation_std = evaluation_std

        return gradient, gradient_std, evaluation_std

    def load_gradient_data(self, filename: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        gradient = data['gradient']
        gradient_std = data['gradient_std']
        evaluation_std = data['evaluation_std']
        return gradient, gradient_std, evaluation_std

    def set_solver(self, solver: Solver):
        self.solver = solver

    def autotune(self):
        raise NotImplementedError


class ChargeDiagramAutotuner(Autotuner):
    def __init__(self, dqd: BasicDQD, solver: Solver = None, evaluators: Tuple[Evaluator, ...] = (),
                 desired_values: pd.Series = pd.Series(), charge_diagram_gradient=None,
                 charge_diagram_covariance=None, charge_diagram_noise=None):
        super().__init__(experiment=dqd, solver=solver, evaluators=evaluators, desired_values=desired_values)
        self.charge_diagram = ChargeDiagram(dqd=dqd)
        self.tunable_gates = self.tunable_gates.drop(['RFA', 'RFB', 'BA', 'BB'])
        if charge_diagram_gradient is not None or charge_diagram_covariance is not None or charge_diagram_noise is not None:
            self.initialize_charge_diagram_kalman(charge_diagram_gradient=charge_diagram_gradient,
                                                  charge_diagram_covariance=charge_diagram_covariance,
                                                  charge_diagram_noise=charge_diagram_noise,
                                                  heuristic_measurement=False, save_to_file=False, load_file=False)

    def initialize_charge_diagram_kalman(self, charge_diagram_gradient=None, charge_diagram_covariance=None,
                                         charge_diagram_noise=None, heuristic_measurement: bool = False, n_noise=15,
                                         n_cov=15, save_to_file: bool = False, filename: str = None,
                                         load_file: bool = False):
        if heuristic_measurement:
            if save_to_file:
                if filename is None:
                    print('Cannot save the measured data without filename!')
                    save_to_file = False
            charge_diagram_covariance, charge_diagram_noise = self.measure_charge_diagram_histogram(n_noise=n_noise,
                                                                                                    n_cov=n_cov,
                                                                                                    save_to_file=save_to_file,
                                                                                                    filename=filename)
        elif load_file:
            if filename is None:
                print('Please insert a file name!')
                return
            charge_diagram_gradient, charge_diagram_covariance, charge_diagram_noise = \
                load_charge_diagram_gradient_covariance_noise_from_histogram(filename)
        self.charge_diagram.initialize_kalman(initX=charge_diagram_gradient, initP=charge_diagram_covariance,
                                              initR=charge_diagram_noise)

    def set_gate_voltages(self, new_voltages):
        self.experiment.set_gate_voltages(new_gate_voltages=new_voltages)
        self.charge_diagram.center_diagram()

    def measure_charge_diagram_histogram(self, n_noise=30, n_cov=30, save_to_file=True, filename: str = None) -> Tuple[
        np.array, np.array, np.array]:
        position_histo, grad_histo = save_charge_diagram_histogram_position_gradient(ch_diag=self.charge_diagram,
                                                                                     n_noise=n_noise, n_cov=n_cov,
                                                                                     savetofile=save_to_file,
                                                                                     filename=filename)
        gradient = np.nanmean(grad_histo)
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
        return gradient, heuristic_covariance, heuristic_noise

