import numpy as np
import pandas as pd
from qtune.experiment import Measurement
from qtune.basic_dqd import BasicQQD
from qtune.GradKalman import GradKalmanFilter
from qtune.util import find_lead_transition


class DQQChargeDiagram:
    charge_line_scan_A_AB = Measurement('line_scan', center=0., range=4e-3,
                                        gate='RFA', N_points=320,
                                        ramptime=.001,
                                        N_average=7,
                                        AWGorDecaDAC='DecaDAC')
    charge_line_scan_B_AB = Measurement('line_scan', center=0., range=4e-3,
                                        gate='RFB', N_points=320,
                                        ramptime=.001,
                                        N_average=7,
                                        AWGorDecaDAC='DecaDAC')

    charge_line_scan_C_CD = Measurement('line_scan', center=0., range=4e-3,
                                        gate='RFC', N_points=320,
                                        ramptime=.001,
                                        N_average=7,
                                        AWGorDecaDAC='DecaDAC')
    charge_line_scan_D_CD = Measurement('line_scan', center=0., range=4e-3,
                                        gate='RFD', N_points=320,
                                        ramptime=.001,
                                        N_average=7,
                                        AWGorDecaDAC='DecaDAC')

    delta_u_gradient_calculation = 3e-3
    centering_accuracy = .2e-3
    centering_plunger_step_size = 3e-3
    n_gates = 6
    charge_diagram_width = 4e-3
    shift_gates = ["RFB", "RFA", "RFD", "RFC"]
    index = ["A_AB", "B_AB", "C_CD", "D_CD"]

    def __init__(self, qqd: BasicQQD,
                 central_position=np.asarray([-0.15e-3, -0.77e-3, -0.15e-3, -0.77e-3])):
        self.qqd = qqd
        self.current_position = pd.Series()
        self.central_position = central_position
        charge_line_scans = [self.charge_line_scan_A_AB, self.charge_line_scan_B_AB, self.charge_line_scan_B_BC,
                             self.charge_line_scan_C_BC, self.charge_line_scan_C_CD, self.charge_line_scan_D_CD]

        self.measurements = pd.DataFrame(data=[[self.shift_gates], [charge_line_scans], [central_position]],
                                         columns=["shift_gate", "charge_line_scan", "central_position"],
                                         index=self.index)

        self.grad_kalman = GradKalmanFilter(self.n_gates, self.n_gates,
                                            initX=np.zeros((self.n_gates, self.n_gates), dtype=float))

        self.centralizing_gates = qqd.centralizing_gates

    def measure_positions(self) -> pd.Series:
        current_gate_voltages = self.qqd.read_gate_voltages()
        for i in self.index:
            shift = pd.Series(-1. * self.charge_diagram_width, [self.measurements["shift_gate"][i]])
            self.qqd.set_gate_voltages(current_gate_voltages.add(shift))
            data = self.qqd.measure(self.measurements["charge_line_scan"][i])
            self.current_position[i] = find_lead_transition(data,
                                                            float(self.measurements["charge_line_scan"][i]["center"]),
                                                            float(self.measurements["charge_line_scan"][i]["range"]),
                                                            self.measurements["charge_line_scan"][i]["N_points"])
        self.qqd.set_gate_voltages(current_gate_voltages)
        return self.current_position

    def calculate_gradient(self, n_repetitions: int=4):
        initial_gate_voltages = self.qqd.read_gate_voltages()
        delta_u = self.delta_u_gradient_calculation

        values_positive_detune = pd.DataFrame(index=self.index, columns=self.centralizing_gates)
        values_negative_detune = pd.DataFrame(index=self.index, columns=self.centralizing_gates)
        for centralizing_gate in self.centralizing_gates:
            for position_index in self.index:
                values_positive_detune[centralizing_gate][position_index] = np.zeros((n_repetitions, ))
                values_negative_detune[centralizing_gate][position_index] = np.zeros((n_repetitions,))

        for centralizing_gate in self.centralizing_gates:
            for i in range(n_repetitions):
                delta_v = pd.Series(data=[delta_u], index=[centralizing_gate])

                current_gate_voltages = initial_gate_voltages.add(delta_v, fill_index=0.)
                self.qqd.set_gate_voltages(current_gate_voltages)
                current_position_pos = self.measure_positions()

                current_gate_voltages = initial_gate_voltages.add(-1. * delta_v, fill_index=0.)
                self.qqd.set_gate_voltages(current_gate_voltages)
                current_position_neg = self.measure_positions()

                for position_index in self.index:
                    values_positive_detune[centralizing_gate][position_index][i] = current_position_pos[position_index]
                    values_negative_detune[centralizing_gate][position_index][i] = current_position_neg[position_index]

        values_positive_detune_std = values_positive_detune.apply(np.nanstd)
        values_negative_detune_std = values_negative_detune.apply(np.nanstd)

        noise = pd.Series(self.index)
        for index in self.index:
            noise[index] = np.nanmean(values_negative_detune_std[:][index])
            noise[index] += np.nanmean(values_positive_detune_std[:][index])
            noise[index] /= 2.

        gradient = (values_positive_detune - values_negative_detune) / delta_u
        gradient_std = gradient.apply(np.nanstd)
        gradient = gradient.apply(np.nanmean)

        gradient.sort_index(0)
        gradient.sort_index(1)
        gradient_std.sort_index(0)
        gradient_std.sort_index(1)
        noise.sort_index()

        return gradient, gradient_std, noise

    def initialize_kalman(self, initX=None, initP=None, initR=None, alpha=1.02):
        if initX is None:
            initX = self.calculate_gradient()[1]
        self.grad_kalman = GradKalmanFilter(len(self.qqd.centralizing_gates), len(self.index), initX=initX, initP=initP,
                                            initR=initR, alpha=alpha)

    def center_diagram(self, remeasure_positions: bool=True):
        if remeasure_positions:
            positions = np.asarray((self.measure_positions()))
        else:
            positions = (self.current_position.sort_index()).as_matrix()
        while np.linalg.norm(positions - self.central_position) > self.centering_accuracy:
            current_positions = positions
            du = np.linalg.solve(self.grad_kalman.grad, positions - self.central_position)
            if np.linalg.norm(du) > self.centering_plunger_step_size:
                du = du * self.centering_plunger_step_size / np.linalg.norm(du)
            diff = pd.Series(-1. * du, self.centralizing_gates)
            new_gate_voltages = self.qqd.read_gate_voltages().add(diff, fill_value=0.)
            self.qqd.set_gate_voltages(new_gate_voltages)

            positions = list(self.measure_positions())
            dpos = positions - current_positions
            self.grad_kalman.update(-1*du, dpos, hack=False)


class DQQPredictionChargeDiagram(DQQChargeDiagram):
    def __init__(self, qqd: BasicQQD,
                 central_position=np.asarray([-0.15e-3, -0.77e-3, -0.15e-3, -0.77e-3])):
        super().__init__(qqd=qqd, central_position=central_position)
        self.tunable_gates = qqd.tunable_gates
        number_tunable_gates = len(self.tunable_gates)
        self.grad_kalman_prediction = GradKalmanFilter(nGates=number_tunable_gates, nParams=len(self.index),
                                                       initX=np.zeros((len(self.index), number_tunable_gates),
                                                                      dtype=float))

    def calculate_prediction_gradient(self, n_repetitions: int=5, delta_u: float=2e-3):
        initial_gate_voltages = self.qqd.read_gate_voltages()

        values_positive_detune = pd.DataFrame(index=self.index, columns=self.tunable_gates)
        values_negative_detune = pd.DataFrame(index=self.index, columns=self.tunable_gates)

        for tunable_gate in self.tunable_gates:
            for position_index in self.index:
                values_positive_detune[tunable_gate][position_index] = np.zeros((n_repetitions,))
                values_negative_detune[tunable_gate][position_index] = np.zeros((n_repetitions,))

        for tunable_gate in self.tunable_gates:
            for i in range(n_repetitions):
                delta_v = pd.Series(data=[delta_u], index=[tunable_gate])

                current_gate_voltages = initial_gate_voltages.add(delta_v, fill_index=0.)
                self.qqd.set_gate_voltages(current_gate_voltages)
                current_position_pos = self.measure_positions()

                current_gate_voltages = initial_gate_voltages.add(-1. * delta_v, fill_index=0.)
                self.qqd.set_gate_voltages(current_gate_voltages)
                current_position_neg = self.measure_positions()

                for position_index in self.index:
                    values_positive_detune[tunable_gate][position_index][i] = current_position_pos[position_index]
                    values_negative_detune[tunable_gate][position_index][i] = current_position_neg[position_index]

        values_positive_detune_std = values_positive_detune.apply(np.nanstd)
        values_negative_detune_std = values_negative_detune.apply(np.nanstd)

        noise = pd.Series(self.index)
        for index in self.index:
            noise[index] = np.nanmean(values_negative_detune_std[:][index])
            noise[index] += np.nanmean(values_positive_detune_std[:][index])
            noise[index] /= 2.

        gradient = (values_positive_detune - values_negative_detune) / delta_u
        gradient_std = gradient.apply(np.nanstd)
        gradient = gradient.apply(np.nanmean)

        gradient.sort_index(0)
        gradient.sort_index(1)
        gradient_std.sort_index(0)
        gradient_std.sort_index(1)
        noise.sort_index()

        return gradient, gradient_std, noise

    def initialize_prediction_kalman(self, gradient=None, covariance=None, noise=None, alpha=1.02):
        if gradient is None:
            gradient = self.calculate_prediction_gradient()[0].as_matrix()
        self.grad_kalman_prediction = GradKalmanFilter(nGates=len(self.tunable_gates), nParams=len(self.index),
                                                       initX=gradient, initP=covariance, initR=noise, alpha=alpha)

    def prediction_center_diagram(self, d_voltages: pd.Series):
        for key in d_voltages.index:
            if key not in self.tunable_gates.index:
                d_voltages = d_voltages.drop(key)

        d_voltages.sort_index()
        d_voltages_vector = np.asarray(d_voltages)
        d_parameter_vector = np.dot(self.grad_kalman_prediction.grad, d_voltages_vector.transpose())
        neg_position_shift = -1. * d_parameter_vector

        correction = np.linalg.solve(self.grad_kalman.grad, neg_position_shift)
        correction_pd = pd.Series(data=correction, index=self.centralizing_gates)

        current_voltages = self.qqd.read_gate_voltages()
        new_voltages = current_voltages.add(correction_pd, fill_value=0.)
        self.qqd.set_gate_voltages(new_voltages)

        actual_position = np.asarray(self.measure_positions())
        total_position_shift = actual_position - self.central_position - neg_position_shift
        self.grad_kalman_prediction.update(dU=d_voltages_vector, dT=total_position_shift)


        # Retune the sensing dots

        self.center_diagram(remeasure_positions=True)
