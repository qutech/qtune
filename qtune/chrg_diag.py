# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:39:28 2017

@author: teske

The chargediagram will be implemented as class.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from qtune.experiment import Measurement
from qtune.Basic_DQD import BasicDQD
from qtune.GradKalman import GradKalmanFilter


class ChargeDiagram:
    charge_line_scan_lead_A = Measurement('line_scan', center=0., range=3e-3,
                                          gate='RFA', N_points=1280,
                                          ramptime=.0005,
                                          N_average=3,
                                          AWGorDecaDAC='DecaDAC')

    charge_line_scan_lead_B = Measurement('line_scan', center=0., range=3e-3,
                                          gate='RFB', N_points=1280,
                                          ramptime=.0005,
                                          N_average=3,
                                          AWGorDecaDAC='DecaDAC')

    def __init__(self, dqd: BasicDQD,
                 charge_line_scan_lead_A: Measurement = None,
                 charge_line_scan_lead_B: Measurement = None):
        self.dqd = dqd

        self.position_lead_A = 0.
        self.position_lead_B = 0.
        self.grad_kalman = GradKalmanFilter(2, 2, initX=np.zeros((2, 2), dtype=float))

        if charge_line_scan_lead_A is not None:
            self.charge_line_scan_lead_A = charge_line_scan_lead_A

        if charge_line_scan_lead_B is not None:
            self.charge_line_scan_lead_B = charge_line_scan_lead_B

    def measure_positions(self) -> Tuple[float, float]:
        current_gate_voltages = self.dqd.read_gate_voltages()
        RFA_eps = pd.Series(1e-3, ['RFA'])
        RFB_eps = pd.Series(1e-3, ['RFB'])
        voltages_for_pos_a = current_gate_voltages.add(-4*RFB_eps, fill_value=0)
        self.dqd.set_gate_voltages(voltages_for_pos_a)
        data_A = self.dqd.measure(self.charge_line_scan_lead_A)
        self.position_lead_A = find_lead_transition(data_A,
                                                    float(self.charge_line_scan_lead_A.parameter["center"]),
                                                    float(self.charge_line_scan_lead_A.parameter["range"]),
                                                    self.charge_line_scan_lead_A.parameter["N_points"])

        voltages_for_pos_b = current_gate_voltages.add(-4*RFA_eps, fill_value=0)
        self.dqd.set_gate_voltages(voltages_for_pos_b)
        data_B = self.dqd.measure(self.charge_line_scan_lead_B)
        self.position_lead_B = find_lead_transition(data_B,
                                                    float(self.charge_line_scan_lead_B.parameter["center"]),
                                                    float(self.charge_line_scan_lead_B.parameter["range"]),
                                                    self.charge_line_scan_lead_B.parameter["N_points"])
        self.dqd.set_gate_voltages(current_gate_voltages)
        return self.position_lead_A, self.position_lead_B

    def calculate_gradient(self):
        current_gate_voltages = self.dqd.read_gate_voltages()

        BA_eps = pd.Series(2e-3, ['BA'])
        BB_eps = pd.Series(2e-3, ['BB'])

        BA_inc = current_gate_voltages.add(BA_eps, fill_value=0)
        BA_dec = current_gate_voltages.add(-BA_eps, fill_value=0)

        BB_inc = current_gate_voltages.add(BB_eps, fill_value=0)
        BB_dec = current_gate_voltages.add(-BB_eps, fill_value=0)

        self.dqd.set_gate_voltages(BA_inc)
        pos_A_BA_inc, pos_B_BA_inc = self.measure_positions()

        self.dqd.set_gate_voltages(BA_dec)
        pos_A_BA_dec, pos_B_BA_dec = self.measure_positions()

        self.dqd.set_gate_voltages(BB_inc)
        pos_A_BB_inc, pos_B_BB_inc = self.measure_positions()

        self.dqd.set_gate_voltages(BB_dec)
        pos_A_BB_dec, pos_B_BB_dec = self.measure_positions()

        gradient = np.zeros((2, 2), dtype=float)
        gradient[0, 0] = (pos_A_BA_inc - pos_A_BA_dec) / 2e-3
        gradient[0, 1] = (pos_A_BB_inc - pos_A_BB_dec) / 2e-3
        gradient[1, 0] = (pos_B_BA_inc - pos_B_BA_dec) / 2e-3
        gradient[1, 1] = (pos_B_BB_inc - pos_B_BB_dec) / 2e-3

        self.dqd.set_gate_voltages(current_gate_voltages)

        return gradient.copy()

    def initialize_kalman(self, initX=None, initP=None, initR=None, alpha=1.02):
        if initX is None:
            initX = self.calculate_gradient()
        self.grad_kalman = GradKalmanFilter(2, 2, initX=initX, initP=initP, initR=initR, alpha=alpha)

    def center_diagram(self):
        positions = self.measure_positions()
        while np.linalg.norm(positions) > 0.2e-3:
            current_position = (self.position_lead_A, self.position_lead_B)
            du = np.linalg.solve(self.grad_kalman.grad, current_position)
            print("solve the lsg:")
            print(current_position)
            print(self.grad_kalman.grad)
            print("the solution is:")
            print(du)
            if np.linalg.norm(du) > 3e-3:
                du = du*3e-3/np.linalg.norm(du)

            diff = pd.Series(-1*du, ['BA', 'BB'])
            new_gate_voltages = self.dqd.read_gate_voltages().add(diff, fill_value=0)
            self.dqd.set_gate_voltages(new_gate_voltages)

            positions = self.measure_positions()
            dpos = (positions[0] - current_position[0], positions[1] - current_position[1])
            self.grad_kalman.update(-1*du, dpos, hack=False)


def find_lead_transition(data: np.ndarray, center: float, scan_range: float, npoints: int) -> float:
    if len(data.shape) == 2:
        y = np.mean(data, 0)
    elif len(data.shape) == 1:
        y = data
    else:
        print('data must be a one or two dimensional array!')
        return np.nan

    x = np.linspace(center - scan_range, center + scan_range, npoints)

    n = int(.2e-3/scan_range*npoints)
    for i in range(0, len(y)-n-1):
        y[i] -= y[i+n]

    y_red = y[0:len(y) - n - 1]
    x_red = x[0:len(y) - n - 1]

    y_red = np.absolute(y_red)
    max_index = np.argmax(y_red)

    return x_red[max_index]
