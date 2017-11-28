# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:39:28 2017

@author: teske

The chargediagram will be implemented as class.
"""

class chrg_diag:
    
    position_lead_A=0
    position_lead_B=0
    gradient=[[0,0],[0,0]]
    charge_line_scan_lead_A = Measurement('line_scan', center=0, range=3e-3, 
                                          gate='RFA', N_points=1280, 
                                          ramptime=.0005, 
                                          N_average=3, 
                                          AWGorDecaDAC='AWG', 
                                          file_name=str(now))
    
    charge_line_scan_lead_A = Measurement('line_scan', center=0, range=3e-3, 
                                          gate='RFB', N_points=1280, 
                                          ramptime=.0005, 
                                          N_average=3, 
                                          AWGorDecaDAC='AWG', 
                                          file_name=str(now))
    
    def __init__(self, exp: Experiment, charge_line_scan_lead_A=None: Measurement, charge_line_scan_lead_B=None: Measurement):
        self.experiment=exp
        
        if charge_line_scan_lead_A is not None:
            self.charge_line_scan_lead_A=charge_line_scan_lead_A
        
        if charge_line_scan_lead_B is not None:
            self.charge_line_scan_lead_B=charge_line_scan_lead_B
    
    def measure_positions(self):        
        data_A=self.exp.measure(self.charge_line_scan_lead_A)
        self.position_lead_A=atune.at_find_lead_trans(data_A,
                                                      charge_line_scan_lead_A["center"], 
                                                      charge_line_scan_A["range"], 
                                                      charge_line_scan_A["N_points"])
        
        data_B=self.exp.measure(self.charge_line_scan_lead_B)
        self.position_lead_B=atune.at_find_lead_trans(data_B,
                                                      charge_line_scan_lead_B["center"], 
                                                      charge_line_scan_B["range"], 
                                                      charge_line_scan_B["N_points"])
        
        

    def calculate_gradient(self):
        current_gate_voltages=self.experiment.gate_voltages
        BA_inc=current_gate_voltages
        BA_inc["BA"] = BA_inc["BA"] + 1e-3
        BA_red=current_gate_voltages
        BA_red["BA"] = BA_inc["BA"] - 1e-3
        BB_inc=current_gate_voltages
        BB_inc["BB"] = BA_inc["BB"] + 1e-3
        BB_red=current_gate_voltages
        BB_red["BB"] = BA_inc["BB"] - 1e-3
        
        self.experiment.set_gate(BA_inc)
        self.measure_positions
        pos_A_BA_inc=self.position_lead_A 
        pos_B_BA_inc=self.position_lead_B        
        
        self.experiment.set_gate(BA_red)
        self.measure_positions
        pos_A_BA_red=self.position_lead_A
        pos_B_BA_red=self.position_lead_B  
        
        self.experiment.set_gate(BB_inc)
        self.measure_positions
        pos_A_BB_inc=self.position_lead_A 
        pos_B_BB_inc=self.position_lead_B        
        
        self.experiment.set_gate(BB_red)
        self.measure_positions
        pos_A_BB_red=self.position_lead_A
        pos_B_BB_red=self.position_lead_B    
        
        self.gradient[0,0]=(pos_A_BA_inc-pos_A_BA_red)/2e-3
        self.gradient[0,1]=(pos_A_BB_inc-pos_A_BB_red)/2e-3
        self.gradient[1,0]=(pos_B_BA_inc-pos_B_BA_red)/2e-3
        self.gradient[1,1]=(pos_B_BB_inc-pos_B_BB_red)/2e-3

        
        
    def center_diagram(self):

    
