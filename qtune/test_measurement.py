# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:40:43 2017

@author: teske
"""

import matlab.engine
eng = matlab.engine.start_matlab

test_matlab_instance = 'insert matlab instance' 

test_DQD = PrototypeDQD(test_matlab_instance)

#test_DQD.read_gate_voltages

#test_DQD.set_gate_voltages(test_DQD.gate_voltages)

#first scan=test_DQD.measure(default_line_scan)