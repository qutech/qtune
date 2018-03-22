function [gate_voltages] = read_qqd_gate_voltages()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
gates = ["SA", "SD", "PA", "PB", "PC", "PD", "NAB", "NBC", "NCD", "TAB", "TBC", "TCD"];
gate_voltages_list = smget(gates);
n_gates = size(gates);
n_gates = n_gates(2);
gate_voltages = struct;
for i = 1:n_gates
    gate_voltage = gate_voltages_list(i);
    gate_voltages.(gates(i)) = gate_voltage;
end
end

