function [actual_gate_voltages] = set_qqd_gate_voltages(gate_voltages)

global pretuned_qqd_gate_voltages
global pretuned_qqd_sensing_dot_voltages
gate_names = fieldnames(gate_voltages);
n_gates = size(gate_names);
n_gates = n_gates(2);
voltage_values = cell2mat(struct2cell(gate_voltages));
actual_gate_voltages = struct;

for gate = gate_names(1:end)
    if abs(pretuned_qqd_gate_voltages.(gate) - gate_voltages.(gate)) > 70e-3
        qtune.set_qqd_gate_voltages(pretuned_qqd_gate_voltages);
        qtune.set_qqd_sensing_dot_voltages(pretuned_qqd_sensing_dot_voltages);
        error(['The Program tried to detune the gate ' gate ' by more than 70meV!'])
    end
end

smset(gate_names, voltage_values)
actual_gate_voltage_values = smget(gate_names);
for i = 1:n_gates
    actual_gate_voltages.(gate_names(i)) = actual_gate_voltage_values(i); 
end

end