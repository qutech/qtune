function [output] = change_sensing_dot_gates(input)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

global sensor_pretuned
global sensor_gate1_pretuned

if abs(input.gate1) > 21e-3
error('Program wants to change SDB1 by more than 20meV. Thats more than the previous scan range!')
end
if abs(input.gate2) > 21e-3
error('Program wants to change SDB2 by more than 20meV. Thats more than the previous scan range!')
end
sdb1_voltage = smget('SDB1');
sdb1_voltage = sdb1_voltage{1};
sdb2_voltage = smget('SDB2');
sdb2_voltage = sdb2_voltage{1};

if abs(sdb1_voltage + input.gate1 - sensor_gate1_pretuned) > 70e-3
	error('Program wants to tune SDB1 more than 70meV away from pretuned point!')
elseif abs(sdb2_voltage + input.gate2 - sensor_pretuned) > 70e-3
	error('Program wants to tune SDB2 more than 70meV away from pretuned point!')
end
sminc('SDB1', input.gate1)
sminc('SDB2', input.gate2)

output= 'empty';
end

