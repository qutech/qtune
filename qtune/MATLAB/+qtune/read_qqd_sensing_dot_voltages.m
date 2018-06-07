function [gate_voltages] = read_qqd_sensing_dot_voltages()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

global tunedata
ri = tunedata.runindex;
nSensors = numel(tunedata.run{ri}.opts.sensors); 
sideGates = cell(1,nSensors);

for ii=1:nSensors
sideGates{ii} = tunedata.run{1}.opts.sensors(ii).gates.s.names;
end
sideGates = sideGates{:};

gateVoltages = smget(gateSide);

gate_voltages = cell2mat(gateVoltages,sideGates);


