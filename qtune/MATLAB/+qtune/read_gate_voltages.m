function [ voltages ] = read_gate_voltages( )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
global tunedata
global smdata


gate_voltages = cell2mat(smget({smdata.channels(tunedata.gatechan).name}));
voltages=struct;
voltages.SB=gate_voltages(1);
voltages.BB=gate_voltages(2);
voltages.T=gate_voltages(3);
voltages.N=gate_voltages(4);
voltages.SA=gate_voltages(5);
voltages.BA=gate_voltages(6);
voltages.RFA=gate_voltages(7);
voltages.RFB=gate_voltages(8);
end

