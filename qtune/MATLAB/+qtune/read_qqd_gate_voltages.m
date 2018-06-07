function [gate_voltages] = read_qqd_gate_voltages(gates)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
global tunedata
if nargin<1
  ri = tunedata.runIndex;
  dcGates = tunedata.run{ri}.opts.gates.names;
  rfGates = tunedata.run{ri}.opts.rfGates.names;
  gates = [dcGates,rfGates];
elseif ~iscell(gates)
  gates={gates};
end

gateVoltages = smget(gates);

gate_voltages = cell2struct(gateVoltages',gates');



