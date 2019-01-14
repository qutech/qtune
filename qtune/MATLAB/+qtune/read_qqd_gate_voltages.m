function [gate_voltages] = read_qqd_gate_voltages(gates)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
global tunedata
if nargin<1
	% error('this might return gate names in a differnt order since struct2cell uses the struct order. best use orderfields or something similar first');
  ri = tunedata.runIndex;
  dcGates = tunedata.run{ri}.opts.gates.names;
  %rfGates = tunedata.run{ri}.opts.rfGates.names;
  %gates = [dcGates,rfGates];
	gates = dcGates;
elseif ~iscell(gates)
  gates={gates};
end

gateVoltages = smget(gates);

gate_voltages = cell2struct(gateVoltages',gates');



