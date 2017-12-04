function [ output_args ] = readout_qpc( )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
qpc_voltage = smget('SDB2');
output_args = struct;
output_args.qpc = qpc_voltage;

end

