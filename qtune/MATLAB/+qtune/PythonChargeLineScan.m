function [ data ] = PythonChargeLineScan( args )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
voltage_on_sweeping_gate = smget(args.gate);
scan = qtune.makePythonChargeLineScan(args.center, args.range, args.gate, args.N_points, args.ramptime, args.N_average, args.AWGorDecaDAC);
data = smrun(scan, args.file_name);
smset(args.gate, voltage_on_sweeping_gate{1});
%data=mean(data{1});
data = data{1};
figure(3668)
plot(mean(data))
end

