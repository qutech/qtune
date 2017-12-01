function [ data ] = PythonChargeLineScan( args )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

scan = qtune.makePythonChargeLineScan(args.center, args.range, args.gate, args.N_points, args.ramptime, args.N_average, args.AWGorDecaDAC);
data = smrun(scan, args.file_name);

data=mean(data{1});

end

