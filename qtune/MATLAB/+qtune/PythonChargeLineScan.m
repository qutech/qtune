function [ data ] = PythonChargeLineScan( args )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

scan = makePythonChargeLineScan(args.center, args.range, args.gate, args.npoints, args.ramptime, args.N_average, args.AWGorDecaDAC);
data = smrun(scan, file_name);

data=mean(data{1});

end

