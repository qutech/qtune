function [ data ] = PythonChargeLineScan( center, range, gate, ramptime, N_average, AWGorDecaDAC, file_name )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

scan = makePythonChargeLineScan(center, range, gate, npoints, ramptime, N_average, AWGorDecaDAC);
data = smrun(scan, file_name);

data=mean(data{1});

end

