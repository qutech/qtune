function [ data ] = PythonLineScan( args )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

scan = qtune.makePythonLineScan(args.center, args.range, args.N_average, args.ramptime, args.N_points, args.AWGorDecaDAC);
data = smrun(scan, args.file_name);

data=mean(data{1});

end

