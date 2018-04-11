function [data] = PythonQPCScan2D(args)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
if nargin < 1
    rng = [-10e-3 10e-3];
else
    rng = [-args.range args.range];
end

scan = makePythonQPCScan2D(rng);
data = smrun(scan, args.file_name);
data = data{1};
end

