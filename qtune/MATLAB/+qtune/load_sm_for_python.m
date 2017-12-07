function [ output ] = load_sm_for_python( filename )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
s=load(filename);
data=s.data{1};
minrange=s.scan.loops(1).rng(1);
maxrange=s.scan.loops(1).rng(2);
center=(minrange+maxrange)/2;
range=(maxrange-minrange)/2;
npoints=s.scan.loops(1).npoints;
output=struct;
output.data=data;
output.center=center;
output.range=range;
output.npoints=npoints;
end

