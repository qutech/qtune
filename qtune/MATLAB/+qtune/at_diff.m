function [ array ] = at_diff( array, n )
%computes finite differences with a step size of n
%   Detailed explanation goes here
for i = 1 : length(array)-n
    array(i)=array(i+n)-array(i);
end
array=array(1:length(array)-n);
end

