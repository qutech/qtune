function [fitresult] = load_fit(data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    try
        x = data(end,2:end);
        data = data(1:end-1,2:end);
        beta = fitwrap('plinit plfit samefig', x, ...
            data, [min(data), range(data), 10],  @(p, x)p(1)+p(2)*exp(-x./p(3)));
        fitresult.parameter_time_load = beta(3);
        fitresult.failed = 0;
    catch
        fitresult.parameter_time_load = nan;
        fitresult.failed = 1;
    end
    
end

