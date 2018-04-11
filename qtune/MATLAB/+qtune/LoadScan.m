function [data] = LoadScan(args)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    global tunedata
    scan = tunedata.load.scanpython;
    scan.loops(1).prefn(2).args{1} = awgseqind(tunedata.load.plsgrp);
    data = smrun(scan, args.file_name);
    x =  [-1 round(logspace(0,2,49)*5)]; % from plsgrp.m
    if any(isnan(data{1}(:))); return; end
    if ndims(data{1}) == 3
        data = diff(squeeze(mean(data{1})));
    else
        data = -mean(data{1},1);
    end
    figure(3667);
    try
        fitwrap('plinit plfit samefig', x, ...
            data, [min(data), range(data), 200],  @(p, x)p(1)+p(2)*exp(-x./p(3)))
    end
    figure(3666)
    data = cat(1, data, x);
end

