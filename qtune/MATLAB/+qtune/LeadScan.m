function [ data ] = LeadScan( args )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
if strcmp(args.AWGorDecaDAC,'DecaDAC')
scan = qtune.make_lead_scan(args.gate);
elseif strcmp(args.AWGorDecaDAC,'AWG')
    warning('The charge line scan via AWG has not been implemented yet! You will recieve a DecaDAC scan!!!')
    scan = qtune.make_lead_scan(args.gate);
else
    warning('the variable AWGorDecaDAC must be AWG or DecaDAC! You will recieve a DecaDAC scan!!!')
    scan = qtune.make_lead_scan(args.gate);
end
scan.figure = 1233126;
data = smrun(scan, args.file_name);
data = diff(squeeze(mean(data{1})));
end

