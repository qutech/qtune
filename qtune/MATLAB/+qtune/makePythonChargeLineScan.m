function scan = makePythonChargeLineScan(center, range, gate, npoints, ramptime, numb_rep, AWGorDecaDAC)
% global tunedata
global smdata

%samplerate is hardcoded 100Mhz
samplerate=1e8;

%%%% default values
% npoints=1280; points for the scan
% ramptime=0.0005; time per point
% numb_rep = 3; number of repetitions
if strcmp(AWGorDecaDAC,'DecaDAC')

inst_index = sminstlookup('ATS9440Python');
config = AlazarDefaultSettings(inst_index);

masks = {};
masks{1}.type = 'Periodic Mask';
masks{1}.begin = 0;
% masks{1}.end = 50000;
% masks{1}.period = 50000;
masks{1}.end = ramptime*samplerate;
masks{1}.period = ramptime*samplerate;
masks{1}.channel = 'A';

operations = {};
operations{1}.type = 'DS';
operations{1}.mask = 1;

config.masks = masks;
config.operations = operations;

if mod(masks{1}.period * npoints,256) ~= 0
    warning('The total record size is ramptime * samplerate * npoints and must be a multiple of 256!')
end
config.total_record_size = masks{1}.period * npoints; % npoints from loop 1, also has to be a multiple of 256;

scan.consts.setchan = 'PulseLine';
scan.consts.val = 1;

scan.saveloop = 2;
scan.disp(1).loop = 2;
scan.disp(1).channel = 1;
scan.disp(1).dim = 1;
scan.disp(2).loop = 2;
scan.disp(2).channel = 1;
scan.disp(2).dim = 2;

scan.configfn(1).fn = @smaconfigwrap;
scan.configfn(1).args = {smdata.inst(inst_index).cntrlfn [inst_index 0 99] [] [] config};
scan.configfn(2).fn = @smaconfigwrap;
scan.configfn(2).args = {smdata.inst(inst_index).cntrlfn,[inst_index 0 5]};
scan.cleanupfn(1).fn =  @smaconfigwrap;
scan.cleanupfn(1).args = {@smset, {'PulseLine'}, 1};
scan.cleanupfn(2).fn =  @smaconfigwrap;
scan.cleanupfn(2).args = {@smset, {'RFA','RFB'}, 0};

scan.loops(1).setchan = gate;
% scan.loops(1).ramptime = -0.0013;
% scan.loops(1).ramptime = -0.0005;
scan.loops(1).ramptime = -1*ramptime;
scan.loops(1).npoints = npoints;
scan.loops(1).rng = center + range * [-1 1];
scan.loops(1).trigfn.fn = @smatrigAWG;
scan.loops(1).trigfn.args = {sminstlookup('AWG5000')};

scan.loops(2).setchan = 'count';
scan.loops(2).getchan = {'ATS1'};
% scan.loops(2).ramptime = 0.1;
scan.loops(2).npoints = numb_rep;
scan.loops(2).rng = [];
scan.loops(2).prefn(1).fn = @smaconfigwrap;
scan.loops(2).prefn(1).args = {smdata.inst(inst_index).cntrlfn,[inst_index 0 4]};

elseif strcmp(AWGorDecaDAC,'AWG')
    warning('The charge line scan via AWG has not been implemented yet! You will recieve a DecaDAC scan!!!')
    scan = qtune.makePythonChargeLineScan(center, range, gate, npoints, ramptime, numb_rep, 'DecaDAC');
else
    warning('the variable AWGorDecaDAC must be AWG or DecaDAC! You will recieve a DecaDAC scan!!!')
    scan = qtune.makePythonChargeLineScan(center, range, gate, npoints, ramptime, numb_rep, 'DecaDAC');
end
end