function scan = makePythonQPCScan2D(rng)
global tunedata
global smdata

if nargin < 1
    rng = [-10e-3 10e-3];
end
center = smget({'SDB2', 'SDB1'});

inst_index = sminstlookup('ATS9440Python');
config = AlazarDefaultSettings(inst_index);

masks = {};
masks{1}.type = 'Periodic Mask';
masks{1}.begin = 0;
masks{1}.end = 500000;
masks{1}.period = 500000;
masks{1}.channel = 'A';

operations = {};
operations{1}.type = 'DS';
operations{1}.mask = 1;

config.masks = masks;
config.operations = operations;
config.total_record_size = masks{1}.period * 104; % npoints from loop 1, also has to be a multiple of 256;

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
scan.cleanupfn(2).args = {@smset, {'SDB1'}, center{2}};
scan.cleanupfn(3).fn =  @smaconfigwrap;
scan.cleanupfn(3).args = {@smset, {'SDB2'}, center{1}};

scan.loops(1).setchan = 'SDB2';
scan.loops(1).ramptime = -0.005;
scan.loops(1).npoints = 104;
scan.loops(1).rng = center{1} + rng;
scan.loops(1).trigfn.fn = @smatrigAWG;
scan.loops(1).trigfn.args = {sminstlookup('AWG5000')};

scan.loops(2).setchan = 'SDB1';
scan.loops(2).getchan = {'ATS1'};
scan.loops(2).ramptime = 0.05;
scan.loops(2).npoints = 20;
scan.loops(2).rng = center{2} + rng;
scan.loops(2).prefn(1).fn = @smaconfigwrap;
scan.loops(2).prefn(1).args = {smdata.inst(inst_index).cntrlfn,[inst_index 0 4]};
end
