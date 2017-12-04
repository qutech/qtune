function scan = make_lead_scan(ch)

% TODO: replace
% global tunedata
% measp = tunedata.measp;

switch ch
	case 'A'
		pulse = 'sqrC';
		measp = 1e-3 * [-4 -4; 0.15 -3];
	case  'C'
		pulse = 'sqrC';
		measp = 1e-3 * [-4 -4; 0.15 -3];
	case 'B'
		pulse = 'sqrD';
		measp = 1e-3 * [-0.5 2; -4 -4];
	case 'D'
		pulse = 'sqrD';
		measp = 1e-3 * [-0.5 2; -4 -4];
	otherwise
		error('what kind of channel is that? %s', ch);
end

scan = confSeqPython(pulse, 2048*10, 1, 'genMask', true, 'operations', {'RSA'});
scan.loops(1).setchan = {'RFB', 'RFA'};
scan.loops(1).ramptime = [];
scan.loops(1).npoints = [];
scan.loops(1).rng = [1 2];
scan.loops(1).trafofn = {atune.get_at_trafo(1, measp), atune.get_at_trafo(2, measp)};
scan.loops(1).postfn.fn = @postpause;
scan.loops(1).postfn.args = {0.2};

scan.loops(2).setchan = 'count';
scan.loops(2).ramptime = [];
scan.loops(2).npoints = 10;


% reset rf gates after scan
scan.cleanupfn(1).fn = @smaconfigwrap;
scan.cleanupfn(1).args = {@smset, {'RFB', 'RFA',} 0};

% set AWG to 0 pulse
scan.cleanupfn(2).fn = @smaconfigwrap;
scan.cleanupfn(2).args = {@smset, {'PulseLine'}, 1};