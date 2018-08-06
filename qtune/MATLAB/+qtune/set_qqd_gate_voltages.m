function [actual_gate_voltages] = set_qqd_gate_voltages(gate_voltages)

global pretuned_qqd_gate_voltages

stdQuestionChangeGateVoltages = 'Change gate voltages by these amounts?';

gateNames = fieldnames(gate_voltages);

newVoltages = cell2mat(struct2cell(gate_voltages));
oldVoltages = cell2mat(struct2cell(qtune.read_qqd_gate_voltages(gateNames)));
pretunedVoltages = cellfun(@(x)pretuned_qqd_gate_voltages.(x),gateNames);


stepDelta = oldVoltages - newVoltages;
totalDelta = pretunedVoltages - newVoltages;

util.disp_section('Step Voltage Delta')
tune.disp_gate_voltages(gateNames, stepDelta);
util.disp_section('Total Voltage Delta')
tune.disp_gate_voltages(gateNames, totalDelta);

% if any(abs(totalDelta) > 70e-3)
%   error(['The Program tried to detune a gate by more than 70 mV from starting point!'])
% 	
% elseif any(abs(stepDelta) > 30e-3)
%   error(['The Program tried to step a gate by more than 30 mV!'])
% 	
% else %if util.yes_no_input(stdQuestionChangeGateVoltages, [], 'n')
% 	smset(gateNames, newVoltages)
% end

% if any(cellfun(@(x)(strcmp(x, 'IV_ref')), gateNames))
% 	error('Do you really want to use IV_ref?');
% end

if any(newVoltages > 0)
  error(['The Program tried to detune a gate higher than 0V!'])
	
elseif any(newVoltages < -1.3)
  error(['The Program tried to detune a gate lower than -1.3V!'])
	
elseif any(abs(stepDelta) > 10e-3)
  error(['The Program tried to step a gate by more than 10 mV!'])	
	
else %if util.yes_no_input(stdQuestionChangeGateVoltages, [], 'n')
	smset(gateNames, newVoltages)
end

actualGateVoltageValues = smget(gateNames);

actual_gate_voltages = cell2struct(actualGateVoltageValues',gateNames);


end