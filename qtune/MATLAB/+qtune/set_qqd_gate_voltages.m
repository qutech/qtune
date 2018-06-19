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

if any(abs(totalDelta) > 20e-3)
  error(['The Program tried to detune a gate by more than 20 mV from starting point!'])
	
elseif any(abs(stepDelta) > 5e-3)
  error(['The Program tried to step a gate by more than 5 mV!'])
	
elseif util.yes_no_input(stdQuestionChangeGateVoltages, [], 'n')
	smset(gateNames, newVoltages)
end

actualGateVoltageValues = smget(gateNames);

actual_gate_voltages = cell2struct(actualGateVoltageValues',gateNames);


end