function [actual_gate_voltages] = set_qqd_gate_voltages_abs_limit(gate_voltages)

global pretuned_qqd_gate_voltages


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

if any(newVoltages > -.4)
  error(['The Program tried to detune a gate higher than -0.4V!'])
	
elseif any(stepDelta < -1.3)
  error(['The Program tried to detune a gate lower than -1.3V!'])
	
else %if util.yes_no_input(stdQuestionChangeGateVoltages, [], 'n')
	smset(gateNames, newVoltages)
end

actualGateVoltageValues = smget(gateNames);

actual_gate_voltages = cell2struct(actualGateVoltageValues',gateNames);


end