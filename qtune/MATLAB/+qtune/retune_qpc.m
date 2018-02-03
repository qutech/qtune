function [ new_qpc_position ] = retune_qpc( input_args )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
global sensor_pretuned
global pretuned_point
global tunedata
gatechannel= tunedata.gatechan;

current_qpc_position = input_args.qpc_position;
range = input_args.tuning_range;
file_name=input_args.file_name;
scan = qtune.makePythonChargeLineScan(current_qpc_position, range, 'SDB2', 1280, .0005, 1, 'DecaDAC');
scan.figure = 32939;
scan.cleanupfn(2).fn =  @smaconfigwrap;
scan.cleanupfn(2).args = {@smset, {'SDB2'}, current_qpc_position};
qpc_line = smrun(scan, [file_name, '_',  'scan_qpc']);

smoothed = smooth(diff(smooth(qpc_line{1}, 200)), 30);
smoothed = smoothed(50:end-50);
x = current_qpc_position + linspace(-5e-3, 5e-3, 1280);
x = x(50:end-50);

[~, idx] = min(smoothed);
figure(32939);
subplot(122);
plot(x(2:end), smoothed);
hold on;
scatter(x(idx+1), smoothed(idx), 'ro');
if abs(x(idx)-sensor_pretuned)> 50e-3
	for i = 1:8
      smset(gatechannel(i),pretuned_point(i))
  end
	smset('SDB2', sensor_pretuned);
	error('The sensing dot is being tuned more than 50mV away from its original position!')
%     disp('The sensing dot is being tuned more than 50mV away from its original position!')
%         while true
%         decision = input('You have now the possibility to shut down the tuning by typing <<stop>> or continue by typing <<continue>>.');
%         if decision == 'stop'
%             error('The tuning has been shut down')
%         elseif decision == 'continue'
%             smset('SDB2', x(idx));
%             continue
%         end
%         disp('This was not a valid input!')
%         end
else
	smset('SDB2', x(idx));
end
new_qpc_position = struct('qpc', x(idx));


end

