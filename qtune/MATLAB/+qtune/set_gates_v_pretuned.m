function [ actual_point ] = set_gates_v_pretuned( args )
%setting the real gates to new values and read them out to correct numeric
%inaccuracy
%   Detailed explanation goes here
global tunedata
global pretuned_point
global sensor_pretuned
global sensor_gate1_pretuned
global smdata
gatechannels=tunedata.gatechan;

% disp(pretuned_point(1:6)'-new_point);
% 'The gates are changed by the values above away from the pretuned point to the absolute value below!'
% new_point

new_point=[args.SB, args.BB, args.T, args.N, args.SA, args.BA, args.RFA, args.RFB];

for i=1:8
	if abs(pretuned_point(i)-new_point(i)) > 100e-3
 		for j = 1:8
 			smset(gatechannels(j),pretuned_point(j))
		end
		smset('SDB2', sensor_pretuned)
		smset('SDB1', sensor_gate1_pretuned)
		error('emergency: the dot is 100mV away from pretuned point!')
	end
end

% str = input('The program wants to set the gates to the values above! [Y/N]','s')
% if strcmp(str,'Y')
    for i=1:8
        smset(gatechannels(i),new_point(i));
    end
% end
vector_actual_point=cell2mat(smget({smdata.channels(tunedata.gatechan).name})); 
vector_actual_point=vector_actual_point;
actual_point=struct('SB',vector_actual_point(1),'BB',vector_actual_point(2),'T',vector_actual_point(3), ...
	'N',vector_actual_point(4),'SA',vector_actual_point(5),'BA',vector_actual_point(6), 'RFA', vector_actual_point(7), 'RFB', vector_actual_point(8));
end

