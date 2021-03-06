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

max_distance = 30e-3;

new_point=[args.SB, args.BB, args.T, args.N, args.SA, args.BA, args.RFA, args.RFB, args.SDB1, args.SDB2];

go_back = false;
for i=1:8
	if abs(pretuned_point(i)-new_point(i)) > max_distance
 		go_back = true;
        disp(gatechannels)
        disp(' is ')
        disp(max_distance * 1e3)
        disp('mV away from pretuned!')
	end
end

if (args.SDB2 - sensor_pretuned) > max_distance
    go_back = true;
    disp('SDB2 is')
    disp(max_distance * 1e3)
    disp('mV away from pretuned!')
end
if (args.SDB1 - sensor_gate1_pretuned) > max_distance
    go_back = true;
    disp('SDB1 is ')
    disp(max_distance * 1e3)
    disp('mV away from pretuned!')
end


if go_back
    for j = 1:8
        smset(gatechannels(j),pretuned_point(j))
    end
    smset('SDB2', sensor_pretuned)
    smset('SDB1', sensor_gate1_pretuned)
    error('The dot is too far away from pretuned point on a dot defining gate or 20 mV away from a sensing dot gate!')
end

% str = input('The program wants to set the gates to the values above! [Y/N]','s')
% if strcmp(str,'Y')

smset({smdata.channels(tunedata.gatechan).name, 'SDB1', 'SDB2'},new_point);

% end
vector_actual_point=cell2mat(smget({smdata.channels(tunedata.gatechan).name, 'SDB1', 'SDB2'})); 
actual_point=struct('SB',vector_actual_point(1),'BB',vector_actual_point(2),'T',vector_actual_point(3), ...
	'N',vector_actual_point(4),'SA',vector_actual_point(5),'BA',vector_actual_point(6), 'RFA', vector_actual_point(7), 'RFB', vector_actual_point(8), ...
    'SDB1', vector_actual_point(9), 'SDB2', vector_actual_point(10));
end

