% setting up global variables for the matlab Backend


global pretuned_point;
global sensor_pretuned;
global sensor_gate1_pretuned;


pretuned_point = cell2mat(smget({smdata.channels(tunedata.gatechan).name}));
sensor_pretuned = smget('SDB2');
sensor_pretuned = sensor_pretuned{1};
sensor_gate1_pretuned = smget('SDB1');
sensor_gate1_pretuned = sensor_gate1_pretuned{1};

%% setting gates back to pretuned point
global tunedata;
gatechannel=tunedata.gatechan;
for i = 1:8
	smset(gatechannel(i),pretuned_point(i))
end
smset('SDB2', sensor_pretuned)
smset('SDB1', sensor_gate1_pretuned)


cell2mat(smget({smdata.channels(tunedata.gatechan).name})) - pretuned_point