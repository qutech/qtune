function [output] = lead_fit(data)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
failed =0;
data = squeeze(diff(mean(data, 1), 1))';
ydata=data;
samprate = 1e8;
x = (0:length(ydata)-1)./samprate * 1e6;      
x_red=zeros(1,398);
x_red(1:8)=x(1:8);
x_red(9:208)=x(10:209);  
x_red(209:398)=x(211:400);          
y_red=zeros(1,398);
y_red(1:8)=data(1:8);
y_red(9:208)=data(10:209);  
y_red(209:398)=data(211:400);          
try
     leadinfo_B(:) = qtune.fit_lead_scan( x_red,y_red) ;
catch
     leadinfo_B(:) = [nan nan nan nan nan nan];
     failed = 1;
end
t_rise=leadinfo_B(3);
t_fall=leadinfo_B(4);
output = struct;
output.t_rise = t_rise;
output.t_fall = t_fall;
output.failed = failed;
end

