function [ trans_pos ] = at_find_lead_trans( filename_or_data, center, range, npoints )
%This function finds the lead transition in a the data for a lead charge
%scan. Author: Teske
%   The function takes the filename as string as input parameter. A high
%   pass filter is applied to eliminate slow trends in the data. Then a
%   median filter smoothens the noise. Afterwards the data is
%   differentiated, the absolute value is taken and with a second median
%   filter the narrow peaks are supressed. The output is the voltage at
%   which the lead transition occurs. A plot demonstrates the peaks found
%   in the data.
%   Update: The data can be the input argument instead of the filename.
if ischar(filename_or_data)
	experimental_data=load(filename_or_data);


	min_range=experimental_data.scan.loops(1).rng(1);
	max_range=experimental_data.scan.loops(1).rng(2);
	npoints=experimental_data.scan.loops(1).npoints;

	ydata=experimental_data.data{1,1}(1,:);
else
	min_range=center-range;
	max_range=center+range;
	if iscell(filename_or_data)
		ydata=filename_or_data{1,1}(1,:);
	else
		ydata=filename_or_data(1,:);
	end
end

xdata=linspace(min_range,max_range,npoints);


T=1;%(max_range-min_range)/npoints; % distance of data points is one since we plot one dimensional data
tau=1e4; % magical number that promises a good high pass filter
a=T/tau;
n=floor(.4e-3/(max_range-min_range)*npoints);
%n=50;
% figure(3)
% plot(xdata,ydata);

% ydata=filter([1-a a-1],[1 a-1],ydata); % high pass filter
% figure(4)
% plot(xdata,ydata);
ydata=medfilt1(ydata,31); % median filter
% figure(5)
% plot(xdata,ydata);
ydata=atune.at_diff(ydata,n); % we are trying to find the step (peak in the differential)
xdata=xdata(1+floor(n/2):end-(n-floor(n/2)));
% figure(6)
% plot(xdata,ydata);
ydata=abs(ydata); %take absolute value, since we are searching for a negative peak
% figure(7)
% plot(xdata,ydata);
ydata=smooth(ydata,11);
%ydata=medfilt1(ydata,7); % median filter
% figure(8)
% plot(xdata,ydata);


% figure(1);
%  ax1 = subplot(2,1,1);
%  ax2 = subplot(2,1,2);


%  plot(ax1,xdata,ydata);
%  plot(ax2,xdata,smooth(ydata,9));

%[pks1,locs1,w1,p1]=findpeaks(ydata);
[pks2,locs2,w2,p2]=findpeaks(smooth(ydata,11));

% plot(ax1,p1);
% plot(ax2,p2);

% figure(2);
% 
% plot(xdata,ydata); % we have only npoints-1 in ydata after differentiating

%plot(xdata(2:end),smooth(ydata)); % we have only npoints-1 in ydata after differentiating


[M,I]=max(pks2);%We take the peak with the largest value 
I=locs2(I);

%[M,I,w,p]=findpeaks(ydata);

%plot(w);
%plot(p);

trans_pos = xdata(I);
end

