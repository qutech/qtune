function [ beta ] = fit_lead_scan( x,data )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here


beta = fitwrap('plinit plfit samefig pause', x, ...tunedata.lead.period/nsamp * (0:nsamp-1), ...
   data, [mean(data), range(data)*sign(data(round(end/4))-data(round(3*end/4))), .2, .2, .05,-1e-3],  @leadfn);



end

%constant shift for the rising flank
% function y = leadfn(beta, x) %beta([offset, prefactor, width<0, width>0])
% %global tunedata; 
% %x0 = tunedata.lead.period/2;
% x0 = x(end/2+1);
% x = mod(x-0.085, 2*x0);
% 
% y = beta(1) + .5 * beta(2) * ((cosh(.5 * (beta(5)+x0)/beta(3)) - exp((.5*(beta(5)+x0)-x)./beta(3)))./sinh(.5*(beta(5)+x0)/beta(3)) .* (x < (beta(5)+x0)) ...
%     -  (cosh(.5 * (beta(5)+x0)/beta(4)) - exp((1.5*(beta(5)+x0)-x)./beta(4)))./sinh(.5*(beta(5)+x0)/beta(4)) .* (x >= (beta(5)+x0)));
% end

%fixed difference - no beta6
% function y = leadfn(beta, x) %beta([offset, prefactor, width<0, width>0])
% %global tunedata; 
% %x0 = tunedata.lead.period/2;
% x0 = x(floor(end/2+1));
% x = mod(x-beta(5), 2*x0);
% 
% y = beta(1) + .5 * beta(2) * ((cosh(.5 * x0/beta(3)) - exp((.5*x0-x)./beta(3)))./sinh(.5*x0/beta(3)) .* (x < x0) ...
%     -  (cosh(.5 * x0/beta(4)) - exp((1.5*x0-x)./beta(4)))./sinh(.5*x0/beta(4)) .* (x >= x0));
% end


%backup
function y = leadfn(beta, x) %beta([offset, prefactor, width<0, width>0])
%global tunedata; 
%x0 = tunedata.lead.period/2;
x0 = x(floor(end/2+1));
x = mod(x-beta(5), 2*x0);

y = beta(1) + .5 * beta(2) * ((cosh(.5 * (beta(6)+x0)/beta(3)) - exp((.5*(beta(6)+x0)-x)./beta(3)))./sinh(.5*(beta(6)+x0)/beta(3)) .* (x < (beta(6)+x0)) ...
    -  (cosh(.5 * (beta(6)+x0)/beta(4)) - exp((1.5*(beta(6)+x0)-x)./beta(4)))./sinh(.5*(beta(6)+x0)/beta(4)) .* (x >= (beta(6)+x0)));
end