function beta1 = fitwrap(ctrl, x, y, beta0, model, mask, weights)
% beta1 = fitwrap(ctrl, x, y, beta0, model, mask)
% ctrl: plinit, plfit, woff, nofit, pause, samefig, fine
% y = model(beta, x);
%plinit plots the initial guess
%plfit plots the fit
%woff turns off warnings
%no fit does not fit the data
%pause will pause after each plot
%samefig uses current figure. not 500.
% beta0 can be a vector, array, cell array with initialization function handles
% or initial value vectors, or a struct array with fields fn and args, and optionally vals.
% Vinite vals entries override the return of the initialization function.
% Initialization functions are called as beta0 = fn(x, y, args).
% data should be in columns.
% (c) 2010 Hendrik Bluhm.  Please see LICENSE and COPYRIGHT information in plssetup.m.
%modified atune Version by Teske

% This allows the model to be something like '@(p,x) x*p(1)', which
% overcomes issues with saving functions to disk.
if ischar(model)
  model=str2func(model);
end

n = size(y, 1);
if size(y,2) == 1
  fprintf('X is %d x %d, Y is %d x %d\n',size(x,1),size(x,2),size(y,1),size(y,2));
  error('It is unlikely you wanted to fit a single data point.  Try transposing things');
end

if size(x,2) ~= size(y,2)
  fprintf('X is %d x %d, Y is %d x %d\n',size(x,1),size(x,2),size(y,1),size(y,2));
  warning('This is probably not what you want.');
end

if size(x, 1) == 1
  x = repmat(x, n, 1);
end

if isa(beta0, 'function_handle')
  beta0 = {beta0};
end

if isreal(beta0) && size(beta0, 1) == 1 || length(beta0) == 1
  beta0 = repmat(beta0, n, 1);
end

if nargin >= 6
  mask = logical(mask);
end

if strfind(ctrl, 'fine')
  options=optimset('TolX',1e-20,'TolFun',1e-20);
else
  options=optimset();
end

if strfind(ctrl, 'woff')
  ws(1) = warning('query', 'stats:nlinfit:IllConditionedJacobian');
  ws(2) = warning('query', 'stats:nlinfit:IterationLimitExceeded');
  ws2 = ws;
  [ws2.state] = deal('off');
  warning(ws2);
end

for i = 1:n
  if ~isempty(strfind(ctrl, 'pl')) && isempty(strfind(ctrl,'samefig'))
    figure(500);
    clf;
    hold on;
  end
  
  if iscell(beta0)
    if isreal(beta0{i})
      beta2 = beta0{i};
    else
      beta2 = beta0{i}(x(i, :), y(i, :));
    end
  elseif isstruct(beta0)
    if ~isempty(beta0(i).fn)
      beta2 = beta0(i).fn(x(i, :), y(i, :), beta0(i).args{:});
      if isfield(beta0, 'vals');
        beta2(isfinite(beta0(i).vals)) = beta0(i).vals(isfinite(beta0(i).vals));
      end
    else
      beta2 = beta0(i).vals;
    end
  else
    beta2 = beta0(i, :);
  end
  
  if i == 1
    nfp = length(beta2);
    
    if nargin < 6 || isempty(mask)
      mask = true(1, nfp);
    end
    beta1 = zeros(n, nfp);
  end
  
  beta1(i, :) = beta2;
  
  if ~isempty(strfind(ctrl, 'plinit'))
    plot(x(i, :), y(i, :), '.-', x(i, :), model(beta1(i, :), x(i, :)), 'k--');
  end
  
  if isempty(strfind(ctrl, 'nofit'))
    if nargin >= 7
      beta1(i, mask) = nlinfit(x(i, :), y(i, :), @fitfn, beta1(i, mask), options, 'Weights', weights);
    else
      beta1(i, mask) = nlinfit(x(i, :), y(i, :), @fitfn, beta1(i, mask), options);
    end;
  end
  
  if ~isempty(strfind(ctrl, 'plfit'))
    plot(x(i, :), y(i, :), '.-', x(i, :), model(beta1(i, :), x(i, :)), 'r');
  end
  
  if ~isempty(strfind(ctrl, 'pause')) && (i < n)
    pause
  end
  
  if ~isempty(strfind(ctrl,'resid'))
    f=gcf;
    figure(501);
    if isempty(strfind(ctrl,'samefig'))
      clf;
    else
      hold on;
    end
    plot(x(i,:),y(i,:)-model(beta1(i,:),x(i,:)),'rx-');
    figure(f);
  end
end


if strfind(ctrl, 'woff')
  warning(ws);
end


  function y = fitfn(beta, x)
    beta([find(mask), find(~mask)]) = [beta, beta2(~mask)];
    y = model(beta, x);
  end

end


function y = leadfn(beta, x) %beta([offset, prefactor, width<0, width>0])
%global tunedata; 
%x0 = tunedata.lead.period/2;
x0 = x(end/2+1);
x = mod(x-beta(5), 2*x0);

y = beta(1) + .5 * beta(2) * ((cosh(.5 * x0/beta(3)) - exp((.5*x0-x)./beta(3)))./sinh(.5*x0/beta(3)) .* (x < x0) ...
    - (cosh(.5 * x0/beta(4)) - exp((1.5*x0-x)./beta(4)))./sinh(.5*x0/beta(4)) .* (x >= x0));
end
