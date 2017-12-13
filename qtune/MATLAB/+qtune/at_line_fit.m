function [ output ] = at_line_fit( xdata, ydata )
%Fit the data to extract the tunnel coupling
%   if the fit fails, the status 'failed' will be set to 0 otherwise to 1


    failed=0;
    x=xdata;

    pts=1:(floor(length(x)/8));
    pf1 = polyfit(x(pts),ydata(pts),1);
    pts = floor(0.875*length(x)):length(x);
    pf2 = polyfit(x(pts),ydata(pts),1);
    % Look at the skew of the derivative to guess a sign for the atan.
    dd = diff(ydata-pf1(1)*x);
    % Find the step
    ign=floor(length(x)/25);
    dds=abs(smooth(dd));
    [mm, mi] = max(dds(ign:end-ign));   
    cen=x(mi+ign);
    sh=-(pf2(2)+pf2(1)*x(end))+(pf1(2)+pf1(1)*x(end));
    %tunedata.runs(ind).line = fitwrap('plinit plfit woff', x, data, [pf(2)-pf(1)*(cen), pf(1), 0, .0002, .0002], ...
    %    @(p, x)p(1)+p(2)*(x-p(3))-p(4)*tanh((x-p(3))./p(5)));
    %figure(3);
    %msubplot(333);
    try
        
        lineinfo = qtune.fitwrap('plinit plfit fine samefig', x, ydata, [pf1(2),pf1(1), cen, sh, range(x)/16.0], ... 
        @(p, x)p(1)+p(2)*(x)-p(4)*(1+tanh((x-p(3))./p(5)))/2);   

    catch
        try
            
            %if the fit fails, we set the inter dot coupling manually to a
            %magic value which happens to be the desired coupling width
            lineinfo = qtune.fitwrap('plinit plfit fine samefig', x, ydata, [pf1(2),pf1(1), cen, sh, 190e-6], ... 
            @(p, x)p(1)+p(2)*(x)-p(4)*(1+tanh((x-p(3))./p(5)))/2);


        catch
            try
                
                pts=1:(floor(length(x)/20));
                pf1 = polyfit(x(pts),ydata(pts),1);
                pts = floor(0.95*length(x)):length(x);
                pf2 = polyfit(x(pts),ydata(pts),1);
                % Look at the skew of the derivative to guess a sign for the atan.
                dd = diff(ydata-pf1(1)*x);
                % Find the step
                ign=floor(length(x)/25);
                dds=abs(smooth(dd));
                [mm, mi] = max(dds(ign:end-ign));   
                cen=x(mi+ign);
                sh=-(pf2(2)+pf2(1)*x(end))+(pf1(2)+pf1(1)*x(end));
                lineinfo = qtune.fitwrap('plinit plfit fine samefig', x, ydata, [pf1(2),pf1(1), cen, sh, range(x)/16.0], ... 
                @(p, x)p(1)+p(2)*(x)-p(4)*(1+tanh((x-p(3))./p(5)))/2); 
            catch
                lineinfo = [nan nan nan nan nan];            
                failed=1;
            end
        end
    end
    tc=lineinfo(5);
    output = struct;
		output.tc = tc;
		output.failed = failed;

end

