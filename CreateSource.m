function [fmaxx,source]=CreateSource(constSim)
fmaxx = constSim.fctr*constSim.PercentageBW /2;
spread=1/(2*constSim.delT*fmaxx);
tto = constSim.NumOfSidLobes*spread;  % here you deterimin the number of side lobes of the sinc
source(constSim.timeSteps) = 0;
for n = 1:constSim.timeSteps
    if(n<2*tto)
    sinc = sin(pi*2*fmaxx*single(n+0-tto)*constSim.delT)/(pi*2*fmaxx*single(n+0-tto)*constSim.delT+1e-14);
    source(n)=sinc*sin(2*pi*(constSim.fctr*single(n+0-tto)*constSim.delT));%
    else
        source(n)=0;
    end
    source(n)= source(n)*cos(((n-1)/constSim.timeSteps)* pi/2);
end
end