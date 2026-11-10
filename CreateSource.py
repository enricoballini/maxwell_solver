import numpy as np


def CreateSource(constSim):
    fmaxx = constSim.fctr*constSim.PercentageBW /2
    spread=1/(2*constSim.delT*fmaxx)
    tto = constSim.NumOfSidLobes*spread  # here you deterimin the number of side lobes of the sinc
    source = np.zeros(constSim.timeSteps)
    for n in range(1,constSim.timeSteps+1):
        if(n<2*tto):
            sinc = np.sin(np.pi*2*fmaxx*np.single(n+0-tto)*constSim.delT)/(np.pi*2*fmaxx*np.single(n+0-tto)*constSim.delT+1e-14)
            source[n-1]=sinc*np.sin(2*np.pi*(constSim.fctr*np.single(n+0-tto)*constSim.delT))#
        else:
            source[n-1]=0
        source[n-1]= source[n-1]*np.cos(((n-1)/constSim.timeSteps)* np.pi/2)
    return fmaxx,source
