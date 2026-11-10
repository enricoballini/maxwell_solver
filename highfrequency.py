import math
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from CreateSource import CreateSource
from src.simFDTD import simFDTD, reset_gpuDevice

# Python translation of highfrequency.m. Indices are 0-based: MATLAB index k -> k-1, range a:b -> a-1:b.
# The local functions at the end of highfrequency.m are here at the top, because Python needs them
# defined before the script uses them.


def FDTD_solver(P,timeSteps,FullGPU):
    for i in range(0,timeSteps):
        simFDTD('UpdateH_All',P)
        simFDTD('UpdateH_PML',P)
        simFDTD('UpdateE_All',P)
        simFDTD('UpdateE_PML',P)

def FDTD_solverRe(P,timeSteps,FullGPU):
    for i in range(0,timeSteps):
        simFDTD('UpdateH_All',P)
        #UpdateH_PlaneWave(P,TypeInt([i,1]))
        simFDTD('UpdateH_PlaneWave',P,TypeInt([i,2]))
        #UpdateH_PlaneWave(P,TypeInt([i,3]))

        simFDTD('UpdateH_PML',P)
        simFDTD('UpdateE_All',P)
        #UpdateE_PlaneWave(P,TypeInt([i,1]))
        simFDTD('UpdateE_PlaneWave',P,TypeInt([i,2]))
        #UpdateE_PlaneWave(P,TypeInt([i,3]))

        simFDTD('UpdateE_PML',P)





def TypeInt(x):
    x=np.int32(x)
    return x
def TypeReal(x):
    x=np.single(x)
    return x


reset_gpuDevice()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# clear
plt.close('all')



sys.path[0:0] = [p for p,_,_ in os.walk(".") if "venv" not in p.split(os.sep)] + [p for p,_,_ in os.walk("../Functions")]  # addpath(genpath("."),genpath("../Functions")), skipping venv: its folders would shadow Python modules
###optimization parameters initialization
clight=299792458
mu0=4*np.pi*1e-7
eps0=1/clight**2/mu0
fctr=5.5e9
delX=0.2104/2/10**3
delT=0.99*delX/(math.sqrt(3)*clight)
timeSteps=20000
distancePML=10
NumOfSidLobes=2
PercentageBW =8e9/fctr
constSim = SimpleNamespace()
constSim.delT=delT
constSim.fctr=fctr
constSim.timeSteps=timeSteps
constSim.NumOfSidLobes = NumOfSidLobes
constSim.PercentageBW =PercentageBW
fmaxx,constSim.source=CreateSource(constSim)

plt.figure()
plt.plot(constSim.source, linewidth=2)
plt.grid(True)




####save optimization parameters
constSim.delX=delX
constSim.delT=delT
constSim.fctr=fctr
constSim.timeSteps=timeSteps

constSim.Dim=[230,260,30]
numAnts=0
####save path
flag_plot=1

#--------------------project_FDTD_3D-----------------------
pml_size=10
freqRes = [constSim.fctr, delX, delT]
TimeDim = [timeSteps, constSim.Dim[0], constSim.Dim[1], constSim.Dim[2], numAnts]

# # # #  Extend Dielectric profile for the CPML boundary condition
extend_dimX = constSim.Dim[0]+2*pml_size
extend_dimY = constSim.Dim[1]+2*pml_size
extend_dimZ = constSim.Dim[2]+2*pml_size
SimOri=np.round([extend_dimX/2,extend_dimY/2,extend_dimZ/2])
eps_infx = np.ones((extend_dimX,extend_dimY,extend_dimZ))
sigma_sx = 0.0000*np.ones((extend_dimX,extend_dimY,extend_dimZ))
eps_infy=eps_infx.copy()  # .copy(): MATLAB assignment copies the array
eps_infz=eps_infx.copy()
sigma_sy=sigma_sx.copy()
sigma_sz=sigma_sx.copy()
#FDTD
P = simFDTD('CreateSimulation')
simFDTD('SetConst',P,TypeReal(freqRes),TypeInt(TimeDim))
simFDTD('SetSource',P,TypeReal(constSim.source))


simFDTD('UpdateMaterial',P,TypeInt(1),TypeReal(sigma_sx.flatten(order='F')))
simFDTD('UpdateMaterial',P,TypeInt(2),TypeReal(sigma_sy.flatten(order='F')))
simFDTD('UpdateMaterial',P,TypeInt(3),TypeReal(sigma_sz.flatten(order='F')))
simFDTD('UpdateMaterial',P,TypeInt(4),TypeReal(eps_infx.flatten(order='F')))
simFDTD('UpdateMaterial',P,TypeInt(5),TypeReal(eps_infy.flatten(order='F')))
simFDTD('UpdateMaterial',P,TypeInt(6),TypeReal(eps_infz.flatten(order='F')))

simFDTD('ReadySimulation',P)
simFDTD('InitField',P)
antdatasize=numAnts*timeSteps

simFDTD('Init1D',P,TypeReal(1),TypeReal(constSim.source))



simFDTD('UpdateCuda',P)

simFDTD('ResetFieldAll',P)
for i in range(0,timeSteps):
    simFDTD('UpdateH_All',P)
    #simFDTD('UpdateH_PlaneWave',P,TypeInt([i,3]))
    simFDTD('UpdateH_PML',P)
    simFDTD('UpdateE_All',P)
    #simFDTD('UpdateE_PlaneWave',P,TypeInt([i,3]))
    simFDTD('UpdateE_PML',P)
    simFDTD('AddCurrents',P,TypeInt([i,3]),TypeInt([11,239,139,139,26,26]))

    if(np.mod(i,10)==0):
        # aa=simFDTD('OutputVis',P,np.uint32(extend_dimX*extend_dimY*extend_dimZ),np.int32(2))
        # bb=(np.reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ],order='F'))
        # plt.subplot(2,2,1)
        # plt.imshow(np.squeeze((bb[99,:,:])).T,aspect='auto')
        # plt.colorbar()
        # plt.gca().invert_yaxis()
        # plt.gca().set_box_aspect(1)
        # plt.title("Ex--Xsilce"+str(i))
        # aa=simFDTD('OutputVis',P,np.uint32(extend_dimX*extend_dimY*extend_dimZ),np.int32(2))
        # bb=(np.reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ],order='F'))
        # plt.subplot(2,2,2)
        # plt.imshow(np.squeeze((bb[:,:,15])).T,aspect='auto')
        # plt.colorbar()
        # plt.gca().invert_yaxis()
        # plt.gca().set_box_aspect(1)
        # plt.title("Ey--Xsilce"+str(i))

        aa=simFDTD('OutputVis',P,np.uint32(extend_dimX*extend_dimY*extend_dimZ),np.int32(1))
        bb=(np.reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ],order='F'))
        plt.clf()  # MATLAB's imagesc/colorbar replace the previous ones; matplotlib would stack them
        #plt.subplot(2,2,3)
        plt.imshow(np.squeeze(np.abs(bb[:,:,16])).T,aspect='auto')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.gca().set_box_aspect(1)
        plt.title("Ez--Xsilce"+str(i))

        plt.draw();plt.pause(0.1)




simFDTD('DeleteSimulation',P)

reset_gpuDevice()

plt.show()  # MATLAB keeps the figures open after the script ends
