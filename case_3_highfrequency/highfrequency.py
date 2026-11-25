import math
import os
import shutil
import sys
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CreateSource import CreateSource  # noqa: E402
from src.simFDTD import simFDTD, reset_gpuDevice  # noqa: E402

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "text.usetex": shutil.which("latex") is not None,  # the HPsp container has no LaTeX
        "font.family": "serif",
        "mathtext.fontset": "cm",
    }
)

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
os.makedirs("results/figures", exist_ok=True)
# clear
plt.close('all')



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
plt.savefig("results/figures/source.pdf")
plt.close()




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
extend_dimX = constSim.Dim[0]+2*pml_size
extend_dimY = constSim.Dim[1]+2*pml_size
extend_dimZ = constSim.Dim[2]+2*pml_size
TimeDim = [timeSteps, extend_dimX, extend_dimY, extend_dimZ, numAnts]

# # # #  Extend Dielectric profile for the CPML boundary condition

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
antdatasize=numAnts*timeSteps

simFDTD('Init1D',P,TypeReal(1),TypeReal(constSim.source))



simFDTD('UpdateCuda',P)

simFDTD('ResetFieldAll',P)
slice_index_y=119
saved_steps=[]
saved_Ez_slices=[]
for i in range(0,timeSteps):
    simFDTD('UpdateH_All',P)
    simFDTD('UpdateH_PlaneWave',P,TypeInt([i,3]))
    simFDTD('UpdateH_PML',P)
    simFDTD('UpdateE_All',P)
    simFDTD('UpdateE_PlaneWave',P,TypeInt([i,3]))
    simFDTD('UpdateE_PML',P)
    # simFDTD('AddCurrents',P,TypeInt([i,1]),TypeInt([120,145,139,139,26,26]))

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

        aa=simFDTD('OutputVis',P,np.uint32(extend_dimX*extend_dimY*extend_dimZ),np.int32(3))
        bb=(np.reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ],order='F'))
        saved_steps.append(i)
        saved_Ez_slices.append(np.abs(bb[:,slice_index_y,:]).T)




simFDTD('DeleteSimulation',P)

reset_gpuDevice()

Ez_maximum = max(Ez_slice.max() for Ez_slice in saved_Ez_slices)
figure, axes = plt.subplots(layout="constrained")
image = axes.imshow(
    saved_Ez_slices[0],
    aspect="auto",
    cmap=plt.get_cmap("viridis", 10),
    vmin=0,
    vmax=Ez_maximum,
)
figure.colorbar(image, ax=axes, ticks=np.linspace(0, Ez_maximum, 11), label=r"$|E_z|$")
axes.invert_yaxis()
axes.set_box_aspect(1)


def draw_frame(frame):
    image.set_data(saved_Ez_slices[frame])
    axes.set_title(rf"$j = {slice_index_y}$, step ${saved_steps[frame]}$")


FuncAnimation(figure, draw_frame, frames=len(saved_steps)).save(
    "results/figures/Ez.gif", writer=PillowWriter(fps=10)
)
