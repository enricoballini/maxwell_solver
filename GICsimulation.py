import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

from src.simFDTD import simFDTD, reset_gpuDevice

# Python translation of GICsimulation.m. Indices are 0-based: MATLAB index k -> k-1, range a:b -> a-1:b.
# The local functions at the end of GICsimulation.m are here at the top, because Python needs them
# defined before the script uses them.


def FDTD_solver(P, timeSteps, FullGPU):
    for i in range(0, timeSteps):
        simFDTD("UpdateH_All", P)
        simFDTD("UpdateH_PML", P)
        simFDTD("UpdateE_All", P)
        simFDTD("UpdateE_PML", P)


def FDTD_solverRe(P, timeSteps, FullGPU):
    for i in range(0, timeSteps):
        simFDTD("UpdateH_All", P)
        # UpdateH_PlaneWave(P,TypeInt([i,1]))
        simFDTD("UpdateH_PlaneWave", P, TypeInt([i, 2]))
        # UpdateH_PlaneWave(P,TypeInt([i,3]))

        simFDTD("UpdateH_PML", P)
        simFDTD("UpdateE_All", P)
        # UpdateE_PlaneWave(P,TypeInt([i,1]))
        simFDTD("UpdateE_PlaneWave", P, TypeInt([i, 2]))
        # UpdateE_PlaneWave(P,TypeInt([i,3]))

        simFDTD("UpdateE_PML", P)


def TypeInt(x):
    x = np.int32(x)
    return x


def TypeReal(x):
    x = np.single(x)
    return x


reset_gpuDevice()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# No plots here (the cluster has no graphics): the data go to results/, GICpostprocess.py plots them
os.makedirs("results", exist_ok=True)


sys.path[0:0] = [
    p for p, _, _ in os.walk(".") if "venv" not in p.split(os.sep)
]  # addpath(genpath(".")), skipping venv: its folders would shadow Python modules

###optimization parameters initialization
clight = 299792458
mu0 = 4 * np.pi * 1e-7
eps0 = 1 / clight**2 / mu0
unit = 10**3
delX = 10 * unit
delT = 0.99 * delX / (math.sqrt(3) * clight)
fctr = 5.5e9
timeSteps = 300000
distancePML = 10


Amax = 100e3  # 100 kA
t0 = 5  #
FWHM = 10  #
sigma = FWHM / 4 / 2.355
dt = delT
t = np.arange(0, 100, dt)  # 0:dt:100

g = Amax * np.exp(-((t - t0) ** 2) / (2 * sigma**2))


np.savez("results/source.npz", dt=dt, g=g[0:timeSteps])  # the part of g the solver uses


####save optimization parameters
constSim = SimpleNamespace()
constSim.delX = delX
constSim.delT = delT
constSim.fctr = fctr
constSim.timeSteps = timeSteps

constSim.Dim = [230, 260, 30]
numAnts = 0
constSim.source = g[0:timeSteps]
####save path
flag_plot = 1

# --------------------project_FDTD_3D-----------------------
pml_size = 10
freqRes = [constSim.fctr, delX, delT]
TimeDim = [timeSteps, constSim.Dim[0], constSim.Dim[1], constSim.Dim[2], numAnts]

# # # #  Extend Dielectric profile for the CPML boundary condition
extend_dimX = constSim.Dim[0] + 2 * pml_size
extend_dimY = constSim.Dim[1] + 2 * pml_size
extend_dimZ = constSim.Dim[2] + 2 * pml_size
SimOri = np.round([extend_dimX / 2, extend_dimY / 2, extend_dimZ / 2])
eps_infx = np.ones((extend_dimX, extend_dimY, extend_dimZ))
sigma_sx = 0.0000 * np.ones((extend_dimX, extend_dimY, extend_dimZ))
eps_infy = eps_infx.copy()  # .copy(): MATLAB assignment copies the array
eps_infz = eps_infx.copy()
sigma_sy = sigma_sx.copy()
sigma_sz = sigma_sx.copy()
x1 = 107 + 1
x2 = 134 + 1
ind0 = 1
indxy = np.arange(ind0, 16 + 1)  # ind0:16
sigma_sx[ind0 - 1 : x1, :, indxy - 1] = 0.01586
sigma_sx[x2:, :, indxy - 1] = 0.001  # x2+1:end
sigma_sx[ind0 - 1 : x1, :, 17 - 1] = 0.01586 / 2
sigma_sx[x2:, :, 17 - 1] = 0.001 / 2

sigma_sy[ind0 - 1 : x1, :, indxy - 1] = 0.01586
sigma_sy[x2:, :, indxy - 1] = 0.001
sigma_sy[ind0 - 1 : x1, :, 17 - 1] = 0.01586 / 2
sigma_sy[x2:, :, 17 - 1] = 0.001 / 2

sigma_sz[ind0 - 1 : x1, :, indxy - 1] = 0.01586
sigma_sz[x2:, :, indxy - 1] = 0.001

fact = 1.1
for i in range(x1 + 1, x2 + 1):
    sigma_sx[i - 1, :, indxy - 1] = 0.01586 / fact
    sigma_sy[i - 1, :, indxy - 1] = 0.01586 / fact
    sigma_sx[i - 1, :, 17 - 1] = 0.01586 / fact / 2
    sigma_sy[i - 1, :, 17 - 1] = 0.01586 / fact / 2
    sigma_sz[i - 1, :, ind0 - 1 : 16] = 0.01586 / fact
    fact = fact * 1.1
# FDTD
P = simFDTD("CreateSimulation")
simFDTD("SetConst", P, TypeReal(freqRes), TypeInt(TimeDim))
simFDTD("SetSource", P, TypeReal(constSim.source))


simFDTD("UpdateMaterial", P, TypeInt(1), TypeReal(sigma_sx.flatten(order="F")))
simFDTD("UpdateMaterial", P, TypeInt(2), TypeReal(sigma_sy.flatten(order="F")))
simFDTD("UpdateMaterial", P, TypeInt(3), TypeReal(sigma_sz.flatten(order="F")))
simFDTD("UpdateMaterial", P, TypeInt(4), TypeReal(eps_infx.flatten(order="F")))
simFDTD("UpdateMaterial", P, TypeInt(5), TypeReal(eps_infy.flatten(order="F")))
simFDTD("UpdateMaterial", P, TypeInt(6), TypeReal(eps_infz.flatten(order="F")))

simFDTD("ReadySimulation", P)
antdatasize = numAnts * timeSteps


simFDTD("UpdateCuda", P)

simFDTD("ResetFieldAll", P)
tic = time.time()
for i in range(0, timeSteps):
    simFDTD("UpdateH_All", P)
    simFDTD("UpdateH_PML", P)
    simFDTD("UpdateE_All", P)
    simFDTD("UpdateE_PML", P)
    # simFDTD('AddCurrents',P,TypeInt([i,2]),TypeInt([1,249,139,139,26,26]))
    simFDTD("AddCurrents", P, TypeInt([i, 1]), TypeInt([1, 249, 139, 139, 26, 26]))

    if np.mod(i, 1000) == 0:
        print(f"step {i}/{timeSteps}, elapsed {time.time() - tic:.1f} s", flush=True)
        # Ex (component 1) and Ey (component 2) on the plane k = 16, for GICpostprocess.py
        shape = [extend_dimX, extend_dimY, extend_dimZ]
        n_cells = np.uint32(extend_dimX * extend_dimY * extend_dimZ)
        Ex = np.reshape(simFDTD("OutputVis", P, n_cells, np.int32(1)), shape, order="F")
        Ey = np.reshape(simFDTD("OutputVis", P, n_cells, np.int32(2)), shape, order="F")
        np.savez(f"results/fields_{i:06d}.npz", step=i, Ex=Ex[:, :, 16], Ey=Ey[:, :, 16])


print("Elapsed time is %f seconds." % (time.time() - tic))  # toc

simFDTD("DeleteSimulation", P)

reset_gpuDevice()
