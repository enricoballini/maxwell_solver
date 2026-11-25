import math
import os
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simFDTD import simFDTD, reset_gpuDevice  # noqa: E402

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
# No plots here (the cluster has no graphics): the data go to results/ (made by run.sh), GICpostprocess.py turns them into ParaView files

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
extend_dimX = constSim.Dim[0] + 2 * pml_size
extend_dimY = constSim.Dim[1] + 2 * pml_size
extend_dimZ = constSim.Dim[2] + 2 * pml_size
TimeDim = [timeSteps, extend_dimX, extend_dimY, extend_dimZ, numAnts]

# # # #  Extend Dielectric profile for the CPML boundary condition

SimOri = np.round([extend_dimX / 2, extend_dimY / 2, extend_dimZ / 2])
eps_infx = np.ones((extend_dimX, extend_dimY, extend_dimZ))
sigma_sx = 0.0000 * np.ones((extend_dimX, extend_dimY, extend_dimZ))
eps_infy = eps_infx.copy()  # .copy(): MATLAB assignment copies the array
eps_infz = eps_infx.copy()
sigma_sy = sigma_sx.copy()
sigma_sz = sigma_sx.copy()
with h5py.File("data/250x280gem.h5") as soil_file:
    assert (soil_file.attrs["dx_dy_dz"] == delX).all()
    soil_labels = soil_file["data"][:]
# columns of the gprMax materials file: #material: eps_r sigma mu_r sigma_m ID
conductivity_by_label = np.loadtxt(
    "data/cond_250x280_materials.txt", usecols=2, comments=None
)
ground_conductivity = conductivity_by_label[soil_labels]
ground_layers = soil_labels.shape[2]
sigma_sx[:, :, :ground_layers] = ground_conductivity
sigma_sy[:, :, :ground_layers] = ground_conductivity
sigma_sz[:, :, :ground_layers] = ground_conductivity
sigma_sx[:, :, ground_layers] = ground_conductivity[:, :, -1] / 2
sigma_sy[:, :, ground_layers] = ground_conductivity[:, :, -1] / 2
current_indices = [1, 249, 139, 139, 26, 26]
np.savez(
    "results/setup.npz",
    delX=delX,
    delT=delT,
    pml_size=pml_size,
    surface_index=ground_layers,
    current_indices=current_indices,
)
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
    simFDTD("AddCurrents", P, TypeInt([i, 1]), TypeInt(current_indices))

    if np.mod(i, 1000) == 0:
        print(f"step {i}/{timeSteps}, elapsed {time.time() - tic:.1f} s", flush=True)
    if np.mod(i, 10000) == 0:
        shape = [extend_dimX, extend_dimY, extend_dimZ]
        n_cells = np.uint32(extend_dimX * extend_dimY * extend_dimZ)
        np.savez(
            f"results/fields_{i:06d}.npz",
            step=i,
            **{
                name: np.reshape(
                    simFDTD("OutputVis", P, n_cells, np.int32(flag)), shape, order="F"
                )
                for flag, name in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], start=1)
            },
        )


print("Elapsed time is %f seconds." % (time.time() - tic))

simFDTD("DeleteSimulation", P)

reset_gpuDevice()


print("Done!")
