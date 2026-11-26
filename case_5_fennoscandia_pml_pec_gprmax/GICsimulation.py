import json
import math
import os
import time

import h5py
import numpy as np

import gprMax
from gprMax.updates.cuda_updates import CUDAUpdates

# gprMax version of case_4_fennoscandia_pml_pec: same grid and array indices, time step, conductivity on every E edge,
# line current, PML grading, PEC on all faces and snapshot files. Not the same: the PML recursion is gprMax's
# (HORIPML, not CPML), and it also acts on Ey at j = 0 and Ez at k = 0, which the kernels of case_4 skip.

os.chdir(os.path.dirname(os.path.abspath(__file__)))

clight = 299792458
mu0 = 4 * np.pi * 1e-7
eps0 = 1 / clight**2 / mu0
delX = 10e3
delT = 0.99 * delX / (math.sqrt(3) * clight)
timeSteps = 300000
pml_size = 10
Dim = [230, 260, 30]
extend_dimX, extend_dimY, extend_dimZ = [n + 2 * pml_size for n in Dim]  # E and H array sizes
domain_cells = (extend_dimX - 1, extend_dimY - 1, extend_dimZ - 1)

Amax = 100e3  # 100 kA
t0 = 5
FWHM = 10
source = Amax * np.exp(-((np.arange(timeSteps) * delT - t0) ** 2) / (2 * (FWHM / 4 / 2.355) ** 2))
np.savez("results/source.npz", dt=delT, g=source)

with h5py.File("data/250x280gem.h5") as soil_file:
    assert (soil_file.attrs["dx_dy_dz"] == delX).all()
    soil_labels = soil_file["data"][:]
# columns of the gprMax materials file: #material: eps_r sigma mu_r sigma_m ID
ground_conductivity = np.loadtxt("data/cond_250x280_materials.txt", usecols=2, comments=None)[soil_labels]
ground_layers = soil_labels.shape[2]
sigma_sx = np.zeros((extend_dimX, extend_dimY, extend_dimZ))
sigma_sx[:, :, :ground_layers] = ground_conductivity
sigma_sy = sigma_sx.copy()
sigma_sz = sigma_sx.copy()
sigma_sx[:, :, ground_layers] = ground_conductivity[:, :, -1] / 2
sigma_sy[:, :, ground_layers] = ground_conductivity[:, :, -1] / 2
current_indices = [0, extend_dimX - 2, 139, 139, 26, 26]  # i0, i1, j0, j1, k0, k1 of the Ex edges with the current
np.savez(
    "results/setup.npz",
    delX=delX,
    delT=delT,
    pml_size=pml_size,
    surface_index=ground_layers,
    current_indices=current_indices,
)

# One gprMax material per conductivity value, assigned edge by edge (ID arrays), so gprMax does not average it.
conductivities, edge_material = np.unique(np.stack([sigma_sx, sigma_sy, sigma_sz]), return_inverse=True)
assert conductivities[0] == 0  # air
materials = {"free_space": {"model": "builtin", "builtin": "free_space"}}
for number, conductivity in enumerate(conductivities[1:], start=1):
    materials[f"conductivity_{number}"] = {
        "model": "constant",
        "base": {"relative_permittivity": 1.0, "electric_conductivity_s_per_m": float(conductivity)},
    }
with open("results/geometry_materials.json", "w") as database:
    json.dump(
        {
            "schema": "gprMax-material-database",
            "schema_version": 1,
            "database": {"id": "geometry_materials", "name": "Fennoscandia ground", "version": "1"},
            "materials": materials,
        },
        database,
        indent=1,
    )
edge_ID = np.zeros((6, *sigma_sx.shape), dtype=np.int16)  # Ex, Ey, Ez, Hx, Hy, Hz; H in free space (material 0)
edge_ID[:3] = edge_material.reshape(3, *sigma_sx.shape)
with h5py.File("results/geometry.h5", "w") as geometry:
    geometry.attrs["dx_dy_dz"] = (delX, delX, delX)
    geometry.create_dataset("material_keys", data=list(materials), dtype=h5py.string_dtype())
    geometry.create_dataset("ID", data=edge_ID, compression="gzip")
    # the cell materials do not enter the update; rigid: keep ID as given
    geometry.create_dataset("data", data=np.zeros(domain_cells, dtype=np.int16), compression="gzip")
    geometry.create_dataset("rigidE", data=np.ones((12, *domain_cells), dtype=np.int8), compression="gzip")
    geometry.create_dataset("rigidH", data=np.ones((6, *domain_cells), dtype=np.int8), compression="gzip")

scene = gprMax.Scene()
scene.add(gprMax.Domain(p1=tuple(n * delX for n in domain_cells)))
scene.add(gprMax.Discretisation(p1=(delX, delX, delX)))
scene.add(gprMax.TimeStepStabilityFactor(f=0.99))  # time step delT
scene.add(gprMax.TimeWindow(iterations=timeSteps))
# 9 cells give the sigma and alpha of case_4 at the same nodes: there the grading goes from node 9 (zero) to node 0
scene.add(gprMax.PMLThickness(thickness=pml_size - 1))
scene.add(
    gprMax.PMLCFS(
        alphascalingprofile="linear",
        alphascalingdirection="reverse",
        alphamin=0,
        alphamax=1e-8,
        kappascalingprofile="constant",
        kappascalingdirection="forward",
        kappamin=1,
        kappamax=1,
        sigmascalingprofile="cubic",
        sigmascalingdirection="forward",
        sigmamin=0,
        sigmamax=0.8 * (3 + 1) / (math.sqrt(mu0 / eps0) * delX),
    )
)
scene.add(gprMax.GeometryObjectsRead(p1=(0, 0, 0), geofile="results/geometry.h5", material_database="geometry_materials"))
# gprMax samples a dipole at (n + 1/2) delT: shifted so that step n carries source[n], as in case_4
scene.add(
    gprMax.Waveform(
        wave_type="user",
        id="line_current",
        user_values=source,
        user_time=(np.arange(timeSteps) + 0.5) * delT,
        fill_value="extrapolate",
    )
)
for i in range(current_indices[0], current_indices[1] + 1):
    scene.add(
        gprMax.HertzianDipole(
            polarisation="x",
            p1=(i * delX, current_indices[2] * delX, current_indices[4] * delX),
            waveform_id="line_current",
        )
    )


def save_fields_and_print_progress(updates, iteration):
    # at the start of gprMax iteration step + 1 the fields are those of case_4 after its step: E^(step+1), H^(step+1/2)
    global start_time
    if iteration == 0:
        start_time = time.time()
        return
    step = iteration - 1
    if step % 1000 == 0:
        print(f"step {step}/{timeSteps}, elapsed {time.time() - start_time:.1f} s", flush=True)
    if step % 10000 == 0:
        np.savez(
            f"results/fields_{step:06d}.npz",
            step=step,
            **{name: getattr(updates.grid, f"{name}_dev").get() for name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]},
        )


CUDAUpdates.store_snapshots = save_fields_and_print_progress  # gprMax has no per-step callback

gprMax.run(scenes=[scene], outputfile="results/gprmax", gpu=[0], hide_progress_bars=True)

print("Elapsed time is %f seconds." % (time.time() - start_time))
print("Done!")
