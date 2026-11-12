"""Magnetic Gauss law, charge continuity and energy balance (notes, "Checks of the discrete
invariants"). H is read with OutputVis flags 4-6: rebuild with src/Compile.py first."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simFDTD import simFDTD  # noqa: E402

clight = 299792458
mu0 = 4 * np.pi * 1e-7
eps0 = 1 / clight**2 / mu0
delX = 10e3
delT = 0.99 * delX / (math.sqrt(3) * clight)
timeSteps = 100
Nx, Ny, Nz = 26, 25, 24

sigma = np.zeros((Nx, Ny, Nz))
eps = np.ones((Nx, Ny, Nz))
sigma[:, :, :8] = 0.01  # ground
eps[:, :, :8] = 4.0

i0, i1, j0, k0 = 5, 20, 12, 12
source = 100e3 * np.exp(-(((np.arange(timeSteps) - 30) / 10) ** 2))

P = simFDTD("CreateSimulation")
simFDTD("SetConst", P, np.single([0, delX, delT]), np.int32([timeSteps, Nx - 20, Ny - 20, Nz - 20, 0]))
simFDTD("SetSource", P, np.single(source))
for flag, values in [(1, sigma), (2, sigma), (3, sigma), (4, eps), (5, eps), (6, eps)]:
    simFDTD("UpdateMaterial", P, np.int32(flag), np.single(values.flatten(order="F")))
simFDTD("ReadySimulation", P)
simFDTD("ResetFieldAll", P)


def read(flag):
    return np.reshape(
        simFDTD("OutputVis", P, np.uint32(Nx * Ny * Nz), np.int32(flag)), (Nx, Ny, Nz), order="F"
    ).astype(np.float64)


def divergence_at_cell(Fx, Fy, Fz, i, j, k):
    terms = [
        Fx[i + 1, j, k], -Fx[i, j, k],
        Fy[i, j + 1, k], -Fy[i, j, k],
        Fz[i, j, k + 1], -Fz[i, j, k],
    ]
    return sum(terms) / delX, sum(abs(t) for t in terms) / delX


def divergence_at_node(Fx, Fy, Fz, i, j, k):
    terms = [
        Fx[i, j, k], -Fx[i - 1, j, k],
        Fy[i, j, k], -Fy[i, j - 1, k],
        Fz[i, j, k], -Fz[i, j, k - 1],
    ]
    return sum(terms) / delX, sum(abs(t) for t in terms) / delX


def magnetic_gauss(Hx, Hy, Hz, next_to_wall):
    largest_residual, largest_size = 0.0, 0.0
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            for k in range(Nz - 1):
                if (j == Ny - 2 or k == Nz - 2) == next_to_wall:
                    div_H, size = divergence_at_cell(Hx, Hy, Hz, i, j, k)
                    largest_residual = max(largest_residual, abs(div_H))
                    largest_size = max(largest_size, size)
    return largest_residual, largest_size


def charge_continuity(Dx_old, Dy_old, Dz_old, Dx_new, Dy_new, Dz_new, Jx, Jy, Jz):
    largest_residual, largest_size = 0.0, 0.0
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                rho_new, size_new = divergence_at_node(Dx_new, Dy_new, Dz_new, i, j, k)
                rho_old, size_old = divergence_at_node(Dx_old, Dy_old, Dz_old, i, j, k)
                div_J, size_J = divergence_at_node(Jx, Jy, Jz, i, j, k)
                largest_residual = max(largest_residual, abs(rho_new - rho_old + delT * div_J))
                largest_size = max(largest_size, size_new + size_old + delT * size_J)
    return largest_residual, largest_size


gauss, gauss_next_to_wall, continuity = [], [], []
W, loss, work = [], [], []
for n in range(timeSteps):
    Ex_old, Ey_old, Ez_old = read(1), read(2), read(3)  # E^n
    Hx_old, Hy_old, Hz_old = read(4), read(5), read(6)  # H^{n-1/2}
    simFDTD("UpdateH_All", P)
    Hx_new, Hy_new, Hz_new = read(4), read(5), read(6)  # H^{n+1/2}
    simFDTD("UpdateE_All", P)
    simFDTD("AddCurrents", P, np.int32([n, 1]), np.int32([i0, i1, j0, j0, k0, k0]))
    Ex_new, Ey_new, Ez_new = read(1), read(2), read(3)  # E^{n+1}

    Ex_mean, Ey_mean, Ez_mean = (Ex_old + Ex_new) / 2, (Ey_old + Ey_new) / 2, (Ez_old + Ez_new) / 2
    Jx_source = np.zeros((Nx, Ny, Nz))
    Jx_source[i0 : i1 + 1, j0, k0] = source[n] / delX**2

    gauss.append(magnetic_gauss(Hx_new, Hy_new, Hz_new, next_to_wall=False))
    gauss_next_to_wall.append(magnetic_gauss(Hx_new, Hy_new, Hz_new, next_to_wall=True))
    continuity.append(
        charge_continuity(
            eps0 * eps * Ex_old, eps0 * eps * Ey_old, eps0 * eps * Ez_old,
            eps0 * eps * Ex_new, eps0 * eps * Ey_new, eps0 * eps * Ez_new,
            sigma * Ex_mean + Jx_source, sigma * Ey_mean, sigma * Ez_mean,
        )
    )
    W.append(
        delX**3 / 2 * np.sum(
            eps0 * eps * (Ex_old**2 + Ey_old**2 + Ez_old**2)
            + mu0 * (Hx_new * Hx_old + Hy_new * Hy_old + Hz_new * Hz_old)
        )
    )
    loss.append(delT * delX**3 * np.sum(sigma * (Ex_mean**2 + Ey_mean**2 + Ez_mean**2)))
    work.append(delT * delX**3 * np.sum(Jx_source * Ex_mean))

simFDTD("DeleteSimulation", P)

energy = [
    (W[n + 1] - W[n] + loss[n] + work[n], abs(W[n + 1]) + abs(W[n]) + abs(loss[n]) + abs(work[n]))
    for n in range(timeSteps - 1)
]


def relative(steps):
    return max(abs(residual) for residual, size in steps) / max(size for residual, size in steps)


for name, steps in [
    ("magnetic Gauss law", gauss),
    ("charge continuity", continuity),
    ("energy balance", energy),
]:
    value = relative(steps)
    print(f"{name:<20} {value:.1e}  {'OK' if value < 1e-4 else 'FAILED'}")
print(f"magnetic Gauss law next to the walls j = Ny-1, k = Nz-1: {relative(gauss_next_to_wall):.1e} (expected non-zero)")
