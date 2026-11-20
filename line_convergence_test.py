import math
import os

import numpy as np

from src.simFDTD import reset_gpuDevice, simFDTD

speed_of_light = 299792458
vacuum_permeability = 4 * np.pi * 1e-7
cell_size = 10e3
time_step = 0.99 * cell_size / (math.sqrt(3) * speed_of_light)
time_steps = 60
peak_current = 100e3
run_time = time_steps * time_step
refinements = np.array([1, 2, 4])
frequency_shifts_maximum = np.array([0.08, 1e-8])
layer_cells = 5
nodes = np.array([2 * layer_cells + 3, 2 * layer_cells + 17, 2 * layer_cells + 17])
line_index = layer_cells + 8
sample_offsets = np.array([[3, 0], [0, 4], [6, 0], [6, 6]])
sample_times = (1 + np.arange(time_steps)) * time_step


def current(time):
    return peak_current * (time / run_time) ** 3


def exact_Ex(distance_from_line, times):
    travel_time = distance_from_line / speed_of_light
    stretch = np.arccosh(np.maximum(times / travel_time, 1))
    crossed = np.sqrt(np.maximum(times**2 - travel_time**2, 0))
    return (
        -3
        * vacuum_permeability
        * peak_current
        / (2 * np.pi * run_time**3)
        * ((times**2 + travel_time**2 / 2) * stretch - 1.5 * times * crossed)
    )


def line_Ex_at_samples(refinement, frequency_shift_maximum):
    refined_nodes = refinement * (nodes - 1) + 1
    size, step_size = cell_size / refinement, time_step / refinement
    line = refinement * line_index
    sample_x = refinement * (nodes[0] // 2)
    sample_y, sample_z = (line + refinement * sample_offsets).T
    simulation = simFDTD("CreateSimulation")
    simFDTD(
        "SetPML",
        simulation,
        np.int32([refinement * layer_cells, 0]),
        np.single([frequency_shift_maximum]),
    )
    simFDTD(
        "SetConst",
        simulation,
        np.single([0, size, step_size]),
        np.int32([refinement * time_steps, *(refined_nodes - 2 * refinement * layer_cells), 0]),
    )
    simFDTD(
        "SetSource",
        simulation,
        np.single(current((np.arange(refinement * time_steps) + 0.5) * step_size)),
    )
    simFDTD("ReadySimulation", simulation)
    simFDTD("UpdateCuda", simulation)
    simFDTD("ResetFieldAll", simulation)
    sampled_E = []
    for step in range(refinement * time_steps):
        simFDTD("UpdateH_All", simulation)
        simFDTD("UpdateH_PML", simulation)
        simFDTD("UpdateE_All", simulation)
        simFDTD("UpdateE_PML", simulation)
        simFDTD(
            "AddCurrents",
            simulation,
            np.int32([step, 1]),
            np.int32([0, refined_nodes[0] - 2, line, line, line, line]),
        )
        if (step + 1) % refinement == 0:
            Ex = np.reshape(
                simFDTD("OutputVis", simulation, np.uint32(refined_nodes.prod()), np.int32(1)),
                refined_nodes,
                order="F",
            )
            sampled_E.append(Ex[sample_x, sample_y, sample_z])
    simFDTD("DeleteSimulation", simulation)
    return np.array(sampled_E, dtype=np.float64)


reset_gpuDevice()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
exact = np.stack(
    [
        exact_Ex(cell_size * np.sqrt((offsets**2).sum()), sample_times)
        for offsets in sample_offsets
    ],
    axis=1,
)
errors = np.zeros((len(frequency_shifts_maximum), len(refinements)))
for shift_number, frequency_shift_maximum in enumerate(frequency_shifts_maximum):
    for refinement_number, refinement in enumerate(refinements):
        errors[shift_number, refinement_number] = (
            np.abs(line_Ex_at_samples(refinement, frequency_shift_maximum) - exact).max()
            / np.abs(exact).max()
        )
        print(
            f"frequency shift {frequency_shift_maximum:g} S/m, "
            f"cell size {cell_size / refinement / 1e3:g} km: "
            f"error {errors[shift_number, refinement_number]:.2e}",
            flush=True,
        )
    print(f"    orders {np.log2(errors[shift_number, :-1] / errors[shift_number, 1:]).round(2)}")
reset_gpuDevice()
np.savez(
    "results/line_convergence_test.npz",
    cell_sizes=cell_size / refinements,
    frequency_shifts_maximum=frequency_shifts_maximum,
    errors=errors,
)
