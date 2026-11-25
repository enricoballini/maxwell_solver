import glob
import os

import h5py
import numpy as np

# Writes the snapshots of GICsimulation.py as ParaView files in results/paraview/; GICparaview_state.py then writes fields.pvsm, which opens them all.
# E.pvd, B.pvd: vectors on the primal and dual nodes (V/km, nT); line_current.pvd: array current (A); sigma.vti: vector (S/m) on the primal nodes;
# primal_grid.vtp, dual_grid.vtp: grid lines on three planes; physical_domain_box.vtp, pml_outer_box.vtp: the PML (10 cells on every side) is in between.
# The array pml is 1 in the PML (on the grids, boxes, E, B, sigma), dual is 1 on the dual grid. E, B, sigma cover the PML too if the simulation saved it. Coordinates in km, z = 0 at the ground.

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results/paraview/steps", exist_ok=True)
for old_file in glob.glob("results/paraview/*.vt[ipm]") + glob.glob("results/paraview/*.pvd") + glob.glob("results/paraview/steps/*.vt[ip]"):
    os.remove(old_file)

mu0 = 4 * np.pi * 1e-7
setup = np.load("results/setup.npz")
cell_km = float(setup["delX"]) / 1e3
pml_size = int(setup["pml_size"])
surface_index = int(setup["surface_index"])
vector_fields = {
    "E": {"components": ["Ex", "Ey", "Ez"], "scale": 1e3, "node_offset": 0},
    "B": {"components": ["Hx", "Hy", "Hz"], "scale": mu0 * 1e9, "node_offset": 0.5},
}


def position_km(index_x, index_y, index_z):
    return (
        np.array([index_x - pml_size, index_y - pml_size, index_z - surface_index])
        * cell_km
    )


def write_image_data(path, shape, origin, point_arrays):
    extent = " ".join(f"0 {n - 1}" for n in shape)
    offsets = np.cumsum([0] + [8 + array.size * 4 for array in point_arrays.values()])
    data_arrays = "".join(
        f'<DataArray type="Float32" Name="{name}" NumberOfComponents="{array.shape[-1] if array.ndim == 4 else 1}" format="appended" offset="{offset}"/>'
        for (name, array), offset in zip(point_arrays.items(), offsets)
    )
    first_name, first_array = next(iter(point_arrays.items()), (None, None))
    active = "" if first_name is None else f' {"Vectors" if first_array.ndim == 4 else "Scalars"}="{first_name}"'
    with open(path, "wb") as file:
        file.write(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
            f'<ImageData WholeExtent="{extent}" Origin="{" ".join(map(str, origin))}" Spacing="{cell_km} {cell_km} {cell_km}">\n'
            f'<Piece Extent="{extent}">\n<PointData{active}>{data_arrays}</PointData>\n</Piece>\n</ImageData>\n'
            '<AppendedData encoding="raw">\n_'.encode()
        )
        for array in point_arrays.values():
            file.write(np.uint64(array.size * 4).tobytes())
            file.write((np.moveaxis(array, -1, 0) if array.ndim == 4 else array).astype("<f4").tobytes(order="F"))
        file.write(b"\n</AppendedData>\n</VTKFile>\n")


def write_polylines(path, polylines, point_arrays=None, cell_arrays=None):
    point_arrays, cell_arrays = point_arrays or {}, cell_arrays or {}
    points = np.concatenate(polylines)
    ends = np.cumsum([len(polyline) for polyline in polylines])

    def data_arrays(arrays):
        return "".join(
            f'<DataArray type="Float32" Name="{name}" format="ascii">{" ".join(map(str, array))}</DataArray>'
            for name, array in arrays.items()
        )

    def active_scalars(arrays):
        return f' Scalars="{next(iter(arrays))}"' if arrays else ""

    with open(path, "w") as file:
        file.write(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n'
            f'<PolyData>\n<Piece NumberOfPoints="{len(points)}" NumberOfLines="{len(polylines)}">\n'
            f"<PointData{active_scalars(point_arrays)}>{data_arrays(point_arrays)}</PointData>\n"
            f"<CellData{active_scalars(cell_arrays)}>{data_arrays(cell_arrays)}</CellData>\n"
            '<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">'
            f'{" ".join(map(str, points.ravel()))}</DataArray></Points>\n'
            '<Lines><DataArray type="Int64" Name="connectivity" format="ascii">'
            f'{" ".join(map(str, range(len(points))))}</DataArray>'
            f'<DataArray type="Int64" Name="offsets" format="ascii">{" ".join(map(str, ends))}</DataArray></Lines>\n'
            "</Piece>\n</PolyData>\n</VTKFile>\n"
        )


def box_edges(low_corner, high_corner):
    corners = np.array([(x, y, z) for x in (low_corner[0], high_corner[0]) for y in (low_corner[1], high_corner[1]) for z in (low_corner[2], high_corner[2])])
    return [corners[[a, b]] for a in range(8) for b in range(a + 1, 8) if (a ^ b) in (1, 2, 4)]


def grid_planes(is_dual):
    shift = 0.5 if is_dual else 0
    node_count = [n if is_dual else n + 1 for n in extended_shape]
    last_physical_node = [pml_size + n - 1 if is_dual else pml_size + n for n in domain_shape]
    polylines, pml_tags = [], []
    for fixed_axis, plane_node in enumerate([extended_shape[0] // 2, current_node_y, surface_index]):
        for line_axis in [axis for axis in range(3) if axis != fixed_axis]:
            other_axis = 3 - fixed_axis - line_axis
            for other_node in range(node_count[other_axis]):
                other_in_pml = not pml_size <= other_node <= last_physical_node[other_axis]
                for first, last, in_pml in [
                    (0, pml_size, True),
                    (pml_size, last_physical_node[line_axis], other_in_pml),
                    (last_physical_node[line_axis], node_count[line_axis] - 1, True),
                ]:
                    nodes = np.zeros((last - first + 1, 3))
                    nodes[:, fixed_axis] = plane_node
                    nodes[:, other_axis] = other_node
                    nodes[:, line_axis] = np.arange(first, last + 1)
                    polylines.append(position_km(*(nodes + shift).T).T)
                    pml_tags.append(float(in_pml))
    return polylines, {"pml": pml_tags, "dual": [float(is_dual)] * len(pml_tags)}


files = sorted(glob.glob("results/fields_*.npz"))
saved_shape = np.load(files[0])["Ex"].shape
with h5py.File("data/250x280gem.h5") as soil_file:
    soil_labels = soil_file["data"][:]
saved_with_pml = saved_shape[:2] == soil_labels.shape[:2]
domain_shape = [n - 2 * pml_size for n in saved_shape] if saved_with_pml else list(saved_shape)
extended_shape = [n + 2 * pml_size for n in domain_shape]
first_saved_node = 0 if saved_with_pml else pml_size
saved_nodes = tuple(slice(first_saved_node, first_saved_node + n) for n in saved_shape)
in_pml = np.ones(extended_shape)
in_pml[tuple(slice(pml_size, pml_size + n) for n in domain_shape)] = 0
i0, i1, current_node_y, j1, k0, k1 = setup["current_indices"]

for name, is_dual in [("primal_grid", False), ("dual_grid", True)]:
    polylines, cell_arrays = grid_planes(is_dual)
    write_polylines(f"results/paraview/{name}.vtp", polylines, cell_arrays=cell_arrays)
physical_box = box_edges(position_km(pml_size, pml_size, pml_size), position_km(*[pml_size + n for n in domain_shape]))
write_polylines("results/paraview/physical_domain_box.vtp", physical_box, cell_arrays={"pml": [0.0] * len(physical_box)})
outer_box = box_edges(position_km(0, 0, 0), position_km(*extended_shape))
write_polylines("results/paraview/pml_outer_box.vtp", outer_box, cell_arrays={"pml": [1.0] * len(outer_box)})

line_points = np.array([position_km(i, current_node_y, k0) for i in range(i0, i1 + 2)])
source_current = np.load("results/source.npz")["g"]

# same construction of sigma as GICsimulation.py, from the same input files
ground_conductivity = np.loadtxt("data/cond_250x280_materials.txt", usecols=2, comments=None)[soil_labels]
conductivity = {name: np.zeros(extended_shape) for name in ["sigma_x", "sigma_y", "sigma_z"]}
for sigma in conductivity.values():
    sigma[:, :, :surface_index] = ground_conductivity
for name in ["sigma_x", "sigma_y"]:
    conductivity[name][:, :, surface_index] = ground_conductivity[:, :, -1] / 2
write_image_data(
    "results/paraview/sigma.vti",
    saved_shape,
    position_km(first_saved_node, first_saved_node, first_saved_node),
    {"sigma": np.stack([sigma[saved_nodes] for sigma in conductivity.values()], axis=-1), "pml": in_pml[saved_nodes]},
)

time_steps = {"E": [], "B": [], "line_current": []}
for n, file in enumerate(files):
    fields = np.load(file)
    step = int(fields["step"])
    t = (step + 1) * float(setup["delT"])
    print(f"snapshot {n + 1}/{len(files)}: step {step}, t = {t:.3f} s", flush=True)
    for name, field in vector_fields.items():
        values = np.stack([fields[component] for component in field["components"]], axis=-1) * field["scale"]
        node = first_saved_node + field["node_offset"]
        file_name = f"steps/{name}_{step:06d}.vti"
        write_image_data(f"results/paraview/{file_name}", values.shape[:3], position_km(node, node, node), {name: values, "pml": in_pml[saved_nodes]})
        time_steps[name].append((t, file_name))
    file_name = f"steps/line_current_{step:06d}.vtp"
    write_polylines(f"results/paraview/{file_name}", [line_points], point_arrays={"current": np.full(len(line_points), source_current[step])})
    time_steps["line_current"].append((t, file_name))

for name, steps in time_steps.items():
    with open(f"results/paraview/{name}.pvd", "w") as collection:
        collection.write(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
            + "".join(f'<DataSet timestep="{t}" file="{file_name}"/>\n' for t, file_name in steps)
            + "</Collection>\n</VTKFile>\n"
        )
