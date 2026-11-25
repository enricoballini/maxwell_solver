import glob
import os

import numpy as np

# Writes the snapshots of GICsimulation.py as ParaView files in results/paraview/:
# fields.pvd (E in V/km, B in nT, each component at its own Yee position), primal_grid.vti, dual_grid.vti, line_current.vtp.
# Coordinates in km: x, y from the corner of the physical domain, z = 0 at the ground surface (z > 0 is air).

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results/paraview", exist_ok=True)

mu0 = 4 * np.pi * 1e-7
setup = np.load("results/setup.npz")
cell_km = float(setup["delX"]) / 1e3
pml_size = int(setup["pml_size"])
surface_index = int(setup["surface_index"])
yee_offsets = {
    "Ex": (0.5, 0, 0),
    "Ey": (0, 0.5, 0),
    "Ez": (0, 0, 0.5),
    "Bx": (0, 0.5, 0.5),
    "By": (0.5, 0, 0.5),
    "Bz": (0.5, 0.5, 0),
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
        f'<DataArray type="Float32" Name="{name}" format="appended" offset="{offset}"/>'
        for name, offset in zip(point_arrays, offsets)
    )
    with open(path, "wb") as file:
        file.write(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
            f'<ImageData WholeExtent="{extent}" Origin="{" ".join(map(str, origin))}" Spacing="{cell_km} {cell_km} {cell_km}">\n'
            f'<Piece Extent="{extent}">\n<PointData>{data_arrays}</PointData>\n</Piece>\n</ImageData>\n'
            '<AppendedData encoding="raw">\n_'.encode()
        )
        for array in point_arrays.values():
            file.write(np.uint64(array.size * 4).tobytes())
            file.write(array.astype("<f4").tobytes(order="F"))
        file.write(b"\n</AppendedData>\n</VTKFile>\n")


def write_line(path, points):
    with open(path, "w") as file:
        file.write(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n'
            f'<PolyData>\n<Piece NumberOfPoints="{len(points)}" NumberOfLines="1">\n'
            '<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">'
            f'{" ".join(map(str, points.ravel()))}</DataArray></Points>\n'
            '<Lines><DataArray type="Int64" Name="connectivity" format="ascii">'
            f'{" ".join(map(str, range(len(points))))}</DataArray>'
            f'<DataArray type="Int64" Name="offsets" format="ascii">{len(points)}</DataArray></Lines>\n'
            "</Piece>\n</PolyData>\n</VTKFile>\n"
        )


files = sorted(glob.glob("results/fields_*.npz"))
domain_shape = np.load(files[0])["Ex"].shape
write_image_data("results/paraview/primal_grid.vti", domain_shape, position_km(pml_size, pml_size, pml_size), {})
write_image_data("results/paraview/dual_grid.vti", domain_shape, position_km(pml_size + 0.5, pml_size + 0.5, pml_size + 0.5), {})

i0, i1, j0, j1, k0, k1 = setup["current_indices"]
write_line("results/paraview/line_current.vtp", np.array([position_km(i, j0, k0) for i in range(i0, i1 + 2)]))

time_steps = []
for n, file in enumerate(files):
    fields = np.load(file)
    step = int(fields["step"])
    t = (step + 1) * float(setup["delT"])
    print(f"snapshot {n + 1}/{len(files)}: step {step}, t = {t:.3f} s", flush=True)
    for name, offset in yee_offsets.items():
        values = fields[name] * 1e3 if name[0] == "E" else fields["H" + name[1]] * mu0 * 1e9
        write_image_data(
            f"results/paraview/{name}_{step:06d}.vti",
            values.shape,
            position_km(pml_size + offset[0], pml_size + offset[1], pml_size + offset[2]),
            {name: values},
        )
    with open(f"results/paraview/fields_{step:06d}.vtm", "w") as multiblock:
        multiblock.write(
            '<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n<vtkMultiBlockDataSet>\n'
            + "".join(
                f'<DataSet index="{index}" name="{name}" file="{name}_{step:06d}.vti"/>\n'
                for index, name in enumerate(yee_offsets)
            )
            + "</vtkMultiBlockDataSet>\n</VTKFile>\n"
        )
    time_steps.append(f'<DataSet timestep="{t}" file="fields_{step:06d}.vtm"/>\n')

with open("results/paraview/fields.pvd", "w") as collection:
    collection.write(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
        + "".join(time_steps)
        + "</Collection>\n</VTKFile>\n"
    )
