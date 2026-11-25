import os

import numpy as np
from paraview import servermanager
from paraview.simple import (
    AssignViewToLayout,
    ColorBy,
    CreateLayout,
    CreateView,
    GetAnimationScene,
    GetColorTransferFunction,
    GetScalarBar,
    Glyph,
    Hide,
    OpenDataFile,
    Render,
    SaveState,
    Slice,
    Show,
)
from vtkmodules.util.numpy_support import vtk_to_numpy

# Run with ParaView's pvbatch after GICpostprocess.py: writes results/paraview/fields.pvsm, the one file to open (File > Load State).
# Every file of results/paraview/ is a separate source, at the last time step. E and B: outline of the volume, white arrows (direction only,
# fixed length) at about 1000 points spread in the volume, and a slice (E at the ground, B across the line), log colour scale from 1% to 99%.
# The grids are loaded but hidden.

case_folder = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(case_folder, "results", "paraview")
view = CreateView("RenderView")
AssignViewToLayout(view=view, layout=CreateLayout("Layout #1"), hint=0)
sources = {
    name: OpenDataFile(os.path.join(folder, file_name), registrationName=name)
    for name, file_name in [
        ("E", "E.pvd"),
        ("B", "B.pvd"),
        ("line_current", "line_current.pvd"),
        ("sigma", "sigma.vti"),
        ("primal_grid", "primal_grid.vtp"),
        ("dual_grid", "dual_grid.vtp"),
        ("physical_domain_box", "physical_domain_box.vtp"),
        ("pml_outer_box", "pml_outer_box.vtp"),
    ]
}
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.PlayMode = "Snap To TimeSteps"
last_time = scene.TimeKeeper.TimestepValues[-1]
scene.AnimationTime = last_time
displays = {name: Show(source, view) for name, source in sources.items()}
for name in ["primal_grid", "dual_grid"]:
    Hide(sources[name], view)

for name, unit, plane_normal in [("E", "V/km", [0, 0, 1]), ("B", "nT", [1, 0, 0])]:
    sources[name].UpdatePipeline(last_time)
    data = servermanager.Fetch(sources[name])
    magnitude = np.linalg.norm(vtk_to_numpy(data.GetPointData().GetArray(name)), axis=1)
    bounds = data.GetBounds()
    plane = Slice(registrationName=f"{name}_slice", Input=sources[name])
    plane.SliceType.Origin = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, 0]
    plane.SliceType.Normal = plane_normal
    plane_display = Show(plane, view)
    ColorBy(plane_display, ("POINTS", name, "Magnitude"))
    plane_display.SetScalarBarVisibility(view, True)
    color_map = GetColorTransferFunction(name)
    color_map.ApplyPreset("Viridis", True)
    color_map.AutomaticRescaleRangeMode = "Never"
    color_map.RescaleTransferFunction(*np.percentile(magnitude, [1, 99]))
    color_map.MapControlPointsToLogSpace()
    color_map.UseLogScale = 1
    GetScalarBar(color_map, view).Title = f"|{name}| ({unit})"
    GetScalarBar(color_map, view).ComponentTitle = ""
    arrows = Glyph(registrationName=f"{name}_arrows", Input=sources[name], GlyphType="Arrow")
    arrows.OrientationArray = ["POINTS", name]
    arrows.ScaleArray = ["POINTS", "No scale array"]
    arrows.GlyphMode = "Uniform Spatial Distribution (Bounds Based)"
    arrows.MaximumNumberOfSamplePoints = 1000
    arrows.ScaleFactor = 40
    arrows_display = Show(arrows, view)
    arrows_display.ColorArrayName = ["POINTS", ""]
    arrows_display.AmbientColor = arrows_display.DiffuseColor = [1.0, 1.0, 1.0]

current_map = GetColorTransferFunction("current")
current_map.ApplyPreset("Viridis", True)
GetColorTransferFunction("pml").ApplyPreset("Viridis", True)
current_map.AutomaticRescaleRangeMode = "Never"
current_map.RescaleTransferFunction(0.0, float(np.load(os.path.join(case_folder, "results", "source.npz"))["g"].max()))
displays["line_current"].LineWidth = 3
displays["line_current"].SetScalarBarVisibility(view, True)
GetScalarBar(current_map, view).Title = "current (A)"

Render(view)
bounds = sources["pml_outer_box"].GetDataInformation().GetBounds()
view.CameraFocalPoint = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, 0]
view.CameraPosition = [-2500, -4800, 3200]
view.CameraViewUp = [0, 0, 1]
view.ResetCamera(False)
Render(view)
SaveState(os.path.join(folder, "fields.pvsm"))
