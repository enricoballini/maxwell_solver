import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Plots the data that GICsimulation.py saves in results/ as PDFs in results/figures/,
# and joins the frames into the animation results/figures/Ex_Ey.gif.
# Run it where there is matplotlib (e.g. the laptop), after copying results/ from the cluster.

matplotlib.use("Agg")  # only write the PDFs, never open plot windows
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results/figures", exist_ok=True)

source = np.load("results/source.npz")
t = np.arange(len(source["g"])) * source["dt"]
plt.figure()
plt.plot(t, source["g"] / 1e3, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Current (kA)")
plt.title("Decaying Gaussian Source (Peak at t=0)")
plt.grid(True)
plt.savefig("results/figures/source.pdf")
plt.close()

files = sorted(glob.glob("results/fields_*.npz"))
km = 1e3  # the solver saves E in V/m; E * km is in V/km
# x, y in km from the corner of the physical domain (array index 10, after the PML); cells are 10 km.
# Ex[i, j] sits at x = (i - 10 + 1/2) * 10 km, y = (j - 10) * 10 km; Ey[i, j] at x = (i - 10) * 10 km,
# y = (j - 10 + 1/2) * 10 km (Yee grid). extent = [left, right, bottom, top] edges of the plotted pixels.
Ex_extent = [0, 2400, -5, 2595]
Ey_extent = [-5, 2395, 0, 2600]
# One color scale for all frames (the largest value over the whole run), so the frames can be compared
Ex_max = max(np.abs(np.load(file)["Ex"][10:250, 10:270]).max() for file in files) * km
Ey_max = max(np.abs(np.load(file)["Ey"][10:250, 10:270]).max() for file in files) * km
frames = []
for n, file in enumerate(files):
    data = np.load(file)
    i = int(data["step"])
    t = (i + 1) * source["dt"]  # step i (from 0) has advanced E from t = 0 to t = (i + 1) * dt
    print(f"frame {n + 1}/{len(files)}: step {i}, t = {t:.3f} s", flush=True)
    plt.figure(figsize=(5, 8), layout="constrained")  # tall enough for two square maps with labels
    plt.subplot(2, 1, 1)
    plt.imshow(np.abs(data["Ex"][10:250, 10:270] * km).T, aspect="auto", vmin=0, vmax=Ex_max, origin="lower", extent=Ex_extent)
    plt.colorbar(label="|Ex| (V/km)")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.gca().set_box_aspect(1)
    plt.title(f"Ex, t = {t:.3f} s")
    plt.subplot(2, 1, 2)
    plt.imshow(data["Ey"][10:250, 10:270].T * km, aspect="auto", vmin=-Ey_max, vmax=Ey_max, origin="lower", extent=Ey_extent)
    plt.colorbar(label="Ey (V/km)")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.gca().set_box_aspect(1)
    plt.title(f"Ey, t = {t:.3f} s")
    plt.savefig(f"results/figures/Ex_Ey_{i:06d}.pdf")
    plt.gcf().canvas.draw()
    rgba = np.asarray(plt.gcf().canvas.buffer_rgba())
    frames.append(Image.fromarray(rgba).convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE))
    plt.close()

# 0.1 s per frame, as the plt.pause(0.1) of GICsimulation.m; loop=0 repeats forever
frames[0].save("results/figures/Ex_Ey.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
