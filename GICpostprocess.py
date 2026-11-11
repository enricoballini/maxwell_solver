import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Plots the data that GICsimulation.py saves in results/ as PDFs in figures/.
# Run it where there is matplotlib (e.g. the laptop), after copying results/ from the cluster.

matplotlib.use("Agg")  # only write the PDFs, never open plot windows
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

source = np.load("results/source.npz")
t = np.arange(len(source["g"])) * source["dt"]
plt.figure()
plt.plot(t, source["g"] / 1e3, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Current (kA)")
plt.title("Decaying Gaussian Source (Peak at t=0)")
plt.grid(True)
plt.savefig("figures/source.pdf")
plt.close()

files = sorted(glob.glob("results/fields_*.npz"))
for n, file in enumerate(files):
    data = np.load(file)
    i = int(data["step"])
    print(f"frame {n + 1}/{len(files)}: step {i}", flush=True)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(np.abs(data["Ex"][10:250, 10:270]).T, aspect="auto")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.gca().set_box_aspect(1)
    plt.title("Ex--Xsilce" + str(i))
    plt.subplot(2, 1, 2)
    plt.imshow(data["Ey"][10:250, 10:270].T, aspect="auto")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.gca().set_box_aspect(1)
    plt.title("Ey--Xsilce" + str(i))
    plt.savefig(f"figures/Ex_Ey_{i:06d}.pdf")
    plt.close()
