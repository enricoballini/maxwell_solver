import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results/figures", exist_ok=True)

results = np.load("results/line_convergence_test.npz")
cell_sizes_km = results["cell_sizes"] / 1e3
figure, axes = plt.subplots(figsize=(4.2, 3.2), layout="constrained")
for frequency_shift_maximum, errors in zip(
    results["frequency_shifts_maximum"], results["errors"]
):
    shift_text = f"{frequency_shift_maximum:g}"
    if "e" in shift_text:
        shift_text = f"10^{{{int(shift_text.split('e')[1])}}}"
    axes.loglog(cell_sizes_km, errors, "o-", label=rf"$\alpha_{{\max}} = {shift_text}$ S/m")
axes.loglog(
    cell_sizes_km,
    results["errors"][-1, 0] * (cell_sizes_km / cell_sizes_km[0]) ** 2,
    "k--",
    label=r"$\propto\Delta^2$",
)
axes.set_xticks(cell_sizes_km)
axes.set_xticklabels([f"{size:g}" for size in cell_sizes_km])
axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes.set_xlabel(r"$\Delta$ (km), $\Delta t = 0.99\,\Delta/(\sqrt{3}\,c)$")
axes.set_ylabel(r"$e$")
axes.set_ylim(1e-5, 1)
axes.set_title(r"\texttt{maxwell\_solver} (GPU), 3D")
axes.grid(True, which="both")
axes.legend(loc="center left")
figure.savefig("results/figures/line_convergence_test.pdf")
