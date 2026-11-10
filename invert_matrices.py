"""Scalability test: invert many large random matrices in parallel.

The same COUNT matrices are inverted with 1, 2, 4, ... worker processes, up to the
number of CPU cores the job may use, one worker per core; if the job also has a GPU,
they are inverted there too. For each run it prints the time, the speedup over one
worker and the parallel efficiency (speedup divided by the number of workers).

On HPsp the job files invert_matrices_1_gpu.slurm, invert_matrices_2_fast.slurm and
invert_matrices_3_all.slurm run it with the GPU on the 8 efficiency cores, on the
8 performance cores, and on all 24 cores.
"""
import os

# One thread per worker process: the parallelism comes from the workers, not from
# NumPy's linear-algebra library. This must be set before NumPy is imported.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

N = 3000     # each matrix is N x N, in float64
COUNT = 48   # matrices per run; divisible by 1, 2, 4, 8, 16 and 24, so all workers get the same share


def invert(seed):
    """Invert one random matrix and return the largest entry of A A^-1 - I."""
    # random entries in [0, 1) plus N on the diagonal: invertible and well conditioned
    a = np.random.default_rng(seed).random((N, N)) + N * np.eye(N)
    return np.abs(a @ np.linalg.inv(a) - np.eye(N)).max()


def run_cpu(workers):
    """Invert COUNT matrices with the given number of worker processes; return time and largest error."""
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        errors = list(pool.map(invert, range(COUNT)))
    return time.perf_counter() - start, max(errors)


def run_gpu():
    """Invert COUNT matrices on the GPU, if the job has one; return its name, the time and the largest error."""
    try:
        import cupy as cp
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:  # no CuPy, or no GPU visible to this job
        return None
    eye = cp.eye(N)
    cp.linalg.inv(eye)  # warm-up: loads the GPU libraries before the timing starts
    worst = 0.0
    start = time.perf_counter()
    for _ in range(COUNT):
        a = cp.random.rand(N, N) + N * eye
        worst = max(worst, float(cp.abs(a @ cp.linalg.inv(a) - eye).max()))
    return name, time.perf_counter() - start, worst


def as_ranges(numbers):
    """Write sorted integers compactly: [0, 1, 2, 5] -> '0-2,5'."""
    parts = []
    start = prev = numbers[0]
    for n in numbers[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        if n is not None:
            start = prev = n
    return ",".join(parts)


if __name__ == "__main__":
    t0 = time.perf_counter()
    cores = sorted(os.sched_getaffinity(0))
    counts = sorted({w for w in (1, 2, 4, 8, 16, 24) if w <= len(cores)} | {len(cores)})

    print(f"Slurm job: {os.environ.get('SLURM_JOB_ID', 'none (not started by Slurm)')}")
    print(f"CPU cores this job may use: {len(cores)} ({as_ranges(cores)})")
    print(f"Each run inverts {COUNT} random {N} x {N} matrices (float64, {N * N * 8 / 1e6:.0f} MB each)")
    print(f"with {', '.join(map(str, counts))} worker processes, one per core.\n")

    print(f"{'workers':>7} {'time [s]':>9} {'speedup':>8} {'efficiency':>11} {'matrices/s':>11}")
    worst = 0.0
    base = None
    for workers in counts:
        elapsed, error = run_cpu(workers)
        base = base or elapsed
        worst = max(worst, error)
        speedup = base / elapsed
        print(f"{workers:7d} {elapsed:9.1f} {speedup:8.2f} {speedup / workers:11.0%} {COUNT / elapsed:11.2f}")

    gpu = run_gpu()
    if gpu:
        name, elapsed, error = gpu
        worst = max(worst, error)
        print(f"\nGPU {name}: {COUNT} matrices in {elapsed:.1f} s, {COUNT / elapsed:.2f} matrices/s")
    else:
        print("\nNo GPU in this job: GPU part skipped.")

    print(f"\nLargest error |A A^-1 - I| over all inverses: {worst:.1e}")
    print(f"Total time: {time.perf_counter() - t0:.0f} s")
