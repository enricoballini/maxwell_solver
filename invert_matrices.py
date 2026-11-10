"""Scalability test: invert many large random matrices in parallel.

The same COUNT matrices are inverted with 1, 2, 4, ... worker processes, up to the
number of CPU cores the job may use, one worker per core. For each run it prints the
time, the speedup over one worker and the parallel efficiency (speedup divided by the
number of workers).

If the job also has a GPU, it measures how fast the GPU multiplies two large matrices,
in float32 and in float64. On GPUs such as the RTX PRO 4000 Blackwell most cores work
in float32 only, so float64 runs many times slower.

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

N = 3000     # each CPU matrix is N x N, in float64
COUNT = 48   # matrices per run; divisible by 1, 2, 4, 8, 16 and 24, so all workers get the same share
M = 8192     # the GPU multiplies two M x M matrices
REPS = 10    # GPU products per precision


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
    """Multiply two random M x M matrices REPS times on the GPU, in float32 and in float64.

    Returns the GPU name and the speed in TFLOPS (10^12 floating-point operations per
    second) for each precision, or None if the job has no GPU. One product takes about
    2 M^3 operations: M multiplications and M additions for each of the M^2 entries.
    """
    try:
        import cupy as cp
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:  # no CuPy, or no GPU visible to this job
        return None
    tflops = {}
    for dtype in ("float32", "float64"):
        a = cp.random.rand(M, M, dtype=dtype)
        b = cp.random.rand(M, M, dtype=dtype)
        a @ b  # warm-up: loads the GPU libraries before the timing starts
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for _ in range(REPS):
            a @ b
        cp.cuda.Device().synchronize()  # the GPU runs asynchronously: wait until it has finished
        tflops[dtype] = 2 * M**3 * REPS / (time.perf_counter() - start) / 1e12
    return name, tflops


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
        name, tflops = gpu
        print(f"\nGPU {name}: product of two {M} x {M} matrices, {REPS} times per precision")
        for dtype, speed in tflops.items():
            print(f"{dtype:>9}: {speed:6.2f} TFLOPS")
        print(f"float64 is {tflops['float32'] / tflops['float64']:.0f} times slower than float32")
    else:
        print("\nNo GPU in this job: GPU part skipped.")

    print(f"\nLargest error |A A^-1 - I| over all inverses: {worst:.1e}")
    print(f"Total time: {time.perf_counter() - t0:.0f} s")
