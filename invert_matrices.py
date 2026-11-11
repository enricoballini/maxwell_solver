"""Scalability test: multiply large random matrices in parallel, on the CPU and on the GPU.

If the job has a GPU, it first keeps the GPU busy multiplying two large matrices,
SECONDS seconds in float32 and SECONDS in float64, and prints the speed of each; that
is long enough to watch the GPU at 100% with watch -n 1 nvidia-smi. On GPUs such as the
RTX PRO 4000 Blackwell most cores work in float32 only, so float64 runs many times slower.

Then it does the same on the CPU with 1, 2, 4, ... worker processes, up to the number of
CPU cores the job may use, one worker per core: each worker multiplies its own two
matrices for SECONDS seconds per precision (run_cpu). For each run it prints the speed, the speedup
over one worker and the parallel efficiency (speedup divided by the number of workers).

On HPsp the job files invert_matrices_1_gpu_8_cores.slurm, invert_matrices_2_gpu_16_cores.slurm,
invert_matrices_3_fast.slurm and invert_matrices_4_all.slurm run it with the GPU on 8 cores,
with the GPU on 16 cores, without GPU on 8 cores, and without GPU on all 24 cores.
"""
import os

# One thread per worker process: the parallelism comes from the workers, not from
# NumPy's linear-algebra library. This must be set before NumPy is imported.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

N = 3000     # each CPU worker multiplies two N x N matrices
M = 8192     # the GPU multiplies two M x M matrices
SECONDS = 30 # how long the multiplications last, per precision (and on the CPU per number of workers)


def run_cpu():
    """Multiply two random N x N matrices on one CPU core for SECONDS seconds, in float32 and in float64.

    Returns the speed in TFLOPS (10^12 floating-point operations per second) for each
    precision. One product takes about 2 N^3 operations: N multiplications and N additions
    for each of the N^2 entries. The main program runs it in several worker processes at once.
    """
    rng = np.random.default_rng()
    tflops = {}
    for dtype in ("float32", "float64"):
        a = rng.random((N, N), dtype=dtype)
        b = rng.random((N, N), dtype=dtype)
        a @ b  # warm-up: loads the linear-algebra library before the timing starts
        products = 0
        start = time.perf_counter()
        while time.perf_counter() - start < SECONDS:
            a @ b
            products += 1
        tflops[dtype] = 2 * N**3 * products / (time.perf_counter() - start) / 1e12
    return tflops


def run_gpu():
    """Multiply two random M x M matrices on the GPU for SECONDS seconds, in float32 and in float64.

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
        products = 0
        start = time.perf_counter()
        while time.perf_counter() - start < SECONDS:
            a @ b
            cp.cuda.Device().synchronize()  # the GPU runs asynchronously: wait until the product is done
            products += 1
        tflops[dtype] = 2 * M**3 * products / (time.perf_counter() - start) / 1e12
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

    gpu = run_gpu()
    if gpu:
        name, tflops = gpu
        print(f"\nGPU {name}: product of two {M} x {M} matrices, repeated for {SECONDS} s per precision")
        for dtype, speed in tflops.items():
            print(f"{dtype:>9}: {speed:6.2f} TFLOPS")
        print(f"float64 is {tflops['float32'] / tflops['float64']:.0f} times slower than float32")
    else:
        print("\nNo GPU in this job: GPU part skipped.")

    print(f"\nCPU: product of two {N} x {N} matrices in each worker, repeated for {SECONDS} s per precision,")
    print(f"with {', '.join(map(str, counts))} worker processes, one per core.\n")

    print(f"{'':7}  {'float32':^28}  {'float64':^28}")
    print(f"{'workers':>7}" + f"  {'TFLOPS':>8} {'speedup':>8} {'efficiency':>10}" * 2)
    base = None
    for workers in counts:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = [future.result() for future in [pool.submit(run_cpu) for _ in range(workers)]]
        tflops = {dtype: sum(result[dtype] for result in results) for dtype in results[0]}
        base = base or tflops
        row = f"{workers:7d}"
        for dtype, speed in tflops.items():
            speedup = speed / base[dtype]
            row += f"  {speed:8.3f} {speedup:8.2f} {speedup / workers:10.0%}"
        print(row)

    print(f"\nTotal time: {time.perf_counter() - t0:.0f} s")
