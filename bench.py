"""Benchmark shull against scipy.spatial.Delaunay (2D and 3D).

Run with a release build: maturin develop --release && python bench.py

    python bench.py           # sequential vs parallel=True vs scipy
    python bench.py --sweep   # thread scaling of the parallel build

The sweep uses subprocesses because rayon's global thread pool reads
RAYON_NUM_THREADS once, at first use, per process.
"""
import os
import subprocess
import sys
import time

import numpy as np

import shull


def best_of(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def run(title, ndim, sizes):
    import scipy.spatial

    print(title)
    print(f"{'n':>9} {'shull':>10} {'parallel':>10} {'scipy':>10} "
          f"{'par speedup':>12} {'vs scipy':>9}")
    for n, repeats in sizes:
        pts = np.random.default_rng(42).random((n, ndim))
        seq = best_of(lambda: shull.Delaunay(pts), repeats)
        par = best_of(lambda: shull.Delaunay(pts, parallel=True), repeats)
        sp = best_of(lambda: scipy.spatial.Delaunay(pts), repeats)
        print(f"{n:>9} {seq:>9.3f}s {par:>9.3f}s {sp:>9.3f}s "
              f"{seq / par:>11.2f}x {sp / par:>8.2f}x")
    print()


def child(n, ndim, repeats):
    """One timing measurement, isolated so RAYON_NUM_THREADS takes effect."""
    pts = np.random.default_rng(42).random((n, ndim))
    print(best_of(lambda: shull.Delaunay(pts, parallel=True), repeats))


def sweep(ndim, n, repeats=3):
    print(f"{ndim}D parallel build, n = {n:,}: thread scaling")
    pts = np.random.default_rng(42).random((n, ndim))
    base = best_of(lambda: shull.Delaunay(pts), repeats)
    print(f"{'threads':>8} {'time':>10} {'speedup vs sequential':>22}")
    max_threads = os.cpu_count() or 8
    threads = sorted({1, 2, 4, 8, max_threads} & set(range(1, max_threads + 1)))
    for t in threads:
        env = dict(os.environ, RAYON_NUM_THREADS=str(t))
        out = subprocess.run(
            [sys.executable, __file__, "--child", str(n), str(ndim), str(repeats)],
            env=env, capture_output=True, text=True, check=True,
        )
        par = float(out.stdout.strip().splitlines()[-1])
        print(f"{t:>8} {par:>9.3f}s {base / par:>21.2f}x")
    print()


def main():
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        child(int(sys.argv[i + 1]), int(sys.argv[i + 2]), int(sys.argv[i + 3]))
    elif "--sweep" in sys.argv:
        sweep(2, 1_000_000)
        sweep(3, 1_000_000)
    else:
        run("2D", 2, [(10_000, 5), (100_000, 3), (1_000_000, 1)])
        run("3D", 3, [(10_000, 5), (100_000, 3), (1_000_000, 1)])


if __name__ == "__main__":
    main()
