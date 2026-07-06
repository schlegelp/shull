"""Benchmark shull against scipy.spatial.Delaunay (2D and 3D).

Run with a release build: maturin develop --release && python bench.py
"""
import time

import numpy as np
import scipy.spatial

import shull


def best_of(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def run(title, ours_cls, ndim, sizes):
    print(title)
    print(f"{'n':>9} {'shull':>10} {'scipy':>10} {'ratio':>7}")
    for n, repeats in sizes:
        pts = np.random.default_rng(42).random((n, ndim))
        ours = best_of(lambda: ours_cls(pts), repeats)
        theirs = best_of(lambda: scipy.spatial.Delaunay(pts), repeats)
        print(f"{n:>9} {ours:>9.3f}s {theirs:>9.3f}s {theirs / ours:>6.2f}x")
    print()


def main():
    run("2D", shull.Delaunay, 2, [(10_000, 5), (100_000, 3), (1_000_000, 1)])
    run("3D", shull.Delaunay3d, 3, [(10_000, 5), (100_000, 3), (1_000_000, 1)])


if __name__ == "__main__":
    main()
