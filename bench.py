"""Benchmark shull against scipy.spatial.Delaunay (2D and 3D).

Run with a release build: maturin develop --release && python bench.py

    python bench.py           # sequential vs parallel=True vs scipy
    python bench.py --sweep   # thread scaling of the parallel build
    python bench.py --alpha   # alpha shapes vs scipy-DIY and the alphashape pkg

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


def scipy_diy_alpha(pts, alpha):
    """The realistic pure-scipy alternative: triangulate, filter by
    circumradius, extract boundary facets. Same result as shull's alpha shape;
    shull just replaces the slow parts (the Delaunay build and circumradii)."""
    import scipy.spatial

    d = scipy.spatial.Delaunay(pts)
    s = d.simplices
    p = pts[s]
    u = p[:, 1:, :] - p[:, :1, :]
    rhs = 0.5 * np.einsum("mij,mij->mi", u, u)
    y = np.linalg.solve(u, rhs[..., None])[..., 0]
    r = np.linalg.norm(y, axis=1)
    inside = r <= alpha
    nbr_out = (d.neighbors == -1) | ~inside[d.neighbors]
    i, j = np.nonzero(inside[:, None] & nbr_out)
    k = s.shape[1]
    cols = np.array([[c for c in range(k) if c != jj] for jj in range(k)])
    return s[i[:, None], cols[j]]


def alpha_bench(ndim, sizes, alpha=0.05):
    print(f"{ndim}D alpha shape (radius scale {alpha})")
    try:
        import alphashape  # optional; heavy shapely/trimesh stack
    except ImportError:
        alphashape = None
    print(f"{'n':>9} {'shull':>10} {'scipy DIY':>11} {'vs DIY':>8} "
          f"{'alphashape':>12} {'vs pkg':>9}")
    for n, repeats in sizes:
        pts = np.random.default_rng(42).random((n, ndim))
        sh = best_of(lambda: shull.AlphaShape(pts, alpha=alpha).boundary, repeats)
        diy = best_of(lambda: scipy_diy_alpha(pts, alpha), repeats)
        row = f"{n:>9} {sh:>9.4f}s {diy:>10.3f}s {diy / sh:>7.1f}x"
        # The alphashape package uses 1/radius and does not scale past ~1e4.
        if alphashape is not None and n <= 20_000:
            pk = best_of(lambda: alphashape.alphashape(pts, 1.0 / alpha), 1)
            row += f" {pk:>11.3f}s {pk / sh:>8.0f}x"
        else:
            row += f" {'(skipped)':>12} {'':>9}"
        print(row)
    print()


def main():
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        child(int(sys.argv[i + 1]), int(sys.argv[i + 2]), int(sys.argv[i + 3]))
    elif "--sweep" in sys.argv:
        sweep(2, 1_000_000)
        sweep(3, 1_000_000)
    elif "--alpha" in sys.argv:
        alpha_bench(2, [(1_000, 5), (20_000, 3), (100_000, 3), (1_000_000, 1)])
        alpha_bench(3, [(1_000, 5), (20_000, 3), (200_000, 1)], alpha=0.08)
    else:
        run("2D", 2, [(10_000, 5), (100_000, 3), (1_000_000, 1)])
        run("3D", 3, [(10_000, 5), (100_000, 3), (1_000_000, 1)])


if __name__ == "__main__":
    main()
