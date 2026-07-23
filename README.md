# shull

_S-hull: a fast sweep-hull routine for Delaunay triangulation_ by David Sinclair
(see http://www.s-hull.org/) implemented in Rust with Python bindings.

The 2D case implements S-hull proper: a seed triangle with the smallest
circumcircle, a radial sweep over points sorted by distance from its
circumcenter, hull attachment via a linked ring with a pseudo-angle hash, and
in-circle edge flipping to restore the Delaunay condition.

The 3D case generalizes Sinclair's _Newton Apple Wrapper_ sweep-hull algorithm
([arXiv 1602.04707](https://arxiv.org/abs/1602.04707)) one dimension up: points
are lifted onto a 4D paraboloid (`w = x² + y² + z²`), the 4D convex hull is
computed by incremental insertion along a Morton (Z-order) space-filling curve
— every lifted point is extreme on the paraboloid, so any insertion order is
valid, and the spatially local one keeps the hull walk cache-hot — and the
downward-facing facets are exactly the Delaunay tetrahedra.

In both cases all combinatorial decisions (visibility, in-circle/in-sphere
tests) fall back to Shewchuk's exact adaptive predicates (the
[`robust`](https://crates.io/crates/robust) crate) — so the triangulation stays
valid even on adversarial inputs (cospherical grids, tight clusters far apart)
where plain floating point corrupts the result.

## Install

We provide prebuilt wheels for Linux, macOS, and Windows on PyPI:

```bash
pip install shull
```

## Usage

### 2D

```python
>>> import shull
>>> import numpy as np
>>> pts = np.random.default_rng(12345).random((10000, 2))
>>> d = shull.Delaunay(pts)
>>> d.triangles          # alias: d.simplices
array([[5790, 4665, 8764],
       [4665, 9599, 8764],
       [7711,   64, 4665],
       ...,
       [2821, 8418, 1189],
       [1500, 9364, 8681],
       [5462, 1500, 8681]], dtype=int32)
```

### 3D

```python
>>> pts = np.random.default_rng(12345).random((10000, 3))
>>> d = shull.Delaunay(pts)  # dispatches on the number of columns;
>>> d.simplices              # Delaunay3d is kept as an explicit alias
array([[3661, 6693, 7492, 1937],
       ...], dtype=int32)
```

### scipy compatibility

`shull.Delaunay` aims to be a drop-in replacement for
`scipy.spatial.Delaunay`. Beyond `points` and `simplices` (int32, like
scipy) it provides the derived structures. `neighbors` comes straight out
of the triangulation (the hull construction maintains facet adjacency
anyway, so exporting it is essentially free, like qhull) and
`vertex_neighbor_vertices` is built in Rust on first access (~2x faster
than scipy's); the rest are computed lazily in numpy and cached:

- `neighbors` — neighboring simplex opposite each vertex, -1 at the boundary
- `convex_hull` — facets of the convex hull
- `vertex_to_simplex` — a simplex containing each vertex
- `vertex_neighbor_vertices` — CSR `(indptr, indices)` vertex adjacency
- `coplanar` — points not in the triangulation (dropped exact duplicates),
  mapped to their kept representative; recorded by the Rust core during its
  dedup, not reconstructed after the fact
- `transform` — barycentric transforms, same layout as scipy
- `find_simplex(xi, bruteforce=False, tol=None)` — point location via a
  vectorized visibility walk (brute force as option/fallback)
- `npoints`, `nsimplex`, `ndim`, `min_bound`, `max_bound`, `furthest_site`,
  `close()`

Not implemented: `equations` (and `paraboloid_scale`/`paraboloid_shift`,
`plane_distance`, `lift_points`), incremental mode (`add_points`),
`furthest_site=True` and `qhull_options` — the constructor accepts scipy's
keyword arguments but raises `NotImplementedError` for non-default values.
Unlike scipy, float32 `points` are kept as float32 (see below).

Notes (both dimensions):
- Output simplices are positively oriented (counterclockwise triangles in 2D,
  positive-volume tetrahedra in 3D) and index into `points`.
- float32 input is supported natively: no upcast copy is made (`d.points`
  keeps the float32 dtype). Coordinates widen to float64 exactly internally,
  so the result is identical to passing `points.astype(np.float64)`. Other
  dtypes are converted to float64.
- Exact duplicate points are dropped, keeping the first occurrence as the
  representative. scipy/Qhull likewise never includes duplicate indices in
  `simplices` (though which copy Qhull keeps is arbitrary, while shull's
  choice is deterministic). Dropped points are reported in `coplanar`
  (scipy's convention); the raw `calculate_shull_*` functions return the
  `(dropped, kept)` index pairs as a third array.
- Degenerate input (too few distinct points, all points collinear in 2D, all
  points coplanar/cospherical in 3D) raises `ValueError` — a full-dimensional
  triangulation does not exist in those cases.
- Points are triangulated after centering on their centroid (a ≤1-ulp
  perturbation of the coordinates), which makes the result robust to clouds
  positioned far from the origin.

## Use from Rust

The crate is also usable as a plain Rust library: the Python bindings sit
behind an off-by-default `python` cargo feature, so depending on `shull`
pulls in only `ndarray` and `robust` — no pyo3, no Python at build time, no
libpython in your binary.

```toml
[dependencies]
shull = { git = "https://github.com/schlegelp/shull" }
ndarray = "0.17"
```

```rust
use ndarray::Array2;
use shull::{delaunay2d, delaunay4d, csr_adjacency};

let pts: Array2<f64> = /* (n, 2) array */;
// triangles (ccw), neighbor triangle opposite each vertex (-1 on the hull),
// and (dropped, kept) index pairs for exact duplicate points
let (triangles, neighbors, duplicates) = delaunay2d(pts.view())?;
// 3D points (n, 3) -> tetrahedra, same return layout:
let (tetrahedra, neighbors, duplicates) = delaunay4d(pts3.view())?;
```

`csr_adjacency` builds the scipy-style `(indptr, indices)` vertex adjacency
from a simplex array. Degenerate or oversized input is reported as a
`DelaunayError` rather than a panic.

## Parallel builds

For large point clouds the triangulation can be built on multiple threads:

```python
d = shull.Delaunay(pts, parallel=True)   # opt-in; default is False
```

The cloud is partitioned into spatially compact blocks that are triangulated
concurrently — each by the *unchanged* sequential kernel, on its points plus a
one-cell halo of neighbors. A per-point certificate (all incident circumballs
covered by the gathered region, with a rigorously conservative float margin)
decides which local results are provably part of the global triangulation; the
uncertain remainder (block borders, the convex hull, outliers) is
re-triangulated in a single "crust" pass whose output is verified against the
whole cloud with exact predicates. The merge then cross-checks every seam
(exact local-Delaunay tests across block boundaries, face counts, boundary
closure, no missing points) and, if anything is off — typically exactly
cocircular/cospherical inputs split across blocks — the build transparently
falls back to the plain sequential kernel. **The parallel path never returns a
wrong mesh**: for inputs in general position the simplex set is identical to
the sequential build (only the row order differs and is unspecified);
degenerate ties are either resolved identically-to-fallback or consistently
within one block (a valid Delaunay triangulation with different tie-breaks).

Details worth knowing:

- Only worthwhile for large clouds: inputs below ~100k points run
  sequentially even with `parallel=True`.
- `progress=True` renders a self-overwriting progress line on stderr;
  passing a callable instead receives `(stage, done, total)` events
  (`"blocks"` with a running count, then `"crust"`, `"merge"`, `"done"`,
  or `"fallback"` right before the sequential kernel takes over) — easy to
  hook up to tqdm. From Rust: `delaunay2d_par_with_progress` /
  `delaunay4d_par_with_progress` take a `Fn(ParProgress) + Sync` callback.
  The sequential build itself reports no progress (it is untouched by the
  parallel feature).
- Threads come from rayon's global pool (`RAYON_NUM_THREADS` to control);
  the *result* is bit-identical for any thread count.
- Measured on an Apple M3 Max (10 performance + 4 efficiency cores),
  uniform random points, wall-clock speedup over the sequential build:
  ~2.8× (3D, 1M points), ~3.2× (3D, 5–10M), ~2.5× (2D, 1M), ~3.4×
  (2D, 5M) with all 14 threads; ~2.5× (3D, 10M) with 8 threads. The block
  stage scales with cores until memory bandwidth saturates; the merge adds
  a fixed ~O(n) overhead, so speedups grow with cloud size.
- From Rust: enable the off-by-default `parallel` cargo feature and call
  `delaunay2d_par` / `delaunay4d_par` (same signatures and return values as
  the sequential functions). `delaunay2d_par_with_stats` /
  `delaunay4d_par_with_stats` additionally return a `ParStats` with build
  diagnostics (block count, crust size, per-stage times, and whether — and
  why — the build fell back to the sequential kernel). The published wheels
  always enable the feature; pure-Rust consumers who skip it don't pull in
  rayon.

## Build
1. `cd` into directory
2. Activate virtual environment: `source .venv/bin/activate`
3. Run `maturin develop` (use `maturin build --release` to build wheel)

## Test / benchmark
```
cargo test --features parallel # Rust unit tests (incl. hull invariant checks)
python -m pytest tests/        # property tests + comparison against scipy
python bench.py                # benchmark against scipy.spatial.Delaunay
python bench.py --sweep        # thread scaling of the parallel build
```

Benchmark on an Apple-silicon laptop (random uniform points, release build):

**2D** (`Delaunay` vs `scipy.spatial.Delaunay`)

| n | shull | scipy (Qhull) | speedup |
|-----------|--------|--------|-------|
| 10 000 | 0.002 s | 0.018 s | 9.8× |
| 100 000 | 0.029 s | 0.30 s | 10.7× |
| 1 000 000 | 0.46 s | 5.0 s | 10.9× |

**3D** (`Delaunay3d` vs `scipy.spatial.Delaunay`)

| n | shull | scipy (Qhull) | speedup |
|-----------|--------|--------|-------|
| 10 000 | 0.031 s | 0.098 s | 3.2× |
| 100 000 | 0.32 s | 1.69 s | 5.3× |
| 1 000 000 | 3.8 s | 20.8 s | 5.5× |