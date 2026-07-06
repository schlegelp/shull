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
computed by sorted incremental insertion, and the downward-facing facets are
exactly the Delaunay tetrahedra.

In both cases all combinatorial decisions (visibility, in-circle/in-sphere
tests) fall back to Shewchuk's exact adaptive predicates (the
[`robust`](https://crates.io/crates/robust) crate) — so the triangulation stays
valid even on adversarial inputs (cospherical grids, tight clusters far apart)
where plain floating point corrupts the result.

## TODOs
- [x] implementation for the 2d case
- [x] generalize from 2 to N dimensions: 3D Delaunay (tetrahedra) via 4D sweep-hull
- [x] improve 2d performance: ~9.5X faster than scipy's Qhull-based Delaunay
- [x] improve 3d performance: ~1.4–1.8X faster than scipy

## Build
1. `cd` into directory
2. Activate virtual environment: `source .venv/bin/activate`
3. Run `maturin develop` (use `maturin build --release` to build wheel)

## Test / benchmark
```
cargo test                      # Rust unit tests (incl. hull invariant checks)
python -m pytest tests/        # property tests + comparison against scipy
python bench.py                # benchmark against scipy.spatial.Delaunay
```

Benchmark on an Apple-silicon laptop (random uniform points, release build):

**2D** (`Delaunay` vs `scipy.spatial.Delaunay`)

| n | shull | scipy (Qhull) | speedup |
|-----------|--------|--------|-------|
| 10 000 | 0.002 s | 0.015 s | 9.5× |
| 100 000 | 0.021 s | 0.20 s | 9.4× |
| 1 000 000 | 0.35 s | 3.4 s | 9.7× |

**3D** (`Delaunay3d` vs `scipy.spatial.Delaunay`)

| n | shull | scipy (Qhull) | speedup |
|-----------|--------|--------|-------|
| 10 000 | 0.05 s | 0.07 s | 1.4× |
| 100 000 | 0.66 s | 1.19 s | 1.8× |
| 1 000 000 | 8.5 s | 14.7 s | 1.7× |

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
       [5462, 1500, 8681]], dtype=uint64)
```

### 3D

```python
>>> pts = np.random.default_rng(12345).random((10000, 3))
>>> d = shull.Delaunay3d(pts)
>>> d.simplices          # (n, 4) tetrahedra, like scipy.spatial.Delaunay
array([[3661, 6693, 7492, 1937],
       ...], dtype=uint64)
```

Notes (both dimensions):
- Output simplices are positively oriented (counterclockwise triangles in 2D,
  positive-volume tetrahedra in 3D) and index into `points`.
- float32 input is supported natively: no upcast copy is made (`d.points`
  keeps the float32 dtype). Coordinates widen to float64 exactly internally,
  so the result is identical to passing `points.astype(np.float64)`. Other
  dtypes are converted to float64.
- Exact duplicate points are dropped; their indices never appear in the
  output.
- Degenerate input (too few distinct points, all points collinear in 2D, all
  points coplanar/cospherical in 3D) raises `ValueError` — a full-dimensional
  triangulation does not exist in those cases.
- Points are triangulated after centering on their centroid (a ≤1-ulp
  perturbation of the coordinates), which makes the result robust to clouds
  positioned far from the origin.
