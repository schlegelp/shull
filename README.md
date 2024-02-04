# [WIP] shull

S-hull: a fast sweep-hull routine for Delaunay triangulation by David Sinclair
(see http://www.s-hull.org/) implemented in Rust with Python bindings.

Principally based on the pure Python implementation in [pyhull](https://github.com/TimSC/pyshull).

## TODOs
- [x] implementation for the 2d case
- [ ] generalize from 2 to N dimensions
- [ ] improve performance: the expectation is to be ~2X faster than scipy's Qhull-based Delaunay but currently we're about even

## Build
1. `cd` into directory
2. Activate virtual environment: `source .venv/bin/activate`
3. Run `maturin develop` (use `maturin build --release` to build wheel)

## Usage

```python
>>> import shull
>>> import numpy as np
>>> pts = np.random.default_rng(12345).random((10000, 2))
>>> d = shull.Delaunay(pts)
>>> d.triangles
array([[   0, 1953, 1456],
       [1953, 6608, 1456],
       [6608, 4077, 1456],
       ...,
       [9778, 1500, 9364],
       [9364, 1500, 8681],
       [8681, 1500, 5462]], dtype=uint64)
```