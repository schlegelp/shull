import functools

import numpy as np

from shull import _shull

__all__ = ['Delaunay', 'Delaunay3d']


def _facet_columns(m):
    """(m, m-1) column indices: row j = all positions except j.

    Row j selects the facet opposite vertex j of an m-vertex simplex.
    """
    return np.array([[k for k in range(m) if k != j] for j in range(m)])


class Delaunay:
    """Delaunay triangulation of 2D or 3D points via sweep-hull (S-hull).

    Aims to be a drop-in replacement for `scipy.spatial.Delaunay`. The
    triangulation is computed in Rust, which also exports `neighbors` (the
    hull construction maintains that adjacency anyway) and builds
    `vertex_neighbor_vertices` on first access; the remaining derived
    structures are computed lazily in numpy and cached.

    scipy-compatible attributes
    ---------------------------
    - `points`: the input points.
    - `simplices`: (nsimplex, ndim+1) int32 indices into `points`, one
      positively oriented simplex per row (counterclockwise triangles in
      2D, positive-volume tetrahedra in 3D). In 2D also aliased as
      `triangles`.
    - `neighbors`: (nsimplex, ndim+1) int32; the k-th neighbor is opposite
      the k-th vertex, -1 where there is no neighbor.
    - `convex_hull`: (nfacet, ndim) int32 vertex indices of the facets on
      the convex hull.
    - `vertex_to_simplex`: (npoints,) int32 lookup from a vertex to one
      simplex containing it; -1 for points not in the triangulation.
    - `vertex_neighbor_vertices`: CSR tuple `(indptr, indices)` of int32;
      the vertices adjacent to vertex k are
      `indices[indptr[k]:indptr[k+1]]` (sorted ascending).
    - `coplanar`: (ncoplanar, 3) int32 rows of [point index, index of a
      simplex containing the nearest included point, index of the nearest
      included point]. Here these are exactly the dropped duplicate points.
    - `transform`: (nsimplex, ndim+1, ndim) float64 barycentric transforms;
      `transform[i, :ndim, :]` is the inverse of T and `transform[i, ndim, :]`
      is the vector r such that ``c = transform[i, :ndim, :] @ (x - r)``
      gives the first ndim barycentric coordinates of x in simplex i.
    - `npoints`, `nsimplex`, `ndim`, `min_bound`, `max_bound`,
      `furthest_site`.
    - `find_simplex(xi, bruteforce=False, tol=None)` and `close()`.

    Not implemented: `equations`, `paraboloid_scale`/`paraboloid_shift`,
    `plane_distance`, `lift_points`, incremental mode (`add_points`),
    `furthest_site=True` and `qhull_options`.

    Notes (deliberate divergences from scipy)
    -----------------------------------------
    - float32 input is supported natively (no upcast copy is made and
      `points` keeps its dtype, whereas scipy always converts to float64);
      coordinates widen to float64 exactly internally, so the result is
      identical to passing `points.astype(np.float64)`. Any other dtype is
      converted to float64.
    - Exact duplicate points are dropped (one representative is kept);
      dropped points are reported in `coplanar`.
    - Degenerate input (too few distinct points, all points collinear in
      2D, all points coplanar/cospherical in 3D) raises ValueError — a
      full-dimensional triangulation does not exist in those cases.
    """

    furthest_site = False

    def __init__(self, points, furthest_site=False, incremental=False,
                 qhull_options=None):
        if furthest_site:
            raise NotImplementedError("furthest_site=True is not supported")
        if incremental:
            raise NotImplementedError("incremental mode is not supported")
        if qhull_options is not None:
            raise NotImplementedError(
                "qhull_options are not supported (shull does not use Qhull)"
            )

        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("points must have shape (n, 2) or (n, 3)")

        if points.dtype == np.float32:
            self.points = points
            calculate = (_shull.calculate_shull_2d_f32 if points.shape[1] == 2
                         else _shull.calculate_shull_3d_f32)
        else:
            self.points = points.astype(np.float64, copy=False)
            calculate = (_shull.calculate_shull_2d if points.shape[1] == 2
                         else _shull.calculate_shull_3d)
        self.simplices, self.neighbors = calculate(self.points)
        if points.shape[1] == 2:
            self.triangles = self.simplices

    @property
    def ndim(self):
        return self.points.shape[1]

    @property
    def npoints(self):
        return self.points.shape[0]

    @property
    def nsimplex(self):
        return self.simplices.shape[0]

    @property
    def min_bound(self):
        return self._points64.min(axis=0)

    @property
    def max_bound(self):
        return self._points64.max(axis=0)

    @functools.cached_property
    def _points64(self):
        return self.points.astype(np.float64, copy=False)

    @functools.cached_property
    def convex_hull(self):
        i, j = np.nonzero(self.neighbors == -1)
        return self.simplices[i[:, None], _facet_columns(self.simplices.shape[1])[j]]

    @functools.cached_property
    def vertex_to_simplex(self):
        out = np.full(self.npoints, -1, dtype=np.int32)
        out[self.simplices] = np.arange(self.nsimplex, dtype=np.int32)[:, None]
        return out

    @functools.cached_property
    def vertex_neighbor_vertices(self):
        return _shull.vertex_neighbor_vertices(self.simplices, self.npoints)

    @functools.cached_property
    def coplanar(self):
        used = np.zeros(self.npoints, dtype=bool)
        used[self.simplices] = True
        dropped = np.nonzero(~used)[0]
        if dropped.size == 0:
            return np.empty((0, 3), dtype=np.int32)
        # Every dropped point is an exact duplicate of a kept one: group by
        # coordinates and map each dropped point to the kept member of its
        # group.
        _, group = np.unique(self.points, axis=0, return_inverse=True)
        group = group.reshape(-1)
        kept = np.nonzero(used)[0]
        rep_of_group = np.empty(group.max() + 1, dtype=np.int64)
        rep_of_group[group[kept]] = kept
        reps = rep_of_group[group[dropped]]
        out = np.empty((dropped.size, 3), dtype=np.int32)
        out[:, 0] = dropped
        out[:, 1] = self.vertex_to_simplex[reps]
        out[:, 2] = reps
        return out

    @functools.cached_property
    def transform(self):
        pts = self._points64
        s = self.simplices
        d = self.ndim
        r = pts[s[:, d]]
        # Columns of T are (v_k - r); barycentric coords solve T c = x - r.
        T = np.swapaxes(pts[s[:, :d]] - r[:, None, :], 1, 2)
        out = np.empty((self.nsimplex, d + 1, d))
        try:
            out[:, :d, :] = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            # Shouldn't happen (simplices have positive volume), but mirror
            # scipy: NaN for degenerate simplices instead of raising.
            for i in range(self.nsimplex):
                try:
                    out[i, :d, :] = np.linalg.inv(T[i])
                except np.linalg.LinAlgError:
                    out[i, :d, :] = np.nan
        out[:, d, :] = r
        return out

    def find_simplex(self, xi, bruteforce=False, tol=None):
        """Find the simplices containing the given points.

        Parameters
        ----------
        xi : ndarray of double, shape (..., ndim)
            Points to locate.
        bruteforce : bool, optional
            Whether to do a brute-force search instead of a walk.
        tol : float, optional
            Tolerance allowed in the inside-simplex check.
            Default is ``100*eps``.

        Returns
        -------
        i : ndarray of int32, same shape as `xi` without the last axis
            Indices of simplices containing each point; -1 for points
            outside the triangulation.
        """
        xi = np.asarray(xi, dtype=np.float64)
        d = self.ndim
        if xi.shape[-1] != d:
            raise ValueError(f"xi has different dimensionality than the "
                             f"triangulation ({xi.shape[-1]} vs {d})")
        eps = 100 * np.finfo(np.float64).eps if tol is None else float(tol)
        x = xi.reshape(-1, d)
        if bruteforce:
            out = self._find_simplex_bruteforce(x, eps)
        else:
            out = self._find_simplex_walk(x, eps)
        return out.reshape(xi.shape[:-1])

    def _barycentric(self, simplex_idx, x):
        """Full barycentric coordinates of x[k] in simplices[simplex_idx[k]]."""
        Tinv = self.transform[simplex_idx, :self.ndim, :]
        r = self.transform[simplex_idx, self.ndim, :]
        c = np.einsum('nij,nj->ni', Tinv, x - r)
        return np.concatenate([c, 1.0 - c.sum(axis=1, keepdims=True)], axis=1)

    def _find_simplex_walk(self, x, eps):
        out = np.full(x.shape[0], -1, dtype=np.int32)
        cur = np.zeros(x.shape[0], dtype=np.int64)
        alive = np.arange(x.shape[0])
        # A visibility walk crosses O(nsimplex ** (1/ndim)) simplices for
        # points in general position; anything still walking after this many
        # steps (e.g. cycling on a degenerate facet) falls back to brute
        # force.
        max_steps = 4 * int(np.ceil(self.nsimplex ** (1.0 / self.ndim))) + 64
        for _ in range(max_steps):
            if alive.size == 0:
                return out
            c = self._barycentric(cur[alive], x[alive])
            worst = np.argmin(c, axis=1)
            inside = c[np.arange(c.shape[0]), worst] >= -eps
            out[alive[inside]] = cur[alive[inside]]
            # Step through the facet of the most negative coordinate; a
            # missing neighbor there means the point is beyond a hull facet,
            # i.e. outside the triangulation (result stays -1).
            walking = alive[~inside]
            nxt = self.neighbors[cur[walking], worst[~inside]]
            walking = walking[nxt != -1]
            cur[walking] = nxt[nxt != -1]
            alive = walking
        if alive.size:
            out[alive] = self._find_simplex_bruteforce(x[alive], eps)
        return out

    def _find_simplex_bruteforce(self, x, eps):
        out = np.full(x.shape[0], -1, dtype=np.int32)
        Tinv = self.transform[:, :self.ndim, :]
        r = self.transform[:, self.ndim, :]
        # Chunk queries so the (chunk, nsimplex, ndim+1) intermediate stays
        # around ~100 MB.
        chunk = max(1, int(2 ** 24 / max(1, self.nsimplex * (self.ndim + 1))))
        for start in range(0, x.shape[0], chunk):
            xs = x[start:start + chunk]
            c = np.einsum('sij,qsj->qsi', Tinv, xs[:, None, :] - r[None, :, :])
            cmin = np.minimum(c.min(axis=2), 1.0 - c.sum(axis=2))
            best = np.argmax(cmin, axis=1)
            ok = cmin[np.arange(xs.shape[0]), best] >= -eps
            out[start:start + chunk][ok] = best[ok]
        return out

    def close(self):
        """No-op, for compatibility with scipy.spatial.Delaunay."""


class Delaunay3d(Delaunay):
    """3D Delaunay tetrahedralization via a 4D sweep-hull.

    Same as `Delaunay` but only accepts (n, 3) input. Kept for backward
    compatibility; `Delaunay` now handles both 2D and 3D points.
    """

    def __init__(self, points, **kwargs):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        super().__init__(points, **kwargs)
