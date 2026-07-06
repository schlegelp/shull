import numpy as np

from shull import _shull

__all__ = ['Delaunay', 'Delaunay3d']


class Delaunay:
    """2D Delaunay triangulation via S-hull (radial sweep-hull).

    Notes
    -----
    - `triangles` (alias `simplices`) is an (n, 3) array of indices into
      `points`, one counterclockwise triangle per row.
    - float32 input is supported natively (no upcast copy is made and
      `points` keeps its dtype); coordinates widen to float64 exactly
      internally. Any other dtype is converted to float64.
    - Exact duplicate points are dropped (one representative is kept).
    - Degenerate input (fewer than 3 distinct points, or all points
      collinear) raises ValueError.
    """

    def __init__(self, points):
        assert isinstance(points, np.ndarray)
        assert points.ndim == 2
        assert points.shape[1] == 2

        if points.dtype == np.float32:
            self.points = points
            self.triangles = _shull.calculate_shull_2d_f32(points)
        else:
            self.points = points.astype(np.float64, copy=False)
            self.triangles = _shull.calculate_shull_2d(self.points)
        # Alias matching scipy.spatial.Delaunay
        self.simplices = self.triangles


class Delaunay3d:
    """3D Delaunay tetrahedralization via a 4D sweep-hull.

    Notes
    -----
    - `simplices` is an (n, 4) array of indices into `points`, one row per
      tetrahedron, positively oriented (matching scipy.spatial.Delaunay).
    - float32 input is supported natively (no upcast copy is made and
      `points` keeps its dtype); coordinates widen to float64 exactly
      internally, so the result is identical to passing
      `points.astype(np.float64)`. Any other dtype is converted to float64.
    - Exact duplicate points are dropped; their indices never appear in
      `simplices`.
    - Degenerate input (all points coplanar/cospherical, or fewer than 5
      distinct points) raises ValueError.
    """

    def __init__(self, points):
        assert isinstance(points, np.ndarray)
        assert points.ndim == 2
        assert points.shape[1] == 3

        if points.dtype == np.float32:
            self.points = points
            self.simplices = _shull.calculate_shull_3d_f32(points)
        else:
            self.points = points.astype(np.float64, copy=False)
            self.simplices = _shull.calculate_shull_3d(self.points)
