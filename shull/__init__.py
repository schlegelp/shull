import numpy as np

from shull import _shull

class Delaunay:
    def __init__(self, points):
        assert isinstance(points, np.ndarray)
        assert points.ndim == 2
        assert points.shape[1] == 2

        self.points = points.astype(np.float64)
        self.triangles = _shull.calculate_shull_2d(self.points)
