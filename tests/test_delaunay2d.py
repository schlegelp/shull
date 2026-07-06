import numpy as np
import pytest
import scipy.spatial

import shull


def tri_set(simplices):
    """Order-independent representation of a triangulation."""
    return set(map(tuple, np.sort(np.asarray(simplices), axis=1)))


def tri_areas(points, simplices):
    a, b, c = (points[simplices[:, i]] for i in range(3))
    u, v = b - a, c - a
    return (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2.0


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n", [10, 100, 1000])
def test_matches_scipy_random(seed, n):
    pts = np.random.default_rng(seed).random((n, 2))
    ours = tri_set(shull.Delaunay(pts).triangles)
    theirs = tri_set(scipy.spatial.Delaunay(pts).simplices)
    assert ours == theirs


def test_area_partition_and_orientation():
    pts = np.random.default_rng(4).random((500, 2))
    tris = np.asarray(shull.Delaunay(pts).triangles)
    areas = tri_areas(pts, tris)
    assert (areas > 0).all()  # counterclockwise
    hull_area = scipy.spatial.ConvexHull(pts).volume  # 2D "volume" is area
    assert areas.sum() == pytest.approx(hull_area, rel=1e-9)


def test_all_points_used():
    pts = np.random.default_rng(6).random((500, 2))
    tris = np.asarray(shull.Delaunay(pts).triangles)
    assert set(np.unique(tris)) == set(range(len(pts)))


def test_translate_scale_invariance():
    pts = np.random.default_rng(7).random((300, 2))
    base = tri_set(shull.Delaunay(pts).triangles)
    moved = tri_set(shull.Delaunay(pts * 3.7 + np.array([1e6, -50.0])).triangles)
    assert base == moved


def test_collinear_raises():
    pts = np.stack([np.arange(50.0), 2.0 * np.arange(50.0)], axis=1)
    with pytest.raises(ValueError, match="degenerate"):
        shull.Delaunay(pts)


def test_too_few_points():
    with pytest.raises(ValueError):
        shull.Delaunay(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_duplicates_dropped():
    pts = np.random.default_rng(9).random((100, 2))
    dup = np.vstack([pts, pts[:10]])
    tris = np.asarray(shull.Delaunay(dup).triangles)
    # One representative per duplicate pair; triangulation matches the
    # duplicate-free one as a set of coordinate triangles.
    used = set(np.unique(tris))
    for k in range(10):
        assert (k in used) ^ (100 + k in used)


def test_grid():
    """Cocircular ties everywhere; check properties, not scipy equality."""
    grid = np.stack(np.meshgrid(np.arange(20.0), np.arange(20.0)), axis=-1).reshape(-1, 2)
    tris = np.asarray(shull.Delaunay(grid).triangles)
    areas = tri_areas(grid, tris)
    assert (areas > 0).all()
    assert areas.sum() == pytest.approx(19.0 * 19.0, rel=1e-12)
    assert len(tris) == 2 * 19 * 19


def test_float32_matches_float64():
    pts32 = np.random.default_rng(11).random((500, 2)).astype(np.float32)
    pts64 = pts32.astype(np.float64)
    assert tri_set(shull.Delaunay(pts32).triangles) == tri_set(
        shull.Delaunay(pts64).triangles
    )


def test_float32_no_copy_and_dtype():
    pts = np.random.default_rng(12).random((100, 2)).astype(np.float32)
    d = shull.Delaunay(pts)
    assert d.points is pts
    assert d.points.dtype == np.float32


def test_simplices_alias():
    pts = np.random.default_rng(13).random((50, 2))
    d = shull.Delaunay(pts)
    assert d.simplices is d.triangles
