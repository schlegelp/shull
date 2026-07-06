import numpy as np
import pytest
import scipy.spatial

import shull


def tet_set(simplices):
    """Order-independent representation of a tetrahedralization."""
    return set(map(tuple, np.sort(np.asarray(simplices), axis=1)))


def tet_volumes(points, simplices):
    a, b, c, d = (points[simplices[:, i]] for i in range(4))
    return np.linalg.det(np.stack([b - a, c - a, d - a], axis=1)) / 6.0


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n", [10, 100, 1000])
def test_matches_scipy_random(seed, n):
    pts = np.random.default_rng(seed).random((n, 3))
    ours = tet_set(shull.Delaunay3d(pts).simplices)
    theirs = tet_set(scipy.spatial.Delaunay(pts).simplices)
    assert ours == theirs


def test_empty_circumsphere():
    pts = np.random.default_rng(99).random((300, 3))
    simplices = np.asarray(shull.Delaunay3d(pts).simplices)
    a = pts[simplices[:, 0]]
    rows = [pts[simplices[:, i]] - a for i in range(1, 4)]
    m = 2.0 * np.stack(rows, axis=1)
    rhs = np.stack(
        [np.einsum('ij,ij->i', r + a, r + a) - np.einsum('ij,ij->i', a, a) for r in rows],
        axis=1,
    )
    centers = np.linalg.solve(m, rhs[..., None])[..., 0]
    r2 = np.einsum('ij,ij->i', a - centers, a - centers)

    d2 = ((pts[None, :, :] - centers[:, None, :]) ** 2).sum(axis=2)
    inside = d2 < r2[:, None] * (1.0 - 1e-9)
    for t, tet in enumerate(simplices):
        inside[t, tet] = False
    assert not inside.any(), "some circumspheres are not empty"


def test_volume_partition():
    pts = np.random.default_rng(4).random((500, 3))
    simplices = np.asarray(shull.Delaunay3d(pts).simplices)
    total = np.abs(tet_volumes(pts, simplices)).sum()
    hull_volume = scipy.spatial.ConvexHull(pts).volume
    assert total == pytest.approx(hull_volume, rel=1e-9)


def test_orientation():
    pts = np.random.default_rng(5).random((500, 3))
    simplices = np.asarray(shull.Delaunay3d(pts).simplices)
    assert (tet_volumes(pts, simplices) > 0).all()


def test_all_points_used():
    pts = np.random.default_rng(6).random((500, 3))
    simplices = np.asarray(shull.Delaunay3d(pts).simplices)
    assert set(np.unique(simplices)) == set(range(len(pts)))


def test_index_remap_invariance():
    pts = np.random.default_rng(7).random((300, 3))
    base = tet_set(shull.Delaunay3d(pts).simplices)
    moved = tet_set(shull.Delaunay3d(pts * 3.7 + np.array([100.0, -50.0, 1e6])).simplices)
    assert base == moved


def test_too_few_points():
    with pytest.raises(ValueError):
        shull.Delaunay3d(np.random.default_rng(0).random((4, 3)))


def test_coplanar_raises():
    pts = np.random.default_rng(8).random((50, 3))
    pts[:, 2] = 0.0
    with pytest.raises(ValueError, match="degenerate"):
        shull.Delaunay3d(pts)


def test_duplicates_dropped():
    pts = np.random.default_rng(9).random((100, 3))
    dup = np.vstack([pts, pts[:10]])
    simplices = np.asarray(shull.Delaunay3d(dup).simplices)
    assert simplices.max() < 100  # duplicate indices never appear
    assert tet_set(simplices) == tet_set(shull.Delaunay3d(pts).simplices)


def test_float32_matches_float64():
    # f32 values widen to f64 exactly, so the two paths must be identical.
    pts32 = np.random.default_rng(11).random((500, 3)).astype(np.float32)
    pts64 = pts32.astype(np.float64)
    assert tet_set(shull.Delaunay3d(pts32).simplices) == tet_set(
        shull.Delaunay3d(pts64).simplices
    )


def test_float32_no_copy_and_dtype():
    pts = np.random.default_rng(12).random((100, 3)).astype(np.float32)
    d = shull.Delaunay3d(pts)
    assert d.points is pts  # no upcast copy
    assert d.points.dtype == np.float32


def test_float32_matches_scipy():
    pts = np.random.default_rng(15).random((500, 3)).astype(np.float32)
    ours = tet_set(shull.Delaunay3d(pts).simplices)
    theirs = tet_set(scipy.spatial.Delaunay(pts.astype(np.float64)).simplices)
    assert ours == theirs


def test_float32_large_offset():
    # Typical of e.g. nm-scale coordinates: large offset, small extent. f32
    # quantization makes such data grid-like, so check properties rather than
    # scipy set-equality (Qhull breaks the resulting ties differently).
    rng = np.random.default_rng(13)
    pts = (rng.random((300, 3)) * 100 + np.array([4e5, 1.5e5, 8e4])).astype(np.float32)
    simplices = np.asarray(shull.Delaunay3d(pts).simplices)
    pts64 = pts.astype(np.float64)
    vols = tet_volumes(pts64, simplices)
    assert (vols > 0).all()
    hull_volume = scipy.spatial.ConvexHull(pts64).volume
    assert vols.sum() == pytest.approx(hull_volume, rel=1e-9)
    assert set(np.unique(simplices)) == set(range(len(pts)))


def test_float32_noncontiguous():
    base = np.random.default_rng(14).random((400, 3)).astype(np.float32)
    pts = base[::2]
    assert not pts.flags["C_CONTIGUOUS"]
    a = tet_set(shull.Delaunay3d(pts).simplices)
    b = tet_set(shull.Delaunay3d(np.ascontiguousarray(pts)).simplices)
    assert a == b


def test_grid():
    """5x5x5 grid: heavily cospherical. Check properties, not scipy equality
    (Qhull merges cospherical facets differently)."""
    grid = np.stack(np.meshgrid(*[np.arange(5.0)] * 3), axis=-1).reshape(-1, 3)
    simplices = np.asarray(shull.Delaunay3d(grid).simplices)
    vols = tet_volumes(grid, simplices)
    assert (vols > 0).all()
    assert vols.sum() == pytest.approx(64.0, rel=1e-9)
