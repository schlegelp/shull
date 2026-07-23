"""Alpha-shape / concave-hull tests.

The alpha shape is a filtration of the Delaunay triangulation, so most checks
are internal invariants (monotonicity, boundary closure, the alpha=inf limit)
plus a few closed-form circumradii and a cross-check of the filled measure
against scipy's convex-hull volume in the large-alpha limit.
"""

import numpy as np
import pytest
import scipy.spatial

import shull


def facet_set(facets):
    """Order-independent set of facets (each a frozenset of vertex indices)."""
    return set(map(frozenset, np.asarray(facets).tolist()))


# ----------------------------------------------------------------------------
# circumradii
# ----------------------------------------------------------------------------

def numpy_circumradii(points, simplices):
    """Independent circumradius via the least-squares circumcenter."""
    p = points[simplices].astype(np.float64)  # (m, d+1, d)
    a = p[:, :1, :]
    u = p[:, 1:, :] - a  # (m, d, d) edge vectors from vertex 0
    rhs = 0.5 * np.einsum("mij,mij->mi", u, u)  # |u|^2 / 2
    # NumPy 2.x: solve treats b as a stack of matrices, so add a trailing axis.
    y = np.linalg.solve(u, rhs[..., None])[..., 0]  # circumcenter rel. vertex 0
    return np.linalg.norm(y, axis=1)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("dim", [2, 3])
def test_circumradii_match_numpy(seed, dim):
    pts = np.random.default_rng(seed).random((800, dim))
    d = shull.Delaunay(pts)
    ref = numpy_circumradii(pts, np.asarray(d.simplices))
    np.testing.assert_allclose(d.circumradii, ref, rtol=1e-9, atol=1e-12)


def test_circumradii_closed_form_2d():
    # Two right triangles of a unit-ish square: R = half the hypotenuse.
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    d = shull.Delaunay(pts)
    assert np.allclose(d.circumradii, np.sqrt(2.0))


def test_circumradii_f32_matches_f64():
    # Identical coordinate *values* through both paths: f32 widens to f64
    # exactly, so the f32 build must give bit-identical circumradii to feeding
    # the same values as f64. (Rounding f64 random points to f32 would change
    # the coordinates, hence the shared f32 source here.)
    pts32 = np.random.default_rng(7).random((500, 3)).astype(np.float32)
    d32 = shull.Delaunay(pts32)
    d64 = shull.Delaunay(pts32.astype(np.float64))
    key = lambda s: tuple(sorted(s))
    r64 = {key(s): r for s, r in zip(np.asarray(d64.simplices).tolist(), d64.circumradii)}
    r32 = {key(s): r for s, r in zip(np.asarray(d32.simplices).tolist(), d32.circumradii)}
    assert r64.keys() == r32.keys()
    for k in r64:
        assert r64[k] == pytest.approx(r32[k], rel=1e-12)


# ----------------------------------------------------------------------------
# alpha_complex / alpha_shape
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
def test_alpha_shape_inf_is_convex_hull(dim):
    pts = np.random.default_rng(11).random((1500, dim))
    d = shull.Delaunay(pts)
    assert facet_set(d.alpha_shape(np.inf)) == facet_set(d.convex_hull)


@pytest.mark.parametrize("dim", [2, 3])
def test_alpha_complex_is_monotone(dim):
    pts = np.random.default_rng(12).random((1000, dim))
    d = shull.Delaunay(pts)
    alphas = np.linspace(0, float(d.circumradii.max()) * 1.01, 8)
    prev = set()
    for a in alphas:
        cur = set(d.alpha_complex(a).tolist())
        assert prev <= cur  # growing with alpha
        prev = cur
    assert prev == set(range(d.nsimplex))  # eventually the whole triangulation


def test_alpha_complex_empty_and_full():
    pts = np.random.default_rng(13).random((400, 2))
    d = shull.Delaunay(pts)
    assert d.alpha_complex(0.0).size == 0
    assert d.alpha_complex(np.inf).size == d.nsimplex
    assert d.alpha_shape(0.0).shape[0] == 0


@pytest.mark.parametrize("dim", [2, 3])
def test_boundary_is_a_closed_manifold(dim):
    # Every ridge (facet-of-a-facet) of the alpha-shape boundary must be shared
    # by an even number of boundary facets: the surface has no free edges.
    pts = np.random.default_rng(14).random((1200, dim))
    d = shull.Delaunay(pts)
    a = float(np.quantile(d.circumradii, 0.6))
    facets = np.asarray(d.alpha_shape(a))
    assert facets.shape[0] > 0
    ridges = {}
    for f in facets.tolist():
        for k in range(len(f)):
            ridge = frozenset(f[:k] + f[k + 1:])  # drop one vertex
            ridges[ridge] = ridges.get(ridge, 0) + 1
    assert all(c % 2 == 0 for c in ridges.values())


def test_alpha_shape_boundary_matches_complex():
    # A boundary facet must have exactly one incident simplex inside the
    # complex (the definition), cross-checked against alpha_complex.
    pts = np.random.default_rng(15).random((800, 2))
    d = shull.Delaunay(pts)
    a = float(np.quantile(d.circumradii, 0.5))
    inside = np.zeros(d.nsimplex, bool)
    inside[d.alpha_complex(a)] = True
    tris = np.asarray(d.simplices)
    nbr = np.asarray(d.neighbors)
    boundary = facet_set(d.alpha_shape(a))
    # Reconstruct boundary independently from (inside, neighbors).
    ref = set()
    for i in np.nonzero(inside)[0]:
        for j in range(3):
            n = nbr[i, j]
            if n == -1 or not inside[n]:
                ref.add(frozenset(np.delete(tris[i], j).tolist()))
    assert boundary == ref


# ----------------------------------------------------------------------------
# AlphaShape wrapper
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
def test_alphashape_large_alpha_measure_is_hull_volume(dim):
    pts = np.random.default_rng(21).random((2000, dim))
    A = shull.AlphaShape(pts, alpha=1e9)  # effectively the convex hull
    hull_vol = scipy.spatial.ConvexHull(pts).volume  # area in 2D, volume in 3D
    assert A.measure == pytest.approx(hull_vol, rel=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
def test_optimal_alpha_covers_all_points(dim):
    pts = np.random.default_rng(22).random((1500, dim))
    A = shull.AlphaShape(pts)  # auto alpha
    covered = np.unique(np.asarray(A.simplices))
    used = np.unique(np.asarray(A.tri.simplices))  # points in the triangulation
    assert set(used.tolist()) <= set(covered.tolist())


def test_alphashape_reuses_triangulation():
    pts = np.random.default_rng(23).random((600, 2))
    A = shull.AlphaShape(pts, alpha=0.05)
    B = A.at(0.2)
    assert B.tri is A.tri  # no re-triangulation
    assert len(B.simplices) >= len(A.simplices)  # monotone


def test_alphashape_accepts_prebuilt_tri():
    pts = np.random.default_rng(24).random((600, 3))
    d = shull.Delaunay(pts)
    A = shull.AlphaShape(None, alpha=0.1, tri=d)
    assert A.tri is d
    assert A.ndim == 3
