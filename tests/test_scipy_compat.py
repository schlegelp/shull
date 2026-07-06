"""Tests for the scipy.spatial.Delaunay-compatible API surface.

Simplex *sets* match scipy on the random inputs used here (see
test_delaunay2d/3d), but simplex ordering, vertex order within a simplex
and neighbor ordering are implementation-specific — so structural
comparisons below always go through order-independent representations.
"""

import numpy as np
import pytest
import scipy.spatial

import shull


def make(seed, n, ndim):
    pts = np.random.default_rng(seed).random((n, ndim))
    return pts, shull.Delaunay(pts), scipy.spatial.Delaunay(pts)


def tri_set(simplices):
    return set(map(tuple, np.sort(np.asarray(simplices), axis=1)))


@pytest.mark.parametrize("ndim", [2, 3])
def test_basic_attributes(ndim):
    pts, ours, theirs = make(1, 200, ndim)
    assert ours.ndim == ndim
    assert ours.npoints == 200
    assert ours.nsimplex == len(ours.simplices)
    assert ours.simplices.dtype == np.int32
    assert ours.furthest_site is False
    np.testing.assert_array_equal(ours.min_bound, pts.min(axis=0))
    np.testing.assert_array_equal(ours.max_bound, pts.max(axis=0))
    ours.close()  # no-op, must exist


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("seed", [1, 2])
def test_vertex_neighbor_vertices_matches_scipy(ndim, seed):
    _, ours, theirs = make(seed, 300, ndim)
    o_indptr, o_indices = ours.vertex_neighbor_vertices
    t_indptr, t_indices = theirs.vertex_neighbor_vertices
    assert o_indptr.dtype == np.int32 and o_indices.dtype == np.int32
    assert o_indptr[0] == 0 and o_indptr[-1] == len(o_indices)
    for v in range(ours.npoints):
        o = set(o_indices[o_indptr[v]:o_indptr[v + 1]])
        t = set(t_indices[t_indptr[v]:t_indptr[v + 1]])
        assert o == t, f"vertex {v} neighbors differ"


@pytest.mark.parametrize("ndim", [2, 3])
def test_neighbors_matches_scipy(ndim):
    def adjacency(simplices, neighbors):
        keyed = [tuple(sorted(s)) for s in np.asarray(simplices)]
        return {
            frozenset((keyed[i], keyed[k]))
            for i, row in enumerate(np.asarray(neighbors))
            for k in row
            if k != -1
        }

    _, ours, theirs = make(3, 300, ndim)
    assert ours.neighbors.shape == (ours.nsimplex, ndim + 1)
    assert ours.neighbors.dtype == np.int32
    assert adjacency(ours.simplices, ours.neighbors) == adjacency(
        theirs.simplices, theirs.neighbors
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_neighbors_opposite_vertex_convention(ndim):
    _, ours, _ = make(4, 200, ndim)
    s = np.asarray(ours.simplices)
    nb = ours.neighbors
    for i in range(ours.nsimplex):
        for j in range(ndim + 1):
            k = nb[i, j]
            if k == -1:
                continue
            shared = set(s[i]) - {s[i, j]}
            assert shared < set(s[k])  # facet opposite vertex j is shared
            assert s[i, j] not in s[k]


@pytest.mark.parametrize("ndim", [2, 3])
def test_convex_hull_matches_scipy(ndim):
    _, ours, theirs = make(5, 300, ndim)
    assert ours.convex_hull.shape[1] == ndim
    assert tri_set(ours.convex_hull) == tri_set(theirs.convex_hull)


@pytest.mark.parametrize("ndim", [2, 3])
def test_vertex_to_simplex(ndim):
    _, ours, _ = make(6, 300, ndim)
    vts = ours.vertex_to_simplex
    assert vts.dtype == np.int32
    s = np.asarray(ours.simplices)
    for v in range(ours.npoints):
        assert v in s[vts[v]]


@pytest.mark.parametrize("ndim", [2, 3])
def test_transform_barycentric_roundtrip(ndim):
    pts, ours, _ = make(7, 300, ndim)
    t = ours.transform
    assert t.shape == (ours.nsimplex, ndim + 1, ndim)
    # Barycentric coordinates of each simplex's centroid are 1/(ndim+1).
    centroids = pts[np.asarray(ours.simplices)].mean(axis=1)
    c = np.einsum('nij,nj->ni', t[:, :ndim, :], centroids - t[:, ndim, :])
    np.testing.assert_allclose(c, 1.0 / (ndim + 1), rtol=1e-9)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("bruteforce", [False, True])
def test_find_simplex_matches_scipy(ndim, bruteforce):
    pts, ours, theirs = make(8, 300, ndim)
    # Interior queries: shrink toward the centroid to stay inside the hull.
    rng = np.random.default_rng(80)
    xi = pts.mean(axis=0) + (rng.random((500, ndim)) - 0.5) * 0.5
    found = ours.find_simplex(xi, bruteforce=bruteforce)
    expected = theirs.find_simplex(xi)
    assert found.dtype == np.int32
    assert (found >= 0).all() and (expected >= 0).all()
    o = np.sort(np.asarray(ours.simplices)[found], axis=1)
    t = np.sort(theirs.simplices[expected], axis=1)
    np.testing.assert_array_equal(o, t)


@pytest.mark.parametrize("ndim", [2, 3])
def test_find_simplex_outside(ndim):
    pts, ours, _ = make(9, 200, ndim)
    outside = np.vstack([pts + 10.0, pts - 10.0])
    assert (ours.find_simplex(outside) == -1).all()


def test_find_simplex_shapes():
    pts, ours, _ = make(10, 100, 2)
    single = ours.find_simplex(pts.mean(axis=0))
    assert single.shape == ()
    batch = ours.find_simplex(pts[:6].reshape(2, 3, 2))
    assert batch.shape == (2, 3)
    with pytest.raises(ValueError, match="dimensionality"):
        ours.find_simplex(np.zeros((5, 3)))


@pytest.mark.parametrize("ndim", [2, 3])
def test_vertices_own_simplices_found(ndim):
    """Every input point must locate into a simplex that contains it."""
    pts, ours, _ = make(11, 300, ndim)
    found = ours.find_simplex(pts)
    assert (found >= 0).all()
    bary = np.einsum(
        'nij,nj->ni',
        ours.transform[found, :ndim, :],
        pts - ours.transform[found, ndim, :],
    )
    full = np.concatenate([bary, 1 - bary.sum(axis=1, keepdims=True)], axis=1)
    assert (full >= -1e-9).all()


@pytest.mark.parametrize("ndim", [2, 3])
def test_coplanar(ndim):
    pts = np.random.default_rng(12).random((100, ndim))
    dup = np.vstack([pts, pts[:10]])
    d = shull.Delaunay(dup)
    cop = d.coplanar
    assert cop.shape == (10, 3) and cop.dtype == np.int32
    for dropped, simplex, rep in cop:
        np.testing.assert_array_equal(dup[dropped], dup[rep])
        assert dropped != rep
        assert rep in np.asarray(d.simplices)[simplex]
    # Without duplicates: empty, like scipy for points in general position.
    assert shull.Delaunay(pts).coplanar.shape == (0, 3)


def test_delaunay_dispatches_3d():
    pts = np.random.default_rng(13).random((100, 3))
    assert tri_set(shull.Delaunay(pts).simplices) == tri_set(
        shull.Delaunay3d(pts).simplices
    )


def test_array_like_input():
    pts = np.random.default_rng(14).random((50, 2))
    assert tri_set(shull.Delaunay(pts.tolist()).simplices) == tri_set(
        shull.Delaunay(pts).simplices
    )


def test_unsupported_scipy_kwargs_raise():
    pts = np.random.default_rng(15).random((50, 2))
    with pytest.raises(NotImplementedError):
        shull.Delaunay(pts, furthest_site=True)
    with pytest.raises(NotImplementedError):
        shull.Delaunay(pts, incremental=True)
    with pytest.raises(NotImplementedError):
        shull.Delaunay(pts, qhull_options="QJ")


def test_bad_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        shull.Delaunay(np.zeros((10, 4)))
    with pytest.raises(ValueError, match="shape"):
        shull.Delaunay(np.zeros(10))
