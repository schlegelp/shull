"""Tests for the opt-in parallel build (Delaunay(..., parallel=True)).

The parallel path only activates above a size threshold (~100k points by
default); the SHULL_PARALLEL_* environment variables shrink it so these
tests exercise the real multi-block pipeline with manageable point counts.
The overrides are read per triangulation call, so monkeypatching the
environment is enough.
"""

import numpy as np
import pytest

import shull


@pytest.fixture
def small_thresholds(monkeypatch):
    """Make the parallel pipeline kick in for small test clouds."""
    monkeypatch.setenv("SHULL_PARALLEL_MIN_POINTS", "64")
    monkeypatch.setenv("SHULL_PARALLEL_BLOCK_TARGET", "100")


def canonical(simplices):
    """Order-independent form of a simplex array: sorted rows, sorted."""
    rows = np.sort(np.asarray(simplices), axis=1)
    return rows[np.lexsort(rows.T[::-1])]


def assert_same_triangulation(a, b):
    np.testing.assert_array_equal(canonical(a.simplices), canonical(b.simplices))


@pytest.mark.parametrize("ndim", [2, 3])
def test_parallel_matches_sequential(small_thresholds, ndim):
    rng = np.random.default_rng(42)
    pts = rng.random((3000, ndim))
    par = shull.Delaunay(pts, parallel=True)
    seq = shull.Delaunay(pts)
    assert_same_triangulation(par, seq)
    # Neighbors are index-based and row order differs; check the mutual
    # adjacency invariant instead.
    for t in range(min(200, par.nsimplex)):
        for n in par.neighbors[t]:
            if n >= 0:
                assert t in par.neighbors[n]


@pytest.mark.parametrize("ndim", [2, 3])
def test_parallel_default_is_off(ndim):
    # parallel=False (the default) must go through the sequential kernel and
    # keep its exact output, row order included.
    rng = np.random.default_rng(1)
    pts = rng.random((500, ndim))
    a = shull.Delaunay(pts)
    b = shull.Delaunay(pts, parallel=False)
    np.testing.assert_array_equal(a.simplices, b.simplices)
    np.testing.assert_array_equal(a.neighbors, b.neighbors)


def test_parallel_float32(small_thresholds):
    rng = np.random.default_rng(7)
    pts = rng.random((2000, 3), dtype=np.float32)
    par = shull.Delaunay(pts, parallel=True)
    seq = shull.Delaunay(pts.astype(np.float64))
    assert par.points.dtype == np.float32
    assert_same_triangulation(par, seq)


def test_parallel_duplicates_and_coplanar(small_thresholds):
    rng = np.random.default_rng(3)
    pts = rng.random((1500, 3))
    pts = np.vstack([pts, pts[7], pts[7], pts[500]])
    par = shull.Delaunay(pts, parallel=True)
    seq = shull.Delaunay(pts)
    assert_same_triangulation(par, seq)
    # coplanar rows: [dropped point, a simplex containing the kept point,
    # kept point]; row order is unspecified, compare as sets of
    # (dropped, kept) with a valid simplex reference.
    got = {(int(r[0]), int(r[2])) for r in par.coplanar}
    assert got == {(1500, 7), (1501, 7), (1502, 500)}
    for dropped, simplex, kept in par.coplanar:
        assert kept in par.simplices[simplex]
        assert not np.any(par.simplices == dropped)


def test_parallel_degenerate_grid_falls_back(small_thresholds):
    # Exact grids are cocircular/cospherical everywhere: the parallel build
    # detects the degeneracy and falls back to the sequential kernel, so the
    # result must be exactly the sequential one.
    x, y = np.mgrid[0:30, 0:30]
    pts = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)
    par = shull.Delaunay(pts, parallel=True)
    seq = shull.Delaunay(pts)
    np.testing.assert_array_equal(par.simplices, seq.simplices)
    np.testing.assert_array_equal(par.neighbors, seq.neighbors)


def test_parallel_matches_scipy(small_thresholds):
    scipy_spatial = pytest.importorskip("scipy.spatial")
    rng = np.random.default_rng(11)
    pts = rng.random((2000, 3))
    par = shull.Delaunay(pts, parallel=True)
    sp = scipy_spatial.Delaunay(pts)
    np.testing.assert_array_equal(canonical(par.simplices), canonical(sp.simplices))


def test_parallel_find_simplex_works(small_thresholds):
    # Derived structures must work on the parallel output (row order and
    # neighbor layout are internally consistent).
    rng = np.random.default_rng(13)
    pts = rng.random((2000, 2))
    par = shull.Delaunay(pts, parallel=True)
    queries = rng.random((500, 2)) * 0.8 + 0.1
    found = par.find_simplex(queries)
    assert (found >= 0).all()
    brute = par.find_simplex(queries, bruteforce=True)
    # Both must contain the query point; simplex ids may differ on shared
    # facets, so verify containment via barycentric coordinates.
    eps = 100 * np.finfo(np.float64).eps
    for idx in (found, brute):
        bary = par._barycentric(idx, queries)
        assert (bary >= -eps).all()


def test_parallel_below_threshold_is_sequential():
    # Without the env overrides a small cloud must take the sequential path
    # even with parallel=True — identical output, row order included.
    rng = np.random.default_rng(17)
    pts = rng.random((300, 3))
    par = shull.Delaunay(pts, parallel=True)
    seq = shull.Delaunay(pts)
    np.testing.assert_array_equal(par.simplices, seq.simplices)
    np.testing.assert_array_equal(par.neighbors, seq.neighbors)


def test_parallel_kwarg_rejects_positional():
    rng = np.random.default_rng(19)
    pts = rng.random((100, 2))
    with pytest.raises(TypeError):
        shull._shull.calculate_shull_2d(pts, True)


def test_progress_requires_parallel():
    rng = np.random.default_rng(23)
    pts = rng.random((100, 2))
    with pytest.raises(ValueError, match="parallel=True"):
        shull.Delaunay(pts, progress=True)
    with pytest.raises(ValueError, match="parallel=True"):
        shull.Delaunay(pts, parallel=False, progress=lambda *a: None)


def test_progress_callable_receives_ordered_events(small_thresholds):
    rng = np.random.default_rng(29)
    pts = rng.random((2000, 3))
    events = []
    d = shull.Delaunay(pts, parallel=True,
                       progress=lambda *e: events.append(e))
    assert d.nsimplex > 0
    stages = [e[0] for e in events]
    assert stages[0] == "blocks"
    assert stages[-1] == "done"
    assert "crust" in stages and "merge" in stages
    # Block events count up to their total, in order.
    blocks = [e for e in events if e[0] == "blocks"]
    total = blocks[0][2]
    assert total >= 2
    assert [e[1] for e in blocks] == list(range(total + 1))  # 0 (start), 1..total
    # crust/merge come after the last block event.
    assert stages.index("crust") > len(blocks) - 1


def test_progress_reports_fallback_on_degenerate_input(small_thresholds):
    x, y = np.mgrid[0:30, 0:30]
    pts = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)
    events = []
    shull.Delaunay(pts, parallel=True, progress=lambda *e: events.append(e))
    stages = [e[0] for e in events]
    assert "fallback" in stages
    assert stages[-1] == "done"


def test_progress_true_renders_to_stderr(small_thresholds, capsys):
    rng = np.random.default_rng(31)
    pts = rng.random((2000, 2))
    shull.Delaunay(pts, parallel=True, progress=True)
    err = capsys.readouterr().err
    assert "blocks" in err and "merge" in err
    # The bar cleans up after itself: the line ends cleared.
    assert err.endswith("\r")


def test_progress_callback_exception_propagates(small_thresholds):
    rng = np.random.default_rng(37)
    pts = rng.random((2000, 3))

    def boom(stage, done, total):
        raise RuntimeError("stop the presses")

    with pytest.raises(RuntimeError, match="stop the presses"):
        shull.Delaunay(pts, parallel=True, progress=boom)
