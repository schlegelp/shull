//! Python bindings (the `shull._shull` extension module). Compiled only
//! with the `python` feature, which maturin enables via pyproject.toml.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{alpha, csr_adjacency, d2, d4};

// int32 to match scipy.spatial.Delaunay's simplices/neighbors dtype. Called
// inside `detach` so the cast loop runs without the GIL; only the zero-copy
// `into_pyarray` move of the result happens with the GIL held.
fn to_i32_array<T: Copy + Into<i64>, const K: usize>(rows: Vec<[T; K]>) -> Array2<i32> {
    let mut out: Array2<i32> = Array2::zeros((rows.len(), K));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let v: i64 = v.into();
            out[[i, j]] = v as i32;
        }
    }
    out
}

type TriangulationArrays<'py> = (
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray2<i32>>,
);

/// Error for `parallel=True` on a build without the `parallel` feature. The
/// published wheels always enable it (see pyproject.toml); an explicit error
/// beats silently running sequentially.
#[cfg(not(feature = "parallel"))]
fn no_parallel_feature(parallel: bool) -> PyResult<()> {
    if parallel {
        return Err(PyValueError::new_err(
            "shull was built without the 'parallel' cargo feature; \
             rebuild with --features parallel to use parallel=True",
        ));
    }
    Ok(())
}

/// Adapts a Python callable into a parallel-build progress callback. The
/// build runs with the GIL released, so each event re-attaches to the
/// interpreter and calls `cb(stage, done, total)`; events are already
/// serialized by the build. The first exception the callable raises stops
/// further forwarding and is re-raised once the build returns.
#[cfg(feature = "parallel")]
struct PyProgress {
    cb: Py<pyo3::types::PyAny>,
    err: std::sync::Mutex<Option<PyErr>>,
}

#[cfg(feature = "parallel")]
impl PyProgress {
    fn new(cb: Py<pyo3::types::PyAny>) -> Self {
        PyProgress { cb, err: std::sync::Mutex::new(None) }
    }

    fn call(&self, p: crate::parallel::ParProgress) {
        use crate::parallel::ParProgress as P;
        let mut err = self.err.lock().unwrap();
        if err.is_some() {
            return;
        }
        let (stage, done, total) = match p {
            P::Start { n_blocks } => ("blocks", 0, n_blocks),
            P::Blocks { done, total } => ("blocks", done, total),
            P::Crust => ("crust", 0, 0),
            P::Merge => ("merge", 0, 0),
            P::Fallback => ("fallback", 0, 0),
            P::Done { .. } => ("done", 0, 0),
        };
        if let Err(e) = Python::attach(|py| self.cb.call1(py, (stage, done, total)).map(|_| ())) {
            *err = Some(e);
        }
    }

    /// A raising callback wins over the build result.
    fn into_result(self) -> PyResult<()> {
        match self.err.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

fn shull_2d_impl<'py, T: Copy + Into<f64> + numpy::Element + Send + Sync>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, T>,
    parallel: bool,
    progress: Option<Py<pyo3::types::PyAny>>,
) -> PyResult<TriangulationArrays<'py>> {
    let points = points.as_array();
    if points.ncols() != 2 {
        return Err(PyValueError::new_err("input points must have shape (n, 2)"));
    }
    if progress.is_some() && !parallel {
        return Err(PyValueError::new_err(
            "progress reporting requires parallel=True",
        ));
    }
    #[cfg(not(feature = "parallel"))]
    no_parallel_feature(parallel)?;
    // Snapshot the coordinates so the triangulation can run with the GIL
    // released: other Python threads may mutate the input buffer once the
    // GIL is dropped.
    let points = points.to_owned();
    let (tris, nbrs, dups) = py.detach(move || -> PyResult<_> {
        #[cfg(feature = "parallel")]
        let res = if parallel {
            match progress {
                Some(cb) => {
                    let prog = PyProgress::new(cb);
                    let res = crate::parallel::delaunay2d_par_with_progress(points.view(), |p| {
                        prog.call(p)
                    })
                    .map(|(out, _)| out);
                    prog.into_result()?;
                    res
                }
                None => crate::parallel::delaunay2d_par(points.view()),
            }
        } else {
            d2::delaunay2d(points.view())
        };
        #[cfg(not(feature = "parallel"))]
        let res = d2::delaunay2d(points.view());
        res.map(|(t, n, d)| (to_i32_array(t), to_i32_array(n), to_i32_array(d)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((
        tris.into_pyarray(py),
        nbrs.into_pyarray(py),
        dups.into_pyarray(py),
    ))
}

fn shull_3d_impl<'py, T: Copy + Into<f64> + numpy::Element + Send + Sync>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, T>,
    parallel: bool,
    progress: Option<Py<pyo3::types::PyAny>>,
) -> PyResult<TriangulationArrays<'py>> {
    let points = points.as_array();
    if points.ncols() != 3 {
        return Err(PyValueError::new_err("input points must have shape (n, 3)"));
    }
    if progress.is_some() && !parallel {
        return Err(PyValueError::new_err(
            "progress reporting requires parallel=True",
        ));
    }
    #[cfg(not(feature = "parallel"))]
    no_parallel_feature(parallel)?;
    // See shull_2d_impl for why the input is copied before releasing the GIL.
    let points = points.to_owned();
    let (tets, nbrs, dups) = py.detach(move || -> PyResult<_> {
        #[cfg(feature = "parallel")]
        let res = if parallel {
            match progress {
                Some(cb) => {
                    let prog = PyProgress::new(cb);
                    let res = crate::parallel::delaunay4d_par_with_progress(points.view(), |p| {
                        prog.call(p)
                    })
                    .map(|(out, _)| out);
                    prog.into_result()?;
                    res
                }
                None => crate::parallel::delaunay4d_par(points.view()),
            }
        } else {
            d4::delaunay4d(points.view())
        };
        #[cfg(not(feature = "parallel"))]
        let res = d4::delaunay4d(points.view());
        res.map(|(t, n, d)| (to_i32_array(t), to_i32_array(n), to_i32_array(d)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((
        tets.into_pyarray(py),
        nbrs.into_pyarray(py),
        dups.into_pyarray(py),
    ))
}

/// Calculate the S-Hull Delaunay triangulation of a set of 2d points.
///
/// S-hull: a fast sweep-hull routine for Delaunay triangulation by David
/// Sinclair, http://www.s-hull.org/
///
/// Returns three int32 arrays: (n, 3) vertex indices into the input array
/// (one counterclockwise triangle per row), the (n, 3) neighbor triangle
/// opposite each vertex (-1 on the hull) — matching scipy's `simplices` and
/// `neighbors` — and an (m, 2) array of `[dropped index, kept index]` rows
/// for exact duplicate input points (m = 0 when all points are distinct;
/// the kept index is the first occurrence).
///
/// With `parallel=True` the triangulation is built by spatial blocks on
/// rayon's thread pool (worthwhile for large clouds; identical simplex set,
/// unspecified row order; falls back to the sequential build on degenerate
/// inputs). `progress` (parallel only) is a callable receiving
/// `(stage, done, total)` events: `("blocks", done, total)` per finished
/// block, then `("crust", 0, 0)`, `("merge", 0, 0)`, and finally
/// `("done", 0, 0)` — or `("fallback", 0, 0)` right before the sequential
/// kernel takes over.
#[pyfunction]
#[pyo3(signature = (points, *, parallel = false, progress = None))]
pub fn calculate_shull_2d<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    parallel: bool,
    progress: Option<Py<pyo3::types::PyAny>>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_2d_impl(py, points, parallel, progress)
}

/// Same as `calculate_shull_2d` but for float32 coordinates, avoiding the
/// upcast copy of the input array.
#[pyfunction]
#[pyo3(signature = (points, *, parallel = false, progress = None))]
pub fn calculate_shull_2d_f32<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
    parallel: bool,
    progress: Option<Py<pyo3::types::PyAny>>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_2d_impl(py, points, parallel, progress)
}

/// Calculate the 3D Delaunay tetrahedralization of a set of 3d points.
///
/// Uses the sweep-hull ("Newton Apple Wrapper") algorithm of David Sinclair
/// (arXiv 1602.04707) generalized to 4D: points are lifted onto a 4D
/// paraboloid, the 4D convex hull is computed by incremental insertion along
/// a space-filling curve, and the downward-facing facets are the Delaunay
/// tetrahedra.
///
/// Returns three int32 arrays: (n, 4) vertex indices into the input array
/// (one tetrahedron per row), the (n, 4) neighbor tetrahedron opposite each
/// vertex (-1 on the hull boundary) — matching scipy's `simplices` and
/// `neighbors` — and an (m, 2) array of `[dropped index, kept index]` rows
/// for exact duplicate input points (m = 0 when all points are distinct;
/// the kept index is the first occurrence).
///
/// With `parallel=True` the triangulation is built by spatial blocks on
/// rayon's thread pool (worthwhile for large clouds; identical simplex set,
/// unspecified row order; falls back to the sequential build on degenerate
/// inputs). `progress` (parallel only) is a callable receiving
/// `(stage, done, total)` events — see `calculate_shull_2d`.
#[pyfunction]
#[pyo3(signature = (points, *, parallel = false, progress = None))]
pub fn calculate_shull_3d<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    parallel: bool,
    progress: Option<Py<pyo3::types::PyAny>>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_3d_impl(py, points, parallel, progress)
}

/// Same as `calculate_shull_3d` but for float32 coordinates, avoiding the
/// upcast copy of the input array. Coordinates widen to f64 exactly, so the
/// result is identical to converting the input to float64 first.
#[pyfunction]
#[pyo3(signature = (points, *, parallel = false, progress = None))]
pub fn calculate_shull_3d_f32<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
    parallel: bool,
    progress: Option<Py<pyo3::types::PyAny>>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_3d_impl(py, points, parallel, progress)
}

fn circumradii_impl<'py, T: Copy + Into<f64> + numpy::Element + Send + Sync>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, T>,
    simplices: PyReadonlyArray2<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // See shull_2d_impl for why the inputs are copied before releasing the GIL.
    let points = points.as_array().to_owned();
    let simplices = simplices.as_array().to_owned();
    let k = simplices.ncols();
    // Repack the (m, 3)/(m, 4) int32 simplices into the `[u32; K]` row layout
    // the Rust core takes (matching `delaunay2d`/`delaunay4d` output); done off
    // the GIL, only on the alpha path, never during the Delaunay build.
    // Out-of-range or negative indices wrap to a large u32 and are rejected
    // inside the core.
    let radii = py.detach(move || match k {
        3 => {
            let sv: Vec<[u32; 3]> = simplices
                .rows()
                .into_iter()
                .map(|r| [r[0] as u32, r[1] as u32, r[2] as u32])
                .collect();
            alpha::circumradii(points.view(), &sv)
        }
        4 => {
            let sv: Vec<[u32; 4]> = simplices
                .rows()
                .into_iter()
                .map(|r| [r[0] as u32, r[1] as u32, r[2] as u32, r[3] as u32])
                .collect();
            alpha::circumradii(points.view(), &sv)
        }
        other => Err(format!("simplices must have 3 or 4 columns, got {}", other)),
    });
    Ok(radii.map_err(PyValueError::new_err)?.into_pyarray(py))
}

/// Circumradius of each Delaunay simplex, for alpha-shape filtration.
///
/// Takes the `points` (n, 2)/(n, 3) and an (m, 3)/(m, 4) int32 `simplices`
/// array (as produced by `calculate_shull_2d`/`_3d`) and returns an (m,)
/// float64 array of circumradii in row order. Computed from the already-built
/// triangulation — it does not re-triangulate and does not touch the Delaunay
/// build path. Degenerate simplices get an infinite radius.
#[pyfunction]
pub fn circumradii<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    simplices: PyReadonlyArray2<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    circumradii_impl(py, points, simplices)
}

/// Same as `circumradii` but for float32 coordinates, avoiding the upcast
/// copy of the input array. Coordinates widen to f64 exactly, so the radii
/// are identical to converting the points to float64 first.
#[pyfunction]
pub fn circumradii_f32<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
    simplices: PyReadonlyArray2<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    circumradii_impl(py, points, simplices)
}

/// scipy-style `vertex_neighbor_vertices` from an (n, k) int32 simplex
/// array: `(indptr, indices)` CSR arrays (both int32); the neighbors of
/// vertex v are `indices[indptr[v]:indptr[v + 1]]`, sorted ascending.
#[pyfunction]
pub fn vertex_neighbor_vertices<'py>(
    py: Python<'py>,
    simplices: PyReadonlyArray2<'py, i32>,
    n_points: usize,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>)> {
    if simplices.as_array().ncols() < 2 {
        return Err(PyValueError::new_err(
            "simplices must have at least 2 columns",
        ));
    }
    if n_points > i32::MAX as usize {
        return Err(PyValueError::new_err("n_points does not fit int32"));
    }
    // See shull_2d_impl for why the input is copied before releasing the GIL.
    let s = simplices.as_array().to_owned();
    let (indptr, indices) = py
        .detach(move || csr_adjacency(s.view(), n_points))
        .map_err(PyValueError::new_err)?;
    Ok((indptr.into_pyarray(py), indices.into_pyarray(py)))
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_shull")]
fn shull(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_shull_2d, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shull_2d_f32, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shull_3d, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shull_3d_f32, m)?)?;
    m.add_function(wrap_pyfunction!(vertex_neighbor_vertices, m)?)?;
    m.add_function(wrap_pyfunction!(circumradii, m)?)?;
    m.add_function(wrap_pyfunction!(circumradii_f32, m)?)?;
    Ok(())
}
