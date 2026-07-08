//! Python bindings (the `shull._shull` extension module). Compiled only
//! with the `python` feature, which maturin enables via pyproject.toml.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{csr_adjacency, d2, d4};

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

fn shull_2d_impl<'py, T: Copy + Into<f64> + numpy::Element + Send>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, T>,
) -> PyResult<TriangulationArrays<'py>> {
    let points = points.as_array();
    if points.ncols() != 2 {
        return Err(PyValueError::new_err("input points must have shape (n, 2)"));
    }
    // Snapshot the coordinates so the triangulation can run with the GIL
    // released: other Python threads may mutate the input buffer once the
    // GIL is dropped.
    let points = points.to_owned();
    let (tris, nbrs, dups) = py
        .detach(move || {
            d2::delaunay2d(points.view())
                .map(|(t, n, d)| (to_i32_array(t), to_i32_array(n), to_i32_array(d)))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        tris.into_pyarray(py),
        nbrs.into_pyarray(py),
        dups.into_pyarray(py),
    ))
}

fn shull_3d_impl<'py, T: Copy + Into<f64> + numpy::Element + Send>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, T>,
) -> PyResult<TriangulationArrays<'py>> {
    let points = points.as_array();
    if points.ncols() != 3 {
        return Err(PyValueError::new_err("input points must have shape (n, 3)"));
    }
    // See shull_2d_impl for why the input is copied before releasing the GIL.
    let points = points.to_owned();
    let (tets, nbrs, dups) = py
        .detach(move || {
            d4::delaunay4d(points.view())
                .map(|(t, n, d)| (to_i32_array(t), to_i32_array(n), to_i32_array(d)))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
#[pyfunction]
pub fn calculate_shull_2d<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_2d_impl(py, points)
}

/// Same as `calculate_shull_2d` but for float32 coordinates, avoiding the
/// upcast copy of the input array.
#[pyfunction]
pub fn calculate_shull_2d_f32<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_2d_impl(py, points)
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
#[pyfunction]
pub fn calculate_shull_3d<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_3d_impl(py, points)
}

/// Same as `calculate_shull_3d` but for float32 coordinates, avoiding the
/// upcast copy of the input array. Coordinates widen to f64 exactly, so the
/// result is identical to converting the input to float64 first.
#[pyfunction]
pub fn calculate_shull_3d_f32<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_3d_impl(py, points)
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
    Ok(())
}
