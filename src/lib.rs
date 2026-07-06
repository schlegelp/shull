use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod d2;
mod d4;
mod error;

pub use d2::delaunay2d;
pub use d4::delaunay4d;
pub use error::DelaunayError;

fn to_pyarray<'py, const K: usize>(
    py: Python<'py>,
    simplices: Vec<[u32; K]>,
) -> &'py PyArray2<u64> {
    let mut indices: Array2<u64> = Array2::zeros((simplices.len(), K));
    for (i, simplex) in simplices.iter().enumerate() {
        for (j, &v) in simplex.iter().enumerate() {
            indices[[i, j]] = v as u64;
        }
    }
    indices.into_pyarray(py)
}

fn shull_2d_impl<'py, T: Copy + Into<f64> + numpy::Element>(
    py: Python<'py>,
    points: PyReadonlyArray2<T>,
) -> PyResult<&'py PyArray2<u64>> {
    let points = points.as_array();
    if points.ncols() != 2 {
        return Err(PyValueError::new_err("input points must have shape (n, 2)"));
    }
    let tris = d2::delaunay2d(points).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(to_pyarray(py, tris))
}

fn shull_3d_impl<'py, T: Copy + Into<f64> + numpy::Element>(
    py: Python<'py>,
    points: PyReadonlyArray2<T>,
) -> PyResult<&'py PyArray2<u64>> {
    let points = points.as_array();
    if points.ncols() != 3 {
        return Err(PyValueError::new_err("input points must have shape (n, 3)"));
    }
    let tets = d4::delaunay4d(points).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(to_pyarray(py, tets))
}

/// Calculate the S-Hull Delaunay triangulation of a set of 2d points.
///
/// S-hull: a fast sweep-hull routine for Delaunay triangulation by David
/// Sinclair, http://www.s-hull.org/
///
/// Returns an (n, 3) array of vertex indices into the input array, one
/// counterclockwise triangle per row.
#[pyfunction]
pub fn calculate_shull_2d<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<u64>> {
    shull_2d_impl(py, points)
}

/// Same as `calculate_shull_2d` but for float32 coordinates, avoiding the
/// upcast copy of the input array.
#[pyfunction]
pub fn calculate_shull_2d_f32<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
) -> PyResult<&'py PyArray2<u64>> {
    shull_2d_impl(py, points)
}

/// Calculate the 3D Delaunay tetrahedralization of a set of 3d points.
///
/// Uses the sweep-hull ("Newton Apple Wrapper") algorithm of David Sinclair
/// (arXiv 1602.04707) generalized to 4D: points are lifted onto a 4D
/// paraboloid, the 4D convex hull is computed by sorted incremental
/// insertion, and the downward-facing facets are the Delaunay tetrahedra.
///
/// Returns an (n, 4) array of vertex indices into the input array.
#[pyfunction]
pub fn calculate_shull_3d<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<u64>> {
    shull_3d_impl(py, points)
}

/// Same as `calculate_shull_3d` but for float32 coordinates, avoiding the
/// upcast copy of the input array. Coordinates widen to f64 exactly, so the
/// result is identical to converting the input to float64 first.
#[pyfunction]
pub fn calculate_shull_3d_f32<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
) -> PyResult<&'py PyArray2<u64>> {
    shull_3d_impl(py, points)
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_shull")]
fn shull(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_shull_2d, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shull_2d_f32, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shull_3d, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shull_3d_f32, m)?)?;
    Ok(())
}
