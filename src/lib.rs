use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod d2;
mod d4;
mod error;

pub use d2::delaunay2d;
pub use d4::delaunay4d;
pub use error::DelaunayError;

// int32 to match scipy.spatial.Delaunay's simplices/neighbors dtype. Called
// inside `allow_threads` so the cast loop runs without the GIL; only the
// zero-copy `into_pyarray` move of the result happens with the GIL held.
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
    &'py PyArray2<i32>,
    &'py PyArray2<i32>,
    &'py PyArray2<i32>,
);

fn shull_2d_impl<'py, T: Copy + Into<f64> + numpy::Element + Send>(
    py: Python<'py>,
    points: PyReadonlyArray2<T>,
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
        .allow_threads(move || {
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
    points: PyReadonlyArray2<T>,
) -> PyResult<TriangulationArrays<'py>> {
    let points = points.as_array();
    if points.ncols() != 3 {
        return Err(PyValueError::new_err("input points must have shape (n, 3)"));
    }
    // See shull_2d_impl for why the input is copied before releasing the GIL.
    let points = points.to_owned();
    let (tets, nbrs, dups) = py
        .allow_threads(move || {
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
    points: PyReadonlyArray2<f64>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_2d_impl(py, points)
}

/// Same as `calculate_shull_2d` but for float32 coordinates, avoiding the
/// upcast copy of the input array.
#[pyfunction]
pub fn calculate_shull_2d_f32<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
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
    points: PyReadonlyArray2<f64>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_3d_impl(py, points)
}

/// Same as `calculate_shull_3d` but for float32 coordinates, avoiding the
/// upcast copy of the input array. Coordinates widen to f64 exactly, so the
/// result is identical to converting the input to float64 first.
#[pyfunction]
pub fn calculate_shull_3d_f32<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
) -> PyResult<TriangulationArrays<'py>> {
    shull_3d_impl(py, points)
}

/// Build the vertex-adjacency CSR arrays from an (n, k) simplex array.
///
/// Two vertices are adjacent if they share a simplex. Returns
/// `(indptr, indices)` (both int32) where the neighbors of vertex v are
/// `indices[indptr[v]..indptr[v + 1]]`, sorted ascending — the same layout
/// as scipy's `vertex_neighbor_vertices`. Vertices in `0..n_points` not
/// referenced by any simplex (e.g. dropped exact duplicates) get an empty
/// row.
fn csr_adjacency(s: ArrayView2<i32>, n_points: usize) -> Result<(Vec<i32>, Vec<i32>), String> {
    let m = s.ncols();
    for &v in s.iter() {
        if v < 0 || v as usize >= n_points {
            return Err(format!(
                "simplex vertex {} out of range for {} points",
                v, n_points
            ));
        }
    }

    // Pass 1: per-vertex entry counts, duplicates included -- every
    // occurrence of v in a simplex contributes its m-1 co-vertices.
    let mut start = vec![0usize; n_points + 1];
    for &v in s.iter() {
        start[v as usize + 1] += m - 1;
    }
    for v in 0..n_points {
        start[v + 1] += start[v];
    }
    if start[n_points] > i32::MAX as usize {
        return Err("adjacency does not fit int32 indptr".to_string());
    }

    // Pass 2: scatter the co-vertices into one flat buffer.
    let mut buf = vec![0i32; start[n_points]];
    let mut cursor: Vec<usize> = start[..n_points].to_vec();
    for row in s.rows() {
        for a in 0..m {
            let va = row[a] as usize;
            for b in 0..m {
                if b != a {
                    buf[cursor[va]] = row[b];
                    cursor[va] += 1;
                }
            }
        }
    }

    // Pass 3: sort and dedup each vertex's row, compacting in place (the
    // write position never overtakes the read position).
    let mut indptr = vec![0i32; n_points + 1];
    let mut w = 0usize;
    for v in 0..n_points {
        buf[start[v]..start[v + 1]].sort_unstable();
        let mut prev = -1;
        for r in start[v]..start[v + 1] {
            if buf[r] != prev {
                prev = buf[r];
                buf[w] = prev;
                w += 1;
            }
        }
        indptr[v + 1] = w as i32;
    }
    buf.truncate(w);
    Ok((indptr, buf))
}

/// scipy-style `vertex_neighbor_vertices` from an (n, k) int32 simplex
/// array: `(indptr, indices)` CSR arrays (both int32); the neighbors of
/// vertex v are `indices[indptr[v]:indptr[v + 1]]`, sorted ascending.
#[pyfunction]
pub fn vertex_neighbor_vertices<'py>(
    py: Python<'py>,
    simplices: PyReadonlyArray2<i32>,
    n_points: usize,
) -> PyResult<(&'py PyArray1<i32>, &'py PyArray1<i32>)> {
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
        .allow_threads(move || csr_adjacency(s.view(), n_points))
        .map_err(PyValueError::new_err)?;
    Ok((indptr.into_pyarray(py), indices.into_pyarray(py)))
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_shull")]
fn shull(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_shull_2d, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shull_2d_f32, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shull_3d, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shull_3d_f32, m)?)?;
    m.add_function(wrap_pyfunction!(vertex_neighbor_vertices, m)?)?;
    Ok(())
}
