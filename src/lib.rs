use ndarray::ArrayView2;

mod alpha;
mod d2;
mod d4;
mod error;

pub use alpha::circumradii;
pub use d2::delaunay2d;
pub use d4::delaunay4d;
pub use error::DelaunayError;

#[cfg(feature = "parallel")]
mod parallel;
#[cfg(feature = "parallel")]
pub use parallel::{
    delaunay2d_par, delaunay2d_par_with_progress, delaunay2d_par_with_stats, delaunay4d_par,
    delaunay4d_par_with_progress, delaunay4d_par_with_stats, FallbackReason, ParProgress,
    ParStats,
};

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub use python::{
    calculate_shull_2d, calculate_shull_2d_f32, calculate_shull_3d, calculate_shull_3d_f32,
    vertex_neighbor_vertices,
};

/// Build the vertex-adjacency CSR arrays from an (n, k) simplex array.
///
/// Two vertices are adjacent if they share a simplex. Returns
/// `(indptr, indices)` (both int32) where the neighbors of vertex v are
/// `indices[indptr[v]..indptr[v + 1]]`, sorted ascending — the same layout
/// as scipy's `vertex_neighbor_vertices`. Vertices in `0..n_points` not
/// referenced by any simplex (e.g. dropped exact duplicates) get an empty
/// row.
pub fn csr_adjacency(s: ArrayView2<i32>, n_points: usize) -> Result<(Vec<i32>, Vec<i32>), String> {
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
