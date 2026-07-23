//! Alpha shapes: the circumradius filtration of an existing Delaunay
//! triangulation.
//!
//! An alpha shape (Edelsbrunner & Mücke) is the subcomplex of the Delaunay
//! triangulation obtained by keeping every simplex whose circumscribing ball
//! is small enough: as the radius parameter `alpha` grows from 0 the shape
//! interpolates from the empty set up to the full convex hull, so it captures
//! the "concave hull" of a point cloud at a chosen scale.
//!
//! This module does exactly one numeric thing — compute the circumradius of
//! each simplex — and is kept **entirely separate** from the triangulation
//! kernels in `d2`/`d4`. The circumradii are derived from the already-built
//! `simplices`, on demand, so requesting an alpha shape adds nothing to the
//! Delaunay build's time or memory. The combinatorial part of the filtration
//! (which facets bound the complex at a given `alpha`) is a cheap boolean
//! pass over the existing `neighbors` array and lives in the Python layer,
//! reusing the same facet-column machinery as `convex_hull`.
//!
//! `alpha` is a continuous, user-chosen length scale compared against a
//! continuous radius, so — unlike the visibility/in-sphere *sign* decisions
//! of the triangulation itself — there is nothing to make exact here: plain
//! f64 circumradii are the right tool. Coordinates widen to f64 exactly, so
//! f32 input gives the same radii as passing the points as f64. A degenerate
//! (zero-volume) simplex gets an infinite radius; genuine Delaunay simplices
//! have positive volume, and slivers correctly get a large finite radius
//! (they drop out of the shape only at large `alpha`).

use ndarray::ArrayView2;

#[inline(always)]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Circumradius of a triangle, coordinates taken relative to `a` for
/// numerical stability (the shape is translation-invariant, and centering on
/// a vertex avoids cancellation when the cloud sits far from the origin).
#[inline]
fn circumradius_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    let ux = b[0] - a[0];
    let uy = b[1] - a[1];
    let vx = c[0] - a[0];
    let vy = c[1] - a[1];
    let ul = ux * ux + uy * uy;
    let vl = vx * vx + vy * vy;
    // 2 * signed area of (a, b, c); the circumcenter solves u·y = |u|²/2.
    let d = 2.0 * (ux * vy - uy * vx);
    if d == 0.0 {
        return f64::INFINITY;
    }
    let cx = (vy * ul - uy * vl) / d;
    let cy = (ux * vl - vx * ul) / d;
    (cx * cx + cy * cy).sqrt()
}

/// Circumradius of a tetrahedron, coordinates relative to `a` (see the 2D
/// case). Uses the closed-form circumcenter
/// `y = (|u₁|²(u₂×u₃) + |u₂|²(u₃×u₁) + |u₃|²(u₁×u₂)) / (2·u₁·(u₂×u₃))`.
#[inline]
fn circumradius_3d(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let u1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let u2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let u3 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    let l1 = dot3(u1, u1);
    let l2 = dot3(u2, u2);
    let l3 = dot3(u3, u3);
    let c23 = cross3(u2, u3);
    let denom = 2.0 * dot3(u1, c23);
    if denom == 0.0 {
        return f64::INFINITY;
    }
    let c31 = cross3(u3, u1);
    let c12 = cross3(u1, u2);
    let y = [
        (l1 * c23[0] + l2 * c31[0] + l3 * c12[0]) / denom,
        (l1 * c23[1] + l2 * c31[1] + l3 * c12[1]) / denom,
        (l1 * c23[2] + l2 * c31[2] + l3 * c12[2]) / denom,
    ];
    dot3(y, y).sqrt()
}

/// Materialize the point coordinates into a contiguous `Vec<[f64; D]>`,
/// widening exactly. One pass up front keeps the per-simplex lookups
/// cache-friendly (and lets the simplex loop index a plain slice).
fn to_vec<const D: usize, T: Copy + Into<f64>>(points: ArrayView2<T>) -> Vec<[f64; D]> {
    points
        .rows()
        .into_iter()
        .map(|r| {
            let mut p = [0.0f64; D];
            for (k, slot) in p.iter_mut().enumerate() {
                *slot = r[k].into();
            }
            p
        })
        .collect()
}

/// Reject simplex indices that fall outside `0..n`, so the tight per-simplex
/// loop can index the coordinate slice without bounds checks or panics.
/// (Indices arriving as `i32` from Python are cast to `u32`, so a negative
/// value wraps to a large `u32` and is caught here just the same.)
fn validate<const K: usize>(simplices: &[[u32; K]], n: usize) -> Result<(), String> {
    for s in simplices {
        for &v in s {
            if v as usize >= n {
                return Err(format!(
                    "simplex vertex {} out of range for {} points",
                    v, n
                ));
            }
        }
    }
    Ok(())
}

/// Circumradius of each simplex in `simplices` (each a row of vertex indices
/// into `points`), in the same layout `delaunay2d` / `delaunay4d` return.
///
/// Dispatches on the geometry: 2-D points with 3-vertex simplices → triangle
/// circumradii; 3-D points with 4-vertex simplices → tetrahedron circumradii.
/// Any other `(ndim, arity)` combination is rejected. Returns one radius per
/// simplex, in order.
pub fn circumradii<T: Copy + Into<f64>, const K: usize>(
    points: ArrayView2<T>,
    simplices: &[[u32; K]],
) -> Result<Vec<f64>, String> {
    let n = points.nrows();
    match (points.ncols(), K) {
        (2, 3) => {
            validate(simplices, n)?;
            let pv = to_vec::<2, T>(points);
            Ok(simplices
                .iter()
                .map(|s| {
                    circumradius_2d(pv[s[0] as usize], pv[s[1] as usize], pv[s[2] as usize])
                })
                .collect())
        }
        (3, 4) => {
            validate(simplices, n)?;
            let pv = to_vec::<3, T>(points);
            Ok(simplices
                .iter()
                .map(|s| {
                    circumradius_3d(
                        pv[s[0] as usize],
                        pv[s[1] as usize],
                        pv[s[2] as usize],
                        pv[s[3] as usize],
                    )
                })
                .collect())
        }
        (d, k) => Err(format!(
            "circumradii: unsupported {}D points with {}-vertex simplices \
             (expected 2D/3 or 3D/4)",
            d, k
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn unit_right_triangle_has_hypotenuse_radius() {
        // Right triangle: circumradius = half the hypotenuse.
        let pts = arr2(&[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]);
        let r = circumradii(pts.view(), &[[0u32, 1, 2]]).unwrap();
        assert!((r[0] - (8.0f64).sqrt() / 2.0).abs() < 1e-12);
    }

    #[test]
    fn equilateral_triangle_radius() {
        // Side 1 equilateral: R = 1/sqrt(3).
        let h = (3.0f64).sqrt() / 2.0;
        let pts = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.5, h]]);
        let r = circumradii(pts.view(), &[[0u32, 1, 2]]).unwrap();
        assert!((r[0] - 1.0 / (3.0f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn regular_tetrahedron_radius() {
        // Unit-edge regular tetrahedron: R = sqrt(3/8).
        let pts = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, (3.0f64).sqrt() / 2.0, 0.0],
            [0.5, 1.0 / (12.0f64).sqrt(), (2.0f64 / 3.0).sqrt()],
        ]);
        let r = circumradii(pts.view(), &[[0u32, 1, 2, 3]]).unwrap();
        assert!((r[0] - (3.0f64 / 8.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn axis_aligned_box_tet_radius() {
        // Tet on the corner of a 2×2×2 box: circumcenter at (1,1,1), R=sqrt(3).
        let pts = arr2(&[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);
        let r = circumradii(pts.view(), &[[0u32, 1, 2, 3]]).unwrap();
        assert!((r[0] - (3.0f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn radius_is_translation_invariant() {
        let tri = [[0.3, -0.7], [1.9, 0.2], [0.4, 2.1]];
        let base = arr2(&tri);
        let r0 = circumradii(base.view(), &[[0u32, 1, 2]]).unwrap()[0];
        // Shift far from the origin: centering on a vertex keeps this stable.
        let shifted = arr2(&[
            [tri[0][0] + 1e6, tri[0][1] + 1e6],
            [tri[1][0] + 1e6, tri[1][1] + 1e6],
            [tri[2][0] + 1e6, tri[2][1] + 1e6],
        ]);
        let r1 = circumradii(shifted.view(), &[[0u32, 1, 2]]).unwrap()[0];
        assert!((r0 - r1).abs() < 1e-6, "r0={r0} r1={r1}");
    }

    #[test]
    fn f32_matches_f64() {
        let pts64 = arr2(&[[0.0f64, 0.0], [2.0, 0.0], [0.0, 2.0]]);
        let pts32 = arr2(&[[0.0f32, 0.0], [2.0, 0.0], [0.0, 2.0]]);
        let r64 = circumradii(pts64.view(), &[[0u32, 1, 2]]).unwrap();
        let r32 = circumradii(pts32.view(), &[[0u32, 1, 2]]).unwrap();
        assert_eq!(r64, r32);
    }

    #[test]
    fn out_of_range_index_errors() {
        let pts = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert!(circumradii(pts.view(), &[[0u32, 1, 9]]).is_err());
    }

    #[test]
    fn arity_mismatch_errors() {
        let pts = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        // 4-vertex simplex on 2D points
        assert!(circumradii(pts.view(), &[[0u32, 1, 2, 0]]).is_err());
    }
}
