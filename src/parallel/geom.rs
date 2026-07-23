//! Circumball geometry and exact in-ball predicates for the parallel build.
//!
//! # The conservative-radius contract
//!
//! [`Ball::r_search`] is a floating-point *upper bound* on the true
//! circumradius around the *computed* center: every point strictly inside the
//! simplex's true circumball lies within `r_search` of [`Ball::c`]. The
//! parallel build's correctness leans on this in two places:
//!
//! * **Certification** (block pass): a point is only certified when the
//!   `r_search`-inflated balls of its incident simplices are covered by the
//!   gathered cells. Overestimating the radius can only *uncertify* — safe.
//! * **Global verification** (crust pass): candidate points are enumerated
//!   from grid cells within `r_search`; the actual inside/outside decision
//!   per candidate is made with Shewchuk's exact predicates, so enumeration
//!   completeness is the only thing the float ball must provide.
//!
//! The bound is a standard forward error analysis of the Cramer solve with a
//! generous headroom coefficient; when the system is too ill-conditioned to
//! bound (near-degenerate simplex), `circumball` returns `None` and callers
//! treat the simplex conservatively.

use robust::{incircle, insphere, orient2d, orient3d, Coord, Coord3D};

/// Headroom coefficient for all forward error bounds in this module. The
/// hand-derived constants are < 16; the extra margin costs only certification
/// rate, never correctness.
const CB: f64 = 64.0 * f64::EPSILON;

pub(crate) struct Ball<const D: usize> {
    pub c: [f64; D],
    /// Computed circumradius (distance from `c` to the first vertex). The
    /// pipeline only consumes `r_search`; kept for the calibration tests.
    #[allow(dead_code)]
    pub r: f64,
    /// Conservative search radius: contains the true circumball.
    pub r_search: f64,
}

fn c2(p: [f64; 2]) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

fn c3(p: [f64; 3]) -> Coord3D<f64> {
    Coord3D { x: p[0], y: p[1], z: p[2] }
}

/// Exact orientation of a triangle: positive iff counterclockwise.
pub(crate) fn orient2_exact(s: &[[f64; 2]; 3]) -> f64 {
    orient2d(c2(s[0]), c2(s[1]), c2(s[2]))
}

/// Exact orientation of a tetrahedron: positive iff
/// det[v1-v0; v2-v0; v3-v0] > 0 (the kernel's "positively oriented" output
/// convention, i.e. positive volume). robust::orient3d has the opposite
/// sign (see d4.rs `calibrate_predicates`).
pub(crate) fn orient3_exact(s: &[[f64; 3]; 4]) -> f64 {
    -orient3d(c3(s[0]), c3(s[1]), c3(s[2]), c3(s[3]))
}

/// Exact in-circumcircle test: positive iff `q` lies strictly inside the
/// circumcircle of the triangle, zero iff exactly on it. Works for either
/// orientation of the triangle.
pub(crate) fn inball2_exact(s: &[[f64; 2]; 3], q: [f64; 2]) -> f64 {
    let o = orient2_exact(s);
    debug_assert!(o != 0.0, "in-circle test on a degenerate triangle");
    incircle(c2(s[0]), c2(s[1]), c2(s[2]), c2(q)) * o.signum()
}

/// Exact in-circumsphere test: positive iff `q` lies strictly inside the
/// circumsphere of the tetrahedron, zero iff exactly on it. Works for either
/// orientation.
///
/// For a positively oriented tet (det[v1-v0; v2-v0; v3-v0] > 0), `q` inside
/// means the lifted determinant det[v1-v0; v2-v0; v3-v0; q-v0] < 0, and
/// robust::insphere(v1, v2, v3, q, v0) evaluates exactly that determinant
/// (see d4.rs `calibrate_predicates`); pinned by `calibrate_inball` below.
pub(crate) fn inball3_exact(s: &[[f64; 3]; 4], q: [f64; 3]) -> f64 {
    let o = orient3_exact(s);
    debug_assert!(o != 0.0, "in-sphere test on a degenerate tetrahedron");
    -insphere(c3(s[1]), c3(s[2]), c3(s[3]), c3(q), c3(s[0])) * o.signum()
}

/// Circumball of a triangle, with a conservative search radius (see module
/// docs). `None` if the triangle is degenerate or too ill-conditioned to
/// bound the center error.
pub(crate) fn circumball2(s: &[[f64; 2]; 3]) -> Option<Ball<2>> {
    let [a, b, c] = *s;
    let ux = b[0] - a[0];
    let uy = b[1] - a[1];
    let vx = c[0] - a[0];
    let vy = c[1] - a[1];
    let bl = ux * ux + uy * uy;
    let cl = vx * vx + vy * vy;
    let d = ux * vy - uy * vx;
    let sd = (ux * vy).abs() + (uy * vx).abs();
    // Ill-conditioned: the determinant itself is not reliably nonzero.
    if !(d != 0.0 && CB * sd < 0.25 * d.abs()) {
        return None;
    }
    let half = 0.5 / d;
    let numx = vy * bl - uy * cl;
    let numy = ux * cl - vx * bl;
    let x = numx * half;
    let y = numy * half;
    let sx = (vy * bl).abs() + (uy * cl).abs();
    let sy = (ux * cl).abs() + (vx * bl).abs();
    let amp = sd / d.abs();
    let cerr = CB
        * ((sx + sy) * half.abs()
            + (x.abs() + y.abs()) * amp
            + x.abs()
            + y.abs()
            + a[0].abs()
            + a[1].abs());
    let cx = a[0] + x;
    let cy = a[1] + y;
    let r = (x * x + y * y).sqrt();
    let r_search = r * (1.0 + 8.0 * f64::EPSILON) + 2.0 * cerr;
    if !r_search.is_finite() {
        return None;
    }
    Some(Ball { c: [cx, cy], r, r_search })
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Sum of the absolute values of det3's six products: forward error scale.
fn det3_abs(m: [[f64; 3]; 3]) -> f64 {
    m[0][0].abs() * ((m[1][1] * m[2][2]).abs() + (m[1][2] * m[2][1]).abs())
        + m[0][1].abs() * ((m[1][0] * m[2][2]).abs() + (m[1][2] * m[2][0]).abs())
        + m[0][2].abs() * ((m[1][0] * m[2][1]).abs() + (m[1][1] * m[2][0]).abs())
}

/// Circumball of a tetrahedron, with a conservative search radius. `None` if
/// degenerate or too ill-conditioned.
pub(crate) fn circumball3(s: &[[f64; 3]; 4]) -> Option<Ball<3>> {
    let a = s[0];
    // Solve M c' = rhs with rows u_i = v_i - a, rhs_i = |u_i|^2 / 2; the
    // center is a + c'.
    let mut m = [[0.0f64; 3]; 3];
    let mut rhs = [0.0f64; 3];
    for i in 0..3 {
        for k in 0..3 {
            m[i][k] = s[i + 1][k] - a[k];
        }
        rhs[i] = 0.5 * (m[i][0] * m[i][0] + m[i][1] * m[i][1] + m[i][2] * m[i][2]);
    }
    let d = det3(m);
    let da = det3_abs(m);
    if !(d != 0.0 && CB * da < 0.25 * d.abs()) {
        return None;
    }
    let amp = da / d.abs();
    let mut cp = [0.0f64; 3]; // center - a
    let mut cerr = 0.0f64;
    for k in 0..3 {
        let mut mk = m;
        let mut mka = m;
        for i in 0..3 {
            mk[i][k] = rhs[i];
            mka[i][k] = rhs[i];
        }
        let num = det3(mk);
        let numa = det3_abs(mka);
        cp[k] = num / d;
        cerr += CB * (numa / d.abs() + cp[k].abs() * amp + cp[k].abs() + a[k].abs());
    }
    let c = [a[0] + cp[0], a[1] + cp[1], a[2] + cp[2]];
    let r = (cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2]).sqrt();
    let r_search = r * (1.0 + 8.0 * f64::EPSILON) + 2.0 * cerr;
    if !r_search.is_finite() {
        return None;
    }
    Some(Ball { c, r, r_search })
}

/// Squared distance from a point to an axis-aligned box (0 inside).
pub(crate) fn box_dist2<const D: usize>(p: &[f64; D], lo: &[f64; D], hi: &[f64; D]) -> f64 {
    let mut d2 = 0.0;
    for k in 0..D {
        let d = if p[k] < lo[k] {
            lo[k] - p[k]
        } else if p[k] > hi[k] {
            p[k] - hi[k]
        } else {
            0.0
        };
        d2 += d * d;
    }
    d2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Pin the sign conventions of the exact in-ball predicates on known
    /// configurations, for both simplex orientations.
    #[test]
    fn calibrate_inball() {
        // CCW unit right triangle: circumcircle center (0.5, 0.5), r^2 = 0.5.
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let tri_cw = [tri[0], tri[2], tri[1]];
        assert!(orient2_exact(&tri) > 0.0 && orient2_exact(&tri_cw) < 0.0);
        for t in [&tri, &tri_cw] {
            assert!(inball2_exact(t, [0.5, 0.5]) > 0.0);
            assert!(inball2_exact(t, [2.0, 2.0]) < 0.0);
            assert!(inball2_exact(t, [1.0, 1.0]) == 0.0); // cocircular
        }

        // Positively oriented unit tet: circumsphere center (.5,.5,.5).
        let tet = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let tet_neg = [tet[1], tet[0], tet[2], tet[3]];
        assert!(orient3_exact(&tet) > 0.0 && orient3_exact(&tet_neg) < 0.0);
        for t in [&tet, &tet_neg] {
            assert!(inball3_exact(t, [0.5, 0.5, 0.5]) > 0.0);
            assert!(inball3_exact(t, [2.0, 2.0, 2.0]) < 0.0);
            assert!(inball3_exact(t, [1.0, 1.0, 1.0]) == 0.0); // cospherical
        }
    }

    /// The computed ball must be equidistant from all simplex vertices (up to
    /// the reported tolerance) and r_search must dominate the true radius.
    #[test]
    fn circumballs_are_consistent() {
        let mut next = rng(7);
        for _ in 0..2000 {
            let s2 = [[next(), next()], [next(), next()], [next(), next()]];
            if let Some(ball) = circumball2(&s2) {
                for v in &s2 {
                    let d = ((v[0] - ball.c[0]).powi(2) + (v[1] - ball.c[1]).powi(2)).sqrt();
                    assert!(
                        (d - ball.r).abs() <= (ball.r_search - ball.r) + 1e-12 * ball.r,
                        "2D vertex distance {} vs r {} (search {})",
                        d,
                        ball.r,
                        ball.r_search
                    );
                }
                assert!(ball.r_search >= ball.r);
            }
            let s3 = [
                [next(), next(), next()],
                [next(), next(), next()],
                [next(), next(), next()],
                [next(), next(), next()],
            ];
            if let Some(ball) = circumball3(&s3) {
                for v in &s3 {
                    let d = (0..3)
                        .map(|k| (v[k] - ball.c[k]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    assert!(
                        (d - ball.r).abs() <= (ball.r_search - ball.r) + 1e-12 * ball.r,
                        "3D vertex distance {} vs r {} (search {})",
                        d,
                        ball.r,
                        ball.r_search
                    );
                }
                assert!(ball.r_search >= ball.r);
            }
        }
    }

    /// Points the exact predicate says are strictly inside the circumball
    /// must lie within r_search of the computed center: this is the
    /// enumeration-completeness contract the crust verification relies on.
    /// Stress it with well-conditioned, tiny, huge and near-degenerate
    /// simplices.
    #[test]
    fn r_search_contains_all_exact_inside_points() {
        let mut next = rng(13);
        let scales = [1.0, 1e-20, 1e20, 1.0, 1.0];
        for round in 0..5000 {
            let scale = scales[round % scales.len()];
            // Make some simplices nearly degenerate by flattening one vertex
            // towards an edge/face.
            let flat = if round % 3 == 0 { 1e-9 } else { 1.0 };

            let s2 = [
                [next() * scale, next() * scale],
                [next() * scale, next() * scale],
                [next() * scale, next() * scale * flat],
            ];
            if let Some(ball) = circumball2(&s2) {
                for _ in 0..20 {
                    let q = [next() * scale, next() * scale];
                    if inball2_exact(&s2, q) > 0.0 {
                        let d = ((q[0] - ball.c[0]).powi(2) + (q[1] - ball.c[1]).powi(2)).sqrt();
                        assert!(
                            d < ball.r_search,
                            "2D inside point at {} vs r_search {}",
                            d,
                            ball.r_search
                        );
                    }
                }
            }

            let s3 = [
                [next() * scale, next() * scale, next() * scale],
                [next() * scale, next() * scale, next() * scale],
                [next() * scale, next() * scale, next() * scale],
                [next() * scale, next() * scale, next() * scale * flat],
            ];
            if let Some(ball) = circumball3(&s3) {
                for _ in 0..20 {
                    let q = [next() * scale, next() * scale, next() * scale];
                    if inball3_exact(&s3, q) > 0.0 {
                        let d = (0..3)
                            .map(|k| (q[k] - ball.c[k]).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        assert!(
                            d < ball.r_search,
                            "3D inside point at {} vs r_search {}",
                            d,
                            ball.r_search
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn degenerate_simplices_return_none() {
        // Exactly collinear / coplanar.
        assert!(circumball2(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]).is_none());
        assert!(circumball3(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        .is_none());
    }

    #[test]
    fn box_dist2_basics() {
        let lo = [0.0, 0.0];
        let hi = [1.0, 1.0];
        assert_eq!(box_dist2(&[0.5, 0.5], &lo, &hi), 0.0);
        assert_eq!(box_dist2(&[2.0, 0.5], &lo, &hi), 1.0);
        assert_eq!(box_dist2(&[2.0, 3.0], &lo, &hi), 1.0 + 4.0);
    }
}
