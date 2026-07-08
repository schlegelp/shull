//! 2D Delaunay triangulation via S-hull (radial sweep-hull).
//!
//! Implements David Sinclair's S-hull algorithm (http://www.s-hull.org/):
//! pick a seed triangle with the smallest circumcircle, sort the remaining
//! points by distance from its circumcenter, grow the triangulation by
//! attaching each point to the visible part of the advancing convex hull,
//! and restore the Delaunay condition by edge flipping.
//!
//! The implementation uses the data-structure design proven out by the
//! Delaunator family of S-hull implementations: flat halfedge arrays for the
//! triangulation, the hull as a linked ring with a pseudo-angle hash table
//! for O(1) expected visible-edge lookup, and stack-based legalization.
//!
//! As in the 3D module, all combinatorial decisions (edge visibility, the
//! in-circle flip test) use Shewchuk's exact adaptive predicates on centered
//! coordinates, so the triangulation cannot be corrupted by rounding and the
//! flip loop provably terminates -- no angle computations, no configuration
//! history.

use ndarray::ArrayView2;
use robust::{incircle, orient2d, Coord};

use crate::error::DelaunayError;

const EMPTY: u32 = u32::MAX;

fn coord(p: [f64; 2]) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

/// Exact orientation: positive if (a, b, c) are counterclockwise.
fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    orient2d(coord(a), coord(b), coord(c))
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// Squared circumradius of the triangle (a, b, c); infinite if degenerate.
fn circumradius2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let ex = c[0] - a[0];
    let ey = c[1] - a[1];
    let bl = dx * dx + dy * dy;
    let cl = ex * ex + ey * ey;
    let d = dx * ey - dy * ex;
    if d == 0.0 {
        return f64::INFINITY;
    }
    let x = (ey * bl - dy * cl) * (0.5 / d);
    let y = (dx * cl - ex * bl) * (0.5 / d);
    x * x + y * y
}

fn circumcenter(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> [f64; 2] {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let ex = c[0] - a[0];
    let ey = c[1] - a[1];
    let bl = dx * dx + dy * dy;
    let cl = ex * ex + ey * ey;
    let d = dx * ey - dy * ex;
    let x = (ey * bl - dy * cl) * (0.5 / d);
    let y = (dx * cl - ex * bl) * (0.5 / d);
    [a[0] + x, a[1] + y]
}

/// Monotonic proxy for the angle of (dx, dy), in [0, 1).
fn pseudo_angle(dx: f64, dy: f64) -> f64 {
    let p = dx / (dx.abs() + dy.abs());
    (if dy > 0.0 { 3.0 - p } else { 1.0 + p }) / 4.0
}

struct Triangulator {
    pts: Vec<[f64; 2]>,
    /// Vertex id at the start of each halfedge; 3 consecutive per triangle.
    triangles: Vec<u32>,
    /// Twin halfedge of each halfedge (EMPTY on the hull).
    halfedges: Vec<u32>,
    hull_prev: Vec<u32>,
    hull_next: Vec<u32>,
    /// For hull vertex v: the interior halfedge lying on hull edge
    /// v -> hull_next[v].
    hull_tri: Vec<u32>,
    hull_hash: Vec<u32>,
    hull_start: u32,
    hash_size: usize,
    center: [f64; 2],
    /// Legalization work stack (reused).
    stack: Vec<u32>,
}

impl Triangulator {
    fn hash_key(&self, p: [f64; 2]) -> usize {
        let angle = pseudo_angle(p[0] - self.center[0], p[1] - self.center[1]);
        ((angle * self.hash_size as f64) as usize) % self.hash_size
    }

    fn link(&mut self, a: u32, b: u32) {
        self.halfedges[a as usize] = b;
        if b != EMPTY {
            self.halfedges[b as usize] = a;
        }
    }

    /// Add the triangle (i0, i1, i2), twin-linked to halfedges (a, b, c);
    /// returns the id of its first halfedge.
    fn add_triangle(&mut self, i0: u32, i1: u32, i2: u32, a: u32, b: u32, c: u32) -> u32 {
        let t = self.triangles.len() as u32;
        self.triangles.push(i0);
        self.triangles.push(i1);
        self.triangles.push(i2);
        self.halfedges.push(EMPTY);
        self.halfedges.push(EMPTY);
        self.halfedges.push(EMPTY);
        self.link(t, a);
        self.link(t + 1, b);
        self.link(t + 2, c);
        t
    }

    /// Restore the Delaunay condition around halfedge `a` by flipping illegal
    /// edges (exact in-circle test; cocircular counts as legal, so the loop
    /// terminates). Returns the halfedge that ends up along the hull edge the
    /// caller is tracking.
    fn legalize(&mut self, a: u32) -> u32 {
        self.stack.clear();
        let mut a = a;
        let mut ar;
        loop {
            let b = self.halfedges[a as usize];
            let a0 = a - a % 3;
            ar = a0 + (a + 2) % 3;

            if b == EMPTY {
                match self.stack.pop() {
                    Some(e) => {
                        a = e;
                        continue;
                    }
                    None => break,
                }
            }

            let al = a0 + (a + 1) % 3;
            let b0 = b - b % 3;
            let bl = b0 + (b + 2) % 3;

            let p0 = self.triangles[ar as usize];
            let pr = self.triangles[a as usize];
            let pl = self.triangles[al as usize];
            let p1 = self.triangles[bl as usize];

            // (p0, pr, pl) is counterclockwise; the shared edge (pr, pl) is
            // illegal if p1 lies strictly inside its circumcircle.
            let illegal = incircle(
                coord(self.pts[p0 as usize]),
                coord(self.pts[pr as usize]),
                coord(self.pts[pl as usize]),
                coord(self.pts[p1 as usize]),
            ) > 0.0;

            if illegal {
                self.triangles[a as usize] = p1;
                self.triangles[b as usize] = p0;

                let hbl = self.halfedges[bl as usize];
                // The flipped edge exposes bl; if bl was on the hull, repoint
                // the hull's halfedge reference at a.
                if hbl == EMPTY {
                    let mut e = self.hull_start;
                    loop {
                        if self.hull_tri[e as usize] == bl {
                            self.hull_tri[e as usize] = a;
                            break;
                        }
                        e = self.hull_prev[e as usize];
                        if e == self.hull_start {
                            break;
                        }
                    }
                }
                let har = self.halfedges[ar as usize];
                self.link(a, hbl);
                self.link(b, har);
                self.link(ar, bl);

                self.stack.push(b0 + (b + 1) % 3);
            } else {
                match self.stack.pop() {
                    Some(e) => a = e,
                    None => break,
                }
            }
        }
        ar
    }
}

/// Compute the 2D Delaunay triangulation of `points` (an (n, 2) array of f64
/// or f32 coordinates; f32 widens to f64 exactly).
///
/// Returns counterclockwise triangles as indices into the caller's point
/// array, plus the triangle adjacency: `neighbors[t][j]` is the triangle
/// across the edge opposite vertex j of triangle t (scipy's convention),
/// -1 on the hull. Exact duplicate points are dropped (one representative
/// is kept).
pub fn delaunay2d<T: Copy + Into<f64>>(
    points: ArrayView2<T>,
) -> Result<(Vec<[u32; 3]>, Vec<[i32; 3]>), DelaunayError> {
    assert_eq!(points.ncols(), 2, "input points must be 2D");
    let n = points.nrows();
    if n >= EMPTY as usize {
        return Err(DelaunayError::Degenerate(format!(
            "too many points ({}), maximum is {}",
            n,
            EMPTY - 1
        )));
    }
    if n < 3 {
        return Err(DelaunayError::TooFewPoints(n));
    }

    // Center the points on their centroid (Delaunay is translation-invariant
    // and this removes the dominant float cancellation for far-away clouds).
    let mut cx = 0.0;
    let mut cy = 0.0;
    for row in points.rows() {
        cx += row[0].into();
        cy += row[1].into();
    }
    cx /= n as f64;
    cy /= n as f64;
    let pts: Vec<[f64; 2]> = points
        .rows()
        .into_iter()
        .map(|row| [row[0].into() - cx, row[1].into() - cy])
        .collect();

    // Seed selection (S-hull): the point nearest the centroid, its nearest
    // neighbor, and the point completing the smallest circumcircle.
    let i0 = (0..n)
        .min_by(|&a, &b| dist2(pts[a], [0.0, 0.0]).total_cmp(&dist2(pts[b], [0.0, 0.0])))
        .unwrap();
    let i1 = (0..n)
        .filter(|&i| i != i0 && pts[i] != pts[i0])
        .min_by(|&a, &b| dist2(pts[a], pts[i0]).total_cmp(&dist2(pts[b], pts[i0])))
        .ok_or(DelaunayError::TooFewPoints(1))?;

    let mut i2 = usize::MAX;
    let mut best_r2 = f64::INFINITY;
    for i in 0..n {
        if i == i0 || i == i1 {
            continue;
        }
        let r2 = circumradius2(pts[i0], pts[i1], pts[i]);
        if r2 < best_r2 && orient(pts[i0], pts[i1], pts[i]) != 0.0 {
            best_r2 = r2;
            i2 = i;
        }
    }
    if i2 == usize::MAX {
        return Err(DelaunayError::Degenerate(
            "all points are collinear; a 2D triangulation does not exist".to_string(),
        ));
    }
    let (i0, mut i1, mut i2) = (i0 as u32, i1 as u32, i2 as u32);
    // Make the seed triangle counterclockwise.
    if orient(pts[i0 as usize], pts[i1 as usize], pts[i2 as usize]) < 0.0 {
        std::mem::swap(&mut i1, &mut i2);
    }
    let center = circumcenter(pts[i0 as usize], pts[i1 as usize], pts[i2 as usize]);

    // Radial sort: process points by distance from the seed circumcenter.
    // Squared distances are >= 0, so their IEEE bit patterns sort like the
    // values; ties are ordered by (x, y, index) so exact duplicates end up
    // adjacent (making the skip below catch them all).
    let mut ids: Vec<(u64, u32)> = (0..n as u32)
        .map(|i| (dist2(pts[i as usize], center).to_bits(), i))
        .collect();
    ids.sort_unstable();
    let mut s = 0;
    while s < ids.len() {
        let mut e = s + 1;
        while e < ids.len() && ids[e].0 == ids[s].0 {
            e += 1;
        }
        if e - s > 1 {
            ids[s..e].sort_unstable_by(|&(_, a), &(_, b)| {
                let (pa, pb) = (pts[a as usize], pts[b as usize]);
                pa[0]
                    .total_cmp(&pb[0])
                    .then_with(|| pa[1].total_cmp(&pb[1]))
                    .then_with(|| a.cmp(&b))
            });
        }
        s = e;
    }

    let max_triangles = 2 * n - 5;
    let hash_size = (n as f64).sqrt().ceil() as usize;
    let mut tr = Triangulator {
        pts,
        triangles: Vec::with_capacity(max_triangles * 3),
        halfedges: Vec::with_capacity(max_triangles * 3),
        hull_prev: vec![0; n],
        hull_next: vec![0; n],
        hull_tri: vec![0; n],
        hull_hash: vec![EMPTY; hash_size],
        hull_start: i0,
        hash_size,
        center,
        stack: Vec::new(),
    };

    // Seed triangle and initial (counterclockwise) hull ring.
    tr.hull_next[i0 as usize] = i1;
    tr.hull_next[i1 as usize] = i2;
    tr.hull_next[i2 as usize] = i0;
    tr.hull_prev[i2 as usize] = i1;
    tr.hull_prev[i1 as usize] = i0;
    tr.hull_prev[i0 as usize] = i2;
    tr.hull_tri[i0 as usize] = 0;
    tr.hull_tri[i1 as usize] = 1;
    tr.hull_tri[i2 as usize] = 2;
    let key = tr.hash_key(tr.pts[i0 as usize]);
    tr.hull_hash[key] = i0;
    let key = tr.hash_key(tr.pts[i1 as usize]);
    tr.hull_hash[key] = i1;
    let key = tr.hash_key(tr.pts[i2 as usize]);
    tr.hull_hash[key] = i2;
    tr.add_triangle(i0, i1, i2, EMPTY, EMPTY, EMPTY);

    let mut prev: Option<[f64; 2]> = None;
    for &(_, i) in &ids {
        let p = tr.pts[i as usize];

        // Exact duplicates are adjacent in the sorted order: skip them.
        if prev == Some(p) {
            continue;
        }
        prev = Some(p);
        if i == i0 || i == i1 || i == i2 {
            continue;
        }

        // Find a hull edge visible from p, starting near its pseudo-angle.
        let mut start = 0u32;
        let key = tr.hash_key(p);
        for j in 0..tr.hash_size {
            start = tr.hull_hash[(key + j) % tr.hash_size];
            if start != EMPTY && start != tr.hull_next[start as usize] {
                break;
            }
        }
        start = tr.hull_prev[start as usize];
        let mut e = start;
        loop {
            let q = tr.hull_next[e as usize];
            // Visible: p lies strictly to the right of the hull edge e -> q.
            if orient(tr.pts[e as usize], tr.pts[q as usize], p) < 0.0 {
                break;
            }
            e = q;
            if e == start {
                e = EMPTY;
                break;
            }
        }
        if e == EMPTY {
            continue; // point coincides with a hull vertex
        }

        // First triangle from the new point.
        let t = tr.add_triangle(e, i, tr.hull_next[e as usize], EMPTY, EMPTY, tr.hull_tri[e as usize]);
        tr.hull_tri[i as usize] = tr.legalize(t + 2);
        tr.hull_tri[e as usize] = t;

        // Walk forward, adding triangles while the next hull edge is visible.
        let mut nxt = tr.hull_next[e as usize];
        loop {
            let q = tr.hull_next[nxt as usize];
            if orient(tr.pts[nxt as usize], tr.pts[q as usize], p) >= 0.0 {
                break;
            }
            let t = tr.add_triangle(nxt, i, q, tr.hull_tri[i as usize], EMPTY, tr.hull_tri[nxt as usize]);
            tr.hull_tri[i as usize] = tr.legalize(t + 2);
            tr.hull_next[nxt as usize] = nxt; // mark as removed from the hull
            nxt = q;
        }

        // Walk backward the same way (only needed if we started at the edge
        // the hash gave us; otherwise everything behind is invisible).
        if e == start {
            loop {
                let q = tr.hull_prev[e as usize];
                if orient(tr.pts[q as usize], tr.pts[e as usize], p) >= 0.0 {
                    break;
                }
                let t = tr.add_triangle(q, i, e, EMPTY, tr.hull_tri[e as usize], tr.hull_tri[q as usize]);
                tr.legalize(t + 2);
                tr.hull_tri[q as usize] = t;
                tr.hull_next[e as usize] = e; // mark as removed
                e = q;
            }
        }

        // Splice the new point into the hull ring.
        tr.hull_start = e;
        tr.hull_prev[i as usize] = e;
        tr.hull_next[e as usize] = i;
        tr.hull_prev[nxt as usize] = i;
        tr.hull_next[i as usize] = nxt;

        let key = tr.hash_key(p);
        tr.hull_hash[key] = i;
        let key = tr.hash_key(tr.pts[e as usize]);
        tr.hull_hash[key] = e;
    }

    // Halfedge 3t + k runs from the vertex at position k to the one at
    // position (k+1)%3, so the edge opposite vertex position j is halfedge
    // 3t + (j+1)%3; its twin's triangle is the neighbor opposite vertex j.
    let neighbors = (0..tr.triangles.len() / 3)
        .map(|t| {
            [1, 2, 0].map(|k| match tr.halfedges[3 * t + k] {
                EMPTY => -1,
                twin => (twin / 3) as i32,
            })
        })
        .collect();

    let triangles = tr
        .triangles
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect();
    Ok((triangles, neighbors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn to_array(points: &[[f64; 2]]) -> Array2<f64> {
        Array2::from_shape_vec(
            (points.len(), 2),
            points.iter().flatten().copied().collect(),
        )
        .unwrap()
    }

    fn pseudo_random_points(n: usize, seed: u64) -> Vec<[f64; 2]> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n).map(|_| [next(), next()]).collect()
    }

    fn float_orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }

    /// Pin down the predicate conventions: robust::orient2d must match the
    /// counterclockwise-positive float determinant, and robust::incircle must
    /// be positive exactly when the query point is inside the circumcircle of
    /// a counterclockwise triangle.
    #[test]
    fn calibrate_predicates() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0]; // (a, b, c) is counterclockwise
        assert!(float_orient(a, b, c) > 0.0);
        assert!(orient(a, b, c) > 0.0, "robust orient2d is not CCW-positive");
        assert!(orient(a, c, b) < 0.0);

        // (a, b, c) circumcircle is centered at (0.5, 0.5), radius^2 = 0.5.
        let inside = [0.5, 0.5];
        let outside = [2.0, 2.0];
        assert!(incircle(coord(a), coord(b), coord(c), coord(inside)) > 0.0);
        assert!(incircle(coord(a), coord(b), coord(c), coord(outside)) < 0.0);
        // Cocircular: exactly zero.
        assert!(incircle(coord(a), coord(b), coord(c), coord([1.0, 1.0])) == 0.0);
    }

    /// Every triangle's circumcircle must be empty of other points.
    fn assert_delaunay(points: &[[f64; 2]], tris: &[[u32; 3]]) {
        for tri in tris {
            let [a, b, c] = [
                points[tri[0] as usize],
                points[tri[1] as usize],
                points[tri[2] as usize],
            ];
            assert!(
                float_orient(a, b, c) > 0.0,
                "triangle {:?} is not counterclockwise",
                tri
            );
            for (i, &q) in points.iter().enumerate() {
                if tri.contains(&(i as u32)) {
                    continue;
                }
                assert!(
                    incircle(coord(a), coord(b), coord(c), coord(q)) <= 0.0,
                    "point {} is inside the circumcircle of {:?}",
                    i,
                    tri
                );
            }
        }
    }

    #[test]
    fn random_points_are_delaunay() {
        for seed in [1, 2, 3] {
            let points = pseudo_random_points(120, seed);
            let (tris, _) = delaunay2d(to_array(&points).view()).unwrap();
            assert!(!tris.is_empty());
            assert_delaunay(&points, &tris);
            // Every point must appear (no drops on distinct inputs).
            let mut seen = vec![false; points.len()];
            for t in &tris {
                for &v in t {
                    seen[v as usize] = true;
                }
            }
            assert!(seen.iter().all(|&s| s));
        }
    }

    #[test]
    fn triangle_count_matches_euler() {
        // For n points with h on the hull: triangles = 2n - h - 2.
        let points = pseudo_random_points(500, 4);
        let (tris, _) = delaunay2d(to_array(&points).view()).unwrap();
        // Count hull edges: edges that appear in only one triangle.
        let mut edge_count = std::collections::HashMap::new();
        for t in &tris {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *edge_count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        let h = edge_count.values().filter(|&&c| c == 1).count();
        assert_eq!(tris.len(), 2 * points.len() - h - 2);
    }

    #[test]
    fn neighbors_are_mutual_and_share_edges() {
        let points = pseudo_random_points(200, 11);
        let (tris, nbrs) = delaunay2d(to_array(&points).view()).unwrap();
        assert_eq!(tris.len(), nbrs.len());
        let mut boundary = 0;
        for (t, nb) in nbrs.iter().enumerate() {
            for j in 0..3 {
                if nb[j] < 0 {
                    boundary += 1;
                    continue;
                }
                let k = nb[j] as usize;
                // The edge opposite vertex j is shared; vertex j itself is not.
                for (jj, v) in tris[t].iter().enumerate() {
                    assert_eq!(tris[k].contains(v), jj != j);
                }
                assert!(nbrs[k].contains(&(t as i32)), "adjacency is not mutual");
            }
        }
        // Exactly the hull edges (edges in only one triangle) have no neighbor.
        let mut edge_count = std::collections::HashMap::new();
        for t in &tris {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *edge_count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        let hull_edges = edge_count.values().filter(|&&c| c == 1).count();
        assert_eq!(boundary, hull_edges);
    }

    #[test]
    fn grid_points_triangulate() {
        // Cocircular quads everywhere: exercises exact-cocircular ties.
        let mut points = Vec::new();
        for x in 0..30 {
            for y in 0..30 {
                points.push([x as f64, y as f64]);
            }
        }
        let (tris, _) = delaunay2d(to_array(&points).view()).unwrap();
        assert_eq!(tris.len(), 2 * 29 * 29); // grid area tiled by half-squares
        let total: f64 = tris
            .iter()
            .map(|t| {
                float_orient(
                    points[t[0] as usize],
                    points[t[1] as usize],
                    points[t[2] as usize],
                ) / 2.0
            })
            .sum();
        assert!((total - (29.0 * 29.0)).abs() < 1e-9);
    }

    #[test]
    fn duplicates_are_dropped() {
        let mut points = pseudo_random_points(50, 7);
        points.push(points[3]);
        points.push(points[20]);
        let (tris, _) = delaunay2d(to_array(&points).view()).unwrap();
        assert_delaunay(&points[..50], &tris);
        let used: std::collections::HashSet<u32> =
            tris.iter().flatten().copied().collect();
        // One representative of each duplicate pair.
        assert!(used.contains(&3) ^ used.contains(&50));
        assert!(used.contains(&20) ^ used.contains(&51));
    }

    #[test]
    fn collinear_input_errors() {
        let points: Vec<[f64; 2]> = (0..40).map(|i| [i as f64, 2.0 * i as f64]).collect();
        match delaunay2d(to_array(&points).view()) {
            Err(DelaunayError::Degenerate(_)) => {}
            other => panic!("expected Degenerate, got {:?}", other.map(|t| t.0.len())),
        }
    }

    #[test]
    fn too_few_points_errors() {
        let points = [[0.0, 0.0], [1.0, 1.0]];
        match delaunay2d(to_array(&points).view()) {
            Err(DelaunayError::TooFewPoints(_)) => {}
            other => panic!("expected TooFewPoints, got {:?}", other.map(|t| t.0.len())),
        }
    }

    #[test]
    fn f32_input_matches_f64() {
        let points = pseudo_random_points(80, 9);
        let pts32: Vec<f32> = points.iter().flatten().map(|&v| v as f32).collect();
        let arr32 = Array2::from_shape_vec((80, 2), pts32).unwrap();
        let arr64 = arr32.mapv(|v| v as f64);
        assert_eq!(
            delaunay2d(arr32.view()).unwrap(),
            delaunay2d(arr64.view()).unwrap()
        );
    }

    #[test]
    fn tight_clusters_terminate() {
        // The adversarial input class that corrupted the old float-only code.
        let mut state = 0x243F6A8885A308D3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let points: Vec<[f64; 2]> = (0..400)
            .map(|i| {
                let off = if i < 200 { 0.0 } else { 1000.0 };
                [next() * 1e-6 + off, next() * 1e-6 + off]
            })
            .collect();
        let (tris, _) = delaunay2d(to_array(&points).view()).unwrap();
        let mut seen = vec![false; 400];
        for t in &tris {
            for &v in t {
                seen[v as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "some points missing from output");
    }
}
