//! 3D Delaunay tetrahedralization via a 4D sweep-hull.
//!
//! This generalizes David Sinclair's "Newton Apple Wrapper" algorithm
//! (arXiv 1602.04707) one dimension up: the paper computes a 2D Delaunay
//! triangulation by lifting points onto a 3D paraboloid and taking the
//! downward-facing facets of the 3D convex hull. Here we lift 3D points
//! onto a 4D paraboloid (w = x^2 + y^2 + z^2), compute the 4D convex hull
//! with the same sorted-insertion sweep-hull scheme, and keep the
//! downward-facing facets -- those tetrahedra are exactly the 3D Delaunay
//! simplices.
//!
//! # Numerical robustness
//!
//! Every combinatorial decision (facet visibility, facet orientation, the
//! lower-hull filter) is made with a float fast path guarded by a per-facet
//! forward error bound; when the float value falls inside the bound the
//! decision falls back to Shewchuk's exact adaptive predicates (the `robust`
//! crate). The lifted visibility determinant is exactly the `insphere`
//! predicate on the 3D coordinates, so all decisions are consistent with the
//! exact lift of the (centered) input coordinates and the hull can never be
//! corrupted by rounding -- inconsistent float signs would otherwise make the
//! facet count explode on adversarial inputs (e.g. tight point clusters far
//! from each other).

use ndarray::ArrayView2;
use robust::{insphere, orient3d, Coord3D};

use crate::error::DelaunayError;

const NONE: u32 = u32::MAX;

/// Headroom constant for the forward error bound of the visibility test.
const ERRB_COEFF: f64 = 64.0 * f64::EPSILON;

/// A tetrahedral facet of the 4D hull.
///
/// `n[i]` is the id of the facet across the ridge (triangle) opposite `v[i]`,
/// i.e. the ridge formed by the other three vertices.
#[derive(Clone, Copy, Debug)]
struct Tet {
    v: [u32; 4],
    n: [u32; 4],
    /// Outward-facing normal of the facet's hyperplane (not normalized).
    norm: [f64; 4],
    /// Forward error bound for `facet_dist` against any point of the cloud:
    /// |float result| > errb guarantees the float sign is exact.
    errb: f64,
}

/// Facet state, kept in a dense parallel array so the flood fill's
/// dead-neighbor checks touch one byte instead of a whole `Tet`.
/// Bits 0-1: 0 = dead (inside the hull), 1 = live, 2 = newly spawned.
/// Bit 2: whether `norm` is the negated canonical normal (see `normal4`),
/// needed to interpret exact predicate signs.
const F_STATE: u8 = 3;
const F_DEAD: u8 = 0;
const F_LIVE: u8 = 1;
const F_NEW: u8 = 2;
const F_FLIP: u8 = 4;

/// An unmatched ridge of a newly spawned facet, stored in the pairing table.
///
/// All unmatched ridges contain the newly inserted point, so the key is the
/// packed (sorted) horizon edge -- the two ridge vertices that are not the
/// apex. `key == u64::MAX` marks an empty table slot.
#[derive(Clone, Copy)]
struct RidgeSlot {
    key: u64,
    tet: u32,
    slot: u8,
}

const EMPTY_RIDGE: RidgeSlot = RidgeSlot { key: u64::MAX, tet: NONE, slot: 0 };

/// Scratch buffers reused across insertions.
#[derive(Default)]
struct Workspace {
    /// Visible (now dead) facet ids; doubles as the flood-fill queue.
    xlist: Vec<u32>,
    /// Facets created by the most recent insertion.
    tlast: Vec<u32>,
    /// Dead facet slots available for reuse.
    dlist: Vec<u32>,
    /// Open-addressing table pairing the unmatched ridges of new facets.
    /// Every horizon edge occurs exactly twice, so pairs are linked as soon
    /// as the second occurrence arrives; consumed entries keep their key (to
    /// preserve probe chains) and all touched slots are reset afterwards.
    rtab: Vec<RidgeSlot>,
    /// Indices of `rtab` slots written during the current insertion.
    touched: Vec<u32>,
}

fn ridge_hash(key: u64) -> usize {
    (key.wrapping_mul(0x9E3779B97F4A7C15) >> 32) as usize
}

#[cfg(test)]
fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}


fn sub4(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

fn neg4(a: [f64; 4]) -> [f64; 4] {
    [-a[0], -a[1], -a[2], -a[3]]
}

fn dot4(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn c3(pts: &[[f64; 4]], i: u32) -> Coord3D<f64> {
    let p = pts[i as usize];
    Coord3D { x: p[0], y: p[1], z: p[2] }
}

/// Canonical (unoriented) normal of the facet (v0, v1, v2, v3): the
/// generalized cross product of (v1-v0, v2-v0, v3-v0) by cofactor expansion,
/// together with the absolute-value sum of its terms (for error bounds).
///
/// Sign convention: for any point q, dot(q - v0, normal) == -D where
/// D = det[v1-v0; v2-v0; v3-v0; q-v0] (verified in `calibrate_predicates`).
fn normal4(pts: &[[f64; 4]], v: &[u32; 4]) -> ([f64; 4], [f64; 4]) {
    let a = pts[v[0] as usize];
    let u = sub4(pts[v[1] as usize], a);
    let w = sub4(pts[v[2] as usize], a);
    let x = sub4(pts[v[3] as usize], a);

    // The four 3x3 minors share the six 2x2 subdeterminants of (w, x)
    // column pairs; expanding each minor along the `u` row reuses them, and
    // the absolute-value bounds reuse the same products.
    let mut d2 = [[0.0f64; 4]; 4]; // d2[i][j] = w[i]*x[j] - w[j]*x[i]
    let mut d2a = [[0.0f64; 4]; 4]; // |w[i]*x[j]| + |w[j]*x[i]|
    for i in 0..3 {
        for j in (i + 1)..4 {
            let p = w[i] * x[j];
            let q = w[j] * x[i];
            d2[i][j] = p - q;
            d2a[i][j] = p.abs() + q.abs();
        }
    }
    let minor = |i: usize, j: usize, k: usize| {
        (
            u[i] * d2[j][k] - u[j] * d2[i][k] + u[k] * d2[i][j],
            u[i].abs() * d2a[j][k] + u[j].abs() * d2a[i][k] + u[k].abs() * d2a[i][j],
        )
    };

    let (m0, a0) = minor(1, 2, 3);
    let (m1, a1) = minor(0, 2, 3);
    let (m2, a2) = minor(0, 1, 3);
    let (m3, a3) = minor(0, 1, 2);
    ([m0, -m1, m2, -m3], [a0, a1, a2, a3])
}

/// Signed visibility of point `p` against facet `t`
/// (positive = `p` is on the outward side of the facet's hyperplane).
fn facet_dist(pts: &[[f64; 4]], t: &Tet, p: u32) -> f64 {
    let d = sub4(pts[p as usize], pts[t.v[0] as usize]);
    dot4(d, t.norm)
}

/// Exact sign of the visibility of `p` against facet `t`, consistent with the
/// exact paraboloid lift of the stored (centered) 3D coordinates.
///
/// The lifted determinant D = det[v1-v0; v2-v0; v3-v0; p-v0] (with exact
/// w = |q|^2 entries) equals Shewchuk's insphere determinant
/// insphere(v1, v2, v3, p, v0); the stored normal is -canonical when `flip`,
/// and dot(p - v0, canonical) = -D.
fn exact_dist_sign(pts: &[[f64; 4]], t: &Tet, flip: bool, p: u32) -> f64 {
    let ld = insphere(
        c3(pts, t.v[1]),
        c3(pts, t.v[2]),
        c3(pts, t.v[3]),
        c3(pts, p),
        c3(pts, t.v[0]),
    );
    if flip {
        ld
    } else {
        -ld
    }
}

/// Weak visibility: is `p` on or outside the facet's hyperplane? Used for the
/// flood fill so that points cospherical with a facet re-triangulate it.
fn visible_weak(pts: &[[f64; 4]], t: &Tet, flip: bool, p: u32) -> bool {
    let d = facet_dist(pts, t, p);
    if d.abs() > t.errb {
        d > 0.0
    } else {
        exact_dist_sign(pts, t, flip, p) >= 0.0
    }
}

/// Center the points on their centroid, lift onto the paraboloid, sort by
/// (w, x, y, z) and drop exact duplicates. Returns the lifted points and,
/// for each, the index into the caller's original array.
///
/// Generic over the input element type: f32 coordinates convert to f64
/// exactly, so all downstream arithmetic (and its guarantees) is unchanged
/// and the result is identical to converting the input up front.
fn lift_and_sort<T: Copy + Into<f64>>(points: ArrayView2<T>) -> (Vec<[f64; 4]>, Vec<u32>) {
    let n = points.nrows();
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for row in points.rows() {
        cx += row[0].into();
        cy += row[1].into();
        cz += row[2].into();
    }
    let nf = n as f64;
    cx /= nf;
    cy /= nf;
    cz /= nf;

    let mut centered: Vec<[f64; 4]> = Vec::with_capacity(n);
    // w = x^2 + y^2 + z^2 >= 0, so the IEEE bit pattern of w orders exactly
    // like the value -- sort cheap (u64, index) pairs instead of 4 floats.
    let mut order: Vec<(u64, u32)> = Vec::with_capacity(n);
    for (i, row) in points.rows().into_iter().enumerate() {
        let x = row[0].into() - cx;
        let y = row[1].into() - cy;
        let z = row[2].into() - cz;
        let w = x * x + y * y + z * z;
        centered.push([x, y, z, w]);
        order.push((w.to_bits(), i as u32));
    }
    order.sort_unstable();

    let mut lifted: Vec<([f64; 4], u32)> = order
        .into_iter()
        .map(|(_, i)| (centered[i as usize], i))
        .collect();

    // Within runs of equal w, order by (x, y, z, index) so that exact
    // duplicates are adjacent (for dedup) and the result is deterministic
    // (lowest original index survives).
    let mut s = 0;
    while s < lifted.len() {
        let mut e = s + 1;
        while e < lifted.len() && lifted[e].0[3] == lifted[s].0[3] {
            e += 1;
        }
        if e - s > 1 {
            lifted[s..e].sort_unstable_by(|a, b| {
                a.0[0]
                    .total_cmp(&b.0[0])
                    .then_with(|| a.0[1].total_cmp(&b.0[1]))
                    .then_with(|| a.0[2].total_cmp(&b.0[2]))
                    .then_with(|| a.1.cmp(&b.1))
            });
        }
        s = e;
    }
    lifted.dedup_by(|a, b| a.0 == b.0);

    let orig = lifted.iter().map(|&(_, i)| i).collect();
    let pts = lifted.into_iter().map(|(p, _)| p).collect();
    (pts, orig)
}

/// Find the first 5 points whose lifted images are affinely independent and
/// swap them to the front (of both `pts` and `orig`).
///
/// Uses incremental Gram-Schmidt on the difference vectors with a relative
/// tolerance.
fn find_seed(pts: &mut [[f64; 4]], orig: &mut [u32]) -> Result<(), DelaunayError> {
    let n = pts.len();
    let mut basis: Vec<[f64; 4]> = Vec::with_capacity(4);
    let mut chosen = 1; // pts[0] is always the first seed vertex

    for i in 1..n {
        if basis.len() == 4 {
            break;
        }
        let mut d = sub4(pts[i], pts[0]);
        let scale = dot4(d, d).sqrt();
        if scale == 0.0 {
            continue;
        }
        // Project out the existing basis directions.
        for b in &basis {
            let proj = dot4(d, *b);
            for k in 0..4 {
                d[k] -= proj * b[k];
            }
        }
        let residual = dot4(d, d).sqrt();
        if residual > 1e-12 * scale {
            for k in 0..4 {
                d[k] /= residual;
            }
            basis.push(d);
            pts.swap(chosen, i);
            orig.swap(chosen, i);
            chosen += 1;
        }
    }

    if basis.len() < 4 {
        return Err(DelaunayError::Degenerate(
            "points are coplanar or cospherical; a full-dimensional 3D \
             triangulation does not exist"
                .to_string(),
        ));
    }
    Ok(())
}

/// Build a correctly oriented facet with vertices `v`, using `anchor` (a hull
/// vertex known to be on or inside the hull side of the facet) to fix the
/// outward direction: the stored normal satisfies dist(anchor) <= 0.
///
/// If the anchor lies exactly on the facet's hyperplane the orientation is
/// copied from `ref_norm` (the stored normal of a co-hyperplanar neighboring
/// facet); with no reference the input is reported as degenerate.
fn make_facet(
    pts: &[[f64; 4]],
    mabs2: [f64; 4],
    v: [u32; 4],
    anchor: u32,
    ref_norm: Option<[f64; 4]>,
) -> Result<([f64; 4], f64, bool), DelaunayError> {
    let (canon, nabs) = normal4(pts, &v);
    let errb = ERRB_COEFF * dot4(mabs2, nabs);

    let dq = dot4(sub4(pts[anchor as usize], pts[v[0] as usize]), canon);
    let flip = if dq.abs() > errb {
        dq > 0.0
    } else {
        // Exact: dq's true sign is -sign(D) with D the lifted determinant.
        let ld = insphere(
            c3(pts, v[1]),
            c3(pts, v[2]),
            c3(pts, v[3]),
            c3(pts, anchor),
            c3(pts, v[0]),
        );
        if ld != 0.0 {
            ld < 0.0
        } else {
            // Anchor is exactly cospherical: the facet is co-hyperplanar with
            // its neighbor; both must face the same side.
            match ref_norm {
                Some(r) => dot4(canon, r) < 0.0,
                None => {
                    return Err(DelaunayError::Degenerate(
                        "seed simplex is degenerate".to_string(),
                    ))
                }
            }
        }
    };

    let norm = if flip { neg4(canon) } else { canon };
    Ok((norm, errb, flip))
}

/// Build the initial hull: the 5-cell (4-simplex) of the seed points 0..5.
///
/// Facet `i` consists of the seed vertices without vertex `i`; its neighbor
/// across the ridge opposite seed vertex `j` is facet `j`, and vertex `i`
/// itself anchors the outward orientation.
fn init_hull4d(
    pts: &[[f64; 4]],
    mabs2: [f64; 4],
    hull: &mut Vec<Tet>,
    flags: &mut Vec<u8>,
) -> Result<(), DelaunayError> {
    for i in 0..5u32 {
        let mut v = [0u32; 4];
        let mut n = [0u32; 4];
        let mut slot = 0;
        for j in 0..5u32 {
            if j != i {
                v[slot] = j;
                n[slot] = j;
                slot += 1;
            }
        }
        let (norm, errb, flip) = make_facet(pts, mabs2, v, i, None)?;
        hull.push(Tet { v, n, norm, errb });
        flags.push(F_LIVE | if flip { F_FLIP } else { 0 });
    }
    Ok(())
}

/// Find a facet visible from point `p`. Checks the most recently created
/// facets first (the sorted insertion order makes a hit there very likely),
/// then falls back to a full scan.
///
/// Strictly visible facets are preferred; a facet whose hyperplane contains
/// `p` exactly (cospherical) is only returned if nothing is strictly visible.
/// `None` means `p` is not outside any facet (a residual duplicate).
fn find_visible(
    pts: &[[f64; 4]],
    hull: &[Tet],
    flags: &[u8],
    ws: &Workspace,
    p: u32,
) -> Option<u32> {
    // Fast path: strict float visibility only, no exact fallback.
    for &h in ws.tlast.iter().rev() {
        let t = &hull[h as usize];
        if flags[h as usize] & F_STATE == F_LIVE && facet_dist(pts, t, p) > t.errb {
            return Some(h);
        }
    }

    let mut weak = None;
    for (h, t) in hull.iter().enumerate().rev() {
        if flags[h] & F_STATE != F_LIVE {
            continue;
        }
        let d = facet_dist(pts, t, p);
        if d > t.errb {
            return Some(h as u32);
        }
        if d.abs() <= t.errb {
            let s = exact_dist_sign(pts, t, flags[h] & F_FLIP != 0, p);
            if s > 0.0 {
                return Some(h as u32);
            }
            if s == 0.0 && weak.is_none() {
                weak = Some(h as u32);
            }
        }
    }
    weak
}

/// Insert point `p` into the hull: flood-fill the (weakly) visible region,
/// replace it with a fan of new facets from `p` over the horizon ridges, and
/// patch the new facets' mutual adjacencies.
fn insert_point(
    pts: &[[f64; 4]],
    mabs2: [f64; 4],
    hull: &mut Vec<Tet>,
    flags: &mut Vec<u8>,
    ws: &mut Workspace,
    p: u32,
    start: u32,
) -> Result<(), DelaunayError> {
    // Phase A: collect the visible region (a connected set of facets).
    ws.xlist.clear();
    ws.xlist.push(start);
    flags[start as usize] = F_DEAD;
    let mut i = 0;
    while i < ws.xlist.len() {
        let xid = ws.xlist[i];
        let neighbors = hull[xid as usize].n;
        for m in neighbors {
            let f = flags[m as usize];
            if f & F_STATE == F_LIVE
                && visible_weak(pts, &hull[m as usize], f & F_FLIP != 0, p)
            {
                flags[m as usize] = F_DEAD;
                ws.xlist.push(m);
            }
        }
        i += 1;
    }

    // Size the ridge-pairing table: each dead facet has at most 4 horizon
    // ridges, each spawning a facet with 3 unmatched ridges; keep the load
    // factor at or below 0.5. The table only ever grows.
    let cap_needed = (24 * ws.xlist.len()).next_power_of_two();
    if ws.rtab.len() < cap_needed {
        ws.rtab = vec![EMPTY_RIDGE; cap_needed];
    }
    let mask = ws.rtab.len() - 1;

    // Phase B: spawn a new facet over every horizon ridge (a ridge between a
    // dead facet and a live one). The unmatched ridges of new facets all
    // contain `p`, so they are keyed by their horizon edge; every horizon
    // edge belongs to exactly two new facets (the horizon is a closed
    // triangulated 2-manifold), and the pairs are linked as soon as the
    // second one shows up.
    ws.tlast.clear();
    ws.touched.clear();
    for xi in 0..ws.xlist.len() {
        let xid = ws.xlist[xi];
        let xv = hull[xid as usize].v;
        let xn = hull[xid as usize].n;
        for i in 0..4 {
            let m = xn[i];
            if flags[m as usize] & F_STATE == F_DEAD {
                continue; // internal ridge of the visible region
            }

            // Horizon ridge: the vertices of xid without xv[i].
            let mut r = [0u32; 3];
            let mut slot = 0;
            for (j, &vj) in xv.iter().enumerate() {
                if j != i {
                    r[slot] = vj;
                    slot += 1;
                }
            }

            // The outward orientation of the new facet is anchored on the
            // vertex of the live neighbor `m` opposite the shared ridge: it
            // lies on the hull side of the new facet.
            let mt = hull[m as usize];
            let anchor = match mt.v.iter().find(|&&mv| !r.contains(&mv)) {
                Some(&q) => q,
                None => {
                    return Err(DelaunayError::Corrupt(
                        "neighbor facet shares all four vertices".to_string(),
                    ))
                }
            };

            let v = [p, r[0], r[1], r[2]];
            let (norm, errb, flip) = make_facet(pts, mabs2, v, anchor, Some(mt.norm))?;

            let tet = Tet { v, n: [m, NONE, NONE, NONE], norm, errb };
            let fl = F_NEW | if flip { F_FLIP } else { 0 };
            let id = match ws.dlist.pop() {
                Some(slot) => {
                    hull[slot as usize] = tet;
                    flags[slot as usize] = fl;
                    slot
                }
                None => {
                    hull.push(tet);
                    flags.push(fl);
                    (hull.len() - 1) as u32
                }
            };

            // Point the live neighbor's slot (which referenced the dead
            // facet) at the new facet.
            let mn = &mut hull[m as usize].n;
            match mn.iter().position(|&x| x == xid) {
                Some(s) => mn[s] = id,
                None => {
                    return Err(DelaunayError::Corrupt(
                        "horizon neighbor does not link back to visible facet".to_string(),
                    ))
                }
            }

            // The three unmatched ridges of the new facet: ridge opposite
            // v[s] (s in 1..4) omits r[s-1], so its horizon edge is the
            // other two ridge vertices.
            for s in 1..4u8 {
                let (e0, e1) = match s {
                    1 => (r[1], r[2]),
                    2 => (r[0], r[2]),
                    _ => (r[0], r[1]),
                };
                let key = ((e0.min(e1) as u64) << 32) | e0.max(e1) as u64;

                let mut idx = ridge_hash(key) & mask;
                loop {
                    let entry = ws.rtab[idx];
                    if entry.key == key {
                        if entry.tet == NONE {
                            return Err(DelaunayError::Corrupt(
                                "horizon edge shared by more than two facets".to_string(),
                            ));
                        }
                        hull[entry.tet as usize].n[entry.slot as usize] = id;
                        hull[id as usize].n[s as usize] = entry.tet;
                        ws.rtab[idx].tet = NONE; // consumed; key kept for probing
                        break;
                    }
                    if entry.key == u64::MAX {
                        ws.rtab[idx] = RidgeSlot { key, tet: id, slot: s };
                        ws.touched.push(idx as u32);
                        break;
                    }
                    idx = (idx + 1) & mask;
                }
            }
            ws.tlast.push(id);
        }
    }

    if ws.tlast.is_empty() {
        return Err(DelaunayError::Corrupt(
            "no horizon ridges found for inserted point".to_string(),
        ));
    }

    // Phase C: every entry must have been consumed by its partner; reset the
    // table for the next insertion.
    let mut unmatched = false;
    for &ti in &ws.touched {
        unmatched |= ws.rtab[ti as usize].tet != NONE;
        ws.rtab[ti as usize] = EMPTY_RIDGE;
    }
    if unmatched {
        return Err(DelaunayError::Corrupt(
            "unmatched horizon ridge (visible region is not a topological ball)".to_string(),
        ));
    }

    for &id in &ws.tlast {
        flags[id as usize] = (flags[id as usize] & F_FLIP) | F_LIVE;
    }
    // Only now may the dead slots be reused: during Phase B they were still
    // being read for horizon ridges.
    ws.dlist.extend_from_slice(&ws.xlist);

    Ok(())
}

/// Verify mutual adjacency of all live facets (debug builds only).
#[cfg(debug_assertions)]
fn check_hull(hull: &[Tet], flags: &[u8]) -> Result<(), DelaunayError> {
    for (h, t) in hull.iter().enumerate() {
        if flags[h] & F_STATE == F_DEAD {
            continue;
        }
        for &m in &t.n {
            if m == NONE || flags[m as usize] & F_STATE == F_DEAD {
                return Err(DelaunayError::Corrupt(format!(
                    "facet {} has dead or missing neighbor",
                    h
                )));
            }
            if !hull[m as usize].n.contains(&(h as u32)) {
                return Err(DelaunayError::Corrupt(format!(
                    "facets {} and {} are not mutually linked",
                    h, m
                )));
            }
        }
    }
    Ok(())
}

/// Extract the lower-hull facets (the Delaunay tetrahedra), remap vertex ids
/// to the caller's original point order and orient them positively.
///
/// The lower-hull test and the orientation are decided with the exact
/// `orient3d` predicate: the canonical normal's w component is
/// -det[v1-v0; v2-v0; v3-v0] (3D rows), so the stored normal points downward
/// iff (flip XOR the 3D orientation is negative)... concretely: keep the
/// facet iff `flip ? o < 0 : o > 0` where o = det[v1-v0; v2-v0; v3-v0],
/// and o == 0 (a vertical facet, i.e. a zero-volume sliver) is dropped.
fn compact_output(
    pts: &[[f64; 4]],
    hull: &[Tet],
    flags: &[u8],
    orig: &[u32],
) -> Vec<[u32; 4]> {
    let mut tets = Vec::new();
    for (h, t) in hull.iter().enumerate() {
        if flags[h] & F_STATE == F_DEAD {
            continue;
        }
        let flip = flags[h] & F_FLIP != 0;
        // Cheap float pre-filter: clearly upward-facing facets (the vast
        // majority of rejections) need no exact test.
        if t.norm[3] > t.errb {
            continue;
        }
        // Exact 3D orientation o = det[v1-v0; v2-v0; v3-v0] =
        // -orient3d(v0, v1, v2, v3) (Shewchuk convention; verified in
        // `calibrate_predicates`).
        let o = -orient3d(c3(pts, t.v[0]), c3(pts, t.v[1]), c3(pts, t.v[2]), c3(pts, t.v[3]));
        let lower = if flip { o < 0.0 } else { o > 0.0 };
        if !lower {
            continue; // upper hull facet or degenerate (o == 0) sliver
        }
        let mut v = t.v;
        if o < 0.0 {
            v.swap(0, 1); // make det[v1-v0; v2-v0; v3-v0] positive
        }
        tets.push([
            orig[v[0] as usize],
            orig[v[1] as usize],
            orig[v[2] as usize],
            orig[v[3] as usize],
        ]);
    }
    tets
}

/// Compute the 3D Delaunay tetrahedralization of `points` (an (n, 3) array
/// of f64 or f32 coordinates; f32 widens to f64 exactly, so the result is
/// identical to converting the input up front).
///
/// Returns tetrahedra as indices into the caller's point array, positively
/// oriented. Exact duplicate points are dropped (their indices never appear
/// in the output).
pub fn delaunay4d<T: Copy + Into<f64>>(
    points: ArrayView2<T>,
) -> Result<Vec<[u32; 4]>, DelaunayError> {
    assert_eq!(points.ncols(), 3, "input points must be 3D");
    if points.nrows() >= NONE as usize {
        return Err(DelaunayError::Degenerate(format!(
            "too many points ({}), maximum is {}",
            points.nrows(),
            NONE - 1
        )));
    }

    let (mut pts, mut orig) = lift_and_sort(points);
    if pts.len() < 5 {
        return Err(DelaunayError::TooFewPoints(pts.len()));
    }
    find_seed(&mut pts, &mut orig)?;

    // Componentwise magnitude bound of the cloud, doubled: |p_k| + |a_k| <=
    // mabs2[k] for any two points, used in the per-facet error bounds.
    let mut mabs2 = [0.0f64; 4];
    for p in &pts {
        for k in 0..4 {
            mabs2[k] = mabs2[k].max(2.0 * p[k].abs());
        }
    }

    // A 3D Delaunay of n random points has ~6.7n tetrahedra and dead slots
    // are reused, so 7n slots avoids reallocation in the common case.
    let mut hull: Vec<Tet> = Vec::with_capacity(pts.len() * 7);
    let mut flags: Vec<u8> = Vec::with_capacity(pts.len() * 7);
    init_hull4d(&pts, mabs2, &mut hull, &mut flags)?;
    let mut ws = Workspace::default();

    // Safety net: a healthy run reuses dead slots and stays well below this.
    let slot_limit = 1000 + 60 * pts.len();

    for p in 5..pts.len() as u32 {
        if hull.len() > slot_limit {
            return Err(DelaunayError::Corrupt(
                "facet count exploded (numerically adversarial input?)".to_string(),
            ));
        }
        match find_visible(&pts, &hull, &flags, &ws, p) {
            Some(h) => insert_point(&pts, mabs2, &mut hull, &mut flags, &mut ws, p, h)?,
            None => continue, // residual duplicate; skip
        }

        #[cfg(debug_assertions)]
        check_hull(&hull, &flags)?;
    }

    Ok(compact_output(&pts, &hull, &flags, &orig))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn to_array(points: &[[f64; 3]]) -> Array2<f64> {
        Array2::from_shape_vec(
            (points.len(), 3),
            points.iter().flatten().copied().collect(),
        )
        .unwrap()
    }

    /// Simple deterministic pseudo-random points in [0, 1)^3.
    fn pseudo_random_points(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n).map(|_| [next(), next(), next()]).collect()
    }

    /// Plain float evaluation of the lifted determinant
    /// D = det[b-a; c-a; d-a; e-a] (rows are 4D lifted differences).
    fn float_liftdet(pts: &[[f64; 4]], a: u32, b: u32, c: u32, d: u32, e: u32) -> f64 {
        let pa = pts[a as usize];
        let rows: Vec<[f64; 4]> = [b, c, d, e]
            .iter()
            .map(|&i| sub4(pts[i as usize], pa))
            .collect();
        let mut det = 0.0;
        for col in 0..4 {
            let mut sub = [[0.0; 3]; 3];
            for (ri, row) in rows[..3].iter().enumerate() {
                let mut ci = 0;
                for k in 0..4 {
                    if k != col {
                        sub[ri][ci] = row[k];
                        ci += 1;
                    }
                }
            }
            let sign = if col % 2 == 0 { -1.0 } else { 1.0 }; // expansion along last row
            det += sign * rows[3][col] * det3(sub);
        }
        det
    }

    /// Verify the sign conventions this module relies on:
    /// 1. dot(e - a, canonical_normal(a,b,c,d)) == -D
    /// 2. robust::insphere(b, c, d, e, a) == D (as exact evaluation)
    /// 3. det[b-a; c-a; d-a] (3D) == -robust::orient3d(a, b, c, d)
    #[test]
    fn calibrate_predicates() {
        let raw = pseudo_random_points(60, 12345);
        let pts: Vec<[f64; 4]> = raw
            .iter()
            .map(|p| [p[0], p[1], p[2], p[0] * p[0] + p[1] * p[1] + p[2] * p[2]])
            .collect();

        for i in 0..10u32 {
            let (a, b, c, d, e) = (i, i + 7, i + 19, i + 31, i + 43);
            let fd = float_liftdet(&pts, a, b, c, d, e);
            assert!(fd.abs() > 1e-12, "test configuration too degenerate");

            // (1) dot product vs canonical normal
            let (canon, _) = normal4(&pts, &[a, b, c, d]);
            let dq = dot4(sub4(pts[e as usize], pts[a as usize]), canon);
            assert!(
                (dq + fd).abs() <= 1e-9 * fd.abs().max(dq.abs()),
                "dot(e-a, canon) != -D: {} vs {}",
                dq,
                -fd
            );

            // (2) insphere convention
            let is = insphere(c3(&pts, b), c3(&pts, c), c3(&pts, d), c3(&pts, e), c3(&pts, a));
            assert!(
                is.signum() == fd.signum(),
                "insphere sign {} != lifted det sign {}",
                is,
                fd
            );

            // (3) orient3d convention
            let pa = pts[a as usize];
            let o_float = det3([
                [pts[b as usize][0] - pa[0], pts[b as usize][1] - pa[1], pts[b as usize][2] - pa[2]],
                [pts[c as usize][0] - pa[0], pts[c as usize][1] - pa[1], pts[c as usize][2] - pa[2]],
                [pts[d as usize][0] - pa[0], pts[d as usize][1] - pa[1], pts[d as usize][2] - pa[2]],
            ]);
            let o_exact = orient3d(c3(&pts, a), c3(&pts, b), c3(&pts, c), c3(&pts, d));
            assert!(
                o_float.signum() == -o_exact.signum(),
                "orient3d convention mismatch: float {} vs robust {}",
                o_float,
                o_exact
            );
        }
    }

    #[test]
    fn normal4_is_orthogonal_to_facet() {
        let pts = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 1.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, 0.7],
        ];
        let (n, nabs) = normal4(&pts, &[0, 1, 2, 3]);
        for i in 1..4 {
            let d = sub4(pts[i], pts[0]);
            assert!(dot4(n, d).abs() < 1e-12, "normal not orthogonal to edge {}", i);
        }
        for k in 0..4 {
            assert!(nabs[k] >= n[k].abs());
        }
    }

    #[test]
    fn five_cell_invariants() {
        let pts = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.25, 0.25, 0.25, 0.1875],
        ];
        let mabs2 = [2.0, 2.0, 2.0, 2.0];
        let mut hull = Vec::new();
        let mut flags = Vec::new();
        init_hull4d(&pts, mabs2, &mut hull, &mut flags).unwrap();
        assert_eq!(hull.len(), 5);
        // Mutual adjacency; outward normals: the omitted seed vertex must be
        // strictly on the negative side of each facet.
        for (h, t) in hull.iter().enumerate() {
            for &m in &t.n {
                assert!(hull[m as usize].n.contains(&(h as u32)));
            }
            let d = facet_dist(&pts, t, h as u32);
            assert!(d < 0.0, "facet {} does not face away from its seed vertex", h);
        }
    }

    /// Every output tetrahedron must have an empty circumsphere.
    fn assert_delaunay(points: &[[f64; 3]], tets: &[[u32; 4]]) {
        for tet in tets {
            let [a, b, c, d] = [
                points[tet[0] as usize],
                points[tet[1] as usize],
                points[tet[2] as usize],
                points[tet[3] as usize],
            ];
            // Circumcenter from the linear system 2(v - a) . x = |v|^2 - |a|^2.
            let rows = [b, c, d].map(|v| {
                [
                    2.0 * (v[0] - a[0]),
                    2.0 * (v[1] - a[1]),
                    2.0 * (v[2] - a[2]),
                    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
                        - (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]),
                ]
            });
            let m = [
                [rows[0][0], rows[0][1], rows[0][2]],
                [rows[1][0], rows[1][1], rows[1][2]],
                [rows[2][0], rows[2][1], rows[2][2]],
            ];
            let det = det3(m);
            assert!(det.abs() > 0.0, "degenerate tet in output");
            let solve_col = |col: usize| {
                let mut mm = m;
                for r in 0..3 {
                    mm[r][col] = rows[r][3];
                }
                det3(mm) / det
            };
            let center = [solve_col(0), solve_col(1), solve_col(2)];
            let r2 = (0..3).map(|k| (a[k] - center[k]).powi(2)).sum::<f64>();

            for (i, q) in points.iter().enumerate() {
                if tet.contains(&(i as u32)) {
                    continue;
                }
                let d2 = (0..3).map(|k| (q[k] - center[k]).powi(2)).sum::<f64>();
                assert!(
                    d2 > r2 * (1.0 - 1e-9),
                    "point {} is inside the circumsphere of tet {:?}",
                    i,
                    tet
                );
            }
        }
    }

    fn tet_volume(points: &[[f64; 3]], tet: &[u32; 4]) -> f64 {
        let idx = tet.map(|i| i as usize);
        let a = points[idx[0]];
        det3([
            [
                points[idx[1]][0] - a[0],
                points[idx[1]][1] - a[1],
                points[idx[1]][2] - a[2],
            ],
            [
                points[idx[2]][0] - a[0],
                points[idx[2]][1] - a[1],
                points[idx[2]][2] - a[2],
            ],
            [
                points[idx[3]][0] - a[0],
                points[idx[3]][1] - a[1],
                points[idx[3]][2] - a[2],
            ],
        ]) / 6.0
    }

    #[test]
    fn simple_five_point_delaunay() {
        // Four corners of a tetrahedron plus an interior point: the interior
        // point splits the tet into 4.
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.2, 0.2],
        ];
        let tets = delaunay4d(to_array(&points).view()).unwrap();
        assert_eq!(tets.len(), 4);
        assert_delaunay(&points, &tets);
        // Every tet must contain the interior point (index 4).
        for tet in &tets {
            assert!(tet.contains(&4));
        }
    }

    #[test]
    fn random_points_are_delaunay() {
        for seed in [1, 2, 3] {
            let points = pseudo_random_points(80, seed);
            let tets = delaunay4d(to_array(&points).view()).unwrap();
            assert!(!tets.is_empty());
            assert_delaunay(&points, &tets);
            for tet in &tets {
                assert!(
                    tet_volume(&points, tet) > 0.0,
                    "tet {:?} is not positively oriented",
                    tet
                );
            }
        }
    }

    #[test]
    fn duplicates_are_dropped() {
        let mut points = pseudo_random_points(40, 7);
        points.push(points[0]);
        points.push(points[10]);
        let tets = delaunay4d(to_array(&points).view()).unwrap();
        assert_delaunay(&points[..40], &tets);
        for tet in &tets {
            assert!(!tet.contains(&40) && !tet.contains(&41));
        }
    }

    #[test]
    fn coplanar_input_errors() {
        let points: Vec<[f64; 3]> = pseudo_random_points(30, 5)
            .into_iter()
            .map(|p| [p[0], p[1], 0.0])
            .collect();
        match delaunay4d(to_array(&points).view()) {
            Err(DelaunayError::Degenerate(_)) => {}
            other => panic!("expected Degenerate error, got {:?}", other.map(|t| t.len())),
        }
    }

    #[test]
    fn too_few_points_errors() {
        let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        match delaunay4d(to_array(&points).view()) {
            Err(DelaunayError::TooFewPoints(3)) => {}
            other => panic!("expected TooFewPoints, got {:?}", other.map(|t| t.len())),
        }
    }

    #[test]
    fn grid_points_triangulate() {
        // 4x4x4 grid: heavily cospherical, exercises the exact predicates.
        let mut points = Vec::new();
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    points.push([x as f64, y as f64, z as f64]);
                }
            }
        }
        let tets = delaunay4d(to_array(&points).view()).unwrap();
        // The union of tets must tile the cube: total volume = 27; and no
        // inverted or degenerate tets.
        let mut total = 0.0;
        for tet in &tets {
            let vol = tet_volume(&points, tet);
            assert!(vol > 0.0, "non-positive tet {:?}", tet);
            total += vol;
        }
        assert!((total - 27.0).abs() < 1e-9, "tet volumes sum to {}, not 27", total);
    }

    #[test]
    fn f32_input_matches_f64() {
        // f32 coordinates widen to f64 exactly, so both paths must produce
        // identical tetrahedra.
        let points = pseudo_random_points(60, 9);
        let pts32: Vec<f32> = points.iter().flatten().map(|&v| v as f32).collect();
        let arr32 = Array2::from_shape_vec((60, 3), pts32).unwrap();
        let arr64 = arr32.mapv(|v| v as f64);
        let t32 = delaunay4d(arr32.view()).unwrap();
        let t64 = delaunay4d(arr64.view()).unwrap();
        assert!(!t32.is_empty());
        assert_eq!(t32, t64);
    }

    #[test]
    fn tight_clusters_terminate() {
        // Two clusters of scale 1e-6, 1000 apart: extreme cancellation. This
        // input corrupts and explodes a pure-float implementation.
        let mut state = 0x243F6A8885A308D3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut points = Vec::new();
        for i in 0..400 {
            let off = if i < 200 { 0.0 } else { 1000.0 };
            points.push([
                next() * 1e-6 + off,
                next() * 1e-6 + off,
                next() * 1e-6 + off,
            ]);
        }
        let tets = delaunay4d(to_array(&points).view()).unwrap();
        assert!(!tets.is_empty());
        // All 400 points must appear as vertices.
        let mut seen = vec![false; 400];
        for tet in &tets {
            for &v in tet {
                seen[v as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "some points missing from output");
        // The naive float determinant cannot resolve the ~1e-25 sliver
        // volumes this input produces; verify positive orientation exactly on
        // the centered coordinates (the cloud the algorithm triangulates).
        let n = points.len() as f64;
        let mut c = [0.0f64; 3];
        for p in &points {
            for k in 0..3 {
                c[k] += p[k];
            }
        }
        for k in 0..3 {
            c[k] /= n;
        }
        let coord = |i: u32| {
            let p = points[i as usize];
            Coord3D { x: p[0] - c[0], y: p[1] - c[1], z: p[2] - c[2] }
        };
        for tet in &tets {
            let o = -orient3d(coord(tet[0]), coord(tet[1]), coord(tet[2]), coord(tet[3]));
            assert!(o > 0.0, "tet {:?} is not positively oriented", tet);
        }
    }
}
