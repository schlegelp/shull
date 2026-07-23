//! The dimension-generic parallel pipeline: per-block triangulation with
//! certification, the crust pass for the uncertified remainder, and the
//! checked merge.
//!
//! # Why the result is correct (or falls back)
//!
//! Every emitted simplex is justified by one of two arguments:
//!
//! * **Certified star** (block pass): if an owned point `p` is not on its
//!   block's local convex hull and every simplex of its local star has a
//!   circumball covered by the gathered cells (with the conservative
//!   [`Ball::r_search`] margin), then no point outside the gathered subset
//!   can invalidate any of those simplices — `p`'s local star *is* its star
//!   in the global Delaunay triangulation. Emitting every local simplex with
//!   at least one certified vertex therefore emits only global simplices,
//!   and emits *all* global simplices that have at least one certified
//!   vertex.
//! * **Globally verified** (crust pass): the remaining (uncertified) points
//!   are re-triangulated together in one kernel run; a global Delaunay
//!   simplex whose vertices are all uncertified has an empty circumball, so
//!   it appears in this triangulation too regardless of context. Each
//!   candidate is then verified against the *entire* cloud via the grid,
//!   with exact predicates — candidates that fail are simply discarded
//!   (they were artifacts of the subset), candidates that hit an exactly
//!   cospherical point trigger the fallback.
//!
//! In general position the union of the two is exactly the global
//! triangulation. Degenerate inputs (exactly cocircular/cospherical sets)
//! split across sources are caught by the merge checks — every face shared
//! by simplices from *different* sources must be strictly locally Delaunay
//! under exact predicates — plus structural checks (face counts, boundary
//! closure and connectivity, every point present). Any violation abandons
//! the parallel result and reruns the sequential build on the original
//! input, so the parallel path never returns a wrong mesh.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use std::ops::ControlFlow::{self, Break, Continue};

use super::geom::{box_dist2, Ball};
use super::grid::{Grid, SUPER};
use super::{FallbackReason, ParProgress};
use crate::error::DelaunayError;

/// Dimension-specific plumbing: the sequential kernel and the exact/float
/// geometry helpers for its simplex type.
pub(crate) struct Ops<const D: usize, const K: usize> {
    pub kernel: fn(
        ArrayView2<f64>,
    )
        -> Result<(Vec<[u32; K]>, Vec<[i32; K]>, Vec<[u32; 2]>), DelaunayError>,
    pub circumball: fn(&[[f64; D]; K]) -> Option<Ball<D>>,
    /// Exact: positive iff the query point is strictly inside the simplex's
    /// circumball, zero iff exactly on it.
    pub inball_exact: fn(&[[f64; D]; K], [f64; D]) -> f64,
    /// Exact: positive iff the simplex is positively oriented (the kernels'
    /// output convention).
    pub orient_exact: fn(&[[f64; D]; K]) -> f64,
}

/// A ball whose axis-aligned bounding box spans more than this many grid
/// cells is treated as uncoverable (certification) — such simplices belong
/// to the crust anyway.
const MAX_COVER_CELLS: usize = 512;

/// Cap on the nonempty-region cells descended into when globally verifying
/// one crust simplex; beyond this the build falls back. The coarse occupancy
/// level skips empty space, so even the enormous circumballs of hull
/// simplices (which bulge almost entirely into empty space) stay far below
/// this.
const MAX_VERIFY_CELLS: usize = 65_536;

/// Fraction of points allowed to miss certification before the parallel
/// attempt is abandoned as not worth finishing (degenerate/adversarial
/// cloud).
const CRUST_FRACTION_LIMIT: f64 = 0.25;

const CRUST_SOURCE: u32 = u32::MAX;

#[derive(Default)]
pub(crate) struct PipelineStats {
    pub crust_core: usize,
    pub crust_context: usize,
    pub n_block_emitted: usize,
    pub n_crust_emitted: usize,
    pub t_blocks: f64,
    pub t_crust: f64,
    pub t_merge: f64,
}

struct BlockOut<const K: usize> {
    /// Simplices with >= 1 certified owned vertex, as grid dedup ids.
    emitted: Vec<[u32; K]>,
    /// Owned points that failed certification (dedup ids).
    uncertified: Vec<u32>,
    /// All vertices of local simplices incident to an uncertified owned
    /// point: context handed to the crust triangulation.
    context: Vec<u32>,
}

/// Visit every cell in the Chebyshev-1 neighborhood of `coords` (including
/// `coords` itself), clamped to the grid.
fn for_each_neighbor<const D: usize>(
    grid: &Grid<D>,
    coords: [usize; D],
    mut f: impl FnMut([usize; D]),
) {
    let mut lo = [0usize; D];
    let mut hi = [0usize; D];
    for k in 0..D {
        lo[k] = coords[k].saturating_sub(1);
        hi[k] = (coords[k] + 1).min(grid.dims[k] - 1);
    }
    let mut cur = lo;
    loop {
        f(cur);
        let mut k = 0;
        loop {
            if k == D {
                return;
            }
            if cur[k] < hi[k] {
                cur[k] += 1;
                break;
            }
            cur[k] = lo[k];
            k += 1;
        }
    }
}

/// Cell-coordinate range of the ball's bounding box (clamped to the grid),
/// or `None` if it spans more than `max_cells` cells.
fn ball_cell_range<const D: usize>(
    grid: &Grid<D>,
    ball: &Ball<D>,
    max_cells: usize,
) -> Option<([usize; D], [usize; D])> {
    let mut lo = [0usize; D];
    let mut hi = [0usize; D];
    let mut cells = 1usize;
    for k in 0..D {
        lo[k] = grid.axis_cell(k, ball.c[k] - ball.r_search);
        hi[k] = grid.axis_cell(k, ball.c[k] + ball.r_search);
        cells = cells.saturating_mul(hi[k] - lo[k] + 1);
        if cells > max_cells {
            return None;
        }
    }
    Some((lo, hi))
}

/// Visit every coordinate of the D-dimensional integer box `[lo, hi]`
/// (inclusive) until `f` breaks.
fn walk_box<const D: usize, B>(
    lo: [usize; D],
    hi: [usize; D],
    mut f: impl FnMut([usize; D]) -> ControlFlow<B>,
) -> Option<B> {
    let mut cur = lo;
    loop {
        if let Break(b) = f(cur) {
            return Some(b);
        }
        let mut k = 0;
        loop {
            if k == D {
                return None;
            }
            if cur[k] < hi[k] {
                cur[k] += 1;
                break;
            }
            cur[k] = lo[k];
            k += 1;
        }
    }
}

enum Verify {
    /// No cloud point strictly inside: a global Delaunay simplex.
    Keep,
    /// A point is strictly inside: an artifact of the crust subset.
    Discard,
    /// A point lies exactly on the ball: genuine degeneracy.
    Cospherical,
    OverBudget,
}

/// Check the simplex's circumball against the entire cloud: enumerate
/// candidate points via the grid (using the coarse occupancy level to skip
/// empty space) and decide each candidate with the exact predicate.
fn verify_ball<const D: usize, const K: usize>(
    grid: &Grid<D>,
    ops: &Ops<D, K>,
    ps: &[[f64; D]; K],
    global: &[u32; K],
    ball: &Ball<D>,
) -> Verify {
    let r2 = ball.r_search * ball.r_search;
    let mut clo = [0usize; D];
    let mut chi = [0usize; D];
    let mut slo = [0usize; D];
    let mut shi = [0usize; D];
    for k in 0..D {
        clo[k] = grid.axis_cell(k, ball.c[k] - ball.r_search);
        chi[k] = grid.axis_cell(k, ball.c[k] + ball.r_search);
        slo[k] = clo[k] / SUPER;
        shi[k] = chi[k] / SUPER;
    }
    let mut budget = MAX_VERIFY_CELLS;
    let outcome = walk_box(slo, shi, |sc| {
        if !grid.super_occupied[grid.super_id(sc)] {
            return Continue(());
        }
        let mut blo = [0.0; D];
        let mut bhi = [0.0; D];
        for k in 0..D {
            blo[k] = grid.lo[k] + (sc[k] * SUPER) as f64 * grid.cell_size[k];
            bhi[k] = blo[k] + SUPER as f64 * grid.cell_size[k];
        }
        if box_dist2(&ball.c, &blo, &bhi) >= r2 {
            return Continue(());
        }
        // Descend: cells of this super-cell within the ball's AABB.
        let mut ilo = [0usize; D];
        let mut ihi = [0usize; D];
        for k in 0..D {
            ilo[k] = clo[k].max(sc[k] * SUPER);
            ihi[k] = chi[k].min(sc[k] * SUPER + SUPER - 1);
        }
        let inner = walk_box(ilo, ihi, |cc| {
            if budget == 0 {
                return Break(Verify::OverBudget);
            }
            budget -= 1;
            let (cs, ce) = grid.cell_range[grid.cell_id(cc)];
            if cs == ce {
                return Continue(());
            }
            let bl = grid.cell_box_lo(cc);
            let mut bh = bl;
            for k in 0..D {
                bh[k] += grid.cell_size[k];
            }
            if box_dist2(&ball.c, &bl, &bh) >= r2 {
                return Continue(());
            }
            for q in cs..ce {
                if global.contains(&q) {
                    continue;
                }
                let qp = grid.pts[q as usize];
                // Candidates beyond r_search cannot be inside the true ball.
                let mut d2 = 0.0;
                for k in 0..D {
                    let d = qp[k] - ball.c[k];
                    d2 += d * d;
                }
                if d2 >= r2 {
                    continue;
                }
                let sign = (ops.inball_exact)(ps, qp);
                if sign > 0.0 {
                    return Break(Verify::Discard);
                }
                if sign == 0.0 {
                    return Break(Verify::Cospherical);
                }
            }
            Continue(())
        });
        match inner {
            Some(v) => Break(v),
            None => Continue(()),
        }
    });
    outcome.unwrap_or(Verify::Keep)
}

/// Is the (r_search-inflated) ball covered by cells gathered by `block`?
/// Cells without points are always covered — no point of the cloud lives
/// there — as is all space outside the grid.
fn ball_covered<const D: usize>(grid: &Grid<D>, block: u32, ball: &Ball<D>) -> bool {
    let Some((lo, hi)) = ball_cell_range(grid, ball, MAX_COVER_CELLS) else {
        return false;
    };
    let r2 = ball.r_search * ball.r_search;
    let mut covered = true;
    let mut cur = lo;
    'outer: loop {
        let id = grid.cell_id(cur);
        let (s, e) = grid.cell_range[id];
        if s != e && grid.cell_block[id] != block {
            // Nonempty cell not owned by this block: only a problem if the
            // ball actually reaches it and it was not gathered as halo.
            let blo = grid.cell_box_lo(cur);
            let mut bhi = blo;
            for k in 0..D {
                bhi[k] += grid.cell_size[k];
            }
            if box_dist2(&ball.c, &blo, &bhi) < r2 && !grid.is_gathered(block, cur) {
                covered = false;
                break 'outer;
            }
        }
        let mut k = 0;
        loop {
            if k == D {
                break 'outer;
            }
            if cur[k] < hi[k] {
                cur[k] += 1;
                break;
            }
            cur[k] = lo[k];
            k += 1;
        }
    }
    covered
}

fn gather_coords<const D: usize>(grid: &Grid<D>, ids: &[u32]) -> Array2<f64> {
    let mut arr = Array2::zeros((ids.len(), D));
    for (i, &id) in ids.iter().enumerate() {
        let p = grid.pts[id as usize];
        for k in 0..D {
            arr[[i, k]] = p[k];
        }
    }
    arr
}

fn process_block<const D: usize, const K: usize>(
    grid: &Grid<D>,
    ops: &Ops<D, K>,
    b: u32,
) -> BlockOut<K> {
    let blk = grid.blocks[b as usize];
    let own_len = (blk.pts.1 - blk.pts.0) as usize;

    // Gather owned points (locals[0..own_len]) plus the one-cell halo ring,
    // in deterministic order.
    let mut locals: Vec<u32> = (blk.pts.0..blk.pts.1).collect();
    let mut ring: Vec<u32> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for &cell in &grid.cells_morton[blk.cells.0 as usize..blk.cells.1 as usize] {
            for_each_neighbor(grid, grid.cell_coords(cell as usize), |nc| {
                let id = grid.cell_id(nc);
                let (s, e) = grid.cell_range[id];
                if s != e && grid.cell_block[id] != b && seen.insert(id) {
                    ring.push(id as u32);
                }
            });
        }
    }
    ring.sort_unstable();
    for &cell in &ring {
        let (s, e) = grid.cell_range[cell as usize];
        locals.extend(s..e);
    }

    let all_uncertified = |context: Vec<u32>| BlockOut {
        emitted: Vec::new(),
        uncertified: locals[..own_len].to_vec(),
        context,
    };

    if locals.len() < K + 1 {
        return all_uncertified(locals.clone());
    }
    let coords = gather_coords(grid, &locals);
    let Ok((simps, nbrs, dropped)) = (ops.kernel)(coords.view()) else {
        // Degenerate subset (e.g. all collinear/coplanar): let the crust
        // handle every owned point, with the full subset as context.
        return all_uncertified(locals.clone());
    };
    debug_assert!(dropped.is_empty(), "grid dedup must remove all duplicates");

    let n_loc = locals.len();
    let mut on_hull = vec![false; n_loc];
    let mut uncovered = vec![false; n_loc];
    let mut incident = vec![false; n_loc];

    for (t, nb) in nbrs.iter().enumerate() {
        for j in 0..K {
            if nb[j] < 0 {
                // The boundary face opposite vertex j: all vertices but j.
                for (m, &v) in simps[t].iter().enumerate() {
                    if m != j {
                        on_hull[v as usize] = true;
                    }
                }
            }
        }
    }

    for s in &simps {
        // Only simplices with an owned vertex can influence certification;
        // halo-only simplices need no circumball.
        if !s.iter().any(|&v| (v as usize) < own_len) {
            continue;
        }
        let mut ps = [[0.0; D]; K];
        for (i, &v) in s.iter().enumerate() {
            ps[i] = grid.pts[locals[v as usize] as usize];
            incident[v as usize] = true;
        }
        let ok = match (ops.circumball)(&ps) {
            Some(ball) => ball_covered(grid, b, &ball),
            None => false,
        };
        if !ok {
            for &v in s.iter() {
                uncovered[v as usize] = true;
            }
        }
    }

    let certified =
        |v: usize| v < own_len && incident[v] && !on_hull[v] && !uncovered[v];

    let mut emitted = Vec::new();
    let mut context = Vec::new();
    for s in &simps {
        let mut any_certified = false;
        let mut any_uncert_owned = false;
        for &v in s.iter() {
            let v = v as usize;
            if certified(v) {
                any_certified = true;
            } else if v < own_len {
                any_uncert_owned = true;
            }
        }
        if any_certified {
            emitted.push(s.map(|v| locals[v as usize]));
        }
        if any_uncert_owned {
            context.extend(s.iter().map(|&v| locals[v as usize]));
        }
    }
    let uncertified: Vec<u32> = (0..own_len)
        .filter(|&v| !certified(v))
        .map(|v| locals[v])
        .collect();

    BlockOut { emitted, uncertified, context }
}

/// Triangulate the uncertified points (plus context) in one kernel run and
/// return the globally verified simplices whose vertices are all
/// uncertified. See the module docs for why this is complete.
fn crust_pass<const D: usize, const K: usize>(
    grid: &Grid<D>,
    ops: &Ops<D, K>,
    blocks: &[BlockOut<K>],
    stats: &mut PipelineStats,
) -> Result<Vec<[u32; K]>, FallbackReason> {
    let n_d = grid.pts.len();
    let mut is_core = vec![false; n_d];
    // Blocks are ordered by their point ranges, so this core list is sorted
    // and deterministic.
    let mut crust_ids: Vec<u32> = Vec::new();
    for bo in blocks {
        for &v in &bo.uncertified {
            is_core[v as usize] = true;
            crust_ids.push(v);
        }
    }
    stats.crust_core = crust_ids.len();
    if crust_ids.is_empty() {
        return Ok(Vec::new());
    }
    if crust_ids.len() as f64 > CRUST_FRACTION_LIMIT * n_d as f64 {
        return Err(FallbackReason::CrustTooLarge);
    }

    let mut context: Vec<u32> = blocks
        .iter()
        .flat_map(|bo| bo.context.iter().copied())
        .filter(|&v| !is_core[v as usize])
        .collect();
    context.sort_unstable();
    context.dedup();
    stats.crust_context = context.len();
    crust_ids.extend_from_slice(&context);

    let core_len = crust_ids.len() - context.len();
    if crust_ids.len() < K + 1 {
        return Err(FallbackReason::CrustKernelError);
    }
    let coords = gather_coords(grid, &crust_ids);
    let Ok((simps, _, dropped)) = (ops.kernel)(coords.view()) else {
        return Err(FallbackReason::CrustKernelError);
    };
    debug_assert!(dropped.is_empty(), "grid dedup must remove all duplicates");

    // Verify each all-core candidate against the whole cloud.
    let verified: Result<Vec<Option<[u32; K]>>, FallbackReason> = simps
        .par_iter()
        .map(|s| {
            if s.iter().any(|&v| v as usize >= core_len) {
                return Ok(None); // has context vertices: block territory
            }
            let global = s.map(|v| crust_ids[v as usize]);
            let mut ps = [[0.0; D]; K];
            for (i, &v) in global.iter().enumerate() {
                ps[i] = grid.pts[v as usize];
            }
            let Some(ball) = (ops.circumball)(&ps) else {
                return Err(FallbackReason::DegenerateMerge("unverifiable crust ball"));
            };
            match verify_ball(grid, ops, &ps, &global, &ball) {
                Verify::Keep => Ok(Some(global)),
                Verify::Discard => Ok(None), // subset artifact
                Verify::Cospherical => Err(FallbackReason::DegenerateMerge(
                    "cospherical crust simplex",
                )),
                Verify::OverBudget => Err(FallbackReason::DegenerateMerge(
                    "crust verification budget exceeded",
                )),
            }
        })
        .collect();
    Ok(verified?.into_iter().flatten().collect())
}

/// One face for the matching sort, packed into 16 bytes: `a` holds the two
/// smallest sorted face vertices, the high half of `b` holds the third (or
/// u32::MAX in 2D), and the low half of `b` is `row * K + slot` — unique per
/// entry, so it never affects the grouping, which compares `(a, b >> 32)`.
type FaceEntry = (u64, u64);

fn face_entry<const K: usize>(row: &[u32; K], j: usize, r: u32) -> FaceEntry {
    let mut f = [u32::MAX; 4];
    let mut m = 0;
    for (i, &v) in row.iter().enumerate() {
        if i != j {
            f[m] = v;
            m += 1;
        }
    }
    f[..m].sort_unstable();
    (
        (f[0] as u64) << 32 | f[1] as u64,
        (f[2] as u64) << 32 | r as u64,
    )
}

fn same_face(a: &FaceEntry, b: &FaceEntry) -> bool {
    a.0 == b.0 && (a.1 >> 32) == (b.1 >> 32)
}

fn face_ref(e: &FaceEntry) -> u32 {
    e.1 as u32
}

struct Dsu(Vec<u32>);

impl Dsu {
    fn find(&mut self, mut x: u32) -> u32 {
        while self.0[x as usize] != x {
            let p = self.0[x as usize];
            self.0[x as usize] = self.0[p as usize];
            x = p;
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[ra as usize] = rb;
        }
    }
}

/// Merge block and crust emissions into the final mesh: dedup, orient,
/// face-match into the neighbor array, and run the consistency checks.
/// Returns simplices in the caller's *original* indices.
fn merge<const D: usize, const K: usize>(
    grid: &Grid<D>,
    ops: &Ops<D, K>,
    blocks: Vec<BlockOut<K>>,
    crust: Vec<[u32; K]>,
) -> Result<(Vec<[u32; K]>, Vec<[i32; K]>), FallbackReason> {
    // Canonical rows (sorted vertices) + source tag; sources are only needed
    // to decide which adjacencies require the exact cross-source check.
    let mut rows: Vec<([u32; K], u32)> = Vec::new();
    for (b, bo) in blocks.into_iter().enumerate() {
        rows.extend(bo.emitted.into_iter().map(|mut s| {
            s.sort_unstable();
            (s, b as u32)
        }));
    }
    rows.extend(crust.into_iter().map(|mut s| {
        s.sort_unstable();
        (s, CRUST_SOURCE)
    }));
    rows.par_sort_unstable();
    rows.dedup_by_key(|&mut (s, _)| s);

    if rows.is_empty() {
        return Err(FallbackReason::DegenerateMerge("no simplices emitted"));
    }

    // Restore positive orientation (kernel output convention). Exactly
    // degenerate rows cannot come from a kernel; treat as inconsistency.
    let oriented: Result<Vec<[u32; K]>, FallbackReason> = rows
        .par_iter()
        .map(|&(s, _)| {
            let mut ps = [[0.0; D]; K];
            for (i, &v) in s.iter().enumerate() {
                ps[i] = grid.pts[v as usize];
            }
            let o = (ops.orient_exact)(&ps);
            if o == 0.0 {
                return Err(FallbackReason::DegenerateMerge("degenerate simplex"));
            }
            let mut s = s;
            if o < 0.0 {
                s.swap(0, 1);
            }
            Ok(s)
        })
        .collect();
    let simplices = oriented?;
    let sources: Vec<u32> = rows.iter().map(|&(_, src)| src).collect();
    drop(rows);

    // Every deduped point must appear in the mesh.
    let words = grid.pts.len().div_ceil(64);
    let present = simplices
        .par_chunks(64.max(simplices.len() / 256))
        .fold(
            || vec![0u64; words],
            |mut bits, chunk| {
                for s in chunk {
                    for &v in s.iter() {
                        bits[v as usize / 64] |= 1u64 << (v % 64);
                    }
                }
                bits
            },
        )
        .reduce(
            || vec![0u64; words],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x |= y;
                }
                a
            },
        );
    let full = grid.pts.len() / 64;
    let tail = grid.pts.len() % 64;
    if present[..full].iter().any(|&w| w != u64::MAX)
        || (tail > 0 && present[full] != (1u64 << tail) - 1)
    {
        return Err(FallbackReason::DegenerateMerge("point missing from mesh"));
    }
    drop(present);

    // Global face matching.
    let mut faces: Vec<FaceEntry> = (0..simplices.len() as u32)
        .into_par_iter()
        .flat_map_iter(|t| {
            let row = simplices[t as usize];
            (0..K).map(move |j| face_entry(&row, j, t * K as u32 + j as u32))
        })
        .collect();
    faces.par_sort_unstable();

    // Scan the sorted runs in parallel chunks whose starts are aligned to
    // run boundaries (a run has length <= 2, so alignment walks at most one
    // entry in the common case).
    let n_faces = faces.len();
    let n_chunks = (rayon::current_num_threads() * 4).clamp(1, 256);
    let approx = n_faces.div_ceil(n_chunks).max(1);
    let mut starts: Vec<usize> = vec![0];
    for c in 1..n_chunks {
        let mut s = (c * approx).min(n_faces);
        while s < n_faces && s > 0 && same_face(&faces[s], &faces[s - 1]) {
            s += 1;
        }
        if s > *starts.last().unwrap() && s < n_faces {
            starts.push(s);
        }
    }
    starts.push(n_faces);

    struct ChunkOut {
        pairs: Vec<(u32, u32)>,
        boundary: Vec<u32>, // row * K + slot refs
        cross: Vec<(u32, u32)>,
        ok: bool,
    }
    let chunks: Vec<ChunkOut> = starts
        .par_windows(2)
        .map(|w| {
            let (lo, hi) = (w[0], w[1]);
            let mut out = ChunkOut {
                pairs: Vec::new(),
                boundary: Vec::new(),
                cross: Vec::new(),
                ok: true,
            };
            let mut i = lo;
            while i < hi {
                let mut j = i + 1;
                while j < hi && same_face(&faces[j], &faces[i]) {
                    j += 1;
                }
                match j - i {
                    1 => out.boundary.push(face_ref(&faces[i])),
                    2 => {
                        let (ra, rb) = (face_ref(&faces[i]), face_ref(&faces[i + 1]));
                        out.pairs.push((ra, rb));
                        let ta = (ra / K as u32) as usize;
                        let tb = (rb / K as u32) as usize;
                        if sources[ta] != sources[tb] {
                            out.cross.push((ra, rb));
                        }
                    }
                    _ => {
                        out.ok = false;
                        return out;
                    }
                }
                i = j;
            }
            out
        })
        .collect();
    drop(faces);

    let mut neighbors: Vec<[i32; K]> = vec![[-1; K]; simplices.len()];
    let mut boundary: Vec<u32> = Vec::new();
    let mut cross: Vec<(u32, u32)> = Vec::new();
    for c in chunks {
        if !c.ok {
            return Err(FallbackReason::DegenerateMerge(
                "face shared by more than two simplices",
            ));
        }
        for (ra, rb) in c.pairs {
            let (ta, ja) = ((ra / K as u32) as usize, (ra % K as u32) as usize);
            let (tb, jb) = ((rb / K as u32) as usize, (rb % K as u32) as usize);
            neighbors[ta][ja] = tb as i32;
            neighbors[tb][jb] = ta as i32;
        }
        boundary.extend(c.boundary);
        cross.extend(c.cross);
    }

    // Exact local-Delaunay check across every source boundary: the opposite
    // vertex of each side must be strictly outside the other side's
    // circumball. Simplices from the same source are faces of one kernel
    // triangulation and need no check.
    cross
        .par_iter()
        .try_for_each(|&(ra, rb)| {
            for (r_ball, r_q) in [(ra, rb), (rb, ra)] {
                let t = (r_ball / K as u32) as usize;
                let (tq, jq) = ((r_q / K as u32) as usize, (r_q % K as u32) as usize);
                let mut ps = [[0.0; D]; K];
                for (i, &v) in simplices[t].iter().enumerate() {
                    ps[i] = grid.pts[v as usize];
                }
                let q = grid.pts[simplices[tq][jq] as usize];
                if (ops.inball_exact)(&ps, q) >= 0.0 {
                    return Err(FallbackReason::DegenerateMerge(
                        "cross-source adjacency is not strictly Delaunay",
                    ));
                }
            }
            Ok(())
        })?;

    // The boundary must be a single closed surface: every ridge of a
    // boundary face is shared by exactly two boundary faces, and the
    // boundary faces form one connected component (a second component means
    // an internal cavity).
    {
        let mut ridge_owner: std::collections::HashMap<(u64, u64), (u32, u32)> =
            std::collections::HashMap::with_capacity(boundary.len() * (K - 1));
        let mut dsu = Dsu((0..boundary.len() as u32).collect());
        for (bi, &r) in boundary.iter().enumerate() {
            let (t, j) = ((r / K as u32) as usize, (r % K as u32) as usize);
            // The boundary face: vertices except j. Its ridges: drop one
            // more vertex.
            let mut face = [u32::MAX; 4];
            let mut m = 0;
            for (i, &v) in simplices[t].iter().enumerate() {
                if i != j {
                    face[m] = v;
                    m += 1;
                }
            }
            face[..m].sort_unstable();
            for drop_i in 0..m {
                let mut ridge = [u32::MAX; 4];
                let mut w = 0;
                for (i, &v) in face[..m].iter().enumerate() {
                    if i != drop_i {
                        ridge[w] = v;
                        w += 1;
                    }
                }
                let key = (
                    (ridge[0] as u64) << 32 | ridge[1] as u64,
                    (ridge[2] as u64) << 32 | ridge[3] as u64,
                );
                match ridge_owner.entry(key) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert((bi as u32, 1));
                    }
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        let (first, count) = *e.get();
                        if count != 1 {
                            return Err(FallbackReason::DegenerateMerge(
                                "boundary ridge shared by more than two boundary faces",
                            ));
                        }
                        e.insert((first, 2));
                        dsu.union(first, bi as u32);
                    }
                }
            }
        }
        if ridge_owner.values().any(|&(_, count)| count != 2) {
            return Err(FallbackReason::DegenerateMerge(
                "boundary ridge not shared by two boundary faces",
            ));
        }
        let root = dsu.find(0);
        for bi in 1..boundary.len() as u32 {
            if dsu.find(bi) != root {
                return Err(FallbackReason::DegenerateMerge(
                    "boundary is not a single connected surface",
                ));
            }
        }
    }

    // Map dedup ids back to the caller's original indices.
    let out: Vec<[u32; K]> = simplices
        .into_par_iter()
        .map(|s| s.map(|v| grid.orig[v as usize]))
        .collect();
    Ok((out, neighbors))
}

/// Run the full pipeline on a prepared grid. `Err` means "fall back to the
/// sequential build" — the parallel attempt is abandoned, never trusted
/// partially.
pub(crate) fn triangulate_par<const D: usize, const K: usize>(
    grid: &Grid<D>,
    ops: &Ops<D, K>,
    stats: &mut PipelineStats,
    progress: Option<&(dyn Fn(ParProgress) + Sync)>,
) -> Result<(Vec<[u32; K]>, Vec<[i32; K]>), FallbackReason> {
    let total = grid.blocks.len();
    if let Some(cb) = progress {
        cb(ParProgress::Start { n_blocks: total });
    }

    let t0 = std::time::Instant::now();
    // The counter lock also serializes the callback, so events arrive in
    // order even though blocks finish on different worker threads.
    let done = std::sync::Mutex::new(0usize);
    let blocks: Vec<BlockOut<K>> = (0..total as u32)
        .into_par_iter()
        .map(|b| {
            let out = process_block(grid, ops, b);
            if let Some(cb) = progress {
                let mut d = done.lock().unwrap();
                *d += 1;
                cb(ParProgress::Blocks { done: *d, total });
            }
            out
        })
        .collect();
    stats.n_block_emitted = blocks.iter().map(|b| b.emitted.len()).sum();
    stats.t_blocks = t0.elapsed().as_secs_f64();

    if let Some(cb) = progress {
        cb(ParProgress::Crust);
    }
    let t0 = std::time::Instant::now();
    let crust = crust_pass(grid, ops, &blocks, stats)?;
    stats.n_crust_emitted = crust.len();
    stats.t_crust = t0.elapsed().as_secs_f64();

    if let Some(cb) = progress {
        cb(ParProgress::Merge);
    }
    let t0 = std::time::Instant::now();
    let out = merge(grid, ops, blocks, crust);
    stats.t_merge = t0.elapsed().as_secs_f64();
    out
}
