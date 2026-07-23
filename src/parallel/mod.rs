//! Opt-in parallel Delaunay builds (the `parallel` cargo feature).
//!
//! [`delaunay2d_par`] and [`delaunay4d_par`] produce the same triangulation
//! as their sequential counterparts, using a partition + merge scheme: the
//! cloud is split into spatially compact blocks that are triangulated
//! concurrently (each with a one-cell halo of neighboring points) by the
//! *unchanged* sequential kernels; per-point certificates decide which
//! results are provably correct locally; the uncertain remainder is
//! re-triangulated in a "crust" pass and verified against the whole cloud
//! with exact predicates; the merge cross-checks everything and, on any
//! doubt (degenerate inputs, adversarial geometry), transparently falls
//! back to the plain sequential build. The parallel path never returns a
//! wrong mesh.
//!
//! Guarantees and caveats, relative to the sequential functions:
//!
//! * For inputs in general position the simplex *set* is identical; the
//!   *row order* of simplices (and of the duplicate report) differs and is
//!   unspecified.
//! * Exactly cocircular/cospherical point sets are triangulated with
//!   different tie-breaks than the sequential sweep order — either the
//!   whole build falls back to the sequential kernel (detected via exact
//!   predicates), or, when a degenerate pocket is resolved consistently
//!   within one block, the output may be a *valid but different* Delaunay
//!   triangulation of the same points.
//! * The output is bit-identical for any number of rayon threads (the
//!   partition depends only on the point count).
//! * Small inputs (below [`ParConfig::min_points`]) run sequentially even
//!   when the parallel entry points are called.
//!
//! Thread count is controlled by rayon's global pool (e.g. the
//! `RAYON_NUM_THREADS` environment variable).

mod geom;
mod grid;
mod pipeline;

use ndarray::ArrayView2;

use crate::error::DelaunayError;
use pipeline::{triangulate_par, Ops, PipelineStats};

/// Progress events emitted by the parallel build (see
/// [`delaunay2d_par_with_progress`]). Events are delivered in order (the
/// per-block counter is updated under a lock), but from whichever rayon
/// worker thread finished the work — the callback must be `Sync`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParProgress {
    /// The grid is built; `n_blocks` block triangulations follow.
    Start { n_blocks: usize },
    /// A block triangulation finished.
    Blocks { done: usize, total: usize },
    /// The crust pass (re-triangulation + global verification) started.
    Crust,
    /// The merge (dedup, orientation, face matching, checks) started.
    Merge,
    /// The parallel attempt was abandoned (or skipped); the sequential
    /// kernel is about to run on the original input.
    Fallback,
    /// The build finished; `fallback` tells which path produced the result.
    Done { fallback: bool },
}

/// Why a parallel build ran (or finished) sequentially.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    /// The input is too small for the partition to pay off (see
    /// [`ParConfig::min_points`]); the sequential kernel ran directly.
    BelowThreshold,
    /// More than a quarter of the points failed certification; the cloud is
    /// degenerate or adversarial for the partition scheme.
    CrustTooLarge,
    /// The crust subset could not be triangulated by the kernel.
    CrustKernelError,
    /// A merge consistency check failed (typically exactly
    /// cocircular/cospherical inputs split across blocks).
    DegenerateMerge(&'static str),
}

/// Diagnostics of a parallel build attempt.
#[derive(Debug, Clone, Default)]
pub struct ParStats {
    /// Points in the caller's input.
    pub n_points: usize,
    /// Points after exact-duplicate removal.
    pub n_dedup: usize,
    /// Blocks the cloud was partitioned into.
    pub n_blocks: usize,
    /// Points that failed certification and went to the crust pass.
    pub crust_core: usize,
    /// Context points handed to the crust triangulation alongside the core.
    pub crust_context: usize,
    /// Simplices emitted by certified block stars (before dedup).
    pub n_block_emitted: usize,
    /// Simplices emitted by the globally verified crust.
    pub n_crust_emitted: usize,
    /// `None` if the parallel result was used; otherwise why the sequential
    /// kernel produced the returned triangulation.
    pub fallback: Option<FallbackReason>,
    /// Stage wall-clock times in seconds (grid build, block triangulation,
    /// crust pass, merge).
    pub t_grid: f64,
    pub t_blocks: f64,
    pub t_crust: f64,
    pub t_merge: f64,
}

/// Tuning knobs, resolved from defaults and (for hidden test hooks) the
/// environment.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParConfig {
    /// Inputs smaller than this run sequentially outright.
    pub min_points: usize,
    /// Aimed-for points per block.
    pub block_target: usize,
}

const DEFAULT_MIN_POINTS: usize = 100_000;
const BLOCK_TARGET_2D: usize = 150_000;
const BLOCK_TARGET_3D: usize = 125_000;
const MAX_BLOCKS: usize = 256;

impl ParConfig {
    fn from_env(block_target: usize) -> Self {
        let env = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        ParConfig {
            min_points: env("SHULL_PARALLEL_MIN_POINTS", DEFAULT_MIN_POINTS),
            block_target: env("SHULL_PARALLEL_BLOCK_TARGET", block_target),
        }
    }
}

const OPS_2D: Ops<2, 3> = Ops {
    kernel: crate::d2::delaunay2d::<f64>,
    circumball: geom::circumball2,
    inball_exact: geom::inball2_exact,
    orient_exact: geom::orient2_exact,
};

const OPS_3D: Ops<3, 4> = Ops {
    kernel: crate::d4::delaunay4d::<f64>,
    circumball: geom::circumball3,
    inball_exact: geom::inball3_exact,
    orient_exact: geom::orient3_exact,
};

type Triangulation<const K: usize> = (Vec<[u32; K]>, Vec<[i32; K]>, Vec<[u32; 2]>);

fn run_par<const D: usize, const K: usize, T>(
    points: ArrayView2<T>,
    ops: &Ops<D, K>,
    seq: fn(ArrayView2<T>) -> Result<Triangulation<K>, DelaunayError>,
    cfg: ParConfig,
    progress: Option<&(dyn Fn(ParProgress) + Sync)>,
) -> Result<(Triangulation<K>, ParStats), DelaunayError>
where
    T: Copy + Into<f64> + Sync,
{
    assert_eq!(points.ncols(), D, "input points have the wrong dimension");
    let n = points.nrows();
    let mut stats = ParStats { n_points: n, ..Default::default() };
    let emit = |p: ParProgress| {
        if let Some(cb) = progress {
            cb(p);
        }
    };
    let run_seq = |stats: ParStats| {
        emit(ParProgress::Fallback);
        let out = seq(points);
        emit(ParProgress::Done { fallback: true });
        out.map(|out| (out, stats))
    };

    // Small inputs — and inputs the u32-indexed kernels will reject anyway —
    // go straight to the sequential kernel, which also keeps its exact error
    // messages.
    if n < cfg.min_points || n >= u32::MAX as usize {
        stats.fallback = Some(FallbackReason::BelowThreshold);
        return run_seq(stats);
    }

    let t0 = std::time::Instant::now();
    let grid = grid::build_grid::<D, T>(points, cfg.block_target, MAX_BLOCKS);
    stats.t_grid = t0.elapsed().as_secs_f64();
    stats.n_dedup = grid.pts.len();
    stats.n_blocks = grid.blocks.len();
    if grid.blocks.len() < 2 {
        stats.fallback = Some(FallbackReason::BelowThreshold);
        return run_seq(stats);
    }

    let mut ps = PipelineStats::default();
    let result = triangulate_par(&grid, ops, &mut ps, progress);
    stats.crust_core = ps.crust_core;
    stats.crust_context = ps.crust_context;
    stats.n_block_emitted = ps.n_block_emitted;
    stats.n_crust_emitted = ps.n_crust_emitted;
    stats.t_blocks = ps.t_blocks;
    stats.t_crust = ps.t_crust;
    stats.t_merge = ps.t_merge;

    match result {
        Ok((simplices, neighbors)) => {
            emit(ParProgress::Done { fallback: false });
            Ok(((simplices, neighbors, grid.dropped), stats))
        }
        Err(reason) => {
            stats.fallback = Some(reason);
            run_seq(stats)
        }
    }
}

/// Parallel version of [`crate::delaunay2d`]: same input contract, same
/// simplex set (see the module docs for row order and degenerate-input
/// caveats). Uses rayon's global thread pool.
pub fn delaunay2d_par<T: Copy + Into<f64> + Sync>(
    points: ArrayView2<T>,
) -> Result<Triangulation<3>, DelaunayError> {
    delaunay2d_par_with_stats(points).map(|(out, _)| out)
}

/// [`delaunay2d_par`] plus build diagnostics.
pub fn delaunay2d_par_with_stats<T: Copy + Into<f64> + Sync>(
    points: ArrayView2<T>,
) -> Result<(Triangulation<3>, ParStats), DelaunayError> {
    run_par(
        points,
        &OPS_2D,
        crate::d2::delaunay2d::<T>,
        ParConfig::from_env(BLOCK_TARGET_2D),
        None,
    )
}

/// [`delaunay2d_par_with_stats`] with a progress callback (see
/// [`ParProgress`] for the event sequence and threading contract).
pub fn delaunay2d_par_with_progress<T, F>(
    points: ArrayView2<T>,
    progress: F,
) -> Result<(Triangulation<3>, ParStats), DelaunayError>
where
    T: Copy + Into<f64> + Sync,
    F: Fn(ParProgress) + Sync,
{
    run_par(
        points,
        &OPS_2D,
        crate::d2::delaunay2d::<T>,
        ParConfig::from_env(BLOCK_TARGET_2D),
        Some(&progress),
    )
}

/// Parallel version of [`crate::delaunay4d`]: same input contract, same
/// simplex set (see the module docs for row order and degenerate-input
/// caveats). Uses rayon's global thread pool.
pub fn delaunay4d_par<T: Copy + Into<f64> + Sync>(
    points: ArrayView2<T>,
) -> Result<Triangulation<4>, DelaunayError> {
    delaunay4d_par_with_stats(points).map(|(out, _)| out)
}

/// [`delaunay4d_par`] plus build diagnostics.
pub fn delaunay4d_par_with_stats<T: Copy + Into<f64> + Sync>(
    points: ArrayView2<T>,
) -> Result<(Triangulation<4>, ParStats), DelaunayError> {
    run_par(
        points,
        &OPS_3D,
        crate::d4::delaunay4d::<T>,
        ParConfig::from_env(BLOCK_TARGET_3D),
        None,
    )
}

/// [`delaunay4d_par_with_stats`] with a progress callback (see
/// [`ParProgress`] for the event sequence and threading contract).
pub fn delaunay4d_par_with_progress<T, F>(
    points: ArrayView2<T>,
    progress: F,
) -> Result<(Triangulation<4>, ParStats), DelaunayError>
where
    T: Copy + Into<f64> + Sync,
    F: Fn(ParProgress) + Sync,
{
    run_par(
        points,
        &OPS_3D,
        crate::d4::delaunay4d::<T>,
        ParConfig::from_env(BLOCK_TARGET_3D),
        Some(&progress),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Test-sized config: parallel path active from 64 points, tiny blocks
    /// so even small clouds exercise multi-block certification, halos, the
    /// crust and the merge.
    const TEST_CFG: ParConfig = ParConfig { min_points: 64, block_target: 100 };

    fn par2d(points: ArrayView2<f64>) -> Result<(Triangulation<3>, ParStats), DelaunayError> {
        run_par(points, &OPS_2D, crate::d2::delaunay2d::<f64>, TEST_CFG, None)
    }

    fn par3d(points: ArrayView2<f64>) -> Result<(Triangulation<4>, ParStats), DelaunayError> {
        run_par(points, &OPS_3D, crate::d4::delaunay4d::<f64>, TEST_CFG, None)
    }

    fn to_array<const D: usize>(points: &[[f64; D]]) -> Array2<f64> {
        Array2::from_shape_vec(
            (points.len(), D),
            points.iter().flatten().copied().collect(),
        )
        .unwrap()
    }

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn random_points<const D: usize>(n: usize, seed: u64) -> Vec<[f64; D]> {
        let mut next = rng(seed);
        (0..n)
            .map(|_| {
                let mut p = [0.0; D];
                for c in p.iter_mut() {
                    *c = next();
                }
                p
            })
            .collect()
    }

    /// Canonical form: vertices sorted within each simplex, simplices
    /// sorted. Orientation and row order are implementation details; the
    /// simplex *set* is what must match.
    fn canonical<const K: usize>(simps: &[[u32; K]]) -> Vec<[u32; K]> {
        let mut c: Vec<[u32; K]> = simps
            .iter()
            .map(|s| {
                let mut s = *s;
                s.sort_unstable();
                s
            })
            .collect();
        c.sort_unstable();
        c
    }

    /// Neighbor arrays are index-based, so compare them structurally: for
    /// each canonical simplex, the canonical simplex sets of its neighbors
    /// (with -1 for the boundary) must agree.
    fn canonical_adjacency<const K: usize>(
        simps: &[[u32; K]],
        nbrs: &[[i32; K]],
    ) -> Vec<Vec<Vec<u32>>> {
        let canon = |t: usize| {
            let mut s = simps[t].to_vec();
            s.sort_unstable();
            s
        };
        let mut items: Vec<(Vec<u32>, Vec<Vec<u32>>)> = (0..simps.len())
            .map(|t| {
                let mut adj: Vec<Vec<u32>> = nbrs[t]
                    .iter()
                    .map(|&m| if m < 0 { Vec::new() } else { canon(m as usize) })
                    .collect();
                adj.sort();
                (canon(t), adj)
            })
            .collect();
        items.sort();
        items.into_iter().map(|(_, adj)| adj).collect()
    }

    fn assert_equivalent<const K: usize>(
        par: &(Vec<[u32; K]>, Vec<[i32; K]>, Vec<[u32; 2]>),
        seq: &(Vec<[u32; K]>, Vec<[i32; K]>, Vec<[u32; 2]>),
    ) {
        assert_eq!(canonical(&par.0), canonical(&seq.0), "simplex sets differ");
        assert_eq!(
            canonical_adjacency(&par.0, &par.1),
            canonical_adjacency(&seq.0, &seq.1),
            "adjacency differs"
        );
        let dup = |d: &[[u32; 2]]| {
            let mut d = d.to_vec();
            d.sort_unstable();
            d
        };
        assert_eq!(dup(&par.2), dup(&seq.2), "duplicate reports differ");
    }

    #[test]
    fn uniform_2d_matches_sequential() {
        for seed in [1, 2, 3] {
            let pts = random_points::<2>(3000, seed);
            let arr = to_array(&pts);
            let (par, stats) = par2d(arr.view()).unwrap();
            let seq = crate::d2::delaunay2d(arr.view()).unwrap();
            assert_eq!(stats.fallback, None, "unexpected fallback: {:?}", stats);
            assert!(stats.n_blocks >= 2);
            assert_equivalent(&par, &seq);
        }
    }

    #[test]
    fn uniform_3d_matches_sequential() {
        for seed in [1, 2] {
            let pts = random_points::<3>(2500, seed);
            let arr = to_array(&pts);
            let (par, stats) = par3d(arr.view()).unwrap();
            let seq = crate::d4::delaunay4d(arr.view()).unwrap();
            assert_eq!(stats.fallback, None, "unexpected fallback: {:?}", stats);
            assert!(stats.n_blocks >= 2);
            assert_equivalent(&par, &seq);
        }
    }

    #[test]
    fn crust_stays_small_on_uniform_clouds() {
        let pts = random_points::<3>(4000, 5);
        let arr = to_array(&pts);
        let (_, stats) = par3d(arr.view()).unwrap();
        assert_eq!(stats.fallback, None);
        // The parallel result must not silently degrade into "everything is
        // crust": on a uniform cloud the certified block stars carry the
        // build.
        assert!(
            stats.crust_core < stats.n_dedup / 5,
            "crust too large: {:?}",
            stats
        );
        assert!(stats.n_block_emitted > 0);
    }

    #[test]
    fn gaussian_clusters_match_sequential() {
        // Non-uniform density: dense clusters with sparse space between.
        let mut next = rng(11);
        let mut pts: Vec<[f64; 3]> = Vec::new();
        for c in 0..6 {
            let center = [next() * 10.0, next() * 10.0, next() * 10.0];
            let spread = 0.1 + 0.4 * (c as f64 / 6.0);
            for _ in 0..500 {
                // Box-Muller-ish: sums of uniforms are gaussian enough here.
                let g = |next: &mut dyn FnMut() -> f64| {
                    (0..6).map(|_| next()).sum::<f64>() / 6.0 - 0.5
                };
                pts.push([
                    center[0] + g(&mut next) * spread,
                    center[1] + g(&mut next) * spread,
                    center[2] + g(&mut next) * spread,
                ]);
            }
        }
        let arr = to_array(&pts);
        let (par, stats) = par3d(arr.view()).unwrap();
        let seq = crate::d4::delaunay4d(arr.view()).unwrap();
        assert_eq!(stats.fallback, None, "unexpected fallback: {:?}", stats);
        assert_equivalent(&par, &seq);
    }

    #[test]
    fn tight_clusters_match_sequential() {
        // The adversarial two-cluster input from the kernel test suites:
        // scale 1e-6 clusters 1000 apart (2D and 3D).
        let mut next = rng(13);
        let pts2: Vec<[f64; 2]> = (0..600)
            .map(|i| {
                let off = if i < 300 { 0.0 } else { 1000.0 };
                [next() * 1e-6 + off, next() * 1e-6 + off]
            })
            .collect();
        let arr = to_array(&pts2);
        let (par, _) = par2d(arr.view()).unwrap();
        let seq = crate::d2::delaunay2d(arr.view()).unwrap();
        assert_equivalent(&par, &seq);

        let pts3: Vec<[f64; 3]> = (0..600)
            .map(|i| {
                let off = if i < 300 { 0.0 } else { 1000.0 };
                [next() * 1e-6 + off, next() * 1e-6 + off, next() * 1e-6 + off]
            })
            .collect();
        let arr = to_array(&pts3);
        let (par, _) = par3d(arr.view()).unwrap();
        let seq = crate::d4::delaunay4d(arr.view()).unwrap();
        assert_equivalent(&par, &seq);
    }

    #[test]
    fn blob_with_far_outliers_matches_sequential() {
        // Far outliers form giant simplices reaching back to the blob: their
        // circumballs span many blocks, so completeness relies on the crust
        // pass (the "Hole 1" regression case).
        for (n_out, seed) in [(1, 17), (3, 19)] {
            let mut pts = random_points::<3>(2000, seed);
            let mut next = rng(seed + 100);
            for i in 0..n_out {
                let far = 500.0 + 300.0 * i as f64;
                pts.push([far + next(), far * 0.5 + next(), far * 0.25 + next()]);
            }
            let arr = to_array(&pts);
            let (par, stats) = par3d(arr.view()).unwrap();
            let seq = crate::d4::delaunay4d(arr.view()).unwrap();
            assert_eq!(stats.fallback, None, "unexpected fallback: {:?}", stats);
            assert_equivalent(&par, &seq);
        }

        let mut pts = random_points::<2>(2000, 23);
        pts.push([800.0, 750.0]);
        pts.push([-500.0, 620.0]);
        let arr = to_array(&pts);
        let (par, stats) = par2d(arr.view()).unwrap();
        let seq = crate::d2::delaunay2d(arr.view()).unwrap();
        assert_eq!(stats.fallback, None, "unexpected fallback: {:?}", stats);
        assert_equivalent(&par, &seq);
    }

    #[test]
    fn degenerate_grid_falls_back_and_matches() {
        // Exact integer grids are cocircular/cospherical everywhere; split
        // across blocks they must trip the exact cross-source checks (or the
        // crust degeneracy checks) and fall back — bit-identical to
        // sequential.
        let mut pts2 = Vec::new();
        for x in 0..30 {
            for y in 0..30 {
                pts2.push([x as f64, y as f64]);
            }
        }
        let arr = to_array(&pts2);
        let (par, stats) = par2d(arr.view()).unwrap();
        assert!(stats.fallback.is_some(), "expected fallback, got {:?}", stats);
        let seq = crate::d2::delaunay2d(arr.view()).unwrap();
        assert_eq!(par, seq);

        let mut pts3 = Vec::new();
        for x in 0..12 {
            for y in 0..12 {
                for z in 0..12 {
                    pts3.push([x as f64, y as f64, z as f64]);
                }
            }
        }
        let arr = to_array(&pts3);
        let (par, stats) = par3d(arr.view()).unwrap();
        assert!(stats.fallback.is_some(), "expected fallback, got {:?}", stats);
        let seq = crate::d4::delaunay4d(arr.view()).unwrap();
        assert_eq!(par, seq);
    }

    #[test]
    fn duplicates_report_matches_sequential() {
        let mut pts = random_points::<3>(1500, 29);
        pts.push(pts[7]);
        pts.push(pts[7]);
        pts.push(pts[500]);
        let arr = to_array(&pts);
        let (par, stats) = par3d(arr.view()).unwrap();
        let seq = crate::d4::delaunay4d(arr.view()).unwrap();
        assert_eq!(stats.fallback, None, "unexpected fallback: {:?}", stats);
        assert_equivalent(&par, &seq);
        // Kept index is the first occurrence; dropped points appear in no
        // simplex.
        let pairs: std::collections::HashSet<[u32; 2]> = par.2.iter().copied().collect();
        assert_eq!(
            pairs,
            [[1500, 7], [1501, 7], [1502, 500]].into_iter().collect()
        );
        let used: std::collections::HashSet<u32> = par.0.iter().flatten().copied().collect();
        assert!(used.contains(&7) && !used.contains(&1500) && !used.contains(&1501));
    }

    #[test]
    fn below_threshold_runs_sequentially() {
        let pts = random_points::<2>(50, 31);
        let arr = to_array(&pts);
        let (par, stats) = par2d(arr.view()).unwrap();
        assert_eq!(stats.fallback, Some(FallbackReason::BelowThreshold));
        assert_eq!(par, crate::d2::delaunay2d(arr.view()).unwrap());
    }

    #[test]
    fn errors_match_sequential() {
        // Collinear 2D input (above the test threshold) must yield the same
        // Degenerate error as the sequential kernel.
        let pts: Vec<[f64; 2]> = (0..300).map(|i| [i as f64, 2.0 * i as f64]).collect();
        let arr = to_array(&pts);
        match par2d(arr.view()) {
            Err(DelaunayError::Degenerate(_)) => {}
            other => panic!("expected Degenerate, got {:?}", other.map(|(t, _)| t.0.len())),
        }
        // Coplanar 3D input likewise.
        let pts: Vec<[f64; 3]> = random_points::<2>(300, 37)
            .into_iter()
            .map(|p| [p[0], p[1], 0.0])
            .collect();
        let arr = to_array(&pts);
        match par3d(arr.view()) {
            Err(DelaunayError::Degenerate(_)) => {}
            other => panic!("expected Degenerate, got {:?}", other.map(|(t, _)| t.0.len())),
        }
    }

    #[test]
    fn single_thread_pool_matches_default_pool() {
        // Determinism across thread counts: the block partition depends only
        // on the cloud, so a 1-thread pool must produce bit-identical output.
        let pts = random_points::<3>(3000, 41);
        let arr = to_array(&pts);
        let (par_default, _) = par3d(arr.view()).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (par_single, _) = pool.install(|| par3d(arr.view())).unwrap();
        assert_eq!(par_default, par_single);
    }

    #[test]
    fn f32_input_matches_f64() {
        let pts = random_points::<3>(2000, 43);
        let flat32: Vec<f32> = pts.iter().flatten().map(|&v| v as f32).collect();
        let arr32 = Array2::from_shape_vec((pts.len(), 3), flat32).unwrap();
        let arr64 = arr32.mapv(|v| v as f64);
        let (p32, _) = run_par(
            arr32.view(),
            &OPS_3D,
            crate::d4::delaunay4d::<f32>,
            TEST_CFG,
            None,
        )
        .unwrap();
        let (p64, _) = par3d(arr64.view()).unwrap();
        assert_eq!(p32, p64);
    }

    #[test]
    fn progress_events_arrive_in_order() {
        let pts = random_points::<3>(2000, 53);
        let arr = to_array(&pts);
        let events = std::sync::Mutex::new(Vec::new());
        let ((simps, _, _), stats) = run_par(
            arr.view(),
            &OPS_3D,
            crate::d4::delaunay4d::<f64>,
            TEST_CFG,
            Some(&|p| events.lock().unwrap().push(p)),
        )
        .unwrap();
        assert!(!simps.is_empty());
        assert_eq!(stats.fallback, None);

        let events = events.into_inner().unwrap();
        let total = stats.n_blocks;
        assert_eq!(events[0], ParProgress::Start { n_blocks: total });
        // One Blocks event per block, dones strictly 1..=total in order.
        let dones: Vec<usize> = events
            .iter()
            .filter_map(|e| match e {
                ParProgress::Blocks { done, total: t } => {
                    assert_eq!(*t, total);
                    Some(*done)
                }
                _ => None,
            })
            .collect();
        assert_eq!(dones, (1..=total).collect::<Vec<_>>());
        let tail = &events[events.len() - 3..];
        assert_eq!(
            tail,
            [
                ParProgress::Crust,
                ParProgress::Merge,
                ParProgress::Done { fallback: false }
            ]
        );
    }

    #[test]
    fn progress_reports_fallback() {
        // Degenerate grid: parallel attempt runs, then falls back.
        let mut pts = Vec::new();
        for x in 0..12 {
            for y in 0..12 {
                for z in 0..12 {
                    pts.push([x as f64, y as f64, z as f64]);
                }
            }
        }
        let arr = to_array(&pts);
        let events = std::sync::Mutex::new(Vec::new());
        let (_, stats) = run_par(
            arr.view(),
            &OPS_3D,
            crate::d4::delaunay4d::<f64>,
            TEST_CFG,
            Some(&|p| events.lock().unwrap().push(p)),
        )
        .unwrap();
        assert!(stats.fallback.is_some());
        let events = events.into_inner().unwrap();
        assert!(events.contains(&ParProgress::Fallback));
        assert_eq!(*events.last().unwrap(), ParProgress::Done { fallback: true });

        // Below-threshold inputs skip straight to the sequential kernel.
        let pts = random_points::<2>(50, 59);
        let arr = to_array(&pts);
        let events = std::sync::Mutex::new(Vec::new());
        run_par(
            arr.view(),
            &OPS_2D,
            crate::d2::delaunay2d::<f64>,
            TEST_CFG,
            Some(&|p| events.lock().unwrap().push(p)),
        )
        .unwrap();
        assert_eq!(
            events.into_inner().unwrap(),
            [ParProgress::Fallback, ParProgress::Done { fallback: true }]
        );
    }

    /// Every simplex of the parallel output must have an empty circumball —
    /// checked directly, independently of the sequential comparison.
    #[test]
    fn parallel_output_is_delaunay() {
        let pts = random_points::<2>(1200, 47);
        let arr = to_array(&pts);
        let ((tris, _, _), stats) = par2d(arr.view()).unwrap();
        assert_eq!(stats.fallback, None);
        for t in &tris {
            let s = [
                pts[t[0] as usize],
                pts[t[1] as usize],
                pts[t[2] as usize],
            ];
            assert!(geom::orient2_exact(&s) > 0.0, "not CCW: {:?}", t);
            for (i, &q) in pts.iter().enumerate() {
                if t.contains(&(i as u32)) {
                    continue;
                }
                assert!(
                    geom::inball2_exact(&s, q) <= 0.0,
                    "point {} inside circumcircle of {:?}",
                    i,
                    t
                );
            }
        }
    }
}
