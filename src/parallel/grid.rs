//! Spatial preprocessing for the parallel build.
//!
//! Three responsibilities, all deterministic and independent of the thread
//! count:
//!
//! 1. **Exact-duplicate dedup**: the same `[dropped, kept]` semantics as the
//!    sequential kernels (kept = lowest original index), done once globally
//!    so no block ever sees coincident points and the duplicate report does
//!    not depend on the partition.
//! 2. **Uniform grid** over the bounding box (~[`TARGET_PER_CELL`] points per
//!    cell): the deduped points are stored sorted by the Morton code of
//!    their cell, so every cell — and every block — is a contiguous range.
//!    Space outside the bounding box contains no points by construction,
//!    which the coverage test exploits (out-of-grid regions are always
//!    "covered").
//! 3. **Block partition**: the nonempty cells, in Morton order, are split
//!    greedily into spatially compact chunks of roughly equal point count.
//!    The number of blocks depends only on the point count, never on the
//!    thread count, so the output is bit-identical for any RAYON_NUM_THREADS.

use ndarray::ArrayView2;
use rayon::prelude::*;

/// Marker for cells owned by no block (empty cells).
pub(crate) const NO_BLOCK: u32 = u32::MAX;

/// Average number of points per grid cell the resolution aims for. The halo
/// gathered around each block is one cell thick, i.e. ~`TARGET_PER_CELL^(1/D)`
/// point spacings — comfortably more than the circumball radius of a typical
/// interior simplex.
const TARGET_PER_CELL: usize = 32;

/// Edge length of a super-cell in cells: the coarse occupancy level used to
/// skip empty space when enumerating the (often enormous) circumballs of
/// hull-adjacent simplices.
pub(crate) const SUPER: usize = 8;

/// Hard cap on the total number of grid cells (the dense per-cell arrays are
/// `O(cells)`).
const MAX_CELLS: usize = 1 << 24;

/// Spread the low 21 bits of `v` so that bit i lands at position 3*i
/// (Morton interleave, 3D).
fn spread3(mut v: u64) -> u64 {
    v &= 0x1F_FFFF;
    v = (v | (v << 32)) & 0x1F0000_0000FFFF;
    v = (v | (v << 16)) & 0x1F0000_FF0000FF;
    v = (v | (v << 8)) & 0x100F00_F00F00F00F;
    v = (v | (v << 4)) & 0x10C30C_30C30C30C3;
    v = (v | (v << 2)) & 0x1249249249249249;
    v
}

/// Spread the low 32 bits of `v` so that bit i lands at position 2*i
/// (Morton interleave, 2D).
fn spread2(mut v: u64) -> u64 {
    v &= 0xFFFF_FFFF;
    v = (v | (v << 16)) & 0x0000FFFF_0000FFFF;
    v = (v | (v << 8)) & 0x00FF00FF_00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F_0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333_33333333;
    v = (v | (v << 1)) & 0x55555555_55555555;
    v
}

fn morton<const D: usize>(coords: [usize; D]) -> u64 {
    match D {
        2 => spread2(coords[0] as u64) | (spread2(coords[1] as u64) << 1),
        3 => {
            spread3(coords[0] as u64)
                | (spread3(coords[1] as u64) << 1)
                | (spread3(coords[2] as u64) << 2)
        }
        _ => unreachable!("only 2D and 3D grids are supported"),
    }
}

/// One block of the partition: a contiguous run of Morton-ordered nonempty
/// cells and the (contiguous) range of grid-ordered points they own.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Block {
    /// Range into [`Grid::cells_morton`].
    pub cells: (u32, u32),
    /// Range into [`Grid::pts`] (grid-ordered dedup ids).
    pub pts: (u32, u32),
}

pub(crate) struct Grid<const D: usize> {
    pub lo: [f64; D],
    /// Cells per axis (>= 1 each).
    pub dims: [usize; D],
    /// Cell edge length per axis (0.0 on a degenerate axis with dims 1).
    pub cell_size: [f64; D],
    /// 1 / cell_size (0.0 on a degenerate axis: everything maps to cell 0).
    inv_cell: [f64; D],
    /// Deduped points in grid order (sorted by Morton code of their cell,
    /// then coordinates). A point's index in this vector is its "dedup id".
    pub pts: Vec<[f64; D]>,
    /// Dedup id -> index into the caller's original point array.
    pub orig: Vec<u32>,
    /// `[dropped original index, kept original index]` duplicate report.
    pub dropped: Vec<[u32; 2]>,
    /// Row-major cell id -> range of dedup ids ((0, 0)-like empty ranges for
    /// empty cells).
    pub cell_range: Vec<(u32, u32)>,
    /// Row-major cell id -> owning block, NO_BLOCK for empty cells.
    pub cell_block: Vec<u32>,
    /// Nonempty row-major cell ids in Morton order.
    pub cells_morton: Vec<u32>,
    pub blocks: Vec<Block>,
    /// Coarse occupancy: super-cells of SUPER^D cells; `true` iff any cell
    /// inside has points.
    pub super_dims: [usize; D],
    pub super_occupied: Vec<bool>,
}

impl<const D: usize> Grid<D> {
    /// Cell coordinates of a point (clamped into the grid).
    pub fn cell_coords_of(&self, p: &[f64; D]) -> [usize; D] {
        let mut c = [0usize; D];
        for k in 0..D {
            let f = (p[k] - self.lo[k]) * self.inv_cell[k];
            // p >= lo by construction, so f >= 0; clamp the hi corner.
            c[k] = (f as usize).min(self.dims[k] - 1);
        }
        c
    }

    /// Cell index along axis `k` of coordinate `x`, clamped into the grid
    /// (unlike [`Self::cell_coords_of`], `x` may lie outside the bbox).
    pub fn axis_cell(&self, k: usize, x: f64) -> usize {
        let f = (x - self.lo[k]) * self.inv_cell[k];
        if f <= 0.0 {
            0
        } else {
            (f as usize).min(self.dims[k] - 1)
        }
    }

    pub fn cell_id(&self, coords: [usize; D]) -> usize {
        let mut id = 0usize;
        for k in 0..D {
            id = id * self.dims[k] + coords[k];
        }
        id
    }

    pub fn cell_coords(&self, mut id: usize) -> [usize; D] {
        let mut c = [0usize; D];
        for k in (0..D).rev() {
            c[k] = id % self.dims[k];
            id /= self.dims[k];
        }
        c
    }

    pub fn n_cells(&self) -> usize {
        self.dims.iter().product()
    }

    /// Is `cell` (row-major id) gathered by `block`, i.e. within Chebyshev
    /// distance 1 of one of its owned cells? This is exactly the membership
    /// predicate of the point set handed to the block's kernel run.
    pub fn is_gathered(&self, block: u32, coords: [usize; D]) -> bool {
        let mut lo = [0usize; D];
        let mut hi = [0usize; D];
        for k in 0..D {
            lo[k] = coords[k].saturating_sub(1);
            hi[k] = (coords[k] + 1).min(self.dims[k] - 1);
        }
        let mut cur = lo;
        loop {
            if self.cell_block[self.cell_id(cur)] == block {
                return true;
            }
            // Advance the D-dimensional counter.
            let mut k = 0;
            loop {
                if k == D {
                    return false;
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

    /// Lower corner of a cell's box.
    pub fn cell_box_lo(&self, coords: [usize; D]) -> [f64; D] {
        let mut b = [0.0; D];
        for k in 0..D {
            b[k] = self.lo[k] + coords[k] as f64 * self.cell_size[k];
        }
        b
    }

    pub fn super_id(&self, super_coords: [usize; D]) -> usize {
        let mut id = 0usize;
        for k in 0..D {
            id = id * self.super_dims[k] + super_coords[k];
        }
        id
    }
}

/// Build the grid: dedup the input exactly, sort into grid order, and
/// partition into blocks of roughly `block_target` points each (at most
/// `max_blocks`, and never more blocks than nonempty cells). The block count
/// is a function of the deduplicated point count only — never of the thread
/// count — so results are reproducible across machines.
pub(crate) fn build_grid<const D: usize, T: Copy + Into<f64> + Sync>(
    points: ArrayView2<T>,
    block_target: usize,
    max_blocks: usize,
) -> Grid<D> {
    let n = points.nrows();
    debug_assert_eq!(points.ncols(), D);

    // Raw coordinates as [f64; D] (f32 widens exactly, matching the kernels).
    let coords: Vec<[f64; D]> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = points.row(i);
            let mut p = [0.0; D];
            for k in 0..D {
                p[k] = row[k].into();
            }
            p
        })
        .collect();

    // Bounding box.
    let (lo, hi) = coords
        .par_iter()
        .fold(
            || ([f64::INFINITY; D], [f64::NEG_INFINITY; D]),
            |(mut lo, mut hi), p| {
                for k in 0..D {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
                (lo, hi)
            },
        )
        .reduce(
            || ([f64::INFINITY; D], [f64::NEG_INFINITY; D]),
            |(mut lo, mut hi), (l2, h2)| {
                for k in 0..D {
                    lo[k] = lo[k].min(l2[k]);
                    hi[k] = hi[k].max(h2[k]);
                }
                (lo, hi)
            },
        );

    // Grid resolution: aim for TARGET_PER_CELL points per cell with roughly
    // cubic cells (edge from the D-dimensional volume), degenerate axes get a
    // single cell.
    let mut ext = [0.0f64; D];
    let mut volume = 1.0f64;
    let mut n_degenerate = 0;
    for k in 0..D {
        ext[k] = hi[k] - lo[k];
        if ext[k] > 0.0 {
            volume *= ext[k];
        } else {
            n_degenerate += 1;
        }
    }
    let target_cells = (n / TARGET_PER_CELL).clamp(1, MAX_CELLS);
    let live_axes = D - n_degenerate;
    let edge = if live_axes > 0 {
        (volume / target_cells as f64).powf(1.0 / live_axes as f64)
    } else {
        1.0
    };
    let mut dims = [1usize; D];
    for k in 0..D {
        if ext[k] > 0.0 && edge > 0.0 {
            dims[k] = ((ext[k] / edge).ceil() as usize).clamp(1, 1 << 20);
        }
    }
    // Keep the dense per-cell arrays bounded.
    while dims.iter().product::<usize>() > MAX_CELLS {
        let k = (0..D).max_by_key(|&k| dims[k]).unwrap();
        dims[k] = dims[k].div_ceil(2);
    }
    let mut cell_size = [0.0f64; D];
    let mut inv_cell = [0.0f64; D];
    for k in 0..D {
        if ext[k] > 0.0 {
            cell_size[k] = ext[k] / dims[k] as f64;
            inv_cell[k] = dims[k] as f64 / ext[k];
        }
    }

    let mut grid = Grid {
        lo,
        dims,
        cell_size,
        inv_cell,
        pts: Vec::new(),
        orig: Vec::new(),
        dropped: Vec::new(),
        cell_range: Vec::new(),
        cell_block: Vec::new(),
        cells_morton: Vec::new(),
        blocks: Vec::new(),
        super_dims: [1; D],
        super_occupied: Vec::new(),
    };

    // Sort original indices by (Morton(cell), coordinates, index): cells are
    // contiguous, coordinate-equal points are adjacent (dedup), and the
    // lowest original index leads every duplicate run.
    let mut keyed: Vec<(u64, u32)> = (0..n as u32)
        .into_par_iter()
        .map(|i| (morton(grid.cell_coords_of(&coords[i as usize])), i))
        .collect();
    keyed.par_sort_unstable_by(|&(ka, ia), &(kb, ib)| {
        ka.cmp(&kb).then_with(|| {
            let (pa, pb) = (&coords[ia as usize], &coords[ib as usize]);
            for k in 0..D {
                match pa[k].total_cmp(&pb[k]) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            ia.cmp(&ib)
        })
    });

    // Dedup scan + grid-ordered point/orig arrays.
    grid.pts.reserve(n);
    grid.orig.reserve(n);
    let mut kept_orig = u32::MAX;
    let mut prev: Option<[f64; D]> = None;
    for &(_, i) in &keyed {
        let p = coords[i as usize];
        if prev == Some(p) {
            grid.dropped.push([i, kept_orig]);
        } else {
            prev = Some(p);
            kept_orig = i;
            grid.pts.push(p);
            grid.orig.push(i);
        }
    }
    let n_d = grid.pts.len();

    // Per-cell ranges (runs in the grid-ordered array) and the Morton-ordered
    // nonempty cell list.
    grid.cell_range = vec![(0, 0); grid.n_cells()];
    let mut s = 0usize;
    while s < n_d {
        let cell = grid.cell_id(grid.cell_coords_of(&grid.pts[s]));
        let mut e = s + 1;
        while e < n_d && grid.cell_id(grid.cell_coords_of(&grid.pts[e])) == cell {
            e += 1;
        }
        grid.cell_range[cell] = (s as u32, e as u32);
        grid.cells_morton.push(cell as u32);
        s = e;
    }

    // Coarse occupancy level.
    for k in 0..D {
        grid.super_dims[k] = grid.dims[k].div_ceil(SUPER);
    }
    grid.super_occupied = vec![false; grid.super_dims.iter().product()];
    for &cell in &grid.cells_morton {
        let coords = grid.cell_coords(cell as usize);
        let mut sc = [0usize; D];
        for k in 0..D {
            sc[k] = coords[k] / SUPER;
        }
        let sid = grid.super_id(sc);
        grid.super_occupied[sid] = true;
    }

    // Greedy prefix split of the Morton cell order into blocks of roughly
    // equal point count.
    grid.cell_block = vec![NO_BLOCK; grid.n_cells()];
    let n_blocks = (n_d / block_target.max(1))
        .clamp(1, max_blocks.max(1))
        .min(grid.cells_morton.len().max(1));
    let mut remaining_pts = n_d;
    let mut remaining_blocks = n_blocks;
    let mut cell_lo = 0usize;
    let mut pt_lo = 0usize;
    let mut acc = 0usize;
    for (ci, &cell) in grid.cells_morton.iter().enumerate() {
        let (cs, ce) = grid.cell_range[cell as usize];
        acc += (ce - cs) as usize;
        grid.cell_block[cell as usize] = grid.blocks.len() as u32;
        let target = remaining_pts.div_ceil(remaining_blocks);
        let cells_left = grid.cells_morton.len() - ci - 1;
        if (acc >= target && remaining_blocks > 1) || cells_left == 0 {
            grid.blocks.push(Block {
                cells: (cell_lo as u32, ci as u32 + 1),
                pts: (pt_lo as u32, (pt_lo + acc) as u32),
            });
            cell_lo = ci + 1;
            pt_lo += acc;
            remaining_pts -= acc;
            acc = 0;
            if remaining_blocks > 1 {
                remaining_blocks -= 1;
            }
        }
    }
    debug_assert_eq!(pt_lo, n_d);

    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn to_array<const D: usize>(points: &[[f64; D]]) -> Array2<f64> {
        Array2::from_shape_vec(
            (points.len(), D),
            points.iter().flatten().copied().collect(),
        )
        .unwrap()
    }

    fn pseudo_random<const D: usize>(n: usize, seed: u64) -> Vec<[f64; D]> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
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

    #[test]
    fn morton_interleave_is_correct() {
        // x bits at even positions, y at odd: (y2x2 y1x1 y0x0) = 01 10 11.
        assert_eq!(morton::<2>([0b101, 0b011]), 0b01_10_11);
        assert_eq!(morton::<2>([1, 0]), 0b01);
        assert_eq!(morton::<2>([0, 1]), 0b10);
        assert_eq!(morton::<3>([1, 0, 0]), 0b001);
        assert_eq!(morton::<3>([0, 1, 0]), 0b010);
        assert_eq!(morton::<3>([0, 0, 1]), 0b100);
        // Round-trip a few large coordinates through spread3 (21 bits).
        for v in [0x1F_FFFFu64, 0x15_5555, 0x0A_AAAA] {
            let s = spread3(v);
            let mut back = 0u64;
            for i in 0..21 {
                back |= ((s >> (3 * i)) & 1) << i;
            }
            assert_eq!(back, v);
        }
    }

    #[test]
    fn dedup_matches_kernel_semantics() {
        let mut pts = pseudo_random::<3>(200, 1);
        pts.push(pts[7]);
        pts.push(pts[7]);
        pts.push(pts[42]);
        let grid = build_grid::<3, f64>(to_array(&pts).view(), 50, 256);
        assert_eq!(grid.pts.len(), 200);
        let pairs: std::collections::HashSet<[u32; 2]> =
            grid.dropped.iter().copied().collect();
        assert_eq!(pairs, [[200, 7], [201, 7], [202, 42]].into_iter().collect());
        // orig is a bijection onto the survivors.
        let mut seen: Vec<bool> = vec![false; 203];
        for &o in &grid.orig {
            assert!(!seen[o as usize]);
            seen[o as usize] = true;
        }
        assert!(seen[7] && seen[42] && !seen[200] && !seen[201] && !seen[202]);
    }

    #[test]
    fn cells_partition_points_and_blocks_partition_cells() {
        let pts = pseudo_random::<2>(5000, 3);
        let grid = build_grid::<2, f64>(to_array(&pts).view(), 625, 256);
        // Every point is inside its cell's range.
        for (i, p) in grid.pts.iter().enumerate() {
            let cell = grid.cell_id(grid.cell_coords_of(p));
            let (s, e) = grid.cell_range[cell];
            assert!((s as usize) <= i && i < e as usize);
            assert_ne!(grid.cell_block[cell], NO_BLOCK);
        }
        // Blocks: contiguous, disjoint, covering.
        assert!(grid.blocks.len() <= 8 && !grid.blocks.is_empty());
        let mut pt_cursor = 0u32;
        let mut cell_cursor = 0u32;
        for (b, blk) in grid.blocks.iter().enumerate() {
            assert_eq!(blk.pts.0, pt_cursor);
            assert_eq!(blk.cells.0, cell_cursor);
            pt_cursor = blk.pts.1;
            cell_cursor = blk.cells.1;
            for &cell in &grid.cells_morton[blk.cells.0 as usize..blk.cells.1 as usize] {
                assert_eq!(grid.cell_block[cell as usize], b as u32);
            }
        }
        assert_eq!(pt_cursor as usize, grid.pts.len());
        assert_eq!(cell_cursor as usize, grid.cells_morton.len());
        // Block point counts are roughly balanced (within 3x of each other).
        let sizes: Vec<u32> = grid.blocks.iter().map(|b| b.pts.1 - b.pts.0).collect();
        let (min, max) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
        assert!(max / min.max(&1) <= 3, "unbalanced blocks: {:?}", sizes);
    }

    #[test]
    fn gathered_is_chebyshev_one() {
        let pts = pseudo_random::<2>(2000, 9);
        let grid = build_grid::<2, f64>(to_array(&pts).view(), 500, 256);
        for cell in 0..grid.n_cells() {
            let coords = grid.cell_coords(cell);
            for b in 0..grid.blocks.len() as u32 {
                let mut expect = false;
                for di in -1i64..=1 {
                    for dj in -1i64..=1 {
                        let (i, j) = (coords[0] as i64 + di, coords[1] as i64 + dj);
                        if i < 0 || j < 0 || i >= grid.dims[0] as i64 || j >= grid.dims[1] as i64
                        {
                            continue;
                        }
                        expect |= grid.cell_block[grid.cell_id([i as usize, j as usize])] == b;
                    }
                }
                assert_eq!(grid.is_gathered(b, coords), expect);
            }
        }
    }

    #[test]
    fn degenerate_axis_gets_one_cell() {
        // All z equal: 3D grid with dims[2] == 1.
        let pts: Vec<[f64; 3]> = pseudo_random::<2>(500, 5)
            .into_iter()
            .map(|p| [p[0], p[1], 0.25])
            .collect();
        let grid = build_grid::<3, f64>(to_array(&pts).view(), 125, 256);
        assert_eq!(grid.dims[2], 1);
        assert_eq!(grid.pts.len(), 500);
    }
}
