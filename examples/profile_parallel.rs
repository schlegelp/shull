//! Tuning/profiling harness for the parallel build.
//!
//!     cargo run --release --features parallel --example profile_parallel [n] [dim]
//!
//! Set RAYON_NUM_THREADS to control the pool (it is read once, at first use).
//! Prints the sequential time, the parallel time and the build diagnostics
//! (block count, crust size, fallback cause) that drive the tuning knobs in
//! src/parallel/mod.rs.

use ndarray::Array2;

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let dim: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(3);

    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let data: Vec<f64> = (0..n * dim).map(|_| next()).collect();
    let pts = Array2::from_shape_vec((n, dim), data).unwrap();

    println!(
        "n = {}, dim = {}, rayon threads = {}",
        n,
        dim,
        rayon::current_num_threads()
    );

    match dim {
        2 => {
            let t0 = std::time::Instant::now();
            let (tris, _, _) = shull::delaunay2d(pts.view()).unwrap();
            let seq = t0.elapsed();
            println!("sequential: {} triangles in {:?}", tris.len(), seq);

            let t0 = std::time::Instant::now();
            let ((tris, _, _), stats) = shull::delaunay2d_par_with_stats(pts.view()).unwrap();
            let par = t0.elapsed();
            println!("parallel:   {} triangles in {:?}", tris.len(), par);
            println!("speedup:    {:.2}x", seq.as_secs_f64() / par.as_secs_f64());
            println!("stats:      {:?}", stats);
        }
        3 => {
            let t0 = std::time::Instant::now();
            let (tets, _, _) = shull::delaunay4d(pts.view()).unwrap();
            let seq = t0.elapsed();
            println!("sequential: {} tets in {:?}", tets.len(), seq);

            let t0 = std::time::Instant::now();
            let ((tets, _, _), stats) = shull::delaunay4d_par_with_stats(pts.view()).unwrap();
            let par = t0.elapsed();
            println!("parallel:   {} tets in {:?}", tets.len(), par);
            println!("speedup:    {:.2}x", seq.as_secs_f64() / par.as_secs_f64());
            println!("stats:      {:?}", stats);
        }
        other => panic!("dim must be 2 or 3, got {}", other),
    }
}
