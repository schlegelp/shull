use ndarray::Array2;

fn main() {
    let n = 1_000_000usize;
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let data: Vec<f64> = (0..n * 3).map(|_| next()).collect();
    let pts = Array2::from_shape_vec((n, 3), data).unwrap();
    let t0 = std::time::Instant::now();
    let (tets, _, _) = shull::delaunay4d(pts.view()).unwrap();
    println!("{} tets in {:?}", tets.len(), t0.elapsed());
}
