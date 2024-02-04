use core::panic;

use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Euclidean distance between two points (squared)
fn distance(p1: ArrayView1<f64>, p2: ArrayView1<f64>) -> f64 {
    (p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)
}

/// Sort points by radial distance from a single point
///
/// # Arguments
/// * `points` - A 2D array of points
/// * `point` - A single point
///
/// # Returns
/// * A vector of (distance, index) tuples sorted by distance (lowest first)
fn radial_distance_sort(points: ArrayView2<f64>, point: ArrayView1<f64>) -> Vec<(f64, usize)> {
    let mut distances: Vec<(f64, usize)> = Vec::with_capacity(points.nrows());
    for (i, p) in points.rows().into_iter().enumerate() {
        distances.push((distance(p, point), i));
    }
    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    distances
}

/// Find the point that together with two known points produces the smallest triangle circumcircle
///
/// # Arguments
/// * `points` - A 2D array of points
/// * `p1` - Index of the first point
/// * `p2` - Index of the second point
///
/// # Returns
/// * A tuple of the smallest circumcircle diameter and the index of the third point
fn find_smallest_circumcircle(points: ArrayView2<f64>, p1: usize, p2: usize) -> (f64, usize) {
    // http://www.mathopenref.com/trianglecircumcircle.html
    let p1_co = points.row(p1);
    let p2_co = points.row(p2);
    let a = distance(p1_co, p2_co).sqrt();
    assert!(a != 0.0, "Zero distance between duplicate points is not allowed");

    let mut b: f64;
    let mut c: f64;
    let mut x: f64;
    let mut diam: f64;

    let mut smallest_diam = f64::MAX;
    let mut smallest_diam_index = 0;
    for (p3, p3_co) in points.rows().into_iter().enumerate() {
        if p3 == p1 || p3 == p2 {
            continue;
        }
        b = distance(p1_co, p3_co).sqrt();
        c = distance(p2_co, p3_co).sqrt();

        // https://en.wikipedia.org/wiki/Heron%27s_formula#Numerical_stability
        x = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c));

        if x <= 0.0 {
            continue;
        }

        x = x.sqrt();
        diam = 0.5 * a * b * c / x;

        if diam < smallest_diam {
            smallest_diam = diam;
            smallest_diam_index = p3;
        }
    }
    (smallest_diam, smallest_diam_index)
}

/// Calculate the circumcircle center of a triangle
///
/// # Arguments
/// * `pta` - Coordinates of the first point
/// * `ptb` - Coordinates of the second point
/// * `ptc` - Coordinates of the third point
///
/// # Returns
/// * A tuple of the x and y coordinates of the circumcircle center
fn circum_circle_center(
    pta: ArrayView1<f64>,
    ptb: ArrayView1<f64>,
    ptc: ArrayView1<f64>,
) -> (f64, f64) {
    let pta2 = pta[0].powi(2) + pta[1].powi(2);
    let ptb2 = ptb[0].powi(2) + ptb[1].powi(2);
    let ptc2 = ptc[0].powi(2) + ptc[1].powi(2);

    let d = 2.0
        * (pta[0] * (ptb[1] - ptc[1]) + ptb[0] * (ptc[1] - pta[1]) + ptc[0] * (pta[1] - ptb[1]));
    assert!(d != 0.0, "Could not find circumcircle centre");

    let ux = (pta2 * (ptb[1] - ptc[1]) + ptb2 * (ptc[1] - pta[1]) + ptc2 * (pta[1] - ptb[1])) / d;
    let uy = (pta2 * (ptc[0] - ptb[0]) + ptb2 * (pta[0] - ptc[0]) + ptc2 * (ptb[0] - pta[0])) / d;

    (ux, uy)
}

fn right_hand_check(
    pt1_co: ArrayView1<f64>,
    pt2_co: ArrayView1<f64>,
    pt3_co: ArrayView1<f64>,
) -> f64 {
    let vec21 = (pt1_co[0] - pt2_co[0], pt1_co[1] - pt2_co[1]);
    let vec23 = (pt3_co[0] - pt2_co[0], pt3_co[1] - pt2_co[1]);

    vec21.0 * vec23.1 - vec21.1 * vec23.0
}

/// Form triangles from a set of points
///
/// # Arguments
/// * `points` - A 2D array of points
/// * `seed_triangle` - A vector of 3 indices representing the first triangle
/// * `pt_order` - A vector of indices representing the order in which to add points
///
/// # Returns
/// * A tuple of the indices of the hull and a vector of triangles
fn form_triangles(
    points: &ArrayView2<f64>,
    seed_triangle: Vec<usize>,
    pt_order: Vec<usize>,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut triangles = vec![seed_triangle.clone()];
    let mut hull: Vec<usize> = seed_triangle.clone();
    let mut new_hull: Vec<usize> = vec![];

    let mut vis_ind: Vec<usize> = vec![];
    let mut vis_list: Vec<f64> = vec![];
    let mut cursor: usize;

    for pt_to_add in pt_order {
        vis_ind.clear();
        vis_list.clear();
        new_hull.clear();

        // Check which hull faces are visible
        for (h_ind, h_p) in hull.iter().enumerate() {
            let vis = right_hand_check(
                points.row(*h_p),
                points.row(hull[(h_ind + 1) % hull.len()]),
                points.row(pt_to_add),
            );
            vis_list.push(vis);
            if vis <= 0.0 {
                vis_ind.push(h_ind);
            }
        }
        assert!(vis_ind.len() > 0, "No hull sides visible");

        // Check for range of sides that are visible
        let mut first_side: usize = 0;
        while vis_ind.contains(&first_side) {
            first_side += 1;
            assert!(first_side < hull.len(), "No sides are visible to point");
        }

        while !vis_ind.contains(&first_side) {
            first_side = (first_side + 1) % hull.len();
        }

        let mut last_side = first_side;
        while vis_ind.contains(&((last_side + 1) % hull.len())) {
            last_side = (last_side + 1) % hull.len();
        }

        // Get copy of retained section of hull
        cursor = (last_side + 1) % hull.len();
        let mut new_hull: Vec<usize> = vec![];
        loop {
            new_hull.push(hull[cursor]);
            if vis_ind.contains(&cursor) {
                break;
            }
            cursor = (cursor + 1) % hull.len();
        }

        // Add new point to hull
        new_hull.push(pt_to_add);

        // Form new triangles
        cursor = first_side;
        loop {
            triangles.push(vec![
                hull[cursor],
                pt_to_add,
                hull[(cursor + 1) % hull.len()],
            ]);
            if cursor == last_side {
                break;
            }
            cursor = (cursor + 1) % hull.len();
        }
        hull = new_hull;
    }
    (hull, triangles)
}

/// Calculate the angle of a triangle
///
/// # Arguments
/// * `points` - A 2D array of points
/// * `triangle` - A vector of indices representing the triangle
///
/// # Returns
/// * The angle of the triangle
fn calc_triangle_angle(points: &ArrayView2<f64>, triangle: Vec<usize>) -> Result<f64, String> {
    // Angle is computed on pt3, pt1 and pt2 define the side opposite the angle
    let pt1_co = points.row(triangle[0]);
    let pt2_co = points.row(triangle[1]);
    let pt3_co = points.row(triangle[2]);

    let v31 = (pt1_co[0] - pt3_co[0], pt1_co[1] - pt3_co[1]);
    let v32 = (pt2_co[0] - pt3_co[0], pt2_co[1] - pt3_co[1]);

    let mv31 = (v31.0.powi(2) + v31.1.powi(2)).sqrt();
    let mv32 = (v32.0.powi(2) + v32.1.powi(2)).sqrt();

    if mv31 <= 0.0 || mv32 <= 0.0 {
        return Err("Zero length side in triangle".to_string());
    }

    let v31n = (v31.0 / mv31, v31.1 / mv31);
    let v32n = (v32.0 / mv32, v32.1 / mv32);

    let crossp = v31n.1 * v32n.0 - v31n.0 * v32n.1;
    let mut dotp = v31n.0 * v32n.0 + v31n.1 * v32n.1;

    // Limit to valid range
    dotp = dotp.max(-1.0).min(1.0);

    let angle = if crossp < 0.0 {
        2.0 * std::f64::consts::PI - dotp.acos()
    } else {
        dotp.acos()
    };

    Ok(angle)
}

fn check_and_flip_triangle_pairs(
    points: &ArrayView2<f64>,
    tri1: Vec<usize>,
    tri2: Vec<usize>,
) -> (bool, Vec<usize>, Vec<usize>) {
    let quad = (tri1[0], tri1[2], tri2[2], tri2[1]);

    let t1 = calc_triangle_angle(points, vec![quad.0, quad.2, quad.1]);
    let t3 = calc_triangle_angle(points, vec![quad.2, quad.0, quad.3]);

    // Check t1 and t3 for errors and return (false, tri1, tri2) if there is an error
    // else unwrap to get the values
    let t1 = match t1 {
        Ok(t) => t,
        Err(e) => return (false, tri1, tri2),
    };
    let t3 = match t3 {
        Ok(t) => t,
        Err(e) => return (false, tri1, tri2),
    };

    let flip_degenerate_tri = t1 == std::f64::consts::PI || t3 == std::f64::consts::PI;
    let angle_total = t1 + t3;
    let flip_for_delaunay = angle_total > std::f64::consts::PI;

    if flip_degenerate_tri || flip_for_delaunay {
        let t2 = calc_triangle_angle(points, vec![quad.1, quad.3, quad.2]);
        let t4 = calc_triangle_angle(points, vec![quad.3, quad.1, quad.0]);

        // Check t2 and t4 for errors and return (false, tri1, tri2) if there is an error
        // else unwrap to get the values
        let t2 = match t2 {
            Ok(t) => t,
            Err(e) => return (false, tri1, tri2),
        };
        let t4 = match t4 {
            Ok(t) => t,
            Err(e) => return (false, tri1, tri2),
        };

        // Flipping would create an overlap
        if flip_degenerate_tri && (t2 > std::f64::consts::PI || t4 > std::f64::consts::PI) {
            return (false, tri1, tri2);
        }

        // Flipping would create triangle of zero area
        if t2 == std::f64::consts::PI || t4 == std::f64::consts::PI {
            return (false, tri1, tri2);
        }

        let tri1_flip = vec![tri2[1], tri1[2], tri1[0]];
        let tri2_flip = vec![tri1[2], tri2[1], tri1[1]];
        let angle_total_flip = t2 + t4;

        // No improvement when flipped, so abort flip
        if flip_for_delaunay && angle_total_flip >= angle_total {
            return (false, tri1, tri2);
        }

        return (true, tri1_flip, tri2_flip);
    };
    (false, tri1, tri2)
}

/// Check if two triangles have a common edges
fn has_common_edge(
    tri1: &Vec<usize>,
    tri2: &Vec<usize>,
) -> Option<((usize, usize, usize), (usize, usize, usize))> {
    const TRIANGLE_EDGES1: [(usize, usize, usize); 3] = [(0, 1, 2), (1, 2, 0), (2, 0, 1)];
    const TRIANGLE_EDGES2: [(usize, usize, usize); 3] = [(2, 1, 0), (1, 0, 2), (0, 2, 1)];

    for (i, j, k) in TRIANGLE_EDGES1.iter() {
        for (l, m, n) in TRIANGLE_EDGES2.iter() {
            if tri1[*i] == tri2[*l] && tri1[*j] == tri2[*m] {
                return Some(((*i, *j, *k), (*l, *m, *n)));
            }
        }
    }
    return None;
}

/// Remove triangle from `shared_edges`
fn remove_triangle_from_common_edge(
    shared_edges: &mut AHashMap<(usize, usize), Vec<usize>>,
    triangles: &Vec<Vec<usize>>,
    tri_num: usize,
) {
    let tri = &triangles[tri_num];
    const EDGE_INDICES: [(usize, usize, usize); 3] = [(0, 1, 2), (1, 2, 0), (2, 0, 1)];

    for (i, j, k) in EDGE_INDICES {
        let edge = (tri[i], tri[j]);
        let edge_id = (edge.0.min(edge.1), edge.0.max(edge.1));
        // Note: could use swap_remove (faster) here instead of retain if order is
        // not important (need to check)
        if let Some(value) = shared_edges.get_mut(&edge_id) {
            value.retain(|&x| x != tri_num);
        }
    }
}

/// Add triangle to `shared_edges`
fn add_triangle_to_common_edge(
    shared_edges: &mut AHashMap<(usize, usize), Vec<usize>>,
    triangles: &Vec<Vec<usize>>,
    tri_num: usize,
) {
    let tri = &triangles[tri_num];
    const EDGE_INDICES: [(usize, usize, usize); 3] = [(0, 1, 2), (1, 2, 0), (2, 0, 1)];

    for (i, j, k) in EDGE_INDICES {
        let edge = (tri[i], tri[j]);
        let edge_id = (edge.0.min(edge.1), edge.0.max(edge.1));
        if let Some(value) = shared_edges.get_mut(&edge_id) {
            value.push(tri_num);
        } else {
            shared_edges.insert(edge_id, vec![tri_num]);
        }
    }
}

/// Flip triangles to form a Delaunay triangulation
///
/// # Arguments
/// * `points` - A 2D array of points
/// * `triangles` - A vector of triangles
/// * `node_ordering` - A boolean indicating if the node ordering is known
///
/// # Returns
/// * A vector of flipped triangles
fn flip_triangles(
    points: ArrayView2<f64>,
    mut triangles: Vec<Vec<usize>>,
    mut node_ordering: Option<bool>,
) -> Vec<Vec<usize>> {
    // Set all triangle windings the same way
    for tri in triangles.iter_mut() {
        let rh_check = right_hand_check(points.row(tri[0]), points.row(tri[1]), points.row(tri[2]));
        if rh_check < 0.0 {
            // Reverse the winding
            tri.swap(0, 1);
        }
        // Check if `node_ordering`` is not None
        if node_ordering.is_none() && rh_check != 0.0 {
            node_ordering = Some(rh_check > 0.);
        }
    }

    // Catalogue all shared edges
    let mut shared_edges: AHashMap<(usize, usize), Vec<usize>> = AHashMap::new();
    for (tri_num, tri) in triangles.iter().enumerate() {
        add_triangle_to_common_edge(&mut shared_edges, &triangles, tri_num)
    }

    let mut previous_configurations = vec![triangles.clone()];

    loop {
        // Get a copy of the keys in `shared_edges`
        let shared_edge_keys: Vec<(usize, usize)> = shared_edges.keys().cloned().collect();
        let mut count: usize = 0;

        for edge_key in shared_edge_keys {
            let edge = shared_edges.get(&edge_key).unwrap().clone(); // Need to clone here
            if edge.len() < 2 {
                continue;
            }

            let tri1 = &triangles[edge[0]];
            let tri2 = &triangles[edge[1]];

            let common_edge = has_common_edge(tri1, tri2);
            if common_edge.is_none() {
                panic!("Expected common edge: {:?}, {:?}", tri1, tri2);
            }
            let (tri_ind1, tri_ind2) = common_edge.unwrap();

            // Reorder nodes so the common edge is the first two vertices
            let tri1 = vec![tri1[tri_ind1.0], tri1[tri_ind1.1], tri1[tri_ind1.2]]; // 1st and 2nd are common edge
            let tri2 = vec![tri2[tri_ind2.0], tri2[tri_ind2.2], tri2[tri_ind2.1]]; // 1st and 3rd are common edge

            // Check if triangle flip is needed
            let (flip_needed, ft1, ft2) = check_and_flip_triangle_pairs(&points, tri1, tri2);

            if flip_needed {
                // Remove the two triangles from the shared edges
                remove_triangle_from_common_edge(&mut shared_edges, &triangles, edge[0]);
                remove_triangle_from_common_edge(&mut shared_edges, &triangles, edge[1]);

                // Update the triangles
                triangles[edge[0]] = ft1;
                triangles[edge[1]] = ft2;

                // Add the two new triangles to the shared edges
                add_triangle_to_common_edge(&mut shared_edges, &triangles, edge[0]);
                add_triangle_to_common_edge(&mut shared_edges, &triangles, edge[1]);

                count += 1;
            }
        }

        // Prevent an infinite loop of triangle flipping
        if count > 0 && previous_configurations.contains(&triangles) {
            panic!("Cannot find delaunay arrangement");
        }

        previous_configurations.push(triangles.clone());

        if count == 0 {
            break;
        }

        if !node_ordering.unwrap() {
            // Reverse order of triangles to match input node order
            for tri in triangles.iter_mut() {
                tri.reverse();
            }
        }
    }
    triangles
}

/// Remove duplicate points from a set of points
fn remove_duplicate_points(points: ArrayView2<f64>) -> Array2<f64> {
    let mut unique_points: Vec<Vec<f64>> = vec![];
    for p in points.rows() {
        let mut is_unique = true;
        for up in unique_points.iter() {
            if p[0] == up[0] && p[1] == up[1] {
                is_unique = false;
                break;
            }
        }
        if is_unique {
            unique_points.push(p.to_vec());
        }
    }
    Array2::from_shape_vec(
        (unique_points.len(), 2),
        unique_points.into_iter().flatten().collect(),
    )
    .unwrap()
}

/// Calculate triangle area
///
/// # Arguments
/// * `points` - A 2D array of points
/// * `tri` - A vector of indices representing the triangle
///
/// # Returns
/// * The area of the triangle
fn herons_formula(points: &ArrayView2<f64>, tri: &Vec<usize>) -> f64 {
    // https://en.wikipedia.org/wiki/Heron%27s_formula#Numerical_stability
    let a = distance(points.row(tri[0]), points.row(tri[1])).sqrt();
    let b = distance(points.row(tri[1]), points.row(tri[2])).sqrt();
    let c = distance(points.row(tri[2]), points.row(tri[0])).sqrt();

    let x1 = (a + (b + c));
    let x2 = (c - (a - b));
    let x3 = (c + (a - b));
    let x4 = (a + (b - c));
    let x = x1 * x2 * x3 * x4;
    if x < 0.0 {
        return 0.0;
    }
    let area = 0.25 * x.sqrt();
    area
}

/// Remove zero-area triangles
fn remove_zero_area_tris(points: &ArrayView2<f64>, triangles: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut filtered_triangles: Vec<Vec<usize>> = vec![];
    for tri in triangles.iter() {
        let ang1 = calc_triangle_angle(points, vec![tri[2], tri[0], tri[1]]).unwrap();
        let ang2 = calc_triangle_angle(points, vec![tri[0], tri[1], tri[2]]).unwrap();
        let ang3 = calc_triangle_angle(points, vec![tri[1], tri[2], tri[0]]).unwrap();

        if ang1 == 0.0 || ang2 == 0.0 || ang3 == 0.0 {
            continue;
        }

        if ang1 == std::f64::consts::PI
            || ang2 == std::f64::consts::PI
            || ang3 == std::f64::consts::PI
        {
            continue;
        }

        if herons_formula(points, &tri) == 0.0 {
            continue;
        }

        filtered_triangles.push(tri.clone());
    }
    filtered_triangles
}

/// Calculate the S-Hull Delaunay triangulation of a set of 2d points.
///
/// S-hull: a fast sweep-hull routine for Delaunay triangulation by David Sinclair
/// http://www.s-hull.org/
///
/// # Arguments
///
#[pyfunction]
pub fn calculate_shull_2d<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
) -> &'py PyArray2<usize> {
    // Convert the input points to a 2D array
    let points = points.as_array();

    // Sort by radial distance to a seed point
    let seed_index: usize = 0;
    let radial_sorted = radial_distance_sort(points, points.row(seed_index));

    // Nearest point to seed
    let nearest_index = radial_sorted[1].1;

    // Find the third point that creates the smallest circumcircle
    let sorted_circum_circle = find_smallest_circumcircle(points, seed_index, nearest_index);
    if sorted_circum_circle.0 == f64::MAX {
        panic!("Invalid circumcircle error");
    }
    let mut third_index = sorted_circum_circle.1;

    // Order points to be right handed
    let cross_prod = right_hand_check(
        points.row(seed_index),
        points.row(nearest_index),
        points.row(third_index),
    );
    // Swap points if necessary
    let second_index: usize;
    if cross_prod < 0.0 {
        second_index = third_index;
        third_index = nearest_index;
    } else {
        second_index = nearest_index;
    }

    // Center of circumcircle
    let center = circum_circle_center(
        points.row(seed_index),
        points.row(second_index),
        points.row(third_index),
    );
    // Turn center into an array (required for distance calc below)
    let center = Array1::from_vec(vec![center.0, center.1]);

    // Sort points by distance from circum-circle centre
    let mut dists: Vec<(f64, usize)> = vec![];
    for (i, p) in points.rows().into_iter().enumerate() {
        if i == seed_index || i == second_index || i == third_index {
            continue;
        }
        dists.push((distance(p, center.view()), i));
    }
    // Sort `dists` vector from smallest to largest
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Extract indices from `dists` vector
    let order_to_add_pts = dists.iter().map(|x| x.1).collect::<Vec<usize>>();

    // Form triangles by sequentially adding points
    let (hull, triangles) = form_triangles(
        &points,
        vec![seed_index, second_index, third_index],
        order_to_add_pts,
    );

    // Flip adjacent pairs of triangles to meet Delaunay condition
    let delaunay_tris = flip_triangles(
        points, triangles, None, // Node ordering is known
    );

    // Remove zero area triangles
    let filtered_tris = remove_zero_area_tris(&points, delaunay_tris);

    // Turn `filtered_tris` into a 2d array
    let filtered_tris = Array2::from_shape_vec(
        (filtered_tris.len(), 3),
        filtered_tris.into_iter().flatten().collect(),
    )
    .unwrap();

    filtered_tris.into_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_shull")]
fn shull(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_shull_2d, m)?)?;
    Ok(())
}
