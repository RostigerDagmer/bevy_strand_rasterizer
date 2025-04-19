const PI: f32 = 3.14159265358979323846; // TODO: use bevy builtin or move to common.wgsl

fn solve_cubic_3d(a: f32, b: f32, c: f32, d: f32) -> vec3<f32> {
    // Coefficients for the cubic equation
    let eps = 1e-8;
    let alpha = b / (3.0 * a);
    let a_sq = a * a;
    let b_sq = b * b;

    let a_cu = a_sq * a;
    let b_cu = b_sq * b;
    let p = (3.0 * a * c - b_sq) / (3.0 * a_sq);
    let q = (2.0 * b_cu - 9.0 * a * b * c + 27.0 * a_sq * d) / (27.0 * a_cu);

    let half_q = q / 2.0;
    let third_p = p / 3.0;
    let discriminant = half_q * half_q + third_p * third_p * third_p;

    if abs(discriminant) < eps {
        // Three real roots (double root)
        if abs(p) < eps && abs(q) < eps {
            // All roots are equal
            return vec3<f32>(-alpha, -alpha, -alpha);
        }
        let u = pow(-half_q, 1.0 / 3.0);
        let y1 = 2.0 * u;
        let y2 = -u;
        return vec3<f32>(y1, y2, y2) - alpha;
    } else if (discriminant > 0.0) {
        // One real root and two complex conjugate roots
        let sqrt_D = sqrt(discriminant);
        let u = pow(-half_q + sqrt_D, 1.0 / 3.0);
        let v = pow(-half_q - sqrt_D, 1.0 / 3.0);
        let y1 = u + v;
        return vec3<f32>(y1, -1.0, -1.0) - alpha; // we are only interested in roots in [0, 1]
    } else {
        // Three distinct real roots
        let r = 2.0 * sqrt(-third_p);
        let theta = acos(-half_q / sqrt(-third_p * third_p * third_p));
        let y0 = r * cos(theta / 3.0);
        let y1 = r * cos((theta + 2.0 * PI) / 3.0);
        let y2 = r * cos((theta + 4.0 * PI) / 3.0);
        return vec3<f32>(y0, y1, y2) - alpha;
    }
}

fn tj2d(ti: f32, pi: vec2<f32>, pj: vec2<f32>, alpha: f32) -> f32 {
    return ti + pow(length(pi - pj), alpha);
}

// Catmull-Rom spline interpolation function
fn catmull_rom_spline_point2d(
    p0: vec2<f32>,  // Control point 0
    p1: vec2<f32>,  // Control point 1 (start of segment)
    p2: vec2<f32>,  // Control point 2 (end of segment)
    p3: vec2<f32>,  // Control point 3
    t: f32,         // Parameter value [0,1]
    alpha: f32      // Tension parameter, default 0.5
) -> vec2<f32> {
    // Calculate chord lengths with alpha power
    
    let t0 = 0.0;
    let t1 = tj2d(t0, p0, p1, alpha);
    let t2 = tj2d(t1, p1, p2, alpha);
    let t3 = tj2d(t2, p2, p3, alpha);
    
    // Remap t to the local segment [t1, t2]
    let t_remapped = t * (t2 - t1) + t1;
    
    // Interpolate points
    let a1 = ((t1 - t_remapped) / (t1 - t0)) * p0 + ((t_remapped - t0) / (t1 - t0)) * p1;
    let a2 = ((t2 - t_remapped) / (t2 - t1)) * p1 + ((t_remapped - t1) / (t2 - t1)) * p2;
    let a3 = ((t3 - t_remapped) / (t3 - t2)) * p2 + ((t_remapped - t2) / (t3 - t2)) * p3;
    
    let b1 = ((t2 - t_remapped) / (t2 - t0)) * a1 + ((t_remapped - t0) / (t2 - t0)) * a2;
    let b2 = ((t3 - t_remapped) / (t3 - t1)) * a2 + ((t_remapped - t1) / (t3 - t1)) * a3;
    
    return ((t2 - t_remapped) / (t2 - t1)) * b1 + ((t_remapped - t1) / (t2 - t1)) * b2;
}

fn tj3d(ti: f32, pi: vec3<f32>, pj: vec3<f32>, alpha: f32) -> f32 {
    return ti + pow(length(pj - pi), alpha);
}

fn catmull_rom_t(
    p0: vec3<f32>,  // Control point 0
    p1: vec3<f32>,  // Control point 1 (start of segment)
    p2: vec3<f32>,  // Control point 2 (end of segment)
    p3: vec3<f32>,  // Control point 3
    alpha: f32      // Tension parameter, default 0.5
) -> vec4<f32> {
    let t0 = 0.0;
    let t1 = tj3d(t0, p0, p1, alpha);
    let t2 = tj3d(t1, p1, p2, alpha);
    let t3 = tj3d(t2, p2, p3, alpha);
    
    return vec4<f32>(t0, t1, t2, t3);
}
fn catmull_rom_coefficients_3d(
    p0: vec3<f32>,  // Control point 0
    p1: vec3<f32>,  // Control point 1 (start of segment)
    p2: vec3<f32>,  // Control point 2 (end of segment)
    p3: vec3<f32>,  // Control point 3
    t: vec4<f32>,   // Times (should be obtained from tj3d -> catmull_rom_t)
) -> mat4x3<f32> {
    
    let m1 = (t.z - t.y) / (t.z - t.x) * (p2 - p0);
    let m2 = (t.z - t.y) / (t.w - t.y) * (p3 - p1);

    let D = p1;
    let C = m1;
    let B = -3.0 * p1 + 3.0 * p2 - 2.0 * m1 - m2;
    let A = 2.0 * p1 - 2.0 * p2 + m1 + m2;
    
    return mat4x3<f32>(A, B, C, D);
}

fn catmull_rom_spline_roots_3d(
    coeffs: mat4x3<f32>, // Coefficients from catmull_rom_coefficients_3d
    plane: mat3x3<f32>, // Plane defined by offset P and U x V creating normal N
) -> vec3<f32> {
    // returns roots of the cubic equation
    let N = normalize(cross(plane[1], plane[2]));
    let d = dot(-N, plane[0]);
    let a_poly = dot(N, coeffs[0]);
    let b_poly = dot(N, coeffs[1]);
    let c_poly = dot(N, coeffs[2]);
    let d_poly = dot(N, coeffs[3]) + d;
    return solve_cubic_3d(a_poly, b_poly, c_poly, d_poly);
}


fn intersect_catmull_rom_spline_3d(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, plane: mat3x3<f32>, alpha: f32) -> mat4x3<f32> {
    // returns up to three possible intersections of the Catmull-Rom spline with a plane
    // last row in return matrix is a mask indicating which rows are valid.
    // find possible values of t
    let ts = catmull_rom_t(p0, p1, p2, p3, alpha);
    let coeffs = catmull_rom_coefficients_3d(p0, p1, p2, p3, ts);
    let roots = catmull_rom_spline_roots_3d(coeffs, plane);

    var i1 = vec3<f32>(0.0, 0.0, 0.0);
    var i2 = vec3<f32>(0.0, 0.0, 0.0);
    var i3 = vec3<f32>(0.0, 0.0, 0.0);
    var mask = vec3<f32>(0.0, 0.0, 0.0);
    // Check if roots are within the range [0, 1]
    if roots.x >= 0.0 && roots.x <= 1.0 {
        mask.x = 1.0;
        let point = catmull_rom_spline_point3d(p0, p1, p2, p3, ts, roots.x);
        i1 = recover_intersection_point(plane, point);
    }
    if roots.y >= 0.0 && roots.y <= 1.0 {
        mask.y = 1.0;
        let point = catmull_rom_spline_point3d(p0, p1, p2, p3, ts, roots.y);
        i2 = recover_intersection_point(plane, point);
    }
    if roots.z >= 0.0 && roots.z <= 1.0 {
        mask.z = 1.0;
        let point = catmull_rom_spline_point3d(p0, p1, p2, p3, ts, roots.z);
        i3 = recover_intersection_point(plane, point);
    }

    // Return the intersection points and the mask
    return mat4x3<f32>(i1, i2, i3, mask);
}


fn closest_point(points_with_mask: mat4x3<f32>, p: vec3<f32>) -> vec3<f32> {
    // Find the closest point to p from the points_with_mask matrix
    let mask = points_with_mask[3];
    let points = mat3x3<f32>(points_with_mask[0], points_with_mask[1], points_with_mask[2]);
    if length(mask) == 1.0 {
        return points * mask;
    }
    let dot_products = p * points;
    let distances = abs(dot_products); // absolute dot product
    // Conditions (using <= helps prioritize lower indices in ties)
    let x_le_y : bool = mask.x > 0.0 && distances.x <= distances.y;
    let x_le_z : bool = mask.x > 0.0 && distances.x <= distances.z;
    // Need to compare y and z only if x isn't the minimum, prioritize y in case of tie
    let y_le_z : bool = mask.y > 0.0 && distances.y <= distances.z;

    // Determine the index
    let is_x_min : bool = x_le_y && x_le_z;          // True if x is the minimum (or tied for min with lower index)
    let is_y_better_than_z : bool = y_le_z && mask.y > 0.0;         // True if y <= z and y unmasked

    // Select between index 1 and 2, assuming x is NOT the minimum
    let index12 : u32 = select(2u, 1u, is_y_better_than_z); // If y<=z pick 1u, else pick 2u

    // Select between index 0 and the winner of (1 vs 2)
    let min_index : u32 = select(index12, 0u, is_x_min); // If x is min pick 0u, else pick index12

    // Select the column using the found index
    let closest_column : vec3<f32> = points[min_index];
    return closest_column;
}


fn recover_intersection_point(plane: mat3x3<f32>, point: vec3<f32>) -> vec3<f32> {
    // Recover the intersection point on the plane
    let P0 = plane[0]; // Point on the plane
    let U = plane[1];
    let V = plane[2];
    let Usq = dot(U, U);
    let Vsq = dot(V, V);
    let UV = dot(U, V);
    let VU = dot(V, U);

    let det_G = Usq * Vsq - UV * UV;
    let G_inv = mat2x2<f32>(
        Vsq / det_G, -UV / det_G, -VU / det_G, Usq / det_G
    );
    let X_ = point - P0;
    let b = vec2<f32>(dot(U, X_), dot(V, X_));
    let y = G_inv * b;
    return (y.x * U + y.y * V) + P0; // Recover the point on the plane
}


// 3D version of Catmull-Rom spline interpolation
fn catmull_rom_spline_point3d(
    p0: vec3<f32>,  // Control point 0
    p1: vec3<f32>,  // Control point 1 (start of segment)
    p2: vec3<f32>,  // Control point 2 (end of segment)
    p3: vec3<f32>,  // Control point 3
    ts: vec4<f32>, // Times (should be obtained from tj3d -> catmull_rom_t)
    t: f32,         // Parameter value [0,1]
) -> vec3<f32> {
    // Remap t to the local segment [t1, t2]
    let t_remapped = t * (ts.z - ts.y) + ts.y;
    
    let d_t1_t0 = ts.y - ts.x;
    let d_t2_t1 = ts.z - ts.y;
    let d_t3_t2 = ts.w - ts.z;
    let d_t2_t0 = ts.z - ts.x;
    let d_t3_t1 = ts.w - ts.y;

    // Interpolate points
    let a1 = ((ts.y - t_remapped) / d_t1_t0) * p0 + ((t_remapped - ts.x) / d_t1_t0) * p1;
    let a2 = ((ts.z - t_remapped) / d_t2_t1) * p1 + ((t_remapped - ts.y) / d_t2_t1) * p2;
    let a3 = ((ts.w - t_remapped) / d_t3_t2) * p2 + ((t_remapped - ts.z) / d_t3_t2) * p3;
    
    let b1 = ((ts.z - t_remapped) / d_t2_t0) * a1 + ((t_remapped - ts.x) / d_t2_t0) * a2;
    let b2 = ((ts.w - t_remapped) / d_t3_t1) * a2 + ((t_remapped - ts.y) / d_t3_t1) * a3;
    
    return ((ts.z - t_remapped) / d_t2_t1) * b1 + ((t_remapped - ts.y) / d_t2_t1) * b2;
}

fn catmull_rom_C_short(B: mat2x3<f32>, t_l: vec3<f32>, t_r: vec3<f32>) -> vec3<f32> {
    // this is equivalent to catmull_rom_spline_point3d
    let C = B * vec2<f32>(t_l.y, t_r.y);
    return C;
}

fn catmull_rom_T_a(ts: vec4<f32>) -> vec3<f32> {
    return 1.0 / (ts.yzw - ts.xyz);
}

fn catmull_rom_t_l(ts: vec4<f32>, t_remapped: f32) -> vec3<f32> {
    return ts.yzw - t_remapped;
}

fn catmull_rom_t_r(ts: vec4<f32>, t_remapped: f32) -> vec3<f32> {
    return t_remapped - ts.xyz;
}

fn catmull_rom_A(
    p0: vec3<f32>,  // Control point 0
    p1: vec3<f32>,  // Control point 1 (start of segment)
    p2: vec3<f32>,  // Control point 2 (end of segment)
    p3: vec3<f32>,  // Control point 3
    ts: vec4<f32>, // Times (should be obtained from tj3d -> catmull_rom_t)
    t: f32,         // Parameter value [0,1]
) -> mat3x3<f32> {

    let t_remapped = t * (ts.z - ts.y) + ts.y;

    let T_a = catmull_rom_T_a(ts);
    let t_l = catmull_rom_t_l(ts, t_remapped);
    let t_r = catmull_rom_t_r(ts, t_remapped);
    let P_l = mat3x3<f32>(
        p0, p1, p2
    );
    let P_r = mat3x3<f32>(
        p1, p2, p3
    );
    let A = P_l * (t_l * T_a) + P_r * (t_r * T_a);
    return A;
}

fn catmull_rom_A_short(
    p0: vec3<f32>,  // Control point 0
    p1: vec3<f32>,  // Control point 1 (start of segment)
    p2: vec3<f32>,  // Control point 2 (end of segment)
    p3: vec3<f32>,  // Control point 3
    T_a: vec3<f32>, // output T_a(ts)
    t_l: vec3<f32>, // output t_l(ts, t_remapped)
    t_r: vec3<f32>, // output t_r(ts, t_remapped)
) -> mat3x3<f32> {
    let P_l = mat3x3<f32>(
        p0, p1, p2
    );
    let P_r = mat3x3<f32>(
        p1, p2, p3
    );
    let A = P_l * (t_l * T_a) + P_r * (t_r * T_a);
    return A;
}

fn catmull_rom_B(
    A: mat3x3<f32>,
    ts: vec4<f32>, // Times (should be obtained from tj3d -> catmull_rom_t)
    t: f32,         // Parameter value [0,1]
) -> mat2x3<f32> {
    let t_remapped = t * (ts.z - ts.y) + ts.y;
    let t_l = catmull_rom_t_l(ts, t_remapped);
    let t_r = catmull_rom_t_r(ts, t_remapped);
    let T_b = 1.0 / (t_l.zw - t_l.xy);
    let B = mat2x3<f32>(A[0], A[1]) * (t_l.yz * T_b) + mat2x3<f32>(A[1], A[2]) * (t_r.xy * T_b);
    return B;
}

fn catmull_rom_B_short(
    A: mat3x3<f32>,
    t_l: vec4<f32>,
    t_r: vec4<f32>,
) -> mat2x3<f32> {
    let T_b = 1.0 / (t_l.zw - t_l.xy);
    let B = mat2x3<f32>(A[0], A[1]) * (t_l.yz * T_b) + mat2x3<f32>(A[1], A[2]) * (t_r.xy * T_b);
    return B;
}

fn spline_derivative_at(
    p0: vec3<f32>,  // Control point 0
    p1: vec3<f32>,  // Control point 1 (start of segment)
    p2: vec3<f32>,  // Control point 2 (end of segment)
    p3: vec3<f32>,  // Control point 3
    A: mat3x3<f32>,
    B: mat2x3<f32>,
    ts: vec4<f32>, // Times (should be obtained from tj3d -> catmull_rom_t)
    t: f32,         // Parameter value [0,1])
) -> vec3<f32> {
    // refer to catmull_rom.ipynb for details
    let T_ = 1.0 / (ts.yzw - ts.xyz);
    let P = mat3x3<f32>(
        p1 - p0,
        p2 - p1,
        p3 - p2
    );
    let A_ = mat3x3<f32>(
        P[0] * T_.x,
        P[1] * T_.y,
        P[2] * T_.z,
    );
    let dt1 = vec4<f32>(ts.zw - t, t - ts.xy);
    let dt2 = ts.zwz - ts.xyy;

    let b1 = vec3<f32>(1.0, dt1.xz) / dt2.x;
    let b2 = vec3<f32>(1.0, dt1.yw) / dt2.y;

    let A_1 = mat3x3<f32>(
        A[1] - A[0],
        A_[0],
        A_[1],
    );
    let A_2 = mat3x3<f32>(
        A[2] - A[1],
        A_[1],
        A_[2]
    );
    let B_ = mat2x3<f32>(
        b1 * A_1,
        b2 * A_2
    );

    let c = vec3<f32>(1.0, dt1.xw) / dt2.z;
    return c * mat3x3<f32>(B[1] - B[0], B_[0], B_[1]);

}


// helper
fn min_root_above(roots: vec3<f32>, t: f32) -> vec3<f32> {
    // all roots below t or 0.0 become 1e38
    // Find the minimum root above t
    let sentinel = 1e38;
    let larger = roots > vec3(t);
    let limit = roots < vec3(1.0);
    let mask = larger && limit;
    let maxxed = select(roots, vec3(sentinel), mask);
    return maxxed;
}