#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::{
    mesh_view_types::POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE,
    mesh_view_bindings as view_bindings,
}
#import bevy_pbr::mesh_view_types as types

const LEG_ROOTS_5 = array<f32, 5>(-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664);
const LEG_WEIGHTS_5 = array<f32, 5>(0.23692688505618897, 0.4786286704993665, 0.568888888888889, 0.4786286704993665, 0.23692688505618897);
const LEG_ROOTS_10 = array<f32, 10>(-0.9739065285171717, -0.8650633666889844, -0.6794095682990244, -0.4333953941292472, -0.14887433898163116, 0.14887433898163116, 0.4333953941292472, 0.6794095682990244, 0.8650633666889844, 0.9739065285171717);
const LEG_WEIGHTS_10 = array<f32, 10>(0.06667134430868714, 0.14945134915058053, 0.21908636251598224, 0.26926671930999674, 0.2955242247147533, 0.2955242247147533, 0.26926671930999674, 0.21908636251598224, 0.14945134915058053, 0.06667134430868714);
const LEG_ROOTS_15 = array<f32, 15>(-0.9879925180204854, -0.937273392400706, -0.8482065834104272, -0.7244177313601701, -0.5709721726085388, -0.3941513470775634, -0.20119409399743454, 0.0, 0.20119409399743454, 0.3941513470775634, 0.5709721726085388, 0.7244177313601701, 0.8482065834104272, 0.937273392400706, 0.9879925180204854);
const LEG_WEIGHTS_15 = array<f32, 15>(0.030753241996118154, 0.07036604748810715, 0.10715922046717176, 0.13957067792615432, 0.16626920581699411, 0.18616100001556224, 0.19843148532711163, 0.20257824192556137, 0.19843148532711163, 0.18616100001556224, 0.16626920581699411, 0.13957067792615432, 0.10715922046717176, 0.07036604748810715, 0.030753241996118154);

// --- Structures ---

struct StrandMeta { // Ensure this matches Rust exactly
    count: u32,     // Number of vertices in strand
    offset: u32,    // Start index in the original indices buffer
    pad1: u32,      // Padding to align to 16 bytes
    pad2: u32,      // Padding to align to 16 bytes
}

struct PushConstants { // Ensure this matches Rust and range covers all fields
    workgroup_offset: u32, // Abused for column width in this case
    num_elements: u32,    // Generic count (in this case num_strands)
    scan_load_base: u32,
    scan_save_base: u32,
    // Add other needed constants
}

var<push_constant> pc: PushConstants;

const PI = 3.14159265359;
const PI_HALF = PI / 2.0;
const SQRT_2_PI = sqrt(2.0 * PI);

const MAX_TEXTURE_EXT: u32 = #MAX_TEXTURE_EXTENT;
const WORKGROUP_SIZE: u32 = 64; // TODO: shaderdef

@group(0) @binding(#{VERTEX_BUFFER}) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(#{INDEX_BUFFER}) var<storage, read> indices: array<u32>;
@group(0) @binding(#{META_BUFFER}) var<storage, read> strand_metadata: array<StrandMeta>;
@group(0) @binding(#{VIEW_UNIFORM}) var<uniform> view: View;
@group(0) @binding(#{LIGHT_UNIFORM}) var<uniform> lights: types::Lights;
@group(0) @binding(#{CLUSTER_INDICES}) var<storage> clusterable_object_index_lists: types::ClusterLightIndexLists;
@group(0) @binding(#{CLUSTERABLE_OBJECTS}) var<storage> clusterable_objects: types::ClusterableObjects;
@group(0) @binding(#{CLUSTER_OFFSETS_AND_COUNTS}) var<storage> cluster_offsets_and_counts: types::ClusterOffsetsAndCounts;
@group(0) @binding(#{POINT_LIGHT_DEPTH_TEXTURE}) var point_shadow_textures_linear_sampler: sampler;
@group(0) @binding(#{DIRECTIONAL_LIGHT_DEPTH_TEXTURE}) var directional_shadow_textures_linear_sampler: sampler;
@group(0) @binding(#{OUTPUT_TEXTURE}) var output_texture: texture_storage_2d<rgba8unorm, write>;

// For reference because VsCode wgsl analyzer is broken.

// struct DirectionalCascade {
//     clip_from_world: mat4x4<f32>,
//     texel_size: f32,
//     far_bound: f32,
// }

// struct DirectionalLight {
//     cascades: array<DirectionalCascade, #{MAX_CASCADES_PER_LIGHT}>,
//     color: vec4<f32>,
//     direction_to_light: vec3<f32>,
//     // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
//     flags: u32,
//     soft_shadow_size: f32,
//     shadow_depth_bias: f32,
//     shadow_normal_bias: f32,
//     num_cascades: u32,
//     cascades_overlap_proportion: f32,
//     depth_texture_base_index: u32,
//     skip: u32,
// };

// const DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT: u32                  = 1u;
// const DIRECTIONAL_LIGHT_FLAGS_VOLUMETRIC_BIT: u32                       = 2u;
// const DIRECTIONAL_LIGHT_FLAGS_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE_BIT: u32 = 4u;

// struct Lights {
//     // NOTE: this array size must be kept in sync with the constants defined in bevy_pbr/src/render/light.rs
//     directional_lights: array<DirectionalLight, #{MAX_DIRECTIONAL_LIGHTS}u>,
//     ambient_color: vec4<f32>,
//     // x/y/z dimensions and n_clusters in w
//     cluster_dimensions: vec4<u32>,
//     // xy are vec2<f32>(cluster_dimensions.xy) / vec2<f32>(view.width, view.height)
//     //
//     // For perspective projections:
//     // z is cluster_dimensions.z / log(far / near)
//     // w is cluster_dimensions.z * log(near) / log(far / near)
//     //
//     // For orthographic projections:
//     // NOTE: near and far are +ve but -z is infront of the camera
//     // z is -near
//     // w is cluster_dimensions.z / (-far - -near)
//     cluster_factors: vec4<f32>,
//     n_directional_lights: u32,
//     spot_light_shadowmap_offset: i32,
//     environment_map_smallest_specular_mip_level: u32,
//     environment_map_intensity: f32,
// };

// --- Helpers ---

fn csch(x: f32) -> f32 {
    return 1.0 / sinh(x);
}

fn sin2(x: f32) -> f32 {
    return (1.0 - cos(2.0 * x)) / 2.0;
}

// Bravais Index
fn ior(eta: f32, theta: f32) -> f32 {
    return sqrt(pow(eta, 2.0) - sin2(theta)) / cos(theta);
}

fn fresnel(eta: f32, cos_theta: f32) -> f32 {
    // Schlick's approximation or full Fresnel equations
    let f0 = pow((1.0 - eta), 2.0) / pow((1.0 + eta), 2.0);
    let cos_theta_clamped = clamp(0.0, 1.0, cos_theta); // Ensure non-negative base for pow
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta_clamped, 5.0);
}

fn gaussian(x: f32, sigma: f32, mu: f32) -> f32 {
    let exponent = -0.5 * pow((x - mu) / sigma, 2.0);
    return (1.0 / (sigma * SQRT_2_PI)) * exp(exponent);
}

fn signed_angle_between(a: vec3<f32>, b: vec3<f32>, n: vec3<f32>) -> f32 {
    let angle = acos(dot(normalize(a), normalize(b)));
    let sign = sign(dot(cross(a, b), n));
    return angle * sign;
}

const FACTORIALS = array<f32, 10>(
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
); // steps 0-9

fn bessel_first_approx(x: f32, steps: u32) -> f32 {
    var y = 0.0;
    for (var i = 0u; i < steps; i = i + 1) {
        let numerator = pow(x, 2.0 * f32(i));
        let denominator = pow(4.0, f32(i)) * FACTORIALS[i] * FACTORIALS[i];
        y = y + numerator / denominator;
    }
    return y;
}

// σa,e = {0.419,0.697,1.37} // Eumelanin absorption
const SIGMA_AE = vec3<f32>(0.419, 0.697, 1.37);
// σa,p = {0.187,0.4,1.05} // Pheomelanin absorption
const SIGMA_AP = vec3<f32>(0.187, 0.4, 1.05);

const BETA = 0.4;
const BETA_SQR = BETA * BETA;
const BETA_P = 0.3;
const BETA_P_SQR = BETA_P * BETA_P;


const PATH_COUNT = 3u;
const QUAD_COUNT = 10u;
const LEG_ROOTS = LEG_ROOTS_10;
const LEG_WEIGHTS = LEG_WEIGHTS_10;

// notes:   mu_a :  absorption
//          h :     incident shift
//          alpha_p : cuticle tilt

// NOTE: Full fresnel equation for unpolarized light. Could be simplified with Schlick
fn fresnel_reflectance(eta_i: f32, eta_t: f32, cos_theta_i: f32) -> f32 {
    // Ref: model_building.ipynb

    let cos_theta_incident = abs(cos_theta_i);
    let cos_theta_incident_clamped = clamp(0.0, 1.0, cos_theta_incident);
    let sin_theta_incident_sq = 1.0 - cos_theta_incident_clamped * cos_theta_incident_clamped;
    let eta_ratio = eta_i / eta_t;

    let sin_theta_transmitted_sq = eta_ratio * eta_ratio * sin_theta_incident_sq;
    let cos_theta_transmitted = sqrt(1.0 - sin_theta_transmitted_sq);

    let r_parallel = ((eta_t * cos_theta_incident) - (eta_i * cos_theta_transmitted)) /
                     ((eta_t * cos_theta_incident) + (eta_i * cos_theta_transmitted));
    
    let r_perpendicular = ((eta_i * cos_theta_incident) - (eta_t * cos_theta_transmitted)) /
                          ((eta_i * cos_theta_incident) + (eta_t * cos_theta_transmitted));

    return (r_parallel * r_parallel + r_perpendicular * r_perpendicular) * 0.5;
}


fn F(eta: f32, cos_theta: f32) -> f32 {
    return fresnel_reflectance(1.0, eta, cos_theta);
}

fn A(p: u32, h_: f32, eta: f32, eta_prime: f32, mu_a_prime: vec3<f32>, theta_d: f32) -> vec3<f32> {
    // Ref: model_building.ipynb
    // h = np.clip(h, -1.0 + 1e-6, 1.0 - 1e-6) # Avoid domain errors at h = +/- 1
    let h = clamp(-1.0 + 1e-6, 1.0 - 1e-6, h_);
    // let sin_gamma_i = h;
    // let cos_gamma_i = sqrt(1.0 - sin_gamma_i * sin_gamma_i);
    // let sin_gamma_t = h / eta_prime;
    // let cos_gamma_t = sqrt(max(0.0, 1.0 - sin_gamma_t * sin_gamma_t));

    // let cos_incidence_angle_arg = cos(theta_d) * cos_gamma_i;
    // let cos_incidence_angle = clamp(-1.0, 1.0, cos_incidence_angle_arg);

    let gamma_t = asin(h / eta_prime);
    let gamma_i = asin(h);

    // let f = F(eta_prime, cos_incidence_angle);
    // fresnel alt
    let f = F(eta, acos(cos(theta_d) * cos(gamma_i)));

    let absorption_term_exponent = -2.0 * mu_a_prime * (1.0 + cos(2.0 * gamma_t)); // there likely is an error here. (cos^2)

    let T = exp(absorption_term_exponent);
    if p == 0 {
        return f * T * T; //vec3<f32>(1.0, 1.0, 1.0); // (1.0 - mu_a_prime);
    } else {
        let f_ = (1.0 - f);
        let fp = pow(f, f32(p - 1));
        let Tp = vec3<f32>(pow(T.x, f32(p)), pow(T.y, f32(p)), pow(T.z, f32(p)));
        return (f_ * f_) * fp * Tp;
    }
}
// Ref: model_building.ipynb
// # Gaussian Detector D_p (Equation 11) [cite: 119]
// # Implemented using a limited sum as suggested in paper Sec 6. [cite: 128]
// def gaussian_detector(beta_p, phi_delta, k_limit=15):
//     """Calculates the wrapped Gaussian detector D_p."""
//     variance = beta_p**2
//     if variance < 1e-6: # Avoid division by zero for smooth surface
//         # Approximate Dirac delta if variance is tiny
//         # A proper implementation would return infinity at phi_delta=0 and 0 otherwise
//         # For numerical purposes, return a large value if close to 0, else 0.
//         return np.where(np.abs(phi_delta % (2 * np.pi)) < 1e-3, 1e6, 0.0)

//     norm_factor = 1.0 / (np.sqrt(2 * np.pi) * beta_p) # Normalization for a single Gaussian
//     total_val = 0.0
//     for k in range(-k_limit, k_limit + 1):
//         angle = phi_delta - 2 * np.pi * k
//         total_val += np.exp(-angle**2 / (2 * variance))

//     # The paper mentions this relates to Elliptic Theta functions,
//     # suggesting the normalization might differ from sum of simple Gaussians.
//     # Let's stick to the sum of Gaussians approach implied by Eq 11 for now.
//     # Need to ensure it integrates to 1 over [0, 2pi]. This sum doesn't guarantee that.
//     # A common practice is to normalize the *result* numerically if needed,
//     # or use the proper Theta function implementation if accuracy is paramount.
//     # For now, return the unnormalized sum * norm_factor.
//     return norm_factor * total_val
fn gaussian_detector(beta_p: f32, phi_delta: f32, k_limit: u32) -> f32 {
    let variance = beta_p * beta_p;
    if variance < 1e-6 {
        return select(1e6, 0.0, abs(phi_delta % (2.0 * PI)) < 1e-3);
    };

    let norm_factor = 1.0 / (sqrt(2.0 * PI) * beta_p); // check whats better precision wise (inverseSqrt * 1/beta_p or this)
    var total_val = 0.0;
    let ik = i32(k_limit);
    for (var k = -ik; k <= ik; k = k + 1) {
        let angle = phi_delta - 2.0 * PI * f32(k); // NOTE: can save a mul with PI lut.
        total_val = total_val + exp(-angle * angle / (2.0 * variance));
    }
    return norm_factor * total_val;
}

// Ref: model_building.ipynb
// # Azimuthal shift Phi(p, h) (Equation 8) [cite: 95]
// def Phi(p, h, eta_prime):
//     """Calculates the azimuthal shift angle Phi."""
//     h = np.clip(h, -1.0 + 1e-6, 1.0 - 1e-6) # Avoid domain errors
//     gamma_t = np.arcsin(np.clip(h / eta_prime, -1.0, 1.0)) # Clip argument for safety
//     return 2 * p * gamma_t - 2 * np.arcsin(h) + p * np.pi

fn Phi(p: u32, h_: f32, eta_prime: f32) -> f32 {
    let h = clamp(-1.0 + 1e-6, 1.0 - 1e-6, h_);
    let gamma_t = asin(clamp(h / eta_prime, -1.0, 1.0));
    return 2.0 * f32(p) * gamma_t - 2.0 * asin(h) + f32(p) * PI;
}

fn Mp(v_long_val: f32, theta_i_: f32, theta_r: f32, alpha_p_shift: f32) -> f32 {
    var theta_r_shifted = theta_r + alpha_p_shift; // cuticle shift

    // Ref: model_building.ipynb
    // # Ensure angles are within valid range [-pi/2, pi/2] if needed, though paper doesn't explicitly state clipping here.
    // # Clamp arguments to avoid domain errors in trig functions if necessary
    // theta_i = np.clip(theta_i, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
    // theta_r_shifted = np.clip(theta_r_shifted, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)

    // cos_theta_i = np.cos(theta_i)
    // cos_theta_r = np.cos(theta_r_shifted)
    // sin_theta_i = np.sin(theta_i)
    // sin_theta_r = np.sin(theta_r_shifted)

    // # Check for zero variance (smooth surface) -> Dirac delta
    // if v < 1e-6:
    //     # Return large value if theta_r_shifted is near -theta_i, else 0
    //     # Need a tolerance delta
    //     delta_tolerance = np.radians(0.1) # e.g., 0.1 degrees
    //     return np.where(np.abs(theta_r_shifted + theta_i) < delta_tolerance, 1e6 / delta_tolerance, 0.0)

    let theta_i = clamp(-PI_HALF + 1e-6, PI_HALF - 1e-6, theta_i_);
    theta_r_shifted = clamp(-PI_HALF + 1e-6, PI_HALF - 1e-6, theta_r_shifted);

    let cos_theta_i = cos(-theta_i);
    let cos_theta_r = cos(theta_r_shifted);
    let sin_theta_i = sin(-theta_i);
    let sin_theta_r = sin(theta_r_shifted);

    if v_long_val < 1e-6 {
        let delta_tolerance = radians(0.1);
        return select(1e6 / delta_tolerance, 0.0, abs(theta_r_shifted + theta_i) < delta_tolerance);
    }

    let bessel_arg = (cos_theta_i * cos_theta_r) / v_long_val;
    let exp_arg = (sin_theta_i * sin_theta_r) / v_long_val;

    // var csch_val = 0.0;
    // if ((1.0 / v_long_val) < 1e-6) {
    //     csch_val = 2.0 * exp(-1.0 / v_long_val);
    // } else {
    //     csch_val = 1.0 / sinh(1.0 / v_long_val);
    // }
    let csch_val = csch(1.0 / v_long_val);

    let bessel_val = bessel_first_approx(bessel_arg, 5u);
    // let m_val = (csch_val / (2.0 * v_long_val)) * bessel_val * exp(exp_arg);
    // return m_val;

    // Debug
    // return exp(exp_arg); // looks good
    return bessel_val * exp(exp_arg); //
    // return (csch_val / (2.0 * v_long_val)); // 
}

fn Np(p: u32, phi: f32, theta_i: f32, theta_r: f32, eta_val: f32, mu_a_rgb_val: vec3<f32>, v_azim_val: f32, beta_azim_val: f32) -> vec3<f32> {
    let q_roots = LEG_ROOTS;
    let q_weights = LEG_WEIGHTS;
    
    // Ref:
    // sin_theta_i = np.sin(theta_i)
    // cos_theta_d = np.cos(theta_d)
    // # Clamp argument under sqrt to avoid domain error
    // eta_prime_arg_sq = eta_val**2 - sin_theta_i**2
    // eta_prime_arg = np.sqrt(np.maximum(0.0, eta_prime_arg_sq))
    // eta_prime = eta_prime_arg / cos_theta_d

    // # Reduced absorption mu_a' (Section 6, after Eq 14) [cite: 128]
    // cos_theta_i = np.cos(theta_i) # Paper uses cos(theta_t) but that depends on h. Using cos(theta_i) as approximation or per Zinke? Revisit paper. Text says mu_a' = mu_a / cos(theta_d) but uses theta_i in eta' derivation... Using cos(theta_d) based on text after Eq 14.
    // mu_a_prime = mu_a_rgb_val / cos_theta_d

    let theta_d = (theta_r - theta_i) * 0.5;
    let sin_theta_i = sin(theta_i);
    let cos_theta_d = cos(theta_d);
    let eta_prime_arg_sq = eta_val * eta_val - sin_theta_i * sin_theta_i;
    let eta_prime_arg = sqrt(max(0.0, eta_prime_arg_sq));
    let eta_prime = eta_prime_arg / cos_theta_d;
    let cos_theta_i = cos(theta_i);
    let mu_a_prime = mu_a_rgb_val / cos_theta_d;

    var integral_val = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < QUAD_COUNT; i = i + 1) {
        let h_val = q_roots[i];
        let weight = q_weights[i];
        let attenuation = A(p, h_val, eta_val, eta_prime, mu_a_prime, theta_d);
        let phi_shift = Phi(p, h_val, eta_prime);
        let detector = gaussian_detector(beta_azim_val, phi - phi_shift, 10u);
        integral_val = integral_val + attenuation * detector * weight;
    }

    // Apply the 1/2 factor from Equation 10 [cite: 113]
    return integral_val * 0.5;

}


fn weta_strand_bsdf(theta_i: f32, phi_i: f32, theta_r: f32, phi_r: f32, eta_val: f32, mu_a_rgb_val: vec3<f32>, v_long_val: f32, v_azim_val: f32, alpha_p_val: f32, specular_a_rgb_val: vec4<f32>) -> vec3<f32> {
    // for more information on this see: model_building.ipynb
    let beta_azim_val = sqrt(v_azim_val);
    var total_reflectance = Mp(v_long_val, theta_i, theta_r, alpha_p_val) * specular_a_rgb_val.xyz * specular_a_rgb_val.w;
    // var total_reflectance = vec3<f32>(0.0, 0.0, 0.0);
    for (var p = 0u; p < PATH_COUNT; p = p + 1) {
        let Np_val = Np(p, phi_i, theta_i, theta_r, eta_val, mu_a_rgb_val, v_azim_val, beta_azim_val);
        total_reflectance = total_reflectance + Np_val;
    }
    return total_reflectance;
}

fn marschner(point: vec4<f32>, direction: vec3<f32>, view_normal: vec3<f32>, light_normal: vec3<f32>, hair_color: vec4<f32>, specular_color: vec4<f32>, ao_intensity: f32) -> vec3<f32> {

    let u = direction;
    
    // Compute the key angles needed for Marschner model
    let theta_i = acos(dot(direction, light_normal));
    let theta_r = acos(dot(direction, view_normal));

    let light_projected = normalize(light_normal - dot(light_normal, u) * u);
    let view_projected = normalize(view_normal - dot(view_normal, u) * u);

    let phi = signed_angle_between(light_projected, view_projected, u);

    let sigma_a = 1.0 - hair_color.xyz; // Artist adjustable absorption coefficient
    // let sigma_a = SIGMA_AE * 0.5 + SIGMA_AP * 0.5; // TODO: use hair color to mix between eumelanin and pheomelanin
    let eta = 1.55; // Refractive index of hair
    let beta = 0.5; // Roughness of hair
    let alpha = 0.35; // Roughness of hair
    let shift = 0.02; // specular shift

    let v_long_val = alpha * alpha;
    let v_azim_val = beta * beta;

    let bcsdf = weta_strand_bsdf(theta_i, phi, theta_r, phi, eta, sigma_a, v_long_val, v_azim_val, shift, specular_color);
    return bcsdf;

    // Debug
    // let alpha_p_val = shift;
    // let Mp_val = Mp(v_long_val, theta_i, theta_r, alpha_p_val);
    // return Mp_val * vec3<f32>(1.0, 1.0, 1.0); // TODO: remove this debug line

    // return Np(0u, phi, theta_i, theta_r, eta, sigma_a, v_long_val, sqrt(v_azim_val)) + Np(1u, phi, theta_i, theta_r, eta, sigma_a, v_long_val, sqrt(v_azim_val)); // TODO: remove this debug line

}



@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn shade_strands(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3u,         
    @builtin(local_invocation_id) local_id: vec3u          
) {

    let strand_id = workgroup_id.x;
    let segment_id = local_id.x;
    if strand_id >= pc.num_elements {
        return;
    }

    let strand_meta = strand_metadata[strand_id];
    let strand_offset = strand_meta.offset;
    let strand_count = strand_meta.count;

    if segment_id >= strand_count {
        return;
    }

    let light_count = lights.n_directional_lights;

    // var strand_absorption_color = vec4<f32>(0.44, 0.15, 0.05, 0.5);
    var strand_absorption_color = vec4<f32>(0.6, 0.1, 0.05, 0.5);
    var strand_specular_color = vec4<f32>(0.93, 0.48, 0.375, 2.0);

    let ambient_factor = 0.05;
    let ao_factor = 0.3;

    for (var i = 0u; i < strand_count; i = i + WORKGROUP_SIZE) { // in case we have more segments than workgroup size
        let segment_offset = segment_id + i;
        var next_point = segment_offset + 1u;
        if next_point >= strand_count {
            // we use the previous point to indicate fibre direction and reverse it
            // we still have to shade the tip of the strand
            next_point = strand_count - 1u;
        }
        let index = indices[strand_offset + segment_offset];
        let i_dir = indices[strand_offset + next_point];

        let vertex = vertices[index];
        let next_vertex = vertices[i_dir];

        // fiber direction
        var U = normalize(next_vertex.xyz - vertex.xyz);
        if next_point < segment_offset {
            // invert direction for the tip
            U = -U;
        }

        // view direction
        let tangent = normalize(next_vertex.xyz - vertex.xyz);
        let camera_dir = normalize(view.world_position - (vertex.xyz + next_vertex.xyz) / 2.0);
        let binormal = normalize(cross(tangent, camera_dir));
        let V = normalize(cross(binormal, tangent));

        let ao_intensity = smoothstep(1.0, 0.0, max(0.0, 1.0 - (f32(segment_offset) / f32(strand_count))));

        // TODO: we can theoretically split this across multiple workgroups
        for (var j = 0u; j < light_count; j = j+1) {
            let light: types::DirectionalLight = lights.directional_lights[j];
            let light_flags = light.flags;
            let L = normalize(light.direction_to_light); // TODO: point lights, spot lights etc. this would be normalize(light.position - strand_point.position);
            
            let bcsdf = marschner(vertex, L, V, U, strand_absorption_color, strand_specular_color, ao_intensity);

            var c = bcsdf * (light.color.xyz * 0.005); // * dot(V, L);
            c = mix(c, strand_absorption_color.xyz * ambient_factor + (lights.ambient_color.xyz / 255.0) * ambient_factor, ambient_factor); // ambient TODO: ambient lighting

            strand_absorption_color =  vec4<f32>(c.xyz, strand_absorption_color.w);

        }
        let out_row = strand_id % MAX_TEXTURE_EXT;
        let out_col = strand_id / MAX_TEXTURE_EXT;
        let y_coord = out_row;
        let x_coord = out_col * (pc.workgroup_offset + 1) + segment_offset; // remember that workgroup_offset is abused for column width in this context
        textureStore(output_texture, vec2<i32>(i32(x_coord), i32(y_coord)), strand_absorption_color);
    }
}