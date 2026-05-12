#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::{
    mesh_view_types::POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE,
}
#import bevy_pbr::mesh_view_types as types
#import "shaders/shading_LUTs.wgsl"::{
    LEG_ROOTS_5,
    LEG_WEIGHTS_5,
    LEG_ROOTS_10,
    LEG_WEIGHTS_10,
    LEG_ROOTS_15,
    LEG_WEIGHTS_15,
}
#import "shaders/types.wgsl"::{
    DevicePtr,
    FroxelConfig,
    StrandMeta,
    StrandMaterial,
    Vertices,
    Indices,
    Materials,
    Meta,
    PushConstants,
    StrandInstance,
}
#import bevy_vsms::virtual_surface_types::{
    VirtualPageTableEntry,
    VirtualPageTableMetaRow,
    vsms_virtual_page_table_address,
}

#import "shaders/task_contract.wgsl"::{
    BinningTask,
    unpack_binning_frustum,
}

#import "shaders/queues.wgsl"::{
    BinningQueue,
}

#import "shaders/common.wgsl"::{
    DOM_GAMMA,
    // PI,
    // PI_HALF,
    // SQRT_2_PI,
    is_valid_ptr,
    find_clip_bounds,
    normalize_depth01,
    world_to_screen_raw,
}

const PI = 3.14159265359;
const PI_HALF = PI / 2.0;
const SQRT_2_PI = 2.5066282746310002;

const MAX_TEXTURE_EXT: u32 = #MAX_TEXTURE_EXTENT;
const WORKGROUP_SIZE: u32 = #WORKGROUP_SIZE;
const SIZEOF_METADATA: u32 = #SIZEOF_METADATA;
const SIZEOF_MATERIAL: u32 = #SIZEOF_MATERIAL;
const SIZEOF_VERTEX: u32 = #SIZEOF_VERTEX;
const DOM_SLICES: u32 = #{NUM_DOM_SLICES};
const INVALID_PTR: u32 = 0xFFFFFFFFu;
const VSMS_OPACITY_POOL_TEXTURE_COUNT: u32 = #{VSMS_OPACITY_POOL_TEXTURE_COUNT};
const VSMS_DEPTH_POOL_TEXTURE_COUNT: u32 = #{VSMS_DEPTH_POOL_TEXTURE_COUNT};

var<push_constant> pc: PushConstants;
struct FrustumDesc {
    screen_width: u32,
    screen_height: u32,
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
    bucket_base: u32,
    bucket_count: u32,
    kind: u32,
    coarse_depth_tile_base: u32,
    coarse_depth_tile_count: u32,
    coarse_tiles_x: u32,
    coarse_tiles_y: u32,
    cascade_index: u32,
    fine_depth_tile_base: u32,
    pad1: u32,
    pad2: u32,
}

@group(#{BIND_ARRAYS}) @binding(#{VERTICES}) var<storage, read_write> vertices: binding_array<Vertices>;
@group(#{BIND_ARRAYS}) @binding(#{INDICES}) var<storage, read_write> indices: binding_array<Indices>;
@group(#{BIND_ARRAYS}) @binding(#{STRAND_METADATA}) var<storage, read_write> strand_metadata: binding_array<Meta>;
@group(#{BIND_ARRAYS}) @binding(#{STRAND_MATERIALS}) var<storage, read_write> materials: binding_array<Materials>;

@group(#{PAGE_TABLES}) @binding(#{VERTICES}) var<storage, read_write> t_vertices: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{INDICES}) var<storage, read_write> t_indices: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{STRAND_METADATA}) var<storage, read_write> t_strand_metadata: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{STRAND_MATERIALS}) var<storage, read_write> t_materials: array<DevicePtr>;

@group(#{SHADING_GROUP}) @binding(#{VIEW_UNIFORM}) var<uniform> view: View;
@group(#{SHADING_GROUP}) @binding(#{LIGHT_UNIFORM}) var<uniform> lights: types::Lights;
@group(#{SHADING_GROUP}) @binding(#{BINNING_QUEUE}) var<storage, read> binning_queue: BinningQueue;
@group(#{SHADING_GROUP}) @binding(#{FRUSTUM_TABLE}) var<storage, read> frustum_table: array<FrustumDesc>;
@group(#{SHADING_GROUP}) @binding(#{OUTPUT_TEXTURE}) var output_texture: texture_storage_2d_array<rgba8unorm, write>;
@group(#{SHADING_GROUP}) @binding(#{STRAND_INSTANCES}) var<storage, read> strand_instances: array<StrandInstance>;
@group(#{SHADING_GROUP}) @binding(#{SHADOW_DOM_SURFACE_IDS}) var<storage, read> shadow_dom_surface_ids: array<vec2<u32>>;
@group(#{SHADING_GROUP}) @binding(#{SHADOW_HISTORY_PREV}) var shadow_history_prev: texture_2d_array<f32>;
@group(#{SHADING_GROUP}) @binding(#{SHADOW_HISTORY_NEXT}) var shadow_history_next: texture_storage_2d_array<rgba16float, write>;

@group(#{VSMS_OPACITY_WRITE_GROUP}) @binding(#{VSMS_POOL_TEXTURE_BINDING}) var shadow_opacity_maps: binding_array<texture_3d<f32> >;
@group(#{VSMS_OPACITY_WRITE_GROUP}) @binding(#{VSMS_POOL_SAMPLER_BINDING}) var shadow_opacity_sampler: sampler;
@group(#{VSMS_DEPTH_WRITE_GROUP}) @binding(#{VSMS_POOL_TEXTURE_BINDING}) var shadow_depth_maps: binding_array<texture_2d_array<f32> >;
@group(#{VSMS_DEPTH_WRITE_GROUP}) @binding(#{VSMS_POOL_SAMPLER_BINDING}) var shadow_depth_sampler: sampler;

@group(#{VSMS_OPACITY_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_META_BINDING}) var<storage, read> opacity_virtual_meta: array<VirtualPageTableMetaRow>;
@group(#{VSMS_OPACITY_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_PAGE_TABLE_BINDING}) var<storage, read> opacity_virtual_pages: array<VirtualPageTableEntry>;
@group(#{VSMS_DEPTH_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_META_BINDING}) var<storage, read> depth_virtual_meta: array<VirtualPageTableMetaRow>;
@group(#{VSMS_DEPTH_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_PAGE_TABLE_BINDING}) var<storage, read> depth_virtual_pages: array<VirtualPageTableEntry>;

fn shading_atlas_coord(linear_idx: u32, dims: vec2<u32>) -> vec2<i32> {
    return vec2<i32>(i32(linear_idx % dims.x), i32(linear_idx / dims.x));
}

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

fn light_frustum_from_layer(light_layer: u32) -> u32 {
    var layer = 0u;
    for (var frustum_id = 0u; frustum_id < arrayLength(&frustum_table); frustum_id = frustum_id + 1u) {
        if frustum_table[frustum_id].kind != 1u {
            continue;
        }
        if layer == light_layer {
            return frustum_id;
        }
        layer = layer + 1u;
    }
    return INVALID_PTR;
}

fn cascade_index_for_frustum(frustum_id: u32, light_layer: u32) -> u32 {
    if frustum_id >= arrayLength(&frustum_table) || light_layer >= lights.n_directional_lights {
        return 0u;
    }
    let num_cascades = lights.directional_lights[light_layer].num_cascades;
    if num_cascades == 0u {
        return 0u;
    }
    return min(frustum_table[frustum_id].cascade_index, num_cascades - 1u);
}

fn sample_shadow_dom_visibility(p_world: vec3<f32>, light_layer: u32) -> f32 {
    let frustum_id = light_frustum_from_layer(light_layer);
    if frustum_id == INVALID_PTR || frustum_id >= arrayLength(&frustum_table) || frustum_id >= arrayLength(&shadow_dom_surface_ids) {
        return 1.0;
    }
    let surface_ids = shadow_dom_surface_ids[frustum_id];
    let opacity_surface_id = surface_ids.x;
    let depth_surface_id = surface_ids.y;
    if opacity_surface_id == INVALID_PTR || depth_surface_id == INVALID_PTR {
        return 1.0;
    }
    if opacity_surface_id >= arrayLength(&opacity_virtual_meta) || depth_surface_id >= arrayLength(&depth_virtual_meta) {
        return 1.0;
    }
    let opacity_row = opacity_virtual_meta[opacity_surface_id];
    let depth_row = depth_virtual_meta[depth_surface_id];
    if opacity_row.entry_count == 0u || depth_row.entry_count == 0u {
        return 1.0;
    }

    let desc = frustum_table[frustum_id];
    let viewport = vec4<f32>(0.0, 0.0, f32(desc.screen_width), f32(desc.screen_height));
    let cascade_index = cascade_index_for_frustum(frustum_id, light_layer);
    let light_clip_from_world = lights.directional_lights[light_layer].cascades[cascade_index].clip_from_world;
    let raw = world_to_screen_raw(vec4<f32>(p_world, 1.0), light_clip_from_world, viewport);
    if raw.x < 0.0 || raw.y < 0.0 || raw.x >= viewport.z || raw.y >= viewport.w || raw.z < 0.0 || raw.z > 1.0 {
        return 1.0;
    }

    let px = vec2<u32>(u32(raw.x), u32(raw.y));
    let opacity_page_size = vec3<u32>(
        max(opacity_row.page_size_x, 1u),
        max(opacity_row.page_size_y, 1u),
        max(opacity_row.page_size_z, 1u),
    );
    let depth_page_size = vec2<u32>(
        max(depth_row.page_size_x, 1u),
        max(depth_row.page_size_y, 1u),
    );
    let opacity_tile = vec3<u32>(px.x / opacity_page_size.x, px.y / opacity_page_size.y, 0u);
    let depth_tile = vec3<u32>(px.x / depth_page_size.x, px.y / depth_page_size.y, 0u);
    let opacity_addr = vsms_virtual_page_table_address(opacity_row, opacity_tile, 0u, 0u);
    let depth_addr = vsms_virtual_page_table_address(depth_row, depth_tile, 0u, 0u);
    if opacity_addr.valid == 0u || depth_addr.valid == 0u {
        return 1.0;
    }
    if opacity_addr.entry_index >= arrayLength(&opacity_virtual_pages) || depth_addr.entry_index >= arrayLength(&depth_virtual_pages) {
        return 1.0;
    }

    let opacity_entry = opacity_virtual_pages[opacity_addr.entry_index];
    let depth_entry = depth_virtual_pages[depth_addr.entry_index];
    if opacity_entry.valid == 0u || depth_entry.valid == 0u {
        return 1.0;
    }
    if opacity_entry.physical_index >= VSMS_OPACITY_POOL_TEXTURE_COUNT || depth_entry.physical_index >= VSMS_DEPTH_POOL_TEXTURE_COUNT {
        return 1.0;
    }

    let opacity_px = vec2<i32>(i32(px.x % opacity_page_size.x), i32(px.y % opacity_page_size.y));
    let depth_px = vec2<i32>(i32(px.x % depth_page_size.x), i32(px.y % depth_page_size.y));
    let z0 = textureLoad(shadow_depth_maps[i32(depth_entry.physical_index)], depth_px, 0, 0).x;
    if z0 <= 0.0 {
        return 1.0;
    }

    let z = normalize_depth01(raw.z);
    if z > z0 - 1e-3 {
        return 1.0;
    }
    let u = pow(clamp(max(0.0, z0 - z) / max(z0, 1e-6), 0.0, 1.0), DOM_GAMMA);
    let slice_f = u * f32(DOM_SLICES);
    let slice = min(u32(floor(slice_f)), min(DOM_SLICES, opacity_page_size.z) - 1u);
    let opacity = textureLoad(
        shadow_opacity_maps[i32(opacity_entry.physical_index)],
        vec3<i32>(opacity_px, i32(slice)),
        0,
    ).x;
    return clamp(1.0 - opacity, 0.08, 1.0);
}

fn sample_average_shadow_visibility(p_world: vec3<f32>) -> f32 {
    let light_count = min(lights.n_directional_lights, 4u);
    if light_count == 0u {
        return 1.0;
    }
    var visibility = 0.0;
    for (var i = 0u; i < light_count; i = i + 1u) {
        visibility = visibility + sample_shadow_dom_visibility(p_world, i);
    }
    return visibility / f32(light_count);
}

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

const PATH_COUNT = 3u; // 3 paths for integration (R, TRT, TRRT etc.)
const QUAD_COUNT = 10u; // 10 quadrature points for integration
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

    let r_parallel = ((eta_t * cos_theta_incident) - (eta_i * cos_theta_transmitted)) / ((eta_t * cos_theta_incident) + (eta_i * cos_theta_transmitted));

    let r_perpendicular = ((eta_i * cos_theta_incident) - (eta_t * cos_theta_transmitted)) / ((eta_i * cos_theta_incident) + (eta_t * cos_theta_transmitted));

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
    // if (1.0 / v_long_val) < 1e-6 {
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


fn weta_strand_bsdf(theta_i: f32, phi_i: f32, theta_r: f32, phi_r: f32, eta_val: f32, mu_a_rgb_val: vec3<f32>, v_long_val: f32, v_azim_val: f32, alpha_p_val: f32, specular_a_rgb_val: vec4<f32>, occlusion: f32) -> vec3<f32> {
    // for more information on this see: model_building.ipynb
    let beta_azim_val = sqrt(v_azim_val);
    let scattering_visibility = clamp(1.0 - occlusion, 0.0, 1.0);
    let direct_lobe_visibility = mix(0.18, 1.0, scattering_visibility);
    var total_reflectance = Mp(v_long_val, theta_i, theta_r, alpha_p_val) * specular_a_rgb_val.xyz * specular_a_rgb_val.w * direct_lobe_visibility;
    // var total_reflectance = vec3<f32>(0.0, 0.0, 0.0);
    for (var p = 0u; p < PATH_COUNT; p = p + 1) {
        let Np_val = Np(p, phi_i, theta_i, theta_r, eta_val, mu_a_rgb_val, v_azim_val, beta_azim_val);
        total_reflectance = total_reflectance + Np_val * ((f32(p + 1u) * scattering_visibility) / f32(PATH_COUNT));
    }
    return total_reflectance;
}

fn marschner(point: vec4<f32>, direction: vec3<f32>, view_normal: vec3<f32>, light_normal: vec3<f32>, material: StrandMaterial, occlusion: f32) -> vec3<f32> {

    let u = direction;

    // Compute the key angles needed for Marschner model
    let theta_i = acos(dot(direction, light_normal));
    let theta_r = acos(dot(direction, view_normal));

    let light_projected = normalize(light_normal - dot(light_normal, u) * u);
    let view_projected = normalize(view_normal - dot(view_normal, u) * u);

    let phi = signed_angle_between(light_projected, view_projected, u);
    let sigma_a = 1.0 - material.absorption_color.xyz; // Artist adjustable absorption coefficient

    let v_long_val = material.alpha * material.alpha;
    let v_azim_val = material.beta * material.beta;

    let bcsdf = weta_strand_bsdf(theta_i, phi, theta_r, phi, material.eta, sigma_a, v_long_val, v_azim_val, material.shift, material.specular_color, occlusion);
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
) {
    let task_idx = global_id.y * pc.workgroup_offset + global_id.x;
    if task_idx >= pc.num_elements {
        return;
    }
    if task_idx >= atomicLoad(&binning_queue.tail) {
        return;
    }
    let task: BinningTask = binning_queue.tasks[task_idx];
    let frustum_id = unpack_binning_frustum(task.packed_field);
    if frustum_id >= arrayLength(&frustum_table) {
        return;
    }
    let frustum = frustum_table[frustum_id];
    if frustum.kind != 0u {
        return; // Camera frusta only
    }

    let inst_id = task.id_info;
    if inst_id >= arrayLength(&strand_instances) {
        return;
    }
    let instance = strand_instances[inst_id];
    let vertex_id = instance.vertex_id;
    let index_id = instance.index_id;
    let meta_id = instance.meta_id;
    let material_id = instance.material_id;
    if vertex_id >= arrayLength(&t_vertices) || index_id >= arrayLength(&t_indices) || meta_id >= arrayLength(&t_strand_metadata) || material_id >= arrayLength(&t_materials) {
        return;
    }

    let vertex_ptr = t_vertices[vertex_id];
    let index_ptr = t_indices[index_id];
    let meta_ptr = t_strand_metadata[meta_id];
    let material_ptr = t_materials[material_id];
    if !is_valid_ptr(vertex_ptr) || !is_valid_ptr(index_ptr) || !is_valid_ptr(meta_ptr) || !is_valid_ptr(material_ptr) {
        return;
    }

    let strand_local = task.chunk_id;
    let meta_base = meta_ptr.offset / SIZEOF_METADATA;
    let meta_count = meta_ptr.size / SIZEOF_METADATA;
    if strand_local >= meta_count {
        return;
    }
    let strand_meta = strand_metadata[meta_ptr.slab].ms[meta_base + strand_local];
    if strand_meta.count < 2u {
        return;
    }

    let seg_idx = task.seg_idx;
    let index_count = index_ptr.size / 4u;
    if seg_idx + 1u >= index_count {
        return;
    }
    if seg_idx < strand_meta.offset {
        return;
    }
    let seg_local = seg_idx - strand_meta.offset;
    if seg_local >= (strand_meta.count - 1u) {
        return;
    }

    let index_base = index_ptr.offset / 4u;
    let vertex_base = vertex_ptr.offset / SIZEOF_VERTEX;
    let i0 = indices[index_ptr.slab].is[index_base + seg_idx];
    let i1 = indices[index_ptr.slab].is[index_base + seg_idx + 1u];
    let vertex_count = vertex_ptr.size / SIZEOF_VERTEX;
    if i0 >= vertex_count || i1 >= vertex_count {
        return;
    }

    let v0 = instance.world_from_local * vec4<f32>(vertices[vertex_ptr.slab].vs[vertex_base + i0], 1.0);
    let v1 = instance.world_from_local * vec4<f32>(vertices[vertex_ptr.slab].vs[vertex_base + i1], 1.0);
    let U = normalize(v1.xyz - v0.xyz);
    if all(U == vec3<f32>(0.0)) {
        return;
    }

    let tangent = U;
    let camera_dir = normalize(view.world_position - v0.xyz);
    let binormal = normalize(cross(tangent, camera_dir));
    let V = normalize(cross(binormal, tangent));

    let material_base = material_ptr.offset / SIZEOF_MATERIAL;
    let material_count = material_ptr.size / SIZEOF_MATERIAL;
    if strand_meta.material_idx >= material_count {
        return;
    }
    let material = materials[material_ptr.slab].mats[material_base + strand_meta.material_idx];

    var accum_color = vec4<f32>(0.0, 0.0, 0.0, material.absorption_color.w);
    let layer = inst_id;
    let atlas_dims = textureDimensions(shadow_history_prev, 0);
    let atlas_capacity = atlas_dims.x * atlas_dims.y;
    let atlas_idx = strand_meta.offset + seg_local;
    if atlas_idx >= atlas_capacity {
        return;
    }
    let history_coord = shading_atlas_coord(atlas_idx, atlas_dims);
    let strand_midpoint = mix(v0.xyz, v1.xyz, 0.5);
    let current_scattering_visibility = sample_average_shadow_visibility(strand_midpoint);
    let previous_scattering_visibility = textureLoad(shadow_history_prev, history_coord, i32(layer), 0).x;
    let history_valid = previous_scattering_visibility > 0.0;
    let scattering_visibility = select(
        current_scattering_visibility,
        mix(current_scattering_visibility, previous_scattering_visibility, 0.88),
        history_valid,
    );
    textureStore(
        shadow_history_next,
        history_coord,
        i32(layer),
        vec4<f32>(scattering_visibility, current_scattering_visibility, 0.0, 1.0),
    );
    let scattering_occlusion = 1.0 - scattering_visibility;

    let light_count = lights.n_directional_lights;
    for (var j = 0u; j < light_count; j = j + 1u) {
        let light: types::DirectionalLight = lights.directional_lights[j];
        let L = normalize(light.direction_to_light);
        let bcsdf = marschner(v0, L, V, U, material, scattering_occlusion);
        var c = bcsdf * (light.color.xyz / 255.0);
        c = mix(c, material.absorption_color.xyz * material.ambient_factor + (lights.ambient_color.xyz / 255.0) * material.ambient_factor, material.ambient_factor);
        accum_color += vec4<f32>(c.xyz, 0.0);
    }

    textureStore(output_texture, history_coord, i32(layer), accum_color);
    let next_atlas_idx = atlas_idx + 1u;
    if next_atlas_idx < atlas_capacity && seg_local + 1u < strand_meta.count {
        textureStore(output_texture, shading_atlas_coord(next_atlas_idx, atlas_dims), i32(layer), accum_color);
    }
}
