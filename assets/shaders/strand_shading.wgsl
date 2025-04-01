#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::{
    mesh_view_types::POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE,
    mesh_view_bindings as view_bindings,
}
#import bevy_pbr::mesh_view_types as types

// --- Structures ---

struct StrandMeta { // Ensure this matches Rust exactly
    count: u32,     // Number of vertices in strand
    offset: u32,    // Start index in the original indices buffer
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

@group(0) @binding(0) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> strand_metadata: array<StrandMeta>;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var<uniform> lights: types::Lights;
@group(0) @binding(5) var output_texture: texture_storage_2d<rgba8unorm, write>;

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

fn fresnel(eta: f32, cos_theta: f32) -> f32 {
    // Schlick's approximation or full Fresnel equations
    let r0 = 1.0 / pow((1.0 - eta) / (1.0 + eta), 2.0);
    let cos_theta_clamped = max(0.0, cos_theta); // Ensure non-negative base for pow
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta_clamped, 5.0);
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

// --- Marschner Model ---

fn marschner_R(theta_i: f32, theta_r: f32, phi: f32, attenuation: f32) -> vec2<f32> {
    /// Longitudinal surface reflection (Mr)
    let beta_r = theta_i - theta_r;
    let alpha_r = 0.1309; // TODO: material paramter in radians
    let Mr = gaussian(beta_r, alpha_r, 0.0);

    /// Azimuthal surface reflection (Nr)
    let eta = 1.55; // TODO: material parameter (IOR)
    let theta_d = cos(phi / 2.0);
    let Nr = attenuation * fresnel(eta, theta_d);

    return vec2<f32>(Mr, Nr);
}

fn marschner_TT(theta_i: f32, theta_r: f32, phi: f32) -> vec2<f32> {
    /// Longitudinal transmission (Mtt)
    let shift_tt = 0.0; // TODO: material parameter (shift factor)
    let beta_tt = (theta_i - theta_r) / 2.0 - shift_tt;
    let alpha_tt = 0.21; // TODO: material paramter in radians
    let Mtt = gaussian(beta_tt, alpha_tt, 0.0);

    /// Azimuthal transmission (Ntt)
    let attenuation = 0.65; // TODO: material parameter (attenuation factor)
    let eta = 1.55; // TODO: material parameter (IOR)

    let phi_tt = phi - PI_HALF; // entry ray
    let theta_d = cos(phi_tt / 2.0);
    let theta_d_ = sqrt(max(0.0, 1.0 - (1.0 - theta_d * theta_d) / (eta * eta)));; // TODO: check if this is actually correct
    let f1 = (1.0 - fresnel(eta, theta_d));
    let f2 = (1.0 - fresnel(1.0 / eta, theta_d_));
    let Ntt = attenuation * f1 * f2;

    return vec2<f32>(Mtt, Ntt);
}

fn marschner_TRT(theta_i: f32, theta_r: f32, phi: f32) -> vec2<f32> {
    /// Longitudinal transmission-reflection (Mtrt)
    let beta_trt = -(theta_r - theta_i) * 1.5;
    let alpha_trt = 0.3; // TODO: material paramter in radians
    let Mtrt = gaussian(beta_trt, alpha_trt, 0.0);

    /// Azimuthal transmission-reflection (Ntrt)
    let attenuation = 0.45; // TODO: material parameter (attenuation factor)
    let eta = 1.55; // TODO: material parameter (IOR)
    let phi_trt = phi - PI;
    let theta_d = cos(phi_trt / 2.0);
    let theta_d_ = sqrt(max(0.0, 1.0 - (1.0 - theta_d * theta_d) / (eta * eta)));

    let f1 = (1.0 - fresnel(eta, theta_d));
    let f2 = fresnel(1.0 / eta, theta_d_);
    let f3 = (1.0 - fresnel(1.0 / eta, theta_d_));
    let Ntrt = attenuation * f1 * f2 * f3;

    return vec2<f32>(Mtrt, Ntrt);
}

// Returns the BCSDF color value for the Marschner model
fn marschner(point: vec4<f32>, direction: vec3<f32>, view_normal: vec3<f32>, light_normal: vec3<f32>, hair_color: vec4<f32>, specular_color: vec4<f32>) -> vec3<f32> {

    let u = direction;
    
    // Compute the key angles needed for Marschner model
    let theta_i = acos(dot(direction, light_normal));
    let theta_r = acos(dot(direction, view_normal));

    let light_projected = normalize(light_normal - dot(light_normal, u) * u);
    let view_projected = normalize(view_normal - dot(view_normal, u) * u);

    let phi = signed_angle_between(light_projected, view_projected, u);

    let sigma_a = 1.0 - hair_color.xyz;
    let single_pass_absorption = exp(-2.0 * sigma_a);
    let dual_pass_absorption = exp(-4.0 * sigma_a);
    let specular_attenuation = specular_color.w;

    let R = marschner_R(theta_i, theta_r, phi, specular_attenuation);
    let TT = marschner_TT(theta_i, theta_r, phi);
    let TRT = marschner_TRT(theta_i, theta_r, phi);

    let R_contrib = R.x * R.y * specular_color.xyz;
    let TT_contrib = TT.x * TT.y * single_pass_absorption;
    let TRT_contrib = TRT.x * TRT.y * dual_pass_absorption;

    return R_contrib + TT_contrib + TRT_contrib;

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

    let light_count = lights.n_directional_lights;

    var strand_absorption_color = vec4<f32>(0.7, 0.1, 0.05, 0.3);
    var strand_specular_color = vec4<f32>(1.0, 1.0, 1.0, 0.6);

    for (var i = 0u; i < strand_count; i = i + WORKGROUP_SIZE) {
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
        let V = normalize(view.world_position - vertex.xyz);

        // TODO: we can theoretically split this across multiple workgroups
        for (var j = 0u; j < light_count; j = j+1) {
            let light: types::DirectionalLight = lights.directional_lights[j];
            let light_flags = light.flags;
            let L = light.direction_to_light; // TODO: point lights, spot lights etc. this would be normalize(light.position - strand_point.position);
            
            let bcsdf = marschner(vertex, L, V, U, strand_absorption_color, strand_specular_color);

            var c = bcsdf * light.color.xyz;
            c = mix(c, lights.ambient_color.xyz / 255.0, 0.01); // ambient TODO: ambient lighting
            strand_absorption_color = vec4<f32>(c.xyz, strand_absorption_color.w);

        }
        let out_row = strand_id % MAX_TEXTURE_EXT;
        let out_col = strand_id / MAX_TEXTURE_EXT;
        let y_coord = out_row;
        let x_coord = out_col * pc.workgroup_offset + segment_offset; // remember that workgroup_offset is abused for column width in this context
        textureStore(output_texture, vec2<i32>(i32(x_coord), i32(y_coord)), strand_absorption_color);
    }
}