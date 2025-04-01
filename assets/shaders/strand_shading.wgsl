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

@group(0) @binding(0) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> strand_metadata: array<StrandMeta>;
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> view: View;
@group(0) @binding(5) var<uniform> lights: types::Lights;

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


const MAX_TEXTURE_EXT: u32 = #MAX_TEXTURE_EXTENT;

@compute @workgroup_size(64, 1, 1)
fn shade_strands(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3u,         
    @builtin(local_invocation_id) local_id: vec3u          
) {

    let strand_id = workgroup_id.x * 64 + local_id.x;
    if strand_id >= pc.num_elements {
        return;
    }

    let strand_meta = strand_metadata[strand_id];
    let strand_offset = strand_meta.offset;
    let strand_count = strand_meta.count;

    let light_count = lights.n_directional_lights;
    let light_flags = lights.flags;

    let strand_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let strand_normal = vec3<f32>(0.0, 0.0, 0.0);

    for i in 0..strand_count {
        let vertex = vertices[strand_offset + i];
        let normal = normalize(vec3<f32>(vertex.xyz));
        let color = vec4<f32>(vertex.w, vertex.w, vertex.w, 1.0);

        // TODO: we can theoretically split this across multiple workgroups
        for j in 0..light_count {
            let light: types::DirectionalLight = lights.directional_lights[j];
            // TODO
        }
        let out_row = strand_id % MAX_TEXTURE_EXT;
        let out_col = strand_id / MAX_TEXTURE_EXT;
        let y_coord = out_row;
        let x_coord = out_col * pc.workgroup_offset + i;
        textureStore(output_texture, vec2<i32>(x_coord, y_coord), strand_color);
    }


}