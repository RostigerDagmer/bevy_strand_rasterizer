#import bevy_render::view::View
#import "shaders/types.wgsl"::{
    Aabb,
    FroxelConfig,
    SegmentRef,
    StrandGeo,
    StrandMeta,
    PushConstants,
}

const PI = 3.14159265359;
const PI_HALF = PI / 2.0;
const SQRT_2_PI = sqrt(2.0 * PI);

fn calculate_froxel_index(x: u32, y: u32, z: u32, config: FroxelConfig) -> u32 {
    let froxels_x = (config.screen_width + config.froxel_size_x - 1u) / config.froxel_size_x;
    let froxels_y = (config.screen_height + config.froxel_size_y - 1u) / config.froxel_size_y;
    // Clamp coordinates to valid range before calculating index
    let clamped_x = min(x, froxels_x - 1u);
    let clamped_y = min(y, froxels_y - 1u);
    let clamped_z = min(z, config.depth_slices - 1u);
    return clamped_z * froxels_x * froxels_y + clamped_y * froxels_x + clamped_x;
}


fn find_clip_bounds(clip_from_world: mat4x4<f32>, world_aabb_min: vec3<f32>, world_aabb_max: vec3<f32>) -> mat2x3<f32> {
    
    var min_clip = vec3<f32>(999999.0);
    var max_clip = vec3<f32>(-999999.0);

    let corners = array<vec4<f32>, 8>(
        vec4(world_aabb_min, 1.0),
        vec4(world_aabb_min.x, world_aabb_min.y, world_aabb_max.z, 1.0),
        vec4(world_aabb_min.x, world_aabb_max.y, world_aabb_min.z, 1.0),
        vec4(world_aabb_min.x, world_aabb_max.y, world_aabb_max.z, 1.0),
        vec4(world_aabb_max.x, world_aabb_min.y, world_aabb_min.z, 1.0),
        vec4(world_aabb_max.x, world_aabb_min.y, world_aabb_max.z, 1.0),
        vec4(world_aabb_max.x, world_aabb_max.y, world_aabb_min.z, 1.0),
        vec4(world_aabb_max, 1.0)
    );
    for (var i = 0u; i < 8u; i = i + 1u) {
        let clip_pos = clip_from_world * corners[i];
        // For orthographic projections, w will usually be 1.0;
        let view_pos = clip_pos.xyz / clip_pos.w;
        // Update min and max for the z component:
        min_clip = min(min_clip, view_pos);
        max_clip = max(max_clip, view_pos);
    }
    return mat2x3(min_clip, max_clip);
}

fn world_to_screen(position: vec4<f32>, clip_from_world: mat4x4<f32>, screen_width: f32, screen_height: f32, aabb_znear_zfar: vec2<f32>) -> vec3<f32> {
    // Transform from world to clip space using the view-projection matrix
    let clip_pos = clip_from_world * position;

    if clip_pos.w < 0.0 {
        // Handle point behind camera
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    
    // Perform perspective division to get NDC coordinates
    let ndc = clip_pos.xyz / clip_pos.w;
    
    // Convert NDC to screen coordinates for X and Y
    let screen_x = (ndc.x * 0.5 + 0.5) * screen_width;
    let screen_y = (ndc.y * -0.5 + 0.5) * screen_height; // Flip Y for top-left origin
    
    // Ensure proper ordering (min should be closer to camera)
    let z_near = aabb_znear_zfar.x;
    let z_far = aabb_znear_zfar.y;
    
    // Normalize the depth within the AABB Z range
    let normalized_depth = (ndc.z - z_near) / (z_far - z_near);
    
    // Clamp to ensure we stay in the [0,1] range even if point is outside AABB
    let screen_z = clamp(normalized_depth, 0.0, 1.0);

    return vec3<f32>(screen_x, screen_y, 1.0 - screen_z);
}

fn screen_to_world(
    screen_pos: vec3<f32>,           // (x, y, packed_depth)
    view: View,
    aabb_znear_zfar: vec2<f32>,      // (z_near, z_far)
) -> vec3<f32> {
    // 1) Unpack viewport & near/far
    let screen_width  = view.viewport.z;
    let screen_height = view.viewport.w;
    let z_near = aabb_znear_zfar.x;
    let z_far  = aabb_znear_zfar.y;

    // 2) Reconstruct NDC x/y from screen coords
    let ndc_x = (screen_pos.x / screen_width)  * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_pos.y / screen_height) * 2.0;
    let ndc_z = (1.0 - screen_pos.z) * (z_far - z_near) + z_near;

    // 3) Unproject using the reconstructed NDC Z
    let ndc_point = vec4<f32>(ndc_x, ndc_y, ndc_z, 1.0);
    let world_h = view.world_from_clip * ndc_point;

    return world_h.xyz / world_h.w;
}

fn wang_hash(seed: u32) -> u32 {
    var x = seed;
    x = (x ^ 61u) ^ (x >> 16u);
    x = x + (x << 3u);
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

fn hash_to_unit_float(x: u32) -> f32 {
    // Divide by 2^32 to get a float in [0.0, 1.0)
    return f32(x) / 4294967296.0;
}

// fn canonical_min(v: vec4<f32>) -> f32 {
//     return min(min(v.x, v.y), min(v.z, v.w));
// }

// fn canonical_min_mask(v: vec4<f32>) -> vec4<bool> {
//     // 1. Find the minimum value across all components
//     let min_val = canonical_min(v);
  
//     // 2. Create a mask identifying ALL components equal to the minimum value
//     //    Comparison operators on vectors return boolean vectors in WGSL.
//     let is_min: vec4<bool> = (v == vec4(min_val)); // e.g., (false, true, true, false)
  
//     var prev_cum_mask = vec4<bool>(false);
//     var current_cum = is_min.x;     // Start with cum up to x
//     prev_cum_mask.y = current_cum;  // Set prev for y
//     current_cum = current_cum | is_min.y; // Update cum up to y
//     prev_cum_mask.z = current_cum;  // Set prev for z
//     current_cum = current_cum | is_min.z; // Update cum up to z
//     prev_cum_mask.w = current_cum;  // Set prev for w
//     let final_mask = is_min & !prev_cum_mask; // Same final step
  
//     return final_mask; // e.g., (false, true, false, false) for input (3.0, 1.0, 1.0, 2.0)
// }
fn canonical_min(v: vec3<f32>) -> f32 {
    return min(min(v.x, v.y), v.z);
}

fn canonical_min_mask(v: vec3<f32>) -> vec3<bool> {
    // 1. Find the minimum value across all components
    let min_val = canonical_min(v);
  
    // 2. Create a mask identifying ALL components equal to the minimum value
    //    Comparison operators on vectors return boolean vectors in WGSL.
    let is_min: vec3<bool> = (v == vec3(min_val)); // e.g., (false, true, true)
  
    return select(
        select(vec3(false, false, true), vec3(false, true, false), is_min.y),
        vec3(true, false, false),
        is_min.x
    );

  }