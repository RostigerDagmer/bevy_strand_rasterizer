#import bevy_render::view::View
#import "shaders/types.wgsl"::{
    Aabb,
    FroxelConfig,
    SegmentRef,
    StrandGeo,
    StrandMeta,
    PushConstants,
    DevicePtr,
}

const PI = 3.14159265359;
const PI_HALF = PI / 2.0;
const SQRT_2_PI = 2.5066282746310002;

const DOM_GAMMA: f32 = 1.0; // distribution coefficient for DOM slices.

fn is_valid_ptr(_ptr: DevicePtr) -> bool {
    return _ptr.slab != 0xFFFFFFFFu;
}

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
        if clip_pos.w > 0.0 {
            let view_pos = clip_pos.xyz / clip_pos.w;
            min_clip = min(min_clip, view_pos);
            max_clip = max(max_clip, view_pos);
        }
        // let view_pos = clip_pos.xyz / clip_pos.w;
        // // Update min and max for the z component:
        // min_clip = min(min_clip, view_pos);
        // max_clip = max(max_clip, view_pos);
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

fn world_to_screen_raw(
    position: vec4<f32>,
    clip_from_world: mat4x4<f32>,
    viewport: vec4<f32>, // (x, y, w, h)
) -> vec3<f32> {
    let clip = clip_from_world * position;
    if clip.w <= 0.0 {
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    let ndc = clip.xyz / clip.w; // ndc in [-1,1]^2 x [0,1] (wgpu/Vulkan)
    let sx = viewport.x + (ndc.x * 0.5 + 0.5) * viewport.z;
    let sy = viewport.y + (ndc.y * -0.5 + 0.5) * viewport.w; // flip Y (top-left origin)
    // return NDC z directly (no reversal)
    return vec3<f32>(sx, sy, ndc.z);
}

fn world_to_screen_aabbnorm(position: vec4<f32>, clip_from_world: mat4x4<f32>, screen_width: f32, screen_height: f32, aabb_clip_bounds: mat2x3<f32>) -> vec3<f32> {

    // Transform from world to clip space
    let clip_pos = clip_from_world * position;

    // Check if the point is behind the camera (w <= 0 typically indicates this)
    // Using a small epsilon might be safer depending on the projection matrix, but w < 0 is common.
    if clip_pos.w <= 0.0 {
        // Point is behind the camera or exactly on the near plane in a way that causes issues.
        return vec3<f32>(-1.0, -1.0, -1.0); // Indicate an invalid screen position
    }

    // Perform perspective division to get Normalized Device Coordinates (NDC)
    // NDC range is typically [-1, 1] for x/y and [0, 1] or [-1, 1] for z depending on convention.
    let ndc = clip_pos.xyz / clip_pos.w;

    // Extract the min and max clip bounds for the AABB
    let min_clip = aabb_clip_bounds[0]; // vec3
    let max_clip = aabb_clip_bounds[1]; // vec3

    // Calculate the range (delta) of the AABB in clip space
    // Add a small epsilon to avoid division by zero if the AABB has zero size in an axis
    let epsilon = 0.00001;
    let clip_range = max_clip - min_clip + vec3<f32>(epsilon);

    // Normalize the NDC coordinates relative to the AABB bounds
    // This maps the AABB's clip space volume to a [0, 1] cube
    let norm_coords = (ndc - min_clip) / clip_range;

    // Clamp the normalized coordinates to the [0, 1] range.
    // This ensures points outside the AABB's projection are clamped to its boundaries.
    let clamped_norm_coords = clamp(norm_coords, vec3<f32>(0.0), vec3<f32>(1.0));

    // Convert normalized coordinates to screen coordinates
    // X: [0, 1] -> [0, screen_width]
    let screen_x = clamped_norm_coords.x * screen_width;

    // Y: [0, 1] -> [0, screen_height] (with Y flip for top-left origin)
    // norm_coords.y = 0 corresponds to min_clip.y (bottom in NDC)
    // norm_coords.y = 1 corresponds to max_clip.y (top in NDC)
    // Screen Y = 0 should be the top, so we invert norm_coords.y
    let screen_y = (1.0 - clamped_norm_coords.y) * screen_height;

    // Z: [0, 1] -> [0, 1] (depth, potentially flipped depending on convention)
    // The reference world_to_screen function returns 1.0 - normalized_depth.
    // Assuming norm_coords.z = 0 corresponds to the near plane of the AABB (min_clip.z)
    // and norm_coords.z = 1 corresponds to the far plane of the AABB (max_clip.z).
    // To match the reference (1=near, 0=far), we flip it.
    let screen_z = 1.0 - clamped_norm_coords.z;

    return vec3<f32>(screen_x, screen_y, screen_z);
}

fn screen_to_world(
    screen_pos: vec3<f32>,           // (x, y, packed_depth)
    view: View,
    aabb_znear_zfar: vec2<f32>,      // (z_near, z_far)
) -> vec3<f32> {
    // 1) Unpack viewport & near/far
    let screen_width = view.viewport.z;
    let screen_height = view.viewport.w;
    let z_near = aabb_znear_zfar.x;
    let z_far = aabb_znear_zfar.y;

    // 2) Reconstruct NDC x/y from screen coords
    let ndc_x = (screen_pos.x / screen_width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_pos.y / screen_height) * 2.0;
    let ndc_z = (1.0 - screen_pos.z) * (z_far - z_near) + z_near;

    // 3) Unproject using the reconstructed NDC Z
    let ndc_point = vec4<f32>(ndc_x, ndc_y, ndc_z, 1.0);
    let world_h = view.world_from_clip * ndc_point;

    return world_h.xyz / world_h.w;
}

fn screen_to_world_raw(
    screen_pos: vec3<f32>,      // (x, y, ndc_z)
    view: View,                 // has view.world_from_clip and view.viewport
    viewport: vec4<f32>,
) -> vec3<f32> {
    let vp = viewport; // (x, y, w, h)

    // normalize to viewport-local [0,1]
    let u = (screen_pos.x - vp.x) / vp.z;
    let v = (screen_pos.y - vp.y) / vp.w;

    // NDC
    let ndc_x = u * 2.0 - 1.0;
    let ndc_y = 1.0 - v * 2.0; // undo top-left origin
    let ndc_z = screen_pos.z;  // already NDC z

    let clip_p = vec4<f32>(ndc_x, ndc_y, ndc_z, 1.0);
    let world_h = view.world_from_clip * clip_p;
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

fn l_and(v1: vec3<bool>, v2: vec3<bool>) -> vec3<bool> {
    return vec3<bool>(v1.x && v2.x, v1.y && v2.y, v1.z && v2.z);
}
