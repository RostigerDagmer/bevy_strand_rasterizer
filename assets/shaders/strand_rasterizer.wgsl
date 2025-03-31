#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms

// --- Structures ---

struct FroxelConfig { // Ensure this matches Rust exactly
    screen_width: u32,
    screen_height: u32,
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
    aabb_min_x: u32,
    aabb_min_y: u32,
    aabb_min_z: f32,
    aabb_max_x: u32,
    aabb_max_y: u32,
    aabb_max_z: f32,
}

struct SegmentRef { // Ensure this matches Rust if defined there
    strand_idx: u32,
    segment_start_idx: u32, // Index into strand_meta perhaps? Or original vertex buffer? Define clearly.
}

struct StrandMeta { // Ensure this matches Rust exactly
    count: u32,     // Number of vertices in strand
    offset: u32,    // Start index in the original indices buffer (or vertices buffer?)
}

struct PushConstants { // Ensure this matches Rust and range covers all fields
    workgroup_offset: u32, // For dispatch_workgroup_ext compatibility
    num_elements: u32,    // Generic count (e.g., num_strands or num_tiles)
    scan_load_base: u32,
    scan_save_base: u32,
    // Add other needed constants
}
var<push_constant> pc: PushConstants;


@group(0) @binding(0) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> strand_metadata: array<StrandMeta>;
@group(0) @binding(3) var<storage, read> tile_offsets_buffer: array<u32>;
@group(0) @binding(4) var<storage, read> tile_counts_buffer: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> packed_segments_buffer: array<SegmentRef>; // Read only
@group(0) @binding(6) var render_target: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var<uniform> config: FroxelConfig;
@group(0) @binding(8) var<uniform> view: View;

fn heatmap_precise(value: f32) -> vec3<f32> {
    let v = clamp(value, 0.0, 1.0);
    
    if (v < 0.25) {
        // Blue to cyan
        let t = v / 0.25;
        return vec3<f32>(0.0, t, 1.0);
    } else if (v < 0.5) {
        // Cyan to green
        let t = (v - 0.25) / 0.25;
        return vec3<f32>(0.0, 1.0, 1.0 - t);
    } else if (v < 0.75) {
        // Green to yellow
        let t = (v - 0.5) / 0.25;
        return vec3<f32>(t, 1.0, 0.0);
    } else {
        // Yellow to red
        let t = (v - 0.75) / 0.25;
        return vec3<f32>(1.0, 1.0 - t, 0.0);
    }
}

fn world_to_screen(position: vec3<f32>, view: View, screen_width: f32, screen_height: f32) -> vec3<f32> {
    // Transform from world to clip space using the view-projection matrix
    let clip_pos = view.unjittered_clip_from_world * vec4<f32>(position, 1.0);

    if (clip_pos.w <= 0.0) {
        // Handle point behind camera
        return vec3<f32>(-1.0, -1.0, -1.0); // Indicate invalid screen pos
    }

    // Perform perspective division to get NDC coordinates
    let ndc = clip_pos.xyz / clip_pos.w;

    // Convert NDC to screen coordinates (pixel centers)
    // NDC is [-1,1] for x,y and [0,1] for z in Vulkan/WebGPU convention
    let screen_x = (ndc.x * 0.5 + 0.5) * screen_width;
    let screen_y = (ndc.y * -0.5 + 0.5) * screen_height; // Flip Y for top-left origin

    // Depth is already in [0,1] range
    let screen_z = ndc.z;

    return vec3<f32>(screen_x, screen_y, screen_z);
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

// Define hair properties (can be uniforms later)
const HAIR_RADIUS_PIXELS : f32 = 1.0; // Example: Thickness in pixels
const HAIR_COLOR : vec3<f32> = vec3<f32>(0.8, 0.7, 0.6); // Example: Hair color
const HAIR_ALPHA : f32 = 0.1; // Example: Alpha per covered fragment (lower for softer look)

// Helper: Signed distance from point `p` to line segment `a` -> `b`
// Returns distance. Clamps distance calc to the segment endpoints.
fn point_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let l2 = distance(a, b);
    if (l2 == 0.0) { return distance(p, a); } // Segment is a point
    let l2_sq = l2 * l2;

    // Project p onto the line defined by a, b. t is the projection parameter.
    let t = dot(p - a, b - a) / l2_sq;

    // Clamp t to [0, 1] to stay within the segment
    let t_clamped = clamp(t, 0.0, 1.0);

    // Calculate the closest point on the segment to p
    let closest_point = a + t_clamped * (b - a);

    return distance(p, closest_point);
}

// Helper: Basic alpha blending (foreground "over" background)
fn blend_over(foreground: vec4<f32>, background: vec4<f32>) -> vec4<f32> {
    // Premultiply alpha for foreground
    let fg_rgb = foreground.rgb * foreground.a;
    let final_alpha = foreground.a + background.a * (1.0 - foreground.a);
    if (final_alpha < 1e-6) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let final_rgb = (fg_rgb + background.rgb * background.a * (1.0 - foreground.a)) / final_alpha;
    return vec4<f32>(final_rgb, final_alpha);
}

#define DEBUG


@compute @workgroup_size(8, 8, 1) // Should match froxel_size_x, froxel_size_y
fn rasterize_strands(
    @builtin(global_invocation_id) global_id: vec3<u32>, // Represents the pixel coordinate (x, y, 0)
    @builtin(workgroup_id) workgroup_id: vec3u,         // Represents the tile index (tx, ty, 0)
    @builtin(local_invocation_id) local_id: vec3u          // Represents pixel within tile (lx, ly, 0)
) {
    let pixel_coord_int = vec2<i32>(global_id.xy);

    // // Check if pixel is outside screen bounds
    // if (pixel_coord_int.x >= i32(config.screen_width) || pixel_coord_int.y >= i32(config.screen_height)) {
    //     return;
    // }

    let pixel_center = vec2<f32>(global_id.xy) + vec2<f32>(0.5, 0.5); // Center of the pixel

    // Initialize final pixel color (start transparent black)
    var final_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let tile_coord_x = workgroup_id.x;
    let tile_coord_y = workgroup_id.y;

    var frag_count: u32 = 0;
    for (var dz: u32 = 0; dz < u32(config.depth_slices); dz = dz + 1) {
        frag_count += tile_counts_buffer[calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config)];
    }
    #ifdef DEBUG
        // Debug: Output the tile_count of this tile divided by num_elements
        final_color = vec4<f32>(heatmap_precise(f32(frag_count) / f32(pc.num_elements)), 0.2);
    # endif // DEBUG

    // if (frag_count == 0) {
    //     // Early exit if no strands in this tile
    //     textureStore(render_target, pixel_coord_int, final_color);
    //     return;
    // }
    // Or loop FRONT TO BACK (simpler to start)
    for (var dz: u32 = 0; dz < config.depth_slices; dz = dz + 1) {

        let froxel_idx = calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config);

        // Find the range of segments for this froxel in the packed buffer
        // Ensure read index doesn't go out of bounds
        if (froxel_idx + 1u >= arrayLength(&tile_offsets_buffer)) { continue; } // Safety check

        let start_segment_offset = tile_offsets_buffer[froxel_idx];
        let segment_count_in_froxel = tile_counts_buffer[froxel_idx];

        // Process all segments within this froxel
        for (var s: u32 = 0; s < segment_count_in_froxel; s = s + 1u) {
            if s >= 300 { break; } // Limit number of segments processed per froxel
            let packed_buffer_idx = start_segment_offset + s;
             // Safety check packed buffer bounds
            if (packed_buffer_idx >= arrayLength(&packed_segments_buffer)) { continue; }

            let segment_ref = packed_segments_buffer[packed_buffer_idx];

            // Get strand metadata
            let strand_idx = segment_ref.strand_idx;
            if (strand_idx >= arrayLength(&strand_metadata)) { continue; } // Safety check
            let strand_meta = strand_metadata[strand_idx];

            // Get segment vertex indices within the strand
            let v0_strand_idx = indices[segment_ref.segment_start_idx];
            let v1_strand_idx = indices[segment_ref.segment_start_idx + 1u];

            // Get world-space vertex positions
            let v0_world = vertices[v0_strand_idx].xyz;
            let v1_world = vertices[v1_strand_idx].xyz;

            // Project to screen space (pixels)
            let p0_screen = world_to_screen(v0_world, view, f32(config.screen_width), f32(config.screen_height));
            let p1_screen = world_to_screen(v1_world, view, f32(config.screen_width), f32(config.screen_height));

            // Skip if segment is fully behind camera or off-screen after projection
            if (p0_screen.x < 0.0 && p1_screen.x < 0.0) { continue; } // Basic culling

            // Calculate analytical coverage
            let dist = point_segment_distance(pixel_center, p0_screen.xy, p1_screen.xy);

            // Simple linear falloff based on distance
            let coverage = clamp(1.0 - dist / HAIR_RADIUS_PIXELS, 0.0, 1.0);

            if (coverage > 0.0) {
                // Calculate color/alpha contribution of this hair segment fragment
                let hair_fragment_alpha = HAIR_ALPHA * coverage;
                let hair_fragment = vec4<f32>(HAIR_COLOR, hair_fragment_alpha);

                // Blend this fragment OVER the current accumulated color
                final_color = blend_over(hair_fragment, final_color);
            }
        } // End loop over segments in froxel

        // --- Optional Early Exit ---
        // If pixel becomes nearly opaque, we can stop processing deeper Z slices
        // if (final_color.a > 0.99) {
        //     break; // Stop Z loop
        // }

    } // End loop over depth slices (dz)

    // Write the final accumulated color to the render target
    textureStore(render_target, pixel_coord_int, final_color);
}

// strand_rasterizer.wgsl
// @compute @workgroup_size(8, 8, 1)
// fn rasterize_strands(
//     @builtin(global_invocation_id) id: vec3<u32>,
//     @builtin(workgroup_id) workgroup_id: vec3u,
//     @builtin(num_workgroups) num_workgroups: vec3u,
//     @builtin(local_invocation_id) local_id: vec3u,
// )
// {
//     // TODO: Implement strand rasterization
//     // - Read vertices and indices
//     // - Bin strands into froxels
//     // - Write to output texture

//     // Placeholder: Output the tile_count of this tile divided by num_elements
//     let tile = workgroup_id.xy;
//     var frag_count: u32 = 0;
//     for (var dz: u32 = 0; dz < u32(config.depth_slices); dz = dz + 1) {
//         frag_count += tile_counts_buffer[calculate_froxel_index(tile.x, tile.y, dz, config)];
//     }
//     let tile_color = vec4<f32>(heatmap_precise(f32(frag_count) / f32(pc.num_elements)), 0.2);

//     // Compute the base texture coordinates for this tile
//     let base_x = i32(tile.x * config.froxel_size_x);
//     let base_y = i32(tile.y * config.froxel_size_y);
//     var random_color = random_colors[u32(hash12(vec2<f32>(tile)) * 20.0) % 20];
//     let texture_coord: vec2<i32> = vec2(base_x + i32(local_id.x), base_y + i32(local_id.y));

//     if (texture_coord.x < i32(config.screen_width) && texture_coord.y < i32(config.screen_height)) {
//         textureStore(render_target, texture_coord, tile_color);
//     }
// }