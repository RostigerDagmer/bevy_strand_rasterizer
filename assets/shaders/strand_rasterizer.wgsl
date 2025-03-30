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


@group(0) @binding(0) var<storage, read> vertices: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> strand_metadata: array<StrandMeta>;
@group(0) @binding(2) var<storage, read> tile_counts_buffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> packed_segments_buffer: array<SegmentRef>; // Read only
@group(0) @binding(4) var render_target: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<uniform> config: FroxelConfig;
@group(0) @binding(6) var<uniform> view: View;

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

const random_colors = array<vec3<f32>, 20>(
    vec3<f32>(1.0, 0.0, 0.0), // Red
    vec3<f32>(0.0, 1.0, 0.0), // Green
    vec3<f32>(0.0, 0.0, 1.0), // Blue
    vec3<f32>(1.0, 1.0, 0.0), // Yellow
    vec3<f32>(1.0, 0.0, 1.0), // Magenta
    vec3<f32>(0.0, 1.0, 1.0), // Cyan
    vec3<f32>(0.5, 0.5, 0.5), // Gray
    vec3<f32>(1.0, 0.5, 0.0), // Orange
    vec3<f32>(0.5, 0.0, 1.0), // Purple
    vec3<f32>(0.0, 0.5, 0.5), // Teal
    vec3<f32>(0.8, 0.1, 0.1), // Dark Red
    vec3<f32>(0.1, 0.8, 0.1), // Dark Green
    vec3<f32>(0.1, 0.1, 0.8), // Dark Blue
    vec3<f32>(0.8, 0.8, 0.1), // Olive
    vec3<f32>(0.8, 0.1, 0.8), // Violet
    vec3<f32>(0.1, 0.8, 0.8), // Aqua
    vec3<f32>(0.3, 0.3, 0.3), // Dark Gray
    vec3<f32>(0.9, 0.6, 0.1), // Amber
    vec3<f32>(0.6, 0.1, 0.9), // Deep Purple
    vec3<f32>(0.1, 0.6, 0.6)  // Sea Green
);

fn hash12(p: vec2<f32>) -> f32
{
    var p3  = fract(vec3<f32>(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// strand_rasterizer.wgsl
@compute @workgroup_size(8, 8, 1)
fn rasterize_strands(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
)
{
    // TODO: Implement strand rasterization
    // - Read vertices and indices
    // - Bin strands into froxels
    // - Write to output texture

    // Placeholder: Output the tile_count of this tile divided by num_elements
    let tile = workgroup_id.xy;
    let num_tiles_x = config.screen_width / config.froxel_size_x;
    let num_tiles_y = config.screen_height / config.froxel_size_y;
    let num_tiles_z = config.depth_slices;

    let lin_idx = tile.y * num_tiles_x * num_tiles_z + tile.x * num_tiles_z; // this is technically just the first layer in z-depth.
    var frag_count: u32 = 0;
    for (var dz: u32 = 0; dz < u32(config.depth_slices); dz = dz + 1) {
        frag_count += tile_counts_buffer[lin_idx + dz];
    }
    let tile_color = vec4<f32>(heatmap_precise(f32(frag_count) / f32(pc.num_elements)), 0.2);

    // Compute the base texture coordinates for this tile
    let base_x = i32(tile.x * config.froxel_size_x);
    let base_y = i32(tile.y * config.froxel_size_y);

    var random_color = random_colors[u32(hash12(vec2<f32>(tile)) * 20.0) % 20];

    let texture_coord: vec2<i32> = vec2(base_x + i32(local_id.x), base_y + i32(local_id.y));
    // let debug_uv_color = vec4<f32>(f32(local_id.x) / 8.0, f32(local_id.y) / 8.0, 0.0, 0.2);
    if (texture_coord.x < i32(config.screen_width) && texture_coord.y < i32(config.screen_height)) {
        textureStore(render_target, texture_coord, tile_color); // * 0.5 + vec4<f32>(random_color, 0.2) * 0.5);
    }
}