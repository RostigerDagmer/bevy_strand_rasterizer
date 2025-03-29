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
// @group(0) @binding(2) var<storage, read> tile_offsets_buffer: array<u32>; // Read-only access needed? Maybe not directly
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

// strand_rasterizer.wgsl
@compute @workgroup_size(64, 1, 1)
fn rasterize_strands(@builtin(global_invocation_id) id: vec3<u32>) {
    // TODO: Implement strand rasterization
    // - Read vertices and indices
    // - Bin strands into froxels
    // - Write to output texture

    // Placeholder: Output the tile_count of this tile divided by num_elements
    let tile_idx = id.x;
    let num_tiles_x = config.screen_width / config.froxel_size_x;
    let num_tiles_y = config.screen_height / config.froxel_size_y;
    let num_tiles_z = config.depth_slices;
    let tile_x = tile_idx % num_tiles_x;
    let tile_y = (tile_idx / num_tiles_x) % num_tiles_y;
    let tile_count = tile_counts_buffer[tile_idx];
    let tile_color = vec4<f32>(heatmap_precise(log((f32(tile_count) / f32(pc.num_elements)) + 1.0)), 0.2);
    textureStore(
        render_target,
        vec2<i32>(i32(tile_x), i32(tile_y)),
        tile_color
    );
}