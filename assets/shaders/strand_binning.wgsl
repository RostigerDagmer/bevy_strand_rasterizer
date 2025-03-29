#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
// #import NUMBER_OF_THREADS_PER_WORKGROUP
// #import NUMBER_OF_THREADS_PER_SUBGROUP

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

// --- Common Helper Functions ---

fn calculate_froxel_index(x: u32, y: u32, z: u32, config: FroxelConfig) -> u32 {
    let froxels_x = (config.screen_width + config.froxel_size_x - 1u) / config.froxel_size_x;
    let froxels_y = (config.screen_height + config.froxel_size_y - 1u) / config.froxel_size_y;
    // Clamp coordinates to valid range before calculating index
    let clamped_x = min(x, froxels_x - 1u);
    let clamped_y = min(y, froxels_y - 1u);
    let clamped_z = min(z, config.depth_slices - 1u);
    return clamped_z * froxels_x * froxels_y + clamped_y * froxels_x + clamped_x;
}

fn get_num_tiles(config: FroxelConfig) -> u32 {
    let froxels_x = (config.screen_width + config.froxel_size_x - 1u) / config.froxel_size_x;
    let froxels_y = (config.screen_height + config.froxel_size_y - 1u) / config.froxel_size_y;
    return froxels_x * froxels_y * config.depth_slices;
}

fn world_to_screen(position: vec3<f32>, view_proj: mat4x4<f32>, screen_width: f32, screen_height: f32) -> vec3<f32> {
    let clip_pos = view_proj * vec4<f32>(position, 1.0);
    if (clip_pos.w <= 0.0) {
        // Handle point behind camera - return invalid coords?
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    let ndc = clip_pos.xyz / clip_pos.w;
    // Check Y-direction based on Bevy/WGPU conventions
    let screen_x = (ndc.x * 0.5 + 0.5) * screen_width;
    let screen_y = (ndc.y * -0.5 + 0.5) * screen_height; // Common convention (0,0 top-left)
    let screen_z = ndc.z; // Typically [0, 1] after projection in WGSL/WebGPU
    return vec3<f32>(screen_x, screen_y, screen_z);
}

// --- STAGE_COUNT ---
#ifdef STAGE_COUNT

@group(0) @binding(0) var<storage, read> vertices: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> strand_metadata: array<StrandMeta>; // Use meta buffer
@group(0) @binding(2) var<storage, read_write> tile_counts_buffer: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> config: FroxelConfig;
@group(0) @binding(4) var<uniform> view: View; // Or ViewUniform

fn add_segment_ref_to_froxel(froxel_x: u32, froxel_y: u32, froxel_z: u32, cfg: FroxelConfig) -> bool {
    // Check AABB (example uses screen coords, adjust if AABB is in froxel coords)
    // if (froxel_x < cfg.aabb_min_x || ... ) { return false; }

    let froxel_idx = calculate_froxel_index(froxel_x, froxel_y, froxel_z, cfg);
    // Bounds check froxel_idx before atomic operation
    if (froxel_idx < arrayLength(&tile_counts_buffer)) {
        atomicAdd(&tile_counts_buffer[froxel_idx], 1u);
        return true;
    }
    return false;
}

fn trace_segment_through_froxels_count(p0: vec3<f32>, p1: vec3<f32>, cfg: FroxelConfig) {
    // Input p0, p1 are screen-space coordinates (x, y, depth [0,1])

    // Calculate start/end froxel coords (handle potential negatives from world_to_screen fail)
    if (p0.x < 0.0 || p1.x < 0.0) { return; } // Skip segments starting/ending off-screen

    let fx0 = i32(floor(p0.x / f32(cfg.froxel_size_x)));
    let fy0 = i32(floor(p0.y / f32(cfg.froxel_size_y)));
    let fz0 = i32(floor(p0.z * f32(cfg.depth_slices)));

    let fx1 = i32(floor(p1.x / f32(cfg.froxel_size_x)));
    let fy1 = i32(floor(p1.y / f32(cfg.froxel_size_y)));
    let fz1 = i32(floor(p1.z * f32(cfg.depth_slices)));

    // TODO: Implement Amanatides-Woo Voxel Traversal Algorithm
    // This algorithm steps along the line segment in screen-space,
    // determining which froxel boundary (X, Y, or Z) is crossed next.

    // Simplified Placeholder (Incorrect - Replace with Amanatides-Woo):
    // Add start froxel if valid
    if (fx0 >= 0 && fy0 >= 0 && fz0 >= 0) {
        add_segment_ref_to_froxel(u32(fx0), u32(fy0), u32(fz0), cfg);
    }
    // Add end froxel if different and valid
     if ((fx0 != fx1 || fy0 != fy1 || fz0 != fz1) && fx1 >= 0 && fy1 >= 0 && fz1 >= 0) {
        add_segment_ref_to_froxel(u32(fx1), u32(fy1), u32(fz1), cfg);
    }
    // Need to add ALL intermediate froxels crossed by the line segment.
}

@compute @workgroup_size(64, 1, 1) // Match Rust dispatch size if possible
fn count_strands(@builtin(global_invocation_id) id: vec3<u32>) {
    let strand_idx = id.x; // Assuming dispatching per strand
    let num_strands = arrayLength(&strand_metadata);

    if (strand_idx >= num_strands) {
        return;
    }

    let strand_meta = strand_metadata[strand_idx];
    let num_vertices_in_strand = strand_meta.count;
    let start_vertex_offset = strand_meta.offset; // Offset into vertices buffer

    if (num_vertices_in_strand < 2u) {
        return;
    }

    // Process segments for this strand
    var prev_vtx = vertices[start_vertex_offset];
    var prev_screen_pos = world_to_screen(prev_vtx, view.world_from_view, f32(config.screen_width), f32(config.screen_height));

    for (var i = 1u; i < num_vertices_in_strand; i = i + 1u) {
        let current_vtx_idx = start_vertex_offset + i;
        let current_vtx = vertices[current_vtx_idx];
        let current_screen_pos = world_to_screen(current_vtx, view.world_from_view, f32(config.screen_width), f32(config.screen_height));

        // Trace this segment (prev_screen_pos, current_screen_pos)
        trace_segment_through_froxels_count(prev_screen_pos, current_screen_pos, config);

        prev_screen_pos = current_screen_pos;
    }
}

#endif // STAGE_COUNT

// --- STAGE_SCAN_* ---

// Common code for all scan stages
#ifdef STAGE_SCAN_SUMS
#define SCAN_STAGE
#endif
#ifdef STAGE_SCAN_LAST
#define SCAN_STAGE
#endif
#ifdef STAGE_SCAN_PRFX
#define SCAN_STAGE
#endif

#ifdef SCAN_STAGE

// Common scan bindings
@group(0) @binding(0) var<storage, read> input_counts: array<u32>; // tile_counts_buffer (non-atomic read)
@group(0) @binding(1) var<storage, read_write> output_offsets: array<u32>; // tile_offsets_buffer

// Constants for scan logic (can be shader defs or derived)
const SCAN_THREADS: u32 = #{NUMBER_OF_THREADS_PER_WORKGROUP}; // e.g., 256
const SCAN_SUBGROUP_THREADS: u32 = #{NUMBER_OF_THREADS_PER_SUBGROUP}; // e.g., 64
const SCAN_SUBGROUPS: u32 = SCAN_THREADS / SCAN_SUBGROUP_THREADS;

var<workgroup> subgroup_scan_sums: array<u32, SCAN_SUBGROUPS>;

// --- Scan Helper Functions (Adapt from reference scan shader) ---

fn zeroing_shared_scan(local_id_x: u32) {
    if local_id_x < SCAN_SUBGROUPS { subgroup_scan_sums[local_id_x] = 0u; }
    workgroupBarrier();
}

// Exclusive scan within a workgroup
fn scan_exclusive_workgroup(value: u32, subgroup_id: u32, subgroup_local_id: u32) -> u32 {
    // TODO: Adapt logic from reference `scan_exclusive` using subgroup ops
    // Make sure it matches the hierarchical scan requirements.
    return 0u; // Placeholder
}

// Sum reduction within a workgroup
fn sum_workgroup(value: u32, subgroup_id: u32, subgroup_local_id: u32) -> u32 {
     // TODO: Adapt logic from reference `sum` using subgroup ops
    return 0u; // Placeholder
}

fn get_scan_workgroup_index(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    // Adapt from reference if using dispatch_workgroup_ext
    return workgroup_id.x; // Simplified
}

#endif // Scan stages common parts

#ifdef STAGE_SCAN_SUMS
@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn scan_sums(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
) {
    zeroing_shared_scan(local_id.x);
    let wg_idx = get_scan_workgroup_index(workgroup_id, num_workgroups);
    let base_read_idx = pc.scan_load_base + wg_idx * SCAN_THREADS;
    let current_read_idx = base_read_idx + local_id.x;

    var value = 0u;
    // TODO: Check bounds carefully based on number of elements being scanned this round
    // pc.scan_save_base might represent the *end* of the read range for this pass.
    if (current_read_idx < pc.scan_save_base) { // Example bound check
         value = input_counts[current_read_idx];
    }

    let wg_sum = sum_workgroup(value, subgroup_id, subgroup_local_id);

    if local_id.x == 0u {
        // Write sum to the next level of hierarchy
        output_offsets[pc.scan_save_base + wg_idx] = wg_sum;
    }
}
#endif // STAGE_SCAN_SUMS

#ifdef STAGE_SCAN_LAST
@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn scan_last(
     @builtin(local_invocation_id) local_id: vec3u,
     @builtin(subgroup_id) subgroup_id: u32,
     @builtin(subgroup_invocation_id) subgroup_local_id: u32,
) {
     zeroing_shared_scan(local_id.x);
     let num_to_scan = pc.scan_save_base - pc.scan_load_base;
     let read_idx = pc.scan_load_base + local_id.x;

     var value = 0u;
     if local_id.x < num_to_scan {
         // Read from *output_offsets* as it holds intermediate sums from previous stage
         value = output_offsets[read_idx];
     }

     let prefix_sum = scan_exclusive_workgroup(value, subgroup_id, subgroup_local_id);

     if local_id.x < num_to_scan {
          // Write prefix sum back into output_offsets
         output_offsets[read_idx] = prefix_sum;
     }

     // Write total sum (last element's prefix_sum + last element's value)
     // Needs careful coordination - often done by last thread.
     if local_id.x == SCAN_THREADS - 1u {
         let total_sum = prefix_sum + value; // Sum for this workgroup (only 1 WG in scan_last)
         // Write to the designated total count slot (num_tiles index)
         let num_tiles = pc.num_elements; // Assuming num_elements holds num_tiles
         output_offsets[num_tiles] = total_sum;
     }
}
#endif // STAGE_SCAN_LAST

#ifdef STAGE_SCAN_PRFX
@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn scan_prfx(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
) {
    zeroing_shared_scan(local_id.x);
    let wg_idx = get_scan_workgroup_index(workgroup_id, num_workgroups);
    let base_read_idx = pc.scan_load_base + wg_idx * SCAN_THREADS;
    let current_read_idx = base_read_idx + local_id.x;

    var value = 0u;
     // TODO: Check bounds carefully based on number of elements being scanned this round
    if (current_read_idx < pc.scan_save_base) { // Example bound check
         // Read intermediate values (which were sums before scan_last, prefix sums after)
         value = output_offsets[current_read_idx];
    }

    // Perform exclusive scan on these values *within* the workgroup
    let local_prefix_sum = scan_exclusive_workgroup(value, subgroup_id, subgroup_local_id);

    // Get the prefix sum *from the level above* (calculated in previous scan_last/scan_prfx pass)
    let block_sum = output_offsets[pc.scan_save_base + wg_idx];

    // Write final prefix sum: block_sum + local_prefix_sum
    if (current_read_idx < pc.scan_save_base) {
        output_offsets[current_read_idx] = block_sum + local_prefix_sum;
    }
}
#endif // STAGE_SCAN_PRFX


// --- STAGE_INIT_PLACE ---
#ifdef STAGE_INIT_PLACE

@group(0) @binding(0) var<storage, read> tile_offsets_buffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> current_tile_write_indices_buffer: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1) // Example workgroup size
fn init_placement_idx(@builtin(global_invocation_id) id: vec3<u32>) {
    let tile_idx = id.x;
    let num_tiles = arrayLength(&current_tile_write_indices_buffer); // Assumes buffer is exact size

    if (tile_idx >= num_tiles) {
        return;
    }

    // Read the starting offset for this tile
    let offset_val = tile_offsets_buffer[tile_idx];

    // Atomically store this offset into the write counter buffer
    atomicStore(&current_tile_write_indices_buffer[tile_idx], offset_val);
}

#endif // STAGE_INIT_PLACE

// --- STAGE_PLACE ---
#ifdef STAGE_PLACE

@group(0) @binding(0) var<storage, read> vertices: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> strand_metadata: array<StrandMeta>;
// @group(0) @binding(2) var<storage, read> tile_offsets_buffer: array<u32>; // Read-only access needed? Maybe not directly
@group(0) @binding(3) var<storage, read_write> current_tile_write_indices_buffer: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> packed_segments_buffer: array<SegmentRef>; // Write-only effectively
@group(0) @binding(5) var<uniform> config: FroxelConfig;
@group(0) @binding(6) var<uniform> view: View;

fn add_segment_ref_to_froxel_place(froxel_x: u32, froxel_y: u32, froxel_z: u32, segment_ref: SegmentRef, cfg: FroxelConfig) -> bool {
    // Optional AABB Check
    // if (froxel_x < cfg.aabb_min_x || ...) { return false; }

    let froxel_idx = calculate_froxel_index(froxel_x, froxel_y, froxel_z, cfg);
    let num_tiles = arrayLength(&current_tile_write_indices_buffer);

    if (froxel_idx < num_tiles) {
        // Get the write index for this tile atomically
        let write_index = atomicAdd(&current_tile_write_indices_buffer[froxel_idx], 1u);

        // Write the segment reference to the packed buffer
        // Optional: Add bounds check using tile_offsets_buffer if over-allocation is possible
        // let max_idx_for_tile = tile_offsets_buffer[froxel_idx + 1u];
        // if write_index < max_idx_for_tile && write_index < arrayLength(&packed_segments_buffer) {
        if write_index < arrayLength(&packed_segments_buffer) { // Basic check against buffer end
            packed_segments_buffer[write_index] = segment_ref;
            return true;
        } else {
            // Handle overflow - log error? Skip? Depends on strategy
            return false;
        }
    }
     return false;
}

fn trace_segment_through_froxels_place(p0: vec3<f32>, p1: vec3<f32>, segment_ref: SegmentRef, cfg: FroxelConfig) {
    // Input p0, p1 are screen-space coordinates (x, y, depth [0,1])
    if (p0.x < 0.0 || p1.x < 0.0) { return; } // Skip off-screen

    let fx0 = i32(floor(p0.x / f32(cfg.froxel_size_x)));
    let fy0 = i32(floor(p0.y / f32(cfg.froxel_size_y)));
    let fz0 = i32(floor(p0.z * f32(cfg.depth_slices)));

    let fx1 = i32(floor(p1.x / f32(cfg.froxel_size_x)));
    let fy1 = i32(floor(p1.y / f32(cfg.froxel_size_y)));
    let fz1 = i32(floor(p1.z * f32(cfg.depth_slices)));

    // TODO: Implement Amanatides-Woo Voxel Traversal Algorithm
    // Inside the loop of the traversal algorithm, when entering a new froxel (fx, fy, fz):
    // add_segment_ref_to_froxel_place(u32(fx), u32(fy), u32(fz), segment_ref, cfg);

    // Simplified Placeholder (Incorrect - Replace with Amanatides-Woo):
    if (fx0 >= 0 && fy0 >= 0 && fz0 >= 0) {
         add_segment_ref_to_froxel_place(u32(fx0), u32(fy0), u32(fz0), segment_ref, cfg);
    }
    if ((fx0 != fx1 || fy0 != fy1 || fz0 != fz1) && fx1 >= 0 && fy1 >= 0 && fz1 >= 0) {
         add_segment_ref_to_froxel_place(u32(fx1), u32(fy1), u32(fz1), segment_ref, cfg);
    }
}

@compute @workgroup_size(64, 1, 1)
fn place_strands(@builtin(global_invocation_id) id: vec3<u32>) {
    let strand_idx = id.x;
    let num_strands = arrayLength(&strand_metadata);

    if (strand_idx >= num_strands) {
        return;
    }

    let strand_meta = strand_metadata[strand_idx];
    let num_vertices_in_strand = strand_meta.count;
    let start_vertex_offset = strand_meta.offset;

    if (num_vertices_in_strand < 2u) {
        return;
    }

    var prev_vtx = vertices[start_vertex_offset];
    var prev_screen_pos = world_to_screen(prev_vtx, view.world_from_view, f32(config.screen_width), f32(config.screen_height));

    for (var i = 1u; i < num_vertices_in_strand; i = i + 1u) {
        let current_vtx_idx = start_vertex_offset + i;
        let current_vtx = vertices[current_vtx_idx];
        let current_screen_pos = world_to_screen(current_vtx, view.world_from_view, f32(config.screen_width), f32(config.screen_height));

        // Define SegmentRef - How is segment_start_idx used?
        // Option 1: Index into vertices buffer
        // let segment_start_vtx_idx = start_vertex_offset + i - 1u;
        // Option 2: Index into strand_metadata buffer (doesn't make sense for segment)
        // Option 3: Index relative to start of strand (0, 1, 2...)
        let segment_start_idx_in_strand = i - 1u; // Example
        let segment_ref = SegmentRef(strand_idx, segment_start_idx_in_strand);

        trace_segment_through_froxels_place(prev_screen_pos, current_screen_pos, segment_ref, config);

        prev_screen_pos = current_screen_pos;
    }
}

#endif // STAGE_PLACE