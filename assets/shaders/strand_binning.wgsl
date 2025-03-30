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

fn world_to_screen(position: vec4<f32>, view: View, screen_width: f32, screen_height: f32) -> vec3<f32> {
    // Transform from world to clip space using the view-projection matrix
    let clip_pos = view.unjittered_clip_from_world * position;

    if clip_pos.w <= 0.0 {
        // Handle point behind camera
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    
    // Perform perspective division to get NDC coordinates
    let ndc = clip_pos.xyz / clip_pos.w;
    
    // Convert NDC to screen coordinates
    // NDC is [-1,1] for x,y and [0,1] for z in Vulkan/WebGPU convention
    let screen_x = (ndc.x * 0.5 + 0.5) * screen_width;
    let screen_y = (ndc.y * -0.5 + 0.5) * screen_height; // Flip Y for top-left origin
    
    // For your depth slices that run 0.0 to 1.0, the z value is already in the right range
    let screen_z = ndc.z; // This is in [0,1] range in Vulkan/WebGPU

    return vec3<f32>(screen_x, screen_y, screen_z);
}

// --- STAGE_COUNT ---
#ifdef STAGE_COUNT

@group(0) @binding(0) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> strand_metadata: array<StrandMeta>; // Use meta buffer
@group(0) @binding(3) var<storage, read_write> tile_counts_buffer: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> config: FroxelConfig;
@group(0) @binding(5) var<uniform> view: View; // Or ViewUniform


fn add_segment_ref_to_froxel(froxel_x: u32, froxel_y: u32, froxel_z: u32, cfg: FroxelConfig) -> bool {
    // Check AABB (example uses screen coords, adjust if AABB is in froxel coords)
    // if (froxel_x < cfg.aabb_min_x || ... ) { return false; }

    let froxel_idx = calculate_froxel_index(froxel_x, froxel_y, froxel_z, cfg);
    // Bounds check froxel_idx before atomic operation
    if froxel_idx < arrayLength(&tile_counts_buffer) {
        atomicAdd(&tile_counts_buffer[froxel_idx], 1u);
        return true;
    }
    return false;
}

fn trace_segment_through_froxels_count(p0: vec3<f32>, p1: vec3<f32>, cfg: FroxelConfig) {
    // Input p0, p1 are screen-space coordinates (x, y, depth [0,1])
    if p0.x < 0.0 || p1.x < 0.0 { return; } // Skip off-screen or behind camera

    // Use i32 for stepping, but ensure non-negative before passing to add_segment_ref_to_froxel
    let fx0 = i32(floor(p0.x / f32(cfg.froxel_size_x)));
    let fy0 = i32(floor(p0.y / f32(cfg.froxel_size_y)));
    // Clamp depth index calculation strictly between 0 and depth_slices-1
    let fz0 = clamp(i32(floor(p0.z * f32(cfg.depth_slices))), 0, i32(cfg.depth_slices) - 1);

    let fx1 = i32(floor(p1.x / f32(cfg.froxel_size_x)));
    let fy1 = i32(floor(p1.y / f32(cfg.froxel_size_y)));
    let fz1 = clamp(i32(floor(p1.z * f32(cfg.depth_slices))), 0, i32(cfg.depth_slices) - 1);

    var fx = fx0;
    var fy = fy0;
    var fz = fz0;

    var dir = p1 - p0;
    // Need step in integer grid space, but derivation needs float dir
    var step = vec3<i32>(sgn_i32(dir.x), sgn_i32(dir.y), sgn_i32(dir.z)); // Integer step

    // Use screen space sizes for calculations
    let froxel_dim_x = f32(cfg.froxel_size_x);
    let froxel_dim_y = f32(cfg.froxel_size_y);
    let froxel_dim_z = 1.0 / f32(cfg.depth_slices); // Size of a depth slice in [0,1] range

    // Calculate delta distances - how far along the ray (in units of t) we must move
    // for the coord to change by one froxel size.
    // Avoid division by zero. Use a large number if dir component is zero.
    let safe_dir_x = select(dir.x, 1e-6 * f32(step.x), abs(dir.x) < 1e-6);
    let safe_dir_y = select(dir.y, 1e-6 * f32(step.y), abs(dir.y) < 1e-6);
    let safe_dir_z = select(dir.z, 1e-6 * f32(step.z), abs(dir.z) < 1e-6);

    var delta_dist = vec3<f32>(
        abs(froxel_dim_x / safe_dir_x),
        abs(froxel_dim_y / safe_dir_y),
        abs(froxel_dim_z / safe_dir_z)
    );

    // Calculate initial distances (as t values) to the *next* voxel boundary
    // along the ray's direction from p0.
    let fract_p0_x = p0.x / froxel_dim_x; // How many froxels p0.x is
    let fract_p0_y = p0.y / froxel_dim_y;
    let fract_p0_z = p0.z / froxel_dim_z; // p0.z is already [0,1]

    // Distance to next boundary = (boundary - current_pos) / direction
    // If moving positive (step>0), next boundary is floor(pos)+1. Distance = ( (floor(pos)+1)*size - pos ) / dir
    // If moving negative (step<0), next boundary is floor(pos).   Distance = ( floor(pos)*size - pos ) / dir
    var t_max_x = select(
        (floor(fract_p0_x) * froxel_dim_x - p0.x) / safe_dir_x,            // step < 0
        ((floor(fract_p0_x) + 1.0) * froxel_dim_x - p0.x) / safe_dir_x,    // step > 0
        step.x > 0
    );
    var t_max_y = select(
        (floor(fract_p0_y) * froxel_dim_y - p0.y) / safe_dir_y,
        ((floor(fract_p0_y) + 1.0) * froxel_dim_y - p0.y) / safe_dir_y,
        step.y > 0
    );
    var t_max_z = select(
        (floor(fract_p0_z) * froxel_dim_z - p0.z) / safe_dir_z,
        ((floor(fract_p0_z) + 1.0) * froxel_dim_z - p0.z) / safe_dir_z,
        step.z > 0
    );

    // Correct for zero direction components - they should never be the minimum t_max
    if abs(dir.x) < 1e-6 { t_max_x = 1e38; }
    if abs(dir.y) < 1e-6 { t_max_y = 1e38; }
    if abs(dir.z) < 1e-6 { t_max_z = 1e38; }


    // Pre-calculate screen/froxel bounds for loop check
    let max_fx = i32((cfg.screen_width + cfg.froxel_size_x - 1u) / cfg.froxel_size_x);
    let max_fy = i32((cfg.screen_height + cfg.froxel_size_y - 1u) / cfg.froxel_size_y);
    let max_fz = i32(cfg.depth_slices); // Exclusive bound

    var safety = 0u;
    let max_steps = u32(max_fx + max_fy + max_fz + 3); // Generous upper bound

    loop {
        safety = safety + 1u;
        if safety > max_steps { break; } // Safety break

        // Add segment count to current froxel IF it's within valid bounds
        if fx >= 0 && fx < max_fx && fy >= 0 && fy < max_fy && fz >= 0 && fz < max_fz {
            if !add_segment_ref_to_froxel(u32(fx), u32(fy), u32(fz), cfg) {
                 // Optional: handle case where buffer is full, though unlikely for count
                 break;
            }
        } else {
             // Stop if we step out of bounds entirely
             break;
        }


        // Check if we've reached the end froxel (Manhattan distance check can be faster)
        if fx == fx1 && fy == fy1 && fz == fz1 { break; }

        // Find axis with minimum t_max value to find the next froxel boundary crossed
        if t_max_x < t_max_y && t_max_x < t_max_z {
            // X axis traversal
            fx += step.x;
            t_max_x += delta_dist.x;
        } else if t_max_y < t_max_z {
            // Y axis traversal
            fy += step.y;
            t_max_y += delta_dist.y;
        } else {
            // Z axis traversal
            fz += step.z;
            t_max_z += delta_dist.z;
        }

         // Check if we have stepped past the target froxel along any axis where movement occurs
         // This prevents infinite loops for axis-aligned lines ending exactly on a boundary
        if (step.x > 0 && fx > fx1) || (step.x < 0 && fx < fx1) || (step.y > 0 && fy > fy1) || (step.y < 0 && fy < fy1) || (step.z > 0 && fz > fz1) || (step.z < 0 && fz < fz1) {
             break;
        }
    }
}

// Helper for integer sign needed in traversal
fn sgn_i32(f: f32) -> i32 {
    if f > 1e-6 { return 1; }
    if f < -1e-6 { return -1; }
    return 0;
}


@compute @workgroup_size(64, 1, 1) // Match Rust dispatch size if possible
fn count_strands(@builtin(global_invocation_id) id: vec3<u32>) {
    let strand_idx = id.x; // Assuming dispatching per strand
    let num_strands = arrayLength(&strand_metadata);

    if strand_idx >= num_strands {
        return;
    }

    let strand_meta = strand_metadata[strand_idx];
    let num_vertices_in_strand = strand_meta.count;
    let start_vertex_offset = strand_meta.offset; // Offset into vertices buffer

    if num_vertices_in_strand < 2u {
        return;
    }

    // Process segments for this strand
    var prev_vtx = vertices[indices[start_vertex_offset]];
    var prev_screen_pos = world_to_screen(prev_vtx, view, f32(config.screen_width), f32(config.screen_height));

    for (var i = 1u; i < num_vertices_in_strand; i = i + 1u) {
        let current_vtx_idx = indices[start_vertex_offset + i];
        if current_vtx_idx >= arrayLength(&vertices) { break; } // Bounds check

        let current_vtx = vertices[current_vtx_idx];
        let current_screen_pos = world_to_screen(current_vtx, view, f32(config.screen_width), f32(config.screen_height));

        // Trace this segment using the *correct* traversal logic
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

// Sum reduction within a workgroup
fn sum_workgroup(value: u32, subgroup_id: u32, subgroup_local_id: u32) -> u32 {
    var subgroup_sum = subgroupAdd(value);
    if subgroup_local_id == 0u {
        subgroup_scan_sums[subgroup_id] = subgroup_sum;
    }
    workgroupBarrier();
    subgroup_sum = select(0u, subgroup_scan_sums[subgroup_local_id], subgroup_local_id < SCAN_SUBGROUPS);
    return subgroupAdd(subgroup_sum);
}

// Exclusive scan within a workgroup
fn scan_exclusive_workgroup(value: u32, subgroup_id: u32, subgroup_local_id: u32) -> u32 {
    let subgroup_prefix_sum = subgroupInclusiveAdd(value);
    if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
        subgroup_scan_sums[subgroup_id] = subgroup_prefix_sum;
    }
    workgroupBarrier();
    let prev_subgroup_sum = select(0u, subgroup_scan_sums[subgroup_local_id], subgroup_local_id < subgroup_id);
    let prev_sum = subgroupAdd(prev_subgroup_sum);
    return prev_sum + subgroup_prefix_sum - value;
}

fn get_scan_workgroup_index(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    // Adapt from reference if using dispatch_workgroup_ext
    return workgroup_id.y * num_workgroups.x + workgroup_id.x + pc.workgroup_offset;
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
    if current_read_idx < pc.scan_save_base { // Example bound check
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
    if current_read_idx < pc.scan_save_base { // Example bound check
         // Read intermediate values (which were sums before scan_last, prefix sums after)
        value = output_offsets[current_read_idx];
    }

    // Perform exclusive scan on these values *within* the workgroup
    let local_prefix_sum = scan_exclusive_workgroup(value, subgroup_id, subgroup_local_id);

    // Get the prefix sum *from the level above* (calculated in previous scan_last/scan_prfx pass)
    let block_sum = output_offsets[pc.scan_save_base + wg_idx];

    // Write final prefix sum: block_sum + local_prefix_sum
    if current_read_idx < pc.scan_save_base {
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

    if tile_idx >= num_tiles {
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

@group(0) @binding(0) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> strand_metadata: array<StrandMeta>;
@group(0) @binding(3) var<storage, read_write> current_tile_write_indices_buffer: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> packed_segments_buffer: array<SegmentRef>; // Write-only effectively
@group(0) @binding(5) var<uniform> config: FroxelConfig;
@group(0) @binding(6) var<uniform> view: View;

fn add_segment_ref_to_froxel_place(froxel_x: u32, froxel_y: u32, froxel_z: u32, segment_ref: SegmentRef, cfg: FroxelConfig) -> bool {
    // Optional AABB Check
    // if (froxel_x < cfg.aabb_min_x || ...) { return false; }

    let froxel_idx = calculate_froxel_index(froxel_x, froxel_y, froxel_z, cfg);
    let num_tiles = arrayLength(&current_tile_write_indices_buffer);

    if froxel_idx < num_tiles {
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

fn get_froxel_max_extent(fx: u32, fy: u32, fz: u32, cfg: FroxelConfig) -> vec3<f32> {
    let froxel_max_x = f32((fx + 1) * cfg.froxel_size_x) / f32(cfg.screen_width);
    let froxel_max_y = f32((fy + 1) * cfg.froxel_size_y) / f32(cfg.screen_height);
    let froxel_max_z = f32(fz + 1) / f32(cfg.depth_slices);
    return vec3<f32>(froxel_max_x, froxel_max_y, froxel_max_z);
}

fn get_froxel_min_extent(fx: u32, fy: u32, fz: u32, cfg: FroxelConfig) -> vec3<f32> {
    let froxel_min_x = f32(fx * cfg.froxel_size_x) / f32(cfg.screen_width);
    let froxel_min_y = f32(fy * cfg.froxel_size_y) / f32(cfg.screen_height);
    let froxel_min_z = f32(fz) / f32(cfg.depth_slices);
    return vec3<f32>(froxel_min_x, froxel_min_y, froxel_min_z);
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

    var fx = fx0;
    var fy = fy0;
    var fz = fz0;

    var p = p0;
    var dir = p1 - p0;
    var step = vec3<f32>(sign(dir.x), sign(dir.y), sign(dir.z));
    
    // Calculate delta distances - how far along the ray we must move for a one-voxel change
    var delta_dist = vec3<f32>(
        abs(length(dir) / dir.x),
        abs(length(dir) / dir.y),
        abs(length(dir) / dir.z)
    );
    // Handle divisions by zero
    if (dir.x == 0.0) { delta_dist.x = 1000000.0; }
    if (dir.y == 0.0) { delta_dist.y = 1000000.0; }
    if (dir.z == 0.0) { delta_dist.z = 1000000.0; }
    
    // Calculate initial distances to voxel boundaries
    let min_extent = get_froxel_min_extent(u32(fx), u32(fy), u32(fz), cfg);
    let max_extent = get_froxel_max_extent(u32(fx), u32(fy), u32(fz), cfg);
    
    var next_boundary = vec3<f32>(
        select(min_extent.x, max_extent.x, step.x > 0.0),
        select(min_extent.y, max_extent.y, step.y > 0.0),
        select(min_extent.z, max_extent.z, step.z > 0.0)
    );
    
    var t_max = vec3<f32>(
        abs((next_boundary.x - p0.x) / dir.x),
        abs((next_boundary.y - p0.y) / dir.y),
        abs((next_boundary.z - p0.z) / dir.z)
    );
    // Handle divisions by zero
    if (dir.x == 0.0) { t_max.x = 1000000.0; }
    if (dir.y == 0.0) { t_max.y = 1000000.0; }
    if (dir.z == 0.0) { t_max.z = 1000000.0; }

    var safety = 0u;
    loop {
        safety = safety + 1u;
        if (safety > 2048u) { break; } // Safety check
        
        // Add segment to current froxel
        if (!add_segment_ref_to_froxel_place(u32(fx), u32(fy), u32(fz), segment_ref, cfg)) { break; }

        // Check if we've reached the end froxel
        if (fx == fx1 && fy == fy1 && fz == fz1) { break; }
        
        // Find axis with minimum t_max value
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            // X axis traversal
            fx += i32(step.x);
            t_max.x += delta_dist.x;
        } else if (t_max.y < t_max.z) {
            // Y axis traversal
            fy += i32(step.y);
            t_max.y += delta_dist.y;
        } else {
            // Z axis traversal
            fz += i32(step.z);
            t_max.z += delta_dist.z;
        }
        
        // Optional - check if we're outside bounds
        if (fx < 0 || fy < 0 || fz < 0 || 
            fx >= i32(cfg.screen_width / cfg.froxel_size_x) || 
            fy >= i32(cfg.screen_height / cfg.froxel_size_y) || 
            fz >= i32(cfg.depth_slices)) {
            break;
        }
    }
}


@compute @workgroup_size(64, 1, 1)
fn place_strands(@builtin(global_invocation_id) id: vec3<u32>) {
    let strand_idx = id.x;
    let num_strands = arrayLength(&strand_metadata);

    if strand_idx >= num_strands {
        return;
    }

    let strand_meta = strand_metadata[strand_idx];
    let num_vertices_in_strand = strand_meta.count;
    let start_vertex_offset = strand_meta.offset;

    if num_vertices_in_strand < 2u {
        return;
    }

    var prev_vtx = vertices[indices[start_vertex_offset]];
    var prev_screen_pos = world_to_screen(prev_vtx, view, f32(config.screen_width), f32(config.screen_height));

    for (var i = 1u; i < num_vertices_in_strand; i = i + 1u) {
        let current_vtx_idx = indices[start_vertex_offset + i];
        let current_vtx = vertices[current_vtx_idx];
        let current_screen_pos = world_to_screen(current_vtx, view, f32(config.screen_width), f32(config.screen_height));

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