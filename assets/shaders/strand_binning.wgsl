#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::mesh_view_types as types
#import "shaders/common.wgsl"::{ find_clip_bounds, world_to_screen, screen_to_world, calculate_froxel_index, wang_hash, hash_to_unit_float }
// #import NUMBER_OF_THREADS_PER_WORKGROUP
// #import NUMBER_OF_THREADS_PER_SUBGROUP

const LIGHT_INDEX: u32 = 0u; // Example constant for light index TODO: compute prepass -> indirect dispatch -> light index from uniforms
const CULL_MAX_DIST: f32 = 90.0;
const CULL_MIN_DIST: f32 = 4.0;
const SHADOW_MAP_BOOST_FACTOR: f32 = 10.0;
const MODE: u32 = 3u; // 0 = linear, 1 = adaptive tesselation, 3 = analytical splines

// --- Structures ---

#import "shaders/types.wgsl"::{
    Aabb,
    FroxelConfig,
    SegmentRef,
    StrandGeo,
    StrandMeta,
    PushConstants,
}

var<push_constant> pc: PushConstants;

// --- Common Helper Functions ---

fn sgn_i32(f: f32) -> i32 {
    if f > 1e-6 { return 1; }
    if f < -1e-6 { return -1; }
    return 0;
}

fn get_num_tiles(config: FroxelConfig) -> u32 {
    let froxels_x = (config.screen_width + config.froxel_size_x - 1u) / config.froxel_size_x;
    let froxels_y = (config.screen_height + config.froxel_size_y - 1u) / config.froxel_size_y;
    return froxels_x * froxels_y * config.depth_slices;
}

fn stochastic_cull_camera(view: View, aabb: Aabb, sample_threshold: f32) -> bool {
    let aabb_center = aabb.max - aabb.min;
    let distance = length(view.world_position - aabb_center);
    let norm_distance = max((distance - CULL_MIN_DIST) + 0.0001, 0.0001) / CULL_MAX_DIST;
    if sample_threshold <= pow(norm_distance, 0.3) {
        return true;
    }
    return false;
}

fn stochastic_cull_light(aabb_clip: mat2x3<f32>, sample_threshold: f32) -> bool {
    let coverage = (aabb_clip[1] - aabb_clip[0]) * 0.5; // since NDC is [-1,1]
    let area = coverage.x * coverage.y;
    let lod_threshold = clamp((sqrt(area) * SHADOW_MAP_BOOST_FACTOR), 0.0, 1.0);
    if sample_threshold > lod_threshold {
        return true;
    }
    return false;
}

#ifdef STAGE_PLACE
fn add_segment_ref_to_froxel(froxel_x: u32, froxel_y: u32, froxel_z: u32, segment_ref: SegmentRef, cfg: FroxelConfig) -> bool {
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
#endif

#ifdef STAGE_COUNT
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
#endif

#ifdef STAGE_COUNT
#define COUNT_OR_PLACE
#endif

#ifdef STAGE_PLACE
#define COUNT_OR_PLACE
#endif


#ifdef COUNT_OR_PLACE

#ifdef STAGE_COUNT
fn trace_segment_through_froxels_linear(p0: vec3<f32>, p1: vec3<f32>, cfg: FroxelConfig) {
#else
fn trace_segment_through_froxels_linear(p0: vec3<f32>, p1: vec3<f32>, seg_ref: SegmentRef, cfg: FroxelConfig) {
#endif
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
            #ifdef STAGE_COUNT
            if !add_segment_ref_to_froxel(u32(fx), u32(fy), u32(fz), cfg) {
                 // Optional: handle case where buffer is full, though unlikely for count
                 break;
            }
            #endif
            #ifdef STAGE_PLACE
            if !add_segment_ref_to_froxel(u32(fx), u32(fy), u32(fz), seg_ref, cfg) {
                 // Optional: handle case where buffer is full
                 break;
            }
            #endif
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

#endif // COUNT_OR PLACE



// --- STAGE_COUNT ---
#ifdef STAGE_COUNT

@group(0) @binding(#VERTEX_BUFFER) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(#INDEX_BUFFER) var<storage, read> indices: array<u32>;
@group(0) @binding(#META_BUFFER) var<storage, read> strand_metadata: array<StrandMeta>; // Use meta buffer
@group(0) @binding(#TILE_COUNTS_BUFFER) var<storage, read_write> tile_counts_buffer: array<atomic<u32>>;
@group(0) @binding(#FROXEL_CONFIG) var<uniform> config: FroxelConfig;
@group(0) @binding(#VIEW_UNIFORM) var<uniform> view: View; // Or ViewUniform
@group(0) @binding(#LIGHT_UNIFORM) var<uniform> lights: types::Lights;
@group(0) @binding(#GEO_BUFFER) var<storage, read> geos: array<StrandGeo>; // Has AABB for bounds check

@compute @workgroup_size(64, 1, 1) // Match Rust dispatch size if possible
fn count_strands(@builtin(global_invocation_id) id: vec3<u32>) {
    let strand_idx = id.x; // Assuming dispatching per strand
    let num_strands = arrayLength(&strand_metadata);

    if strand_idx >= num_strands {
        return;
    }
    let aabb = geos[0].aabb;
    let aabb_center = aabb.max - aabb.min;
    let strand_hash = wang_hash(strand_idx);
    let sample_threshold = hash_to_unit_float(strand_hash);
    
    #ifdef SHADOWS
        let light: types::DirectionalLight = lights.directional_lights[LIGHT_INDEX];
        let cascade = light.cascades[0]; // TODO: select cascade based on distance
        let clip_from_world = cascade.clip_from_world;
        let aabb_clip = find_clip_bounds(clip_from_world, aabb.min, aabb.max);
        // culling
        if stochastic_cull_light(aabb_clip, sample_threshold) {
            return;
        }
        let aabb_znear_zfar = vec2<f32>(aabb_clip[0].z, aabb_clip[1].z);
    #else
        let clip_from_world = view.unjittered_clip_from_world;
        // culling
        if stochastic_cull_camera(view, aabb, sample_threshold) {
            return;
        }
        let aabb_clip = find_clip_bounds(clip_from_world, aabb.min, aabb.max);
        let aabb_znear_zfar = vec2<f32>(aabb_clip[0].z, aabb_clip[1].z);
    #endif

    let strand_meta = strand_metadata[strand_idx];
    let num_vertices_in_strand = strand_meta.count;
    let start_vertex_offset = strand_meta.offset; // Offset into vertices buffer

    if num_vertices_in_strand < 2u {
        return;
    }

    var prev_vtx = vertices[indices[start_vertex_offset]];
    var prev_screen_pos = world_to_screen(prev_vtx, clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);

    for (var i = 1u; i < num_vertices_in_strand; i = i + 1u) {
        let current_vtx_idx = indices[start_vertex_offset + i];
        if current_vtx_idx >= arrayLength(&vertices) { break; } // Bounds check

        let current_vtx = vertices[current_vtx_idx];
        let current_screen_pos = world_to_screen(current_vtx, clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);

        switch MODE {
            case 2u: { // Analytical splines
                trace_segment_through_froxels_analytical(prev_screen_pos, current_screen_pos, segment_ref, config);
            }
            default: {
                trace_segment_through_froxels_linear(prev_screen_pos, current_screen_pos, config);
            }
        }

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
@group(0) @binding(#TILE_COUNTS_BUFFER) var<storage, read> input_counts: array<u32>; // tile_counts_buffer (non-atomic read)
@group(0) @binding(#TILE_OFFSETS_BUFFER) var<storage, read_write> output_offsets: array<u32>; // tile_offsets_buffer

const SCAN_THREADS: u32 = #{NUMBER_OF_THREADS_PER_WORKGROUP}; // e.g., 256
const SCAN_SUBGROUP_THREADS: u32 = #{NUMBER_OF_THREADS_PER_SUBGROUP}; // e.g., 32
const SCAN_SUBGROUPS: u32 = SCAN_THREADS / SCAN_SUBGROUP_THREADS; // e.g., 8

// Shared memory for inter-subgroup communication and temporary storage
var<workgroup> subgroup_partials: array<u32, SCAN_SUBGROUPS>;
var<workgroup> wg_scan_storage: array<u32, SCAN_THREADS>; // For Blelloch intermediate/final values
// --- Helper Functions ---

fn get_scan_workgroup_index(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    let flat_grid_index = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    return flat_grid_index + pc.workgroup_offset;
}

// Performs an INCLUSIVE scan within the workgroup using Blelloch method modification.
// Stores the intermediate (potentially zeroed last element) result in wg_scan_storage.
// Returns the total sum of the workgroup (only valid for the last thread).
fn workgroup_inclusive_scan_blelloch(
    local_id: vec3u,
    subgroup_id: u32,
    subgroup_local_id: u32,
    value: u32
) -> u32 {
    let sg_inclusive_sum = subgroupInclusiveAdd(value);

    // Last lane of each subgroup writes its inclusive sum (subgroup total) to shared memory
    if local_id.x < SCAN_SUBGROUPS { subgroup_partials[local_id.x] = 0u; }
    workgroupBarrier();
    if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
        subgroup_partials[subgroup_id] = sg_inclusive_sum;
    }
    workgroupBarrier();

    // Scan the subgroup partial sums (sequentially by thread 0) -> Exclusive Prefix
    if local_id.x == 0u {
        var accumulator = 0u;
        for (var i = 0u; i < SCAN_SUBGROUPS; i = i + 1u) {
            let temp = subgroup_partials[i];
            subgroup_partials[i] = accumulator; // Write exclusive prefix
            accumulator += temp;
        }
    }
    workgroupBarrier();

    // Combine subgroup scan results with the scanned partials (exclusive prefix)
    let prefix_for_my_subgroup = subgroup_partials[subgroup_id];
    let block_local_inclusive_sum = sg_inclusive_sum + prefix_for_my_subgroup;

    // Store intermediate result in workgroup storage (for Blelloch down-sweep prep)
    wg_scan_storage[local_id.x] = block_local_inclusive_sum;
    workgroupBarrier();

    // Blelloch modification: Last thread reads total sum AND zeros its storage slot
    var total_wg_sum = 0u;
    if local_id.x == SCAN_THREADS - 1u {
        total_wg_sum = wg_scan_storage[local_id.x]; // Read total sum before zeroing
        wg_scan_storage[local_id.x] = 0u;          // Zero last element's storage
    }
    // Note: total_wg_sum only valid on last thread here.
    workgroupBarrier();

    // Intermediate results for down-sweep are now in wg_scan_storage (last element zeroed)
    return total_wg_sum;
}

// Performs an EXCLUSIVE scan within the workgroup.
// Stores the final exclusive scan result in wg_scan_storage.
// Returns the total sum of the workgroup (exclusive sum of last element + value of last element).
// Only valid for the last thread.
fn workgroup_exclusive_scan(
    local_id: vec3u,
    subgroup_id: u32,
    subgroup_local_id: u32,
    value: u32
) -> u32 {

    let sg_exclusive_sum = subgroupExclusiveAdd(value);
    let sg_inclusive_sum = subgroupInclusiveAdd(value); // Need inclusive sum for totals

    // Last lane of each subgroup writes its INCLUSIVE sum (subgroup total) to shared memory
    if local_id.x < SCAN_SUBGROUPS { subgroup_partials[local_id.x] = 0u; }
    workgroupBarrier();
    if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
        subgroup_partials[subgroup_id] = sg_inclusive_sum;
    }
    workgroupBarrier();

    // Scan the subgroup partial sums (sequentially by thread 0) -> Exclusive Prefix
    if local_id.x == 0u {
        var accumulator = 0u;
        for (var i = 0u; i < SCAN_SUBGROUPS; i = i + 1u) {
            let temp = subgroup_partials[i]; // Read original inclusive sum
            subgroup_partials[i] = accumulator; // Write exclusive prefix
            accumulator += temp;
        }
    }
    workgroupBarrier();

    // Combine subgroup exclusive scan results with the scanned partials (exclusive prefix)
    let prefix_for_my_subgroup = subgroup_partials[subgroup_id];
    let final_exclusive_sum = sg_exclusive_sum + prefix_for_my_subgroup;

    // Store final exclusive sum in workgroup storage
    wg_scan_storage[local_id.x] = final_exclusive_sum;
    workgroupBarrier();

    // Calculate total sum (last element's exclusive sum + last element's value)
    var total_wg_sum = 0u;
    if local_id.x == SCAN_THREADS - 1u {
        // Read the last element's computed exclusive sum and add its original value
        total_wg_sum = wg_scan_storage[SCAN_THREADS - 1u] + value;
    }
    // Return value only used by last thread.
    return total_wg_sum;
}
#endif // SCAN_STAGE COMMON

// --- Scan Stages ---

#ifdef STAGE_SCAN_SUMS
// Up-Sweep Stage: Calculates block sums and intermediate 'y' values for Blelloch down-sweep.
@compute @workgroup_size(SCAN_THREADS, 1, 1)
fn scan_sums(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32
) {
    let wg_idx = get_scan_workgroup_index(workgroup_id, num_workgroups);
    let element_idx_in_pass = wg_idx * SCAN_THREADS + local_id.x; // Index relative to this pass's data block (0-based)
    let num_elements_this_pass = pc.scan_save_base - pc.scan_load_base; // Number of elements for this pass
    // Index to read from global buffer section for this pass's input
    let read_idx = pc.scan_load_base + element_idx_in_pass;

    var value = 0u;
    // Read input value only if within the declared number of elements for this pass
    if element_idx_in_pass < num_elements_this_pass {
        if pc.scan_load_base == 0u { // First pass reads original counts
            value = input_counts[read_idx];
        } else { 
            // Subsequent passes read intermediate sums from output buffer
            value = output_offsets[read_idx];
        }
    }

    // wg_scan_storage now holds intermediate 'y' values (last element zeroed)
    let total_wg_sum = workgroup_inclusive_scan_blelloch(
        local_id, subgroup_id, subgroup_local_id, value
    );

    // Write intermediate result 'y_i' back to the *same location* it was read from.
    // Only write if the thread processed valid data for this pass
    if element_idx_in_pass < num_elements_this_pass {
        output_offsets[read_idx] = wg_scan_storage[local_id.x];
    }

    // Last thread writes the total workgroup sum for the *next* level's input
    if local_id.x == SCAN_THREADS - 1u {
       // Only write a sum if the workgroup's *first* element was within bounds.
        let first_element_idx_in_pass = wg_idx * SCAN_THREADS;
        if first_element_idx_in_pass < num_elements_this_pass {
            let sum_write_idx = pc.scan_save_base + wg_idx;
            output_offsets[sum_write_idx] = total_wg_sum;
        }
    }
}
#endif // STAGE_SCAN_SUMS

#ifdef STAGE_SCAN_LAST
// Top-Level Scan Stage: Performs exclusive scan on the highest level block sums.
@compute @workgroup_size(SCAN_THREADS, 1, 1)
fn scan_last(
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32
) {
    let element_idx_in_pass = local_id.x; // Index relative to this pass (0..SCAN_THREADS-1)

    // Number of elements = difference between save/load bases (exclusive end for save_base)
    let num_elements_this_pass = pc.scan_save_base - pc.scan_load_base;

    // Index to read sums from
    let read_idx = pc.scan_load_base + element_idx_in_pass;

    var value = 0u;
    if element_idx_in_pass < num_elements_this_pass {
        value = output_offsets[read_idx];
    }

    // wg_scan_storage now holds final exclusive scan results for the workgroup elements
    let total_sum_from_helper = workgroup_exclusive_scan(
        local_id, subgroup_id, subgroup_local_id, value
    );

    // Write exclusive scan result back to the *same location* it was read from.
    if element_idx_in_pass < num_elements_this_pass {
        output_offsets[read_idx] = wg_scan_storage[local_id.x];
    }

    if (element_idx_in_pass == num_elements_this_pass - 1u) && (num_elements_this_pass > 0u) {
        // This thread processed the last valid element. Calculate total based on its results.
        let last_element_exclusive_sum = wg_scan_storage[element_idx_in_pass];
        let last_element_value = value; // Value read by this thread
        output_offsets[pc.scan_save_base] = last_element_exclusive_sum + last_element_value;
    } else if local_id.x == 0u && num_elements_this_pass == 0u {
         // Handle edge case of zero elements: Write 0 total sum. Thread 0 does this.
        output_offsets[pc.scan_save_base] = 0u;
    }
}
#endif // STAGE_SCAN_LAST

#ifdef STAGE_SCAN_PRFX
// Down-Sweep Stage: Propagates block prefixes down and combines with intermediate 'y' values.
@compute @workgroup_size(SCAN_THREADS, 1, 1)
fn scan_prfx(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32, // Unused but keep signature
    @builtin(subgroup_invocation_id) subgroup_local_id: u32 // Unused
) {
    let wg_idx = get_scan_workgroup_index(workgroup_id, num_workgroups);
    let element_idx_in_pass = wg_idx * SCAN_THREADS + local_id.x; // Index relative to this pass's data block
    let num_elements_this_pass = pc.scan_save_base - pc.scan_load_base; // Number of elements for this pass

    // Read the block prefix sum (exclusive sum of the preceding block)
    // These prefixes were generated by the previous level (SCAN_LAST or SCAN_PRFX)
    // and are stored starting at pc.scan_load_base.
    let prefix_read_idx = pc.scan_load_base + wg_idx;
    let block_prefix_sum = output_offsets[prefix_read_idx];

    if element_idx_in_pass < num_elements_this_pass {
        // Calculate index to read intermediate 'y_i' and write final output value.
        // 'y_i' values reside where the corresponding SCAN_SUMS pass wrote them.
        let data_rw_idx = pc.scan_save_base + element_idx_in_pass;

        // Read the intermediate value y_i stored by the corresponding SCAN_SUMS pass
        let y_i = output_offsets[data_rw_idx];

        // Store y_i in shared memory to efficiently get y_{i-1}
        wg_scan_storage[local_id.x] = y_i;
        workgroupBarrier();

        // Get y_{i-1} (value from previous thread in workgroup's intermediate value)
        // If local_id.x is 0, y_im1 should be 0.
        let y_im1 = select(0u, wg_scan_storage[local_id.x - 1u], local_id.x > 0u);

        // Calculate final exclusive sum: block_prefix + y_{i-1}
        let final_value = block_prefix_sum + y_im1;

        // Write final exclusive prefix sum value back to global memory
        output_offsets[data_rw_idx] = final_value;
    }
}
#endif // STAGE_SCAN_PRFX


// --- STAGE_INIT_PLACE ---
#ifdef STAGE_INIT_PLACE

@group(0) @binding(#TILE_OFFSETS_BUFFER) var<storage, read> tile_offsets_buffer: array<u32>;
@group(0) @binding(#CURRENT_TILE_WRITE_INDICES) var<storage, read_write> current_tile_write_indices_buffer: array<atomic<u32>>;

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

@group(0) @binding(#VERTEX_BUFFER) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(#INDEX_BUFFER) var<storage, read> indices: array<u32>;
@group(0) @binding(#META_BUFFER) var<storage, read> strand_metadata: array<StrandMeta>;
@group(0) @binding(#CURRENT_TILE_WRITE_INDICES) var<storage, read_write> current_tile_write_indices_buffer: array<atomic<u32>>;
@group(0) @binding(#FROXEL_TILE_BUFFER) var<storage, read_write> packed_segments_buffer: array<SegmentRef>; // Write-only effectively
@group(0) @binding(#FROXEL_CONFIG) var<uniform> config: FroxelConfig;
@group(0) @binding(#VIEW_UNIFORM) var<uniform> view: View;
@group(0) @binding(#LIGHT_UNIFORM) var<uniform> lights: types::Lights;
@group(0) @binding(#GEO_BUFFER) var<storage, read> geos: array<StrandGeo>; // Has AABB for bounds check

@compute @workgroup_size(64, 1, 1)
fn place_strands(@builtin(global_invocation_id) id: vec3<u32>) {
    let strand_idx = id.x;
    let num_strands = arrayLength(&strand_metadata);

    if strand_idx >= num_strands {
        return;
    }
    let aabb = geos[0].aabb;
    let strand_hash = wang_hash(strand_idx);
    let sample_threshold = hash_to_unit_float(strand_hash);
    
    #ifdef SHADOWS
        let light: types::DirectionalLight = lights.directional_lights[LIGHT_INDEX];
        let cascade = light.cascades[0]; // TODO: select cascade based on distance
        let clip_from_world = cascade.clip_from_world;
        let aabb_clip = find_clip_bounds(clip_from_world, aabb.min, aabb.max);
        // culling
        if stochastic_cull_light(aabb_clip, sample_threshold) {
            return;
        }
        let aabb_znear_zfar = vec2<f32>(aabb_clip[0].z, aabb_clip[1].z);
    #else
        let clip_from_world = view.unjittered_clip_from_world;
        // culling
        if stochastic_cull_camera(view, aabb, sample_threshold) {
            return;
        }
        let aabb_clip = find_clip_bounds(clip_from_world, aabb.min, aabb.max);
        let aabb_znear_zfar = vec2<f32>(aabb_clip[0].z, aabb_clip[1].z);
    #endif

    let strand_meta = strand_metadata[strand_idx];
    let num_vertices_in_strand = strand_meta.count;
    let start_vertex_offset = strand_meta.offset; // Offset into vertices buffer

    if num_vertices_in_strand < 2u {
        return;
    }

    var prev_vtx = vertices[indices[start_vertex_offset]];
    var prev_screen_pos = world_to_screen(prev_vtx, clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);

    for (var i = 1u; i < num_vertices_in_strand; i = i + 1u) {
        let current_vtx_idx = indices[start_vertex_offset + i];
        let current_vtx = vertices[current_vtx_idx];
        let current_screen_pos = world_to_screen(current_vtx, clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);

        // Define SegmentRef - How is segment_start_idx used?
        // Option 1: Index into index buffer
        let segment_start_vtx_idx = start_vertex_offset + i - 1u;
        let segment_ref = SegmentRef(strand_idx, segment_start_vtx_idx);

        switch MODE {
            case 2u: { // Analytical splines
                trace_segment_through_froxels_analytical(prev_screen_pos, current_screen_pos, segment_ref, config);
            }
            default: {
                trace_segment_through_froxels_linear(prev_screen_pos, current_screen_pos, segment_ref, config);
            }
        }

        prev_screen_pos = current_screen_pos;
    }
}

#endif // STAGE_PLACE