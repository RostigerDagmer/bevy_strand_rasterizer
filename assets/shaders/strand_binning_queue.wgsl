#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::mesh_view_types as types
#import "shaders/common.wgsl"::{ find_clip_bounds, world_to_screen, world_to_screen_aabbnorm, screen_to_world, calculate_froxel_index, wang_hash, hash_to_unit_float, canonical_min, canonical_min_mask}
#import "shaders/spline.wgsl"::{
    solve_cubic_3d,
    catmull_rom_t,
    catmull_rom_T_a,
    catmull_rom_t_l,
    catmull_rom_t_r,
    catmull_rom_A_short,
    catmull_rom_B_short,
    catmull_rom_coefficients_3d,
    catmull_rom_spline_roots_3d,
    spline_derivative_at,
    min_root_above }

// --- Structures ---

#import "shaders/types.wgsl"::{
    Aabb,
    FroxelConfig,
    SegmentRef,
    StrandGeo,
    StrandMeta,
    PushConstants,
}

#import "shaders/prefix_sum.wgsl"::{
    get_scan_workgroup_index,
    workgroup_inclusive_scan_blelloch,
    workgroup_exclusive_scan,
    SCAN_THREADS,
    SCAN_SUBGROUP_THREADS,
    SCAN_SUBGROUPS,
    subgroup_partials,
    wg_scan_storage
}

#import "shaders/task_contract.wgsl"::{
    BINNING_NUM_LEVELS,
    BinningTask,
    RasterWorkItem,
    unpack_binning_level,
    unpack_binning_shadow,
    unpack_binning_frustum,
    pack_binning_field,
}

// -----------------------------------------------------------------------------
// STATUS: queue-based hierarchical binning (WIP / design stub)
//
// Intent:
// 1) Consume BinningTask items from `binning_queue` (persistent-kernel style loop)
// 2) Trace segment through a level-dependent froxel grid
// 3) Emit either:
//    - child BinningTask(level+1) for refinement, or
//    - leaf raster work at final level
// 4) Batch global queue writes per workgroup using local chunk chains
//
// Current state:
// - ABI alignment to task_contract is in progress
// - several symbols come from allocator/chunk-pool prototypes and are unresolved
// - algorithmic shape is preserved for reference; implementation is incomplete
//
// TODO tags:
// - TODO(bin-queue/abi): field/contract wiring
// - TODO(bin-queue/chunk): chunk pool integration
// - TODO(bin-queue/leaf): leaf raster emission
// - TODO(bin-queue/wg-batch): workgroup batched commit path
// -----------------------------------------------------------------------------

const INITIAL_ROUND_ALLOC_SIZE: u32 = 32u;
const REGULAR_ROUND_ALLOC_SIZE: u32 = 16u;

// --- Common Helper Functions ---

fn sgn_i32(f: f32) -> i32 {
    if f > 1e-6 { return 1; }
    if f < -1e-6 { return -1; }
    return 0;
}

fn in_bounds(f: vec3<i32>, max_f: vec3<i32>) -> bool {
    return all(f >= vec3(0)) && all(f < max_f);
}

fn add_segment_ref_to_froxel(froxel_x: u32, froxel_y: u32, froxel_z: u32, cfg: FroxelConfig, level: u32, task_hash: u32) -> bool {
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



var<workgroup> wg_chunks: array<atomic<u32>, SCAN_THREADS>;

fn pull_chunk(task_hash: u32, local_id: u32, size: u32) -> u32 {
    // TODO(bin-queue/chunk): alloc/N_HEADS contract is not finalized.
    let pool_address: u32 = alloc(task_hash % N_HEADS, size);
    // atomicStore(&wg_chunks[local_id], pool_address);
    wg_chunks[local_id] = pool_address;
    return pool_address
}

fn get_chunk(task_hash: u32, local_id: u32, size: u32) -> Chunk {
    // TODO(bin-queue/chunk): settle chunk ownership and writeback semantics.
    // var chunk_addr = atomicLoad(&wg_chunks[local_id]);
    var chunk_addr = wg_chunks[local_id];
    if chunk_addr == 0u {
        chunk_addr = pull_chunk(task_hash, local_id, size);
        return chunk_pool[chunk_addr];
    } else {
        let chunk = chunk_pool[chunk_addr];
        if chunk.count < CHUNK_SIZE {
            return chunk;
        }
        chunk.next = pull_chunk(task_hash, local_id, size);
        return chunk_pool[chunk.next];
    }
}

fn trace_segment_through_froxels(task: BinningTask, p0: vec3<f32>, p1: vec3<f32>, cfg: FroxelConfig, level: u32, task_hash: u32) {
    // TODO(bin-queue/abi): use `task.id_info` semantics per level (L0 geo vs Li parent tile).
    // Input p0, p1 are screen-space coordinates (x, y, depth [0,1])
    if p0.x < 0.0 || p1.x < 0.0 { return; } // Skip off-screen or behind camera
    let r_ix = 0; //i32(ceil(half_thickness_px / f32(cfg.froxel_size_x)));
    let r_iy = 0; //i32(ceil(half_thickness_px / f32(cfg.froxel_size_y)));

    // Use i32 for stepping, but ensure non-negative before passing to add_segment_ref_to_froxel
    // Clamp depth index calculation strictly between 0 and depth_slices-1
    let f0 = vec3<i32>(
        i32(floor(p0.x / f32(cfg.froxel_size_x))),
        i32(floor(p0.y / f32(cfg.froxel_size_y))),
        clamp(i32(floor(p0.z * f32(cfg.depth_slices))), 0, i32(cfg.depth_slices) - 1)
    );

    let f1 = vec3<i32>(
        i32(floor(p1.x / f32(cfg.froxel_size_x))),
        i32(floor(p1.y / f32(cfg.froxel_size_y))),
        clamp(i32(floor(p1.z * f32(cfg.depth_slices))), 0, i32(cfg.depth_slices) - 1)
    );

    var f = f0;
    var dir = p1 - p0;
    // Need step in integer grid space, but derivation needs float dir
    var step = vec3<i32>(sgn_i32(dir.x), sgn_i32(dir.y), sgn_i32(dir.z)); // Integer step

    // Use screen space sizes for calculations
    let froxel_dim = vec3<f32>(
        f32(cfg.froxel_size_x),
        f32(cfg.froxel_size_y),
        1.0 / f32(cfg.depth_slices)
    );

    // Calculate delta distances - how far along the ray (in units of t) we must move
    // for the coord to change by one froxel size.
    // Avoid division by zero. Use a large number if dir component is zero.
    let safe_dir = select(
        dir,
        vec3<f32>(1e-6, 1e-6, 1e-6) * vec3<f32>(step),
        abs(dir) < vec3<f32>(1e-6, 1e-6, 1e-6)
    );

    var delta_dist = abs(froxel_dim / safe_dir);

    // Calculate initial distances (as t values) to the *next* voxel boundary
    let fract_p0 = p0 / froxel_dim;

    // Distance to next boundary = (boundary - current_pos) / direction
    var t_max = select(
        (floor(fract_p0) * froxel_dim - p0) / safe_dir,
        ((floor(fract_p0) + 1.0) * froxel_dim - p0) / safe_dir,
        step > vec3(0)
    );

    // Correct for zero direction components - they should never be the minimum t_max
    t_max = select(
        t_max,
        vec3<f32>(1e38, 1e38, 1e38),
        abs(dir) < vec3<f32>(1e-6, 1e-6, 1e-6)
    );

    // Pre-calculate screen/froxel bounds for loop check

    let max_f = vec3<i32>(
        i32((cfg.screen_width + cfg.froxel_size_x - 1u) / cfg.froxel_size_x),
        i32((cfg.screen_height + cfg.froxel_size_y - 1u) / cfg.froxel_size_y),
        i32(cfg.depth_slices)
    );

    var safety = 0u;
    let max_steps = u32(max_f.x + max_f.y + max_f.z + 3); // Generous upper bound
    let zero_thickness = r_ix == 0 && r_iy == 0;

    var chunk = 0xFFFFFFFFu;
    var current_count = 0u;

    loop {
        safety = safety + 1u;
        if safety > max_steps { break; } // Safety break

        // Add segment count to current froxel IF it's within valid bounds
        if in_bounds(f, max_f) {
            // TODO(bin-queue/chunk): finalize local chunk append path.
            if current_count >= CHUNK_SIZE {
                chunk = get_chunk(task_hash, local_id, select(INITIAL_ROUND_ALLOC_SIZE, REGULAR_ROUND_ALLOC_SIZE, level > 0u));
            }
            let froxel_id = calculate_froxel_index(froxel_x, froxel_y, froxel_z, cfg);
            switch level {
                // case 0u: {

                // }
                case (NUM_LEVELS - 1u): {
                    // TODO(bin-queue/leaf): emit raster-ready leaf work item.
                }
                default: {
                    // next level refinement task
                    let is_shadow = unpack_binning_shadow(task.packed_field);
                    let frustum_index = unpack_binning_frustum(task.packed_field);
                    let repacked = pack_binning_field(level + 1u, is_shadow, frustum_index);

                    let task = BinningTask(
                        froxel_id,
                        task.chunk_id,
                        task.seg_idx,
                        repacked
                    )
                    // 
                    chunk.items[count] = seg_idx,
                    chunk.count = chunk.count + 1u;
                }
            }
            current_count = chunk.count;
        } else {
            // Stop if we step out of bounds entirely
            break;
        }

        // Check if we've reached the end froxel (Manhattan distance check can be faster)
        if all(f == f1) { break; }

        let a_min = canonical_min_mask(t_max); // TODO: check if we can avoid canonical min
        // in the more than one min case the line segment passes through a corner.

        let fd = select(vec3<i32>(0), step, a_min);
        let dist = select(vec3<f32>(0), delta_dist, a_min);
        f += fd;
        t_max += dist;

         // Check if we have stepped past the target froxel along any axis where movement occurs
         // This prevents infinite loops for axis-aligned lines ending exactly on a boundary
        let pos_step = step > vec3(0);
        let neg_step = step < vec3(0);
        let g_f1 = f > f1;
        let l_f1 = f < f1;

        if any(pos_step && g_f1) || any(neg_step && l_f1) {
             break;
        }
    }

    workgroupBarrier();

    // TODO(bin-queue/wg-batch): this section is intended to amortize global queue atomics
    // by committing per-lane chunk chains in contiguous ranges reserved once per workgroup.
    let num_chunks = workgroup_exclusive_scan(lid, subgroup_id, subgroup_local_id, wg_chunk_counts[lid]);

    var base: u32 = 0u;
    if lid == WG_SIZE - 1u {
        // tally up work
        base = atomicAdd(&binning_queue.tail, num_chunks * BIN_TASK_SIZE / CHUNK_SIZE);
    }
    base = workgroupBroadcastFirst(base);

    workgroupBarrier();

    if chunk != 0xFFFFFFFFu {
        // submit work
        for (var i = 0u; i < num_chunks * BIN_TASK_SIZE / CHUNK_SIZE; i = i + BIN_TASK_SIZE) {
            let write_idx = base + i;
            let chunk_idx = i / CHUNK_SIZE;
            var chunk_pointer = wg_chunks[lid];
            for (var j = 0u; j < chunk_idx; j = j + 1) {
                chunk_pointer = chunk_pointer.next;
            }
            let chunk = chunk_pool[chunk_pointer];
            let task = BinningTask(chunk.items[0], chunk.items[1], chunk.items[2], chunk.items[3]);
            binning_queue.tasks[write_idx] = task;
        }
    }
}

fn nearest_pow2(n: f32) -> f32 {
    if n <= 0.0 {
        return 1.0;
    }
    return pow(2.0, round(log2(n)));
}

@group(0) @binding(#VERTEX_BUFFER) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(#INDEX_BUFFER) var<storage, read> indices: array<u32>;
@group(0) @binding(#META_BUFFER) var<storage, read> strand_metadata: array<StrandMeta>; // Use meta buffer
@group(0) @binding(#TILE_COUNTS_BUFFER) var<storage, read_write> tile_counts_buffer: array<atomic<u32>>;
@group(0) @binding(#FROXEL_CONFIG) var<uniform> config: FroxelConfig;
@group(0) @binding(#VIEW_UNIFORM) var<uniform> view: View; // Or ViewUniform
@group(0) @binding(#LIGHT_UNIFORM) var<uniform> lights: types::Lights;
@group(0) @binding(#GEO_BUFFER) var<storage, read> geos: array<StrandGeo>; // Has AABB for bounds check

// NOTE: hierarchy depth and packed-field layout are defined in task_contract.wgsl.
const NUM_LEVELS: u32 = BINNING_NUM_LEVELS;

fn process_level(task: BinningTask, local_id: vec3<u32>, task_hash: u32) {
    // Stage contract:
    // - unpack level/shadow/frustum from task.packed_field
    // - derive a level-local grid config
    // - trace one segment and emit child/leaf work
    let level = unpack_binning_level(task.packed_field);
    let is_shadow = unpack_binning_shadow(task.packed_field);
    let config_idx = unpack_binning_frustum(task.packed_field);
    let cfg = config[config_idx];
    let screen_xy = vec2<u32>(cfg.screen_width, cfg.screen_height);
    let fsize_xy = vec2<u32>(cfg.froxel_size_x, cfg.froxel_size_y);
    let grid_xy = screen_xy / pow(fsize_xy, NUM_LEVELS - level);
    let z_slices = u32(nearest_pow2(sqrt(f32(cfg.depth_slices))));
    let level_cfg = FroxelConfig(grid_xy.x, grid_xy.y, fsize_xy.x, fsize_xy.y, z_slices);
    let seg_idx = indices[task.seg_idx];
    let p0 = vertices[seg_idx];
    let p1 = vertices[seg_idx + 1u];
    trace_segment_through_froxels(task, p0.xyz, p1.xyz, level_cfg, level, task_hash);
}

fn kickoff_queue_empty() -> bool {
    // Queue empty predicate for persistent-kernel style loop control.
    return atomicLoad(&binning_queue.head) >= atomicLoad(&binning_queue.tail);
}

@compute @workgroup_size(SCAN_THREADS, 1, 1)
fn kernel(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32
) {
    // Persistent-kernel style dequeue/execute loop (WIP).
    loop {
        // 1. Get work item from queue: BinningTask
        var task_idx: u32;
        let binning_tail: u32 = atomicLoad(&binning_queue.tail);
        if local_id.x == 0u {
            task_idx = atomicAdd(&binning_queue.head, SCAN_THREADS);
        }

        task_idx = task_idx + local_id.x; // local task
        if task_idx >= binning_tail { break; }

        let task = binning_queue.tasks[task_idx];
        var task_hash: u32;
        if local_id.x == 0u {
            task_hash = wang_hash(task.packed_field ^ task.id_info ^ task.seg_idx);
        }
        process_level(task, local_id, task_hash);
        // TODO(bin-queue/flow): add explicit producer/consumer termination protocol
        // once child task emission and leaf emission are both implemented.
    }
}
