#import bevy_render::view::View
#import bevy_pbr::mesh_view_types as types

#import "shaders/common.wgsl"::{
    find_clip_bounds,
    world_to_screen,
    calculate_froxel_index,
    canonical_min_mask,
    wang_hash,
    hash_to_unit_float,
    l_and,
}

#import "shaders/types.wgsl"::{
    Aabb,
    DevicePtr,
    Vertices,
    Indices,
    Geos,
    Meta,
    StrandGeo,
    FroxelConfig,
    SegmentRef,
    PushConstants,
}

#import "shaders/task_contract.wgsl"::{
    FinePrepassTask,
    BinningTask,
    RasterWorkItem,
    pack_binning_field,
    unpack_binning_frustum,
}

#import "shaders/queues.wgsl"::{
    FinePrepassQueue,
    BinningQueue,
}

const WORKGROUP_SIZE: u32 = #WORKGROUP_SIZE;
const FINE_WORKGROUP_SIZE: u32 = #FINE_WORKGROUP_SIZE;
const SIZEOF_METADATA: u32 = #SIZEOF_METADATA;
const SIZEOF_GEO: u32 = #SIZEOF_GEO;
const POOL_CHUNK_SIZE: u32 = #POOL_CHUNK_SIZE;
const POOL_NUM_HEADS: u32 = #POOL_NUM_HEADS;
const CHUNK_WORD_STRIDE: u32 = 2u + POOL_CHUNK_SIZE;
const INVALID_PTR: u32 = 0xFFFFFFFFu;
var<push_constant> pc: PushConstants;

struct FrustumDesc {
    screen_width: u32,
    screen_height: u32,
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
    bucket_base: u32,
    bucket_count: u32,
    kind: u32,
}

struct RasterWorkQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    items: array<RasterWorkItem>,
}

// TODO: adjust allocator api so we can actually bind this as read only where no writes are required.
@group(#{BIND_ARRAYS}) @binding(#{VERTICES}) var<storage, read_write> vertices: binding_array<Vertices>;
@group(#{BIND_ARRAYS}) @binding(#{INDICES}) var<storage, read_write> indices: binding_array<Indices>;
@group(#{BIND_ARRAYS}) @binding(#{STRAND_METADATA}) var<storage, read_write> strand_metadata: binding_array<Meta>;
@group(#{BIND_ARRAYS}) @binding(#{STRAND_GEOS}) var<storage, read_write> geos: binding_array<Geos>;

@group(#{PAGE_TABLES}) @binding(#{VERTICES}) var<storage, read_write> t_vertices: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{INDICES}) var<storage, read_write> t_indices: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{STRAND_METADATA}) var<storage, read_write> t_strand_metadata: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{STRAND_GEOS}) var<storage, read_write> t_geos: array<DevicePtr>;

@group(#{PREPASS_GROUP}) @binding(#{VIEW_UNIFORM}) var<uniform> view: View;
@group(#{PREPASS_GROUP}) @binding(#{LIGHT_UNIFORM}) var<uniform> lights: types::Lights;
@group(#{PREPASS_GROUP}) @binding(#{CLUSTER_INDICES}) var<storage, read> clusterable_object_index_lists: types::ClusterLightIndexLists;
@group(#{PREPASS_GROUP}) @binding(#{CLUSTERABLE_OBJECTS}) var<storage, read> clusterable_objects: types::ClusterableObjects;
@group(#{PREPASS_GROUP}) @binding(#{CLUSTER_OFFSETS_AND_COUNTS}) var<storage, read> cluster_offsets_and_counts: types::ClusterOffsetsAndCounts;
@group(#{PREPASS_GROUP}) @binding(#{PREPASS_QUEUE}) var<storage, read_write> fine_phase_queue: FinePrepassQueue;
@group(#{PREPASS_GROUP}) @binding(#{BINNING_QUEUE}) var<storage, read_write> binning_queue: BinningQueue;
@group(#{PREPASS_GROUP}) @binding(#{VISIBLE_FLAGS}) var<storage, read_write> visible_flags: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{VISIBLE_GEO}) var<storage, read_write> visible_geos: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{GEO_PREFIX}) var<storage, read_write> geo_prefix: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{INDIRECT_BUFFER}) var<storage, read_write> dispatch_args: array<u32, 3u>;
@group(#{PREPASS_GROUP}) @binding(#{TILE_COUNTS_BUFFER}) var<storage, read_write> tile_counts_buffer: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{FROXEL_CONFIG}) var<uniform> config: FroxelConfig;
@group(#{PREPASS_GROUP}) @binding(#{TILE_OFFSETS_BUFFER}) var<storage, read_write> tile_offsets_buffer: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{CURRENT_TILE_WRITE_INDICES}) var<storage, read_write> current_tile_write_indices_buffer: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{FROXEL_TILE_BUFFER}) var<storage, read_write> packed_segments_buffer: array<SegmentRef>;
@group(#{PREPASS_GROUP}) @binding(#{FRUSTUM_TABLE}) var<storage, read> frustum_table: array<FrustumDesc>;
@group(#{PREPASS_GROUP}) @binding(#{FROXEL_BUCKET_HEADS}) var<storage, read_write> froxel_bucket_heads: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{CHUNK_POOL}) var<storage, read_write> chunk_pool_words: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{FREE_HEADS}) var<storage, read_write> free_heads: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{RASTER_WORK_QUEUE}) var<storage, read_write> raster_work_queue: RasterWorkQueue;

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

fn is_valid_ptr(_ptr: DevicePtr) -> bool {
    return _ptr.slab != 0xFFFFFFFFu;
}

fn stochastic_cull_camera(cam: View, aabb: Aabb, sample_threshold: f32) -> bool {
    if pc.stochastic_cull_enabled == 0u {
        return false;
    }
    let aabb_center = (aabb.max + aabb.min) * 0.5;
    let distance_to_cam = length(cam.world_position - aabb_center);
    let cull_max = max(pc.cull_max_dist, 1e-5);
    let norm_distance = max(distance_to_cam - pc.cull_min_dist, 0.0) / cull_max;
    return sample_threshold <= pow(norm_distance, pc.cull_exponent);
}

fn ceil_div_u32(x: u32, y: u32) -> u32 {
    return (x + y - 1u) / y;
}

fn frustum_to_config(desc: FrustumDesc) -> FroxelConfig {
    return FroxelConfig(
        desc.screen_width,
        desc.screen_height,
        desc.froxel_size_x,
        desc.froxel_size_y,
        desc.depth_slices,
    );
}

fn chunk_capacity() -> u32 {
    return arrayLength(&chunk_pool_words) / CHUNK_WORD_STRIDE;
}

fn alloc_chunk(head_id: u32) -> u32 {
    let cap = chunk_capacity();
    if cap == 0u {
        return INVALID_PTR;
    }
    let per_head = max(1u, cap / POOL_NUM_HEADS);
    let head = head_id % POOL_NUM_HEADS;
    let local = atomicAdd(&free_heads[head], 1u);
    if local >= per_head {
        return INVALID_PTR;
    }
    let idx = head * per_head + local;
    if idx >= cap {
        return INVALID_PTR;
    }
    return idx;
}

fn chunk_word_base(chunk_idx: u32) -> u32 {
    return chunk_idx * CHUNK_WORD_STRIDE;
}

fn chunk_store_next(chunk_idx: u32, next_ptr: u32) {
    chunk_pool_words[chunk_word_base(chunk_idx)] = next_ptr;
}

fn chunk_store_count(chunk_idx: u32, count: u32) {
    chunk_pool_words[chunk_word_base(chunk_idx) + 1u] = count;
}

fn chunk_store_item(chunk_idx: u32, local_idx: u32, item: u32) {
    chunk_pool_words[chunk_word_base(chunk_idx) + 2u + local_idx] = item;
}

fn append_sparse_froxel_ref(frustum_id: u32, local_froxel_idx: u32, seg_idx: u32) {
    if frustum_id >= arrayLength(&frustum_table) {
        return;
    }
    let desc = frustum_table[frustum_id];
    if local_froxel_idx >= desc.bucket_count {
        return;
    }
    let bucket_idx = desc.bucket_base + local_froxel_idx;
    if bucket_idx >= arrayLength(&froxel_bucket_heads) {
        return;
    }

    let work_idx = atomicAdd(&raster_work_queue.tail, 1u);
    if work_idx >= arrayLength(&raster_work_queue.items) {
        return;
    }
    raster_work_queue.items[work_idx] = RasterWorkItem(seg_idx, frustum_id, local_froxel_idx);

    let chunk_idx = alloc_chunk(bucket_idx ^ seg_idx);
    if chunk_idx == INVALID_PTR {
        return;
    }
    let prev = atomicExchange(&froxel_bucket_heads[bucket_idx], chunk_idx);
    chunk_store_next(chunk_idx, prev);
    chunk_store_count(chunk_idx, 1u);
    // Payload [0] stores raster_work_queue item index.
    chunk_store_item(chunk_idx, 0u, work_idx);
}

fn sgn_i32(f: f32) -> i32 {
    if f > 1e-6 { return 1; }
    if f < -1e-6 { return -1; }
    return 0;
}

fn in_bounds(f: vec3<i32>, max_f: vec3<i32>) -> bool {
    return all(f >= vec3(0)) && all(f < max_f);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn broad_prepass(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let global_invocation = ((workgroup_id.z * num_workgroups.y + workgroup_id.y) * num_workgroups.x + workgroup_id.x) * WORKGROUP_SIZE + local_id.x;
    let total_invocations = num_workgroups.x * num_workgroups.y * num_workgroups.z * WORKGROUP_SIZE;

    let n_geos = arrayLength(&t_geos);
    var geo_idx = global_invocation;

    while geo_idx < n_geos {
        let geo_ptr = t_geos[geo_idx];
        let meta_ptr = t_strand_metadata[geo_idx];

        if !is_valid_ptr(geo_ptr) || !is_valid_ptr(meta_ptr) {
            geo_idx += total_invocations;
            continue;
        }

        let geo = geos[geo_ptr.slab].gs[geo_ptr.offset / SIZEOF_GEO];
        let strand_count_from_meta = meta_ptr.size / SIZEOF_METADATA;
        let strand_count = min(strand_count_from_meta, geo.strand_count);
        let geo_visible = true; // TODO: frustrum test?

        if geo_idx < arrayLength(&visible_flags) {
            visible_flags[geo_idx] = u32(geo_visible);
        }

        if geo_visible {
            for (var strand_local = 0u; strand_local < strand_count; strand_local = strand_local + 1u) {
                let strand_hash = wang_hash(geo_idx + strand_local);
                let strand_visible = !stochastic_cull_camera(view, geo.aabb, hash_to_unit_float(strand_hash));
                if strand_visible {
                    let task_index = atomicAdd(&fine_phase_queue.tail, 1u);
                    if task_index >= arrayLength(&fine_phase_queue.tasks) {
                        break;
                    }
                    fine_phase_queue.tasks[task_index] = FinePrepassTask(geo_idx, strand_local);
                }
            }
        }

        geo_idx += total_invocations;
    }
}

@compute @workgroup_size(1, 1, 1)
fn finalize_prepass() {
    let fine_task_count = atomicLoad(&fine_phase_queue.tail);
    dispatch_args[0] = ceil_div_u32(fine_task_count, FINE_WORKGROUP_SIZE);
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}

@compute @workgroup_size(1, 1, 1)
fn finalize_binning() {
    let binning_task_count = atomicLoad(&binning_queue.tail);
    dispatch_args[0] = ceil_div_u32(binning_task_count, FINE_WORKGROUP_SIZE);
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}

@compute @workgroup_size(FINE_WORKGROUP_SIZE, 1, 1)
fn fine_prepass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let task_idx = gid.x;
    let fine_task_count = atomicLoad(&fine_phase_queue.tail);

    if task_idx >= fine_task_count {
        return;
    }

    let task = fine_phase_queue.tasks[task_idx];

    if task.inst_id >= arrayLength(&t_strand_metadata) || task.inst_id >= arrayLength(&t_indices) {
        return;
    }

    let meta_ptr = t_strand_metadata[task.inst_id];
    let index_ptr = t_indices[task.inst_id];

    if !is_valid_ptr(meta_ptr) || !is_valid_ptr(index_ptr) {
        return;
    }

    let meta_base = meta_ptr.offset / SIZEOF_METADATA;
    let meta_count = meta_ptr.size / SIZEOF_METADATA;
    if task.strand_local >= meta_count {
        return;
    }

    let strand_meta = strand_metadata[meta_ptr.slab].ms[meta_base + task.strand_local];
    if strand_meta.count < 2u {
        return;
    }

    // Camera frustum only for now (frustum_id = 0, level = 0, non-shadow).
    let packed_field_camera = pack_binning_field(0u, 0u, 0u);
    let segment_count = strand_meta.count - 1u;

    for (var i = 0u; i < segment_count; i = i + 1u) {
        let seg_index = strand_meta.offset + i;
        let write_idx = atomicAdd(&binning_queue.tail, 1u);
        if write_idx < arrayLength(&binning_queue.tasks) {
            binning_queue.tasks[write_idx] = BinningTask(
                task.inst_id,
                task.strand_local,
                seg_index,
                packed_field_camera,
            );
        }
    }
}


fn add_segment_ref_to_froxel(froxel_x: u32, froxel_y: u32, froxel_z: u32, cfg: FroxelConfig) -> bool {
    let froxel_idx = calculate_froxel_index(froxel_x, froxel_y, froxel_z, cfg);
    if froxel_idx < arrayLength(&tile_counts_buffer) {
        atomicAdd(&tile_counts_buffer[froxel_idx], 1u);
        return true;
    }
    return false;
}

fn add_segment_to_packed_froxel(
    froxel_x: u32,
    froxel_y: u32,
    froxel_z: u32,
    seg_ref: SegmentRef,
    cfg: FroxelConfig,
) -> bool {
    let froxel_idx = calculate_froxel_index(froxel_x, froxel_y, froxel_z, cfg);
    if froxel_idx >= arrayLength(&current_tile_write_indices_buffer) {
        return false;
    }
    let write_index = atomicAdd(&current_tile_write_indices_buffer[froxel_idx], 1u);
    if write_index >= arrayLength(&packed_segments_buffer) {
        return false;
    }
    packed_segments_buffer[write_index] = seg_ref;
    return true;
}

// Returns true if the float is NaN
fn isNan(val: f32) -> bool {
    let u_val = bitcast<u32>(val);
    // Check if exponent is all 1s (0x7F800000) and mantissa is non-zero (0x007FFFFF)
    return (u_val & 0x7F800000u) == 0x7F800000u && (u_val & 0x007FFFFFu) != 0u;
}

// Returns true if the float is Infinity (positive or negative)
fn isInf(val: f32) -> bool {
    let u_val = bitcast<u32>(val);
    // Check if exponent is all 1s (0x7F800000) and mantissa is zero
    return (u_val & 0x7F800000u) == 0x7F800000u && (u_val & 0x007FFFFFu) == 0u;
}

// Helper: Check a vec3 for any bad values
fn isValid(v: vec3<f32>) -> bool {
    return !(isNan(v.x) || isNan(v.y) || isNan(v.z) || isInf(v.x) || isInf(v.y) || isInf(v.z));
}

fn trace_segment_through_froxels_linear(p0: vec3<f32>, p1: vec3<f32>, cfg: FroxelConfig) {
    if !isValid(p0) || !isValid(p1) {
        return;
    }
    if p0.x < 0.0 || p1.x < 0.0 {
        return;
    }

    let screen = vec2<f32>(f32(cfg.screen_width), f32(cfg.screen_height));
    let seg_xy_min = min(p0.xy, p1.xy);
    let seg_xy_max = max(p0.xy, p1.xy);
    // Segment does not touch the raster screen rect in XY.
    if seg_xy_max.x < 0.0 || seg_xy_max.y < 0.0 || seg_xy_min.x >= screen.x || seg_xy_min.y >= screen.y {
        return;
    }
    let seg_z_min = min(p0.z, p1.z);
    let seg_z_max = max(p0.z, p1.z);
    // Segment does not touch normalized [0,1] depth interval.
    if seg_z_max < 0.0 || seg_z_min > 1.0 {
        return;
    }

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
    let dir = p1 - p0;
    let step = vec3<i32>(sgn_i32(dir.x), sgn_i32(dir.y), sgn_i32(dir.z));
    let froxel_dim = vec3<f32>(
        f32(cfg.froxel_size_x),
        f32(cfg.froxel_size_y),
        1.0 / f32(cfg.depth_slices)
    );
    let safe_dir = select(
        dir,
        vec3<f32>(1e-6, 1e-6, 1e-6) * vec3<f32>(step),
        abs(dir) < vec3<f32>(1e-6, 1e-6, 1e-6)
    );
    let delta_dist = abs(froxel_dim / safe_dir);
    let fract_p0 = p0 / froxel_dim;
    var t_max = select(
        (floor(fract_p0) * froxel_dim - p0) / safe_dir,
        ((floor(fract_p0) + 1.0) * froxel_dim - p0) / safe_dir,
        step > vec3(0)
    );
    t_max = select(
        t_max,
        vec3<f32>(1e38, 1e38, 1e38),
        abs(dir) < vec3<f32>(1e-6, 1e-6, 1e-6)
    );

    let max_f = vec3<i32>(
        i32((cfg.screen_width + cfg.froxel_size_x - 1u) / cfg.froxel_size_x),
        i32((cfg.screen_height + cfg.froxel_size_y - 1u) / cfg.froxel_size_y),
        i32(cfg.depth_slices)
    );
    var safety = 0u;
    let max_steps = u32(max_f.x + max_f.y + max_f.z + 3);

    loop {
        safety = safety + 1u;
        if safety > max_steps {
            break;
        }

        if in_bounds(f, max_f) {
            if !add_segment_ref_to_froxel(u32(f.x), u32(f.y), u32(f.z), cfg) {
                break;
            }
        } else {
            break;
        }

        if all(f == f1) {
            break;
        }

        let a_min = canonical_min_mask(t_max);
        let fd = select(vec3<i32>(0), step, a_min);
        let dist = select(vec3<f32>(0), delta_dist, a_min);
        f += fd;
        t_max += dist;

        let pos_step = step > vec3(0);
        let neg_step = step < vec3(0);
        let g_f1 = f > f1;
        let l_f1 = f < f1;
        if any(l_and(pos_step, g_f1)) || any(l_and(neg_step, l_f1)) {
            break;
        }
    }
}

fn trace_segment_through_froxels_sparse(
    p0: vec3<f32>,
    p1: vec3<f32>,
    cfg: FroxelConfig,
    frustum_id: u32,
    seg_idx: u32,
) {
    if !isValid(p0) || !isValid(p1) {
        return;
    }
    if p0.x < 0.0 || p1.x < 0.0 {
        return;
    }

    let screen = vec2<f32>(f32(cfg.screen_width), f32(cfg.screen_height));
    let seg_xy_min = min(p0.xy, p1.xy);
    let seg_xy_max = max(p0.xy, p1.xy);
    if seg_xy_max.x < 0.0 || seg_xy_max.y < 0.0 || seg_xy_min.x >= screen.x || seg_xy_min.y >= screen.y {
        return;
    }
    let seg_z_min = min(p0.z, p1.z);
    let seg_z_max = max(p0.z, p1.z);
    if seg_z_max < 0.0 || seg_z_min > 1.0 {
        return;
    }

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
    let dir = p1 - p0;
    let step = vec3<i32>(sgn_i32(dir.x), sgn_i32(dir.y), sgn_i32(dir.z));
    let froxel_dim = vec3<f32>(
        f32(cfg.froxel_size_x),
        f32(cfg.froxel_size_y),
        1.0 / f32(cfg.depth_slices)
    );
    let safe_dir = select(
        dir,
        vec3<f32>(1e-6, 1e-6, 1e-6) * vec3<f32>(step),
        abs(dir) < vec3<f32>(1e-6, 1e-6, 1e-6)
    );
    let delta_dist = abs(froxel_dim / safe_dir);
    let fract_p0 = p0 / froxel_dim;
    var t_max = select(
        (floor(fract_p0) * froxel_dim - p0) / safe_dir,
        ((floor(fract_p0) + 1.0) * froxel_dim - p0) / safe_dir,
        step > vec3(0)
    );
    t_max = select(
        t_max,
        vec3<f32>(1e38, 1e38, 1e38),
        abs(dir) < vec3<f32>(1e-6, 1e-6, 1e-6)
    );

    let max_f = vec3<i32>(
        i32((cfg.screen_width + cfg.froxel_size_x - 1u) / cfg.froxel_size_x),
        i32((cfg.screen_height + cfg.froxel_size_y - 1u) / cfg.froxel_size_y),
        i32(cfg.depth_slices)
    );
    let num_tiles_x = (cfg.screen_width + cfg.froxel_size_x - 1u) / cfg.froxel_size_x;
    let num_tiles_y = (cfg.screen_height + cfg.froxel_size_y - 1u) / cfg.froxel_size_y;

    var safety = 0u;
    let max_steps = u32(max_f.x + max_f.y + max_f.z + 3);

    loop {
        safety = safety + 1u;
        if safety > max_steps {
            break;
        }

        if in_bounds(f, max_f) {
            let fx = u32(f.x);
            let fy = u32(f.y);
            let fz = u32(f.z);
            if !add_segment_ref_to_froxel(fx, fy, fz, cfg) {
                break;
            }
            let froxel_idx = (fz * num_tiles_y + fy) * num_tiles_x + fx;
            append_sparse_froxel_ref(frustum_id, froxel_idx, seg_idx);
        } else {
            break;
        }

        if all(f == f1) {
            break;
        }

        let a_min = canonical_min_mask(t_max);
        let fd = select(vec3<i32>(0), step, a_min);
        let dist = select(vec3<f32>(0), delta_dist, a_min);
        f += fd;
        t_max += dist;

        let pos_step = step > vec3(0);
        let neg_step = step < vec3(0);
        let g_f1 = f > f1;
        let l_f1 = f < f1;
        if any(l_and(pos_step, g_f1)) || any(l_and(neg_step, l_f1)) {
            break;
        }
    }
}

fn trace_segment_through_froxels_linear_place(
    p0: vec3<f32>,
    p1: vec3<f32>,
    seg_ref: SegmentRef,
    cfg: FroxelConfig,
) {
    if !isValid(p0) || !isValid(p1) {
        return;
    }
    if p0.x < 0.0 || p1.x < 0.0 {
        return;
    }

    let screen = vec2<f32>(f32(cfg.screen_width), f32(cfg.screen_height));
    let seg_xy_min = min(p0.xy, p1.xy);
    let seg_xy_max = max(p0.xy, p1.xy);
    if seg_xy_max.x < 0.0 || seg_xy_max.y < 0.0 || seg_xy_min.x >= screen.x || seg_xy_min.y >= screen.y {
        return;
    }
    let seg_z_min = min(p0.z, p1.z);
    let seg_z_max = max(p0.z, p1.z);
    if seg_z_max < 0.0 || seg_z_min > 1.0 {
        return;
    }

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
    let dir = p1 - p0;
    let step = vec3<i32>(sgn_i32(dir.x), sgn_i32(dir.y), sgn_i32(dir.z));
    let froxel_dim = vec3<f32>(
        f32(cfg.froxel_size_x),
        f32(cfg.froxel_size_y),
        1.0 / f32(cfg.depth_slices)
    );
    let safe_dir = select(
        dir,
        vec3<f32>(1e-6, 1e-6, 1e-6) * vec3<f32>(step),
        abs(dir) < vec3<f32>(1e-6, 1e-6, 1e-6)
    );
    let delta_dist = abs(froxel_dim / safe_dir);
    let fract_p0 = p0 / froxel_dim;
    var t_max = select(
        (floor(fract_p0) * froxel_dim - p0) / safe_dir,
        ((floor(fract_p0) + 1.0) * froxel_dim - p0) / safe_dir,
        step > vec3(0)
    );
    t_max = select(
        t_max,
        vec3<f32>(1e38, 1e38, 1e38),
        abs(dir) < vec3<f32>(1e-6, 1e-6, 1e-6)
    );

    let max_f = vec3<i32>(
        i32((cfg.screen_width + cfg.froxel_size_x - 1u) / cfg.froxel_size_x),
        i32((cfg.screen_height + cfg.froxel_size_y - 1u) / cfg.froxel_size_y),
        i32(cfg.depth_slices)
    );
    var safety = 0u;
    let max_steps = u32(max_f.x + max_f.y + max_f.z + 3);

    loop {
        safety = safety + 1u;
        if safety > max_steps {
            break;
        }

        if in_bounds(f, max_f) {
            if !add_segment_to_packed_froxel(u32(f.x), u32(f.y), u32(f.z), seg_ref, cfg) {
                break;
            }
        } else {
            break;
        }

        if all(f == f1) {
            break;
        }

        let a_min = canonical_min_mask(t_max);
        let fd = select(vec3<i32>(0), step, a_min);
        let dist = select(vec3<f32>(0), delta_dist, a_min);
        f += fd;
        t_max += dist;

        let pos_step = step > vec3(0);
        let neg_step = step < vec3(0);
        let g_f1 = f > f1;
        let l_f1 = f < f1;
        if any(l_and(pos_step, g_f1)) || any(l_and(neg_step, l_f1)) {
            break;
        }
    }
}


@compute @workgroup_size(FINE_WORKGROUP_SIZE, 1, 1)
fn binning_queue_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let task_idx = gid.x;
    let binning_task_count = atomicLoad(&binning_queue.tail);
    if task_idx >= binning_task_count {
        return;
    }

    let task = binning_queue.tasks[task_idx];
    let frustum_id = unpack_binning_frustum(task.packed_field);
    if frustum_id >= arrayLength(&frustum_table) {
        return;
    }
    let frustum = frustum_table[frustum_id];
    let frustum_cfg = frustum_to_config(frustum);
    let inst_id = task.id_info;
    if inst_id >= arrayLength(&t_vertices) || inst_id >= arrayLength(&t_indices) || inst_id >= arrayLength(&t_geos) {
        return;
    }

    let vertex_ptr = t_vertices[inst_id];
    let index_ptr = t_indices[inst_id];
    let geo_ptr = t_geos[inst_id];
    if !is_valid_ptr(vertex_ptr) || !is_valid_ptr(index_ptr) || !is_valid_ptr(geo_ptr) {
        return;
    }

    let index_count = index_ptr.size / 4u;
    if task.seg_idx + 1u >= index_count {
        return;
    }

    let geo = geos[geo_ptr.slab].gs[geo_ptr.offset / SIZEOF_GEO];
    let clip_from_world = view.unjittered_clip_from_world;
    let clip_bounds = find_clip_bounds(clip_from_world, geo.aabb.min, geo.aabb.max);
    if !isValid(clip_bounds[0]) || !isValid(clip_bounds[1]) {
        return;
    }
    if abs(clip_bounds[1].z - clip_bounds[0].z) < 1e-6 {
        return;
    }
    let aabb_znear_zfar = vec2<f32>(clip_bounds[0].z, clip_bounds[1].z);

    let vi0 = indices[index_ptr.slab].is[task.seg_idx];
    let vi1 = indices[index_ptr.slab].is[task.seg_idx + 1u];
    let p0_world = vec4<f32>(vertices[vertex_ptr.slab].vs[vi0], 1.0);
    let p1_world = vec4<f32>(vertices[vertex_ptr.slab].vs[vi1], 1.0);
    let p0 = world_to_screen(
        p0_world,
        clip_from_world,
        f32(frustum_cfg.screen_width),
        f32(frustum_cfg.screen_height),
        aabb_znear_zfar,
    );
    let p1 = world_to_screen(
        p1_world,
        clip_from_world,
        f32(frustum_cfg.screen_width),
        f32(frustum_cfg.screen_height),
        aabb_znear_zfar,
    );

    // Sparse queue binning path:
    // consume BinningTask and append leaf references into froxel bucket chains.
    trace_segment_through_froxels_sparse(
        p0,
        p1,
        frustum_cfg,
        frustum_id,
        task.seg_idx,
    );
}

@compute @workgroup_size(SCAN_THREADS, 1, 1)
fn scan_sums(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32
) {
    let wg_idx = get_scan_workgroup_index(workgroup_id, num_workgroups, pc);
    let element_idx_in_pass = wg_idx * SCAN_THREADS + local_id.x;
    let num_elements_this_pass = pc.scan_save_base - pc.scan_load_base;
    let read_idx = pc.scan_load_base + element_idx_in_pass;

    var value = 0u;
    if element_idx_in_pass < num_elements_this_pass {
        if pc.scan_load_base == 0u {
            value = atomicLoad(&tile_counts_buffer[read_idx]);
        } else {
            value = tile_offsets_buffer[read_idx];
        }
    }

    let total_wg_sum = workgroup_inclusive_scan_blelloch(local_id, subgroup_id, subgroup_local_id, value);

    if element_idx_in_pass < num_elements_this_pass {
        tile_offsets_buffer[read_idx] = wg_scan_storage[local_id.x];
    }

    if local_id.x == SCAN_THREADS - 1u {
        let first_element_idx_in_pass = wg_idx * SCAN_THREADS;
        if first_element_idx_in_pass < num_elements_this_pass {
            let sum_write_idx = pc.scan_save_base + wg_idx;
            tile_offsets_buffer[sum_write_idx] = total_wg_sum;
        }
    }
}

@compute @workgroup_size(SCAN_THREADS, 1, 1)
fn scan_last(
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32
) {
    let element_idx_in_pass = local_id.x;
    let num_elements_this_pass = pc.scan_save_base - pc.scan_load_base;
    let read_idx = pc.scan_load_base + element_idx_in_pass;

    var value = 0u;
    if element_idx_in_pass < num_elements_this_pass {
        value = tile_offsets_buffer[read_idx];
    }

    let _total_sum_from_helper = workgroup_exclusive_scan(local_id, subgroup_id, subgroup_local_id, value);

    if element_idx_in_pass < num_elements_this_pass {
        tile_offsets_buffer[read_idx] = wg_scan_storage[local_id.x];
    }

    if (element_idx_in_pass == num_elements_this_pass - 1u) && (num_elements_this_pass > 0u) {
        let last_element_exclusive_sum = wg_scan_storage[element_idx_in_pass];
        tile_offsets_buffer[pc.scan_save_base] = last_element_exclusive_sum + value;
    } else if local_id.x == 0u && num_elements_this_pass == 0u {
        tile_offsets_buffer[pc.scan_save_base] = 0u;
    }
}

@compute @workgroup_size(SCAN_THREADS, 1, 1)
fn scan_prfx(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(subgroup_id) _subgroup_id: u32,
    @builtin(subgroup_invocation_id) _subgroup_local_id: u32
) {
    let wg_idx = get_scan_workgroup_index(workgroup_id, num_workgroups, pc);
    let element_idx_in_pass = wg_idx * SCAN_THREADS + local_id.x;
    let num_elements_this_pass = pc.scan_save_base - pc.scan_load_base;

    let prefix_read_idx = pc.scan_load_base + wg_idx;
    let block_prefix_sum = tile_offsets_buffer[prefix_read_idx];

    if element_idx_in_pass < num_elements_this_pass {
        let data_rw_idx = pc.scan_save_base + element_idx_in_pass;
        let y_i = tile_offsets_buffer[data_rw_idx];

        wg_scan_storage[local_id.x] = y_i;
        workgroupBarrier();

        let y_im1 = select(0u, wg_scan_storage[local_id.x - 1u], local_id.x > 0u);
        tile_offsets_buffer[data_rw_idx] = block_prefix_sum + y_im1;
    }
}

@compute @workgroup_size(256, 1, 1)
fn init_placement_idx(@builtin(global_invocation_id) id: vec3<u32>) {
    let tile_idx = id.x;
    let num_tiles = arrayLength(&current_tile_write_indices_buffer);
    if tile_idx >= num_tiles {
        return;
    }
    atomicStore(&current_tile_write_indices_buffer[tile_idx], tile_offsets_buffer[tile_idx]);
}

@compute @workgroup_size(FINE_WORKGROUP_SIZE, 1, 1)
fn binning_queue_place_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let task_idx = gid.x;
    let binning_task_count = atomicLoad(&binning_queue.tail);
    if task_idx >= binning_task_count {
        return;
    }

    let task = binning_queue.tasks[task_idx];
    let inst_id = task.id_info;
    if inst_id >= arrayLength(&t_vertices) || inst_id >= arrayLength(&t_indices) || inst_id >= arrayLength(&t_geos) {
        return;
    }

    let vertex_ptr = t_vertices[inst_id];
    let index_ptr = t_indices[inst_id];
    let geo_ptr = t_geos[inst_id];
    if !is_valid_ptr(vertex_ptr) || !is_valid_ptr(index_ptr) || !is_valid_ptr(geo_ptr) {
        return;
    }

    let index_count = index_ptr.size / 4u;
    if task.seg_idx + 1u >= index_count {
        return;
    }

    let geo = geos[geo_ptr.slab].gs[geo_ptr.offset / SIZEOF_GEO];
    let clip_from_world = view.unjittered_clip_from_world;
    let clip_bounds = find_clip_bounds(clip_from_world, geo.aabb.min, geo.aabb.max);
    if !isValid(clip_bounds[0]) || !isValid(clip_bounds[1]) {
        return;
    }
    if abs(clip_bounds[1].z - clip_bounds[0].z) < 1e-6 {
        return;
    }
    let aabb_znear_zfar = vec2<f32>(clip_bounds[0].z, clip_bounds[1].z);

    let vi0 = indices[index_ptr.slab].is[task.seg_idx];
    let vi1 = indices[index_ptr.slab].is[task.seg_idx + 1u];
    let p0_world = vec4<f32>(vertices[vertex_ptr.slab].vs[vi0], 1.0);
    let p1_world = vec4<f32>(vertices[vertex_ptr.slab].vs[vi1], 1.0);
    let p0 = world_to_screen(
        p0_world,
        clip_from_world,
        f32(config.screen_width),
        f32(config.screen_height),
        aabb_znear_zfar,
    );
    let p1 = world_to_screen(
        p1_world,
        clip_from_world,
        f32(config.screen_width),
        f32(config.screen_height),
        aabb_znear_zfar,
    );

    trace_segment_through_froxels_linear_place(
        p0,
        p1,
        SegmentRef(task.chunk_id, task.seg_idx),
        config,
    );
}
