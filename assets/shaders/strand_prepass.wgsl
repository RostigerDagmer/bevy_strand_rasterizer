#import bevy_render::view::View
#import bevy_pbr::mesh_view_types as types

#import "shaders/common.wgsl"::{
    is_valid_ptr,
    world_to_screen_raw,
    canonical_min_mask,
    wang_hash,
    hash_to_unit_float,
    l_and,
    normalize_depth01,
    to_log_depth,
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
const COARSE_FINE_TILE_EXTENT: u32 = #COARSE_FINE_TILE_EXTENT;
const COARSE_DEPTH_SLICES: u32 = #COARSE_DEPTH_SLICES;
const COARSE_MAX_SLICES_PER_ASSET_INTERVAL: u32 = #COARSE_MAX_SLICES_PER_ASSET_INTERVAL;
const DEPTH_QUANT_MAX: u32 = 16777215u;
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
    coarse_depth_tile_base: u32,
    coarse_depth_tile_count: u32,
    coarse_tiles_x: u32,
    coarse_tiles_y: u32,
}

struct CoarseDepthLutEntry {
    z_min_q: u32,
    z_max_q: u32,
    virtual_start_q: u32,
    virtual_count_q: u32,
}

struct CoarseAssetRange {
    geo_id: u32,
    frustum_id: u32,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
    z_min_q: u32,
    z_max_q: u32,
}

struct CoarseAssetRangeQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    ranges: array<CoarseAssetRange>,
}

struct CoarseIntervalRef {
    next: u32,
    range_id: u32,
    z_min_q: u32,
    z_max_q: u32,
}

struct CoarseIntervalRefPool {
    tail: atomic<u32>,
    refs: array<CoarseIntervalRef>,
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
@group(#{PREPASS_GROUP}) @binding(#{FRUSTUM_TABLE}) var<storage, read> frustum_table: array<FrustumDesc>;
@group(#{PREPASS_GROUP}) @binding(#{FROXEL_BUCKET_HEADS}) var<storage, read_write> froxel_bucket_heads: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{CHUNK_POOL}) var<storage, read_write> chunk_pool_words: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{FREE_HEADS}) var<storage, read_write> free_heads: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{RASTER_WORK_QUEUE}) var<storage, read_write> raster_work_queue: RasterWorkQueue;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_DEPTH_LUT}) var<storage, read_write> coarse_depth_lut: array<CoarseDepthLutEntry>;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_RANGE_QUEUE}) var<storage, read_write> coarse_range_queue: CoarseAssetRangeQueue;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_INTERVAL_HEADS}) var<storage, read_write> coarse_interval_heads: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_INTERVAL_REFS}) var<storage, read_write> coarse_interval_refs: CoarseIntervalRefPool;

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

fn append_sparse_froxel_ref(frustum_id: u32, local_froxel_idx: u32, inst_id: u32, strand_idx: u32, seg_idx: u32) {
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

    let chunk_idx = alloc_chunk(bucket_idx ^ seg_idx);
    if chunk_idx == INVALID_PTR {
        return;
    }
    let work_idx = atomicAdd(&raster_work_queue.tail, 1u);
    if work_idx >= arrayLength(&raster_work_queue.items) {
        return;
    }
    raster_work_queue.items[work_idx] = RasterWorkItem(seg_idx, strand_idx, frustum_id, inst_id);

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

fn quantize_depth01(z: f32) -> u32 {
    return u32(round(clamp(z, 0.0, 1.0) * f32(DEPTH_QUANT_MAX)));
}

fn emit_coarse_asset_range_for_aabb(geo_id: u32, frustum_id: u32, aabb: Aabb) {
    if frustum_id >= arrayLength(&frustum_table) {
        return;
    }
    let frustum = frustum_table[frustum_id];
    if frustum.coarse_depth_tile_count == 0u {
        return;
    }

    let viewport = vec4<f32>(0.0, 0.0, f32(frustum.screen_width), f32(frustum.screen_height));
    let corners = array<vec4<f32>, 8>(
        vec4<f32>(aabb.min, 1.0),
        vec4<f32>(aabb.min.x, aabb.min.y, aabb.max.z, 1.0),
        vec4<f32>(aabb.min.x, aabb.max.y, aabb.min.z, 1.0),
        vec4<f32>(aabb.min.x, aabb.max.y, aabb.max.z, 1.0),
        vec4<f32>(aabb.max.x, aabb.min.y, aabb.min.z, 1.0),
        vec4<f32>(aabb.max.x, aabb.min.y, aabb.max.z, 1.0),
        vec4<f32>(aabb.max.x, aabb.max.y, aabb.min.z, 1.0),
        vec4<f32>(aabb.max, 1.0),
    );

    var screen_min = vec2<f32>(1e30, 1e30);
    var screen_max = vec2<f32>(-1e30, -1e30);
    var z_min = 1.0;
    var z_max = 0.0;
    var any_corner = false;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let raw = world_to_screen_raw(corners[i], view.unjittered_clip_from_world, viewport);
        if raw.x < 0.0 {
            continue;
        }
        any_corner = true;
        screen_min = min(screen_min, raw.xy);
        screen_max = max(screen_max, raw.xy);
        let z = to_log_depth(raw.z);
        z_min = min(z_min, z);
        z_max = max(z_max, z);
    }

    if !any_corner {
        return;
    }
    if screen_max.x < 0.0 || screen_max.y < 0.0 || screen_min.x >= viewport.z || screen_min.y >= viewport.w {
        return;
    }
    if z_max < 0.0 || z_min > 1.0 {
        return;
    }

    let coarse_tile_px_x = max(1u, frustum.froxel_size_x * COARSE_FINE_TILE_EXTENT);
    let coarse_tile_px_y = max(1u, frustum.froxel_size_y * COARSE_FINE_TILE_EXTENT);
    let clamped_min = clamp(screen_min, vec2<f32>(0.0, 0.0), vec2<f32>(viewport.z - 1.0, viewport.w - 1.0));
    let clamped_max = clamp(screen_max, vec2<f32>(0.0, 0.0), vec2<f32>(viewport.z - 1.0, viewport.w - 1.0));
    let min_cx = min(u32(clamped_min.x) / coarse_tile_px_x, frustum.coarse_tiles_x - 1u);
    let max_cx = min(u32(clamped_max.x) / coarse_tile_px_x, frustum.coarse_tiles_x - 1u);
    let min_cy = min(u32(clamped_min.y) / coarse_tile_px_y, frustum.coarse_tiles_y - 1u);
    let max_cy = min(u32(clamped_max.y) / coarse_tile_px_y, frustum.coarse_tiles_y - 1u);
    let min_q = quantize_depth01(z_min);
    let max_q = quantize_depth01(z_max);

    let range_idx = atomicAdd(&coarse_range_queue.tail, 1u);
    if range_idx >= arrayLength(&coarse_range_queue.ranges) {
        return;
    }
    coarse_range_queue.ranges[range_idx] = CoarseAssetRange(
        geo_id,
        frustum_id,
        min_cx,
        max_cx,
        min_cy,
        max_cy,
        min_q,
        max_q,
    );
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
            let frustum_count = min(pc.frustum_count, arrayLength(&frustum_table));
            for (var fi = 0u; fi < frustum_count; fi = fi + 1u) {
                let frustum = frustum_table[fi];
                if frustum.kind <= 1u {
                    emit_coarse_asset_range_for_aabb(geo_idx, fi, geo.aabb);
                }
            }

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

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn coarse_interval_pass(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let range_idx = workgroup_id.x;
    let range_count = atomicLoad(&coarse_range_queue.tail);
    if range_idx >= pc.num_elements || range_idx >= range_count || range_idx >= arrayLength(&coarse_range_queue.ranges) {
        return;
    }

    let range = coarse_range_queue.ranges[range_idx];
    if range.frustum_id >= arrayLength(&frustum_table) {
        return;
    }
    let frustum = frustum_table[range.frustum_id];
    if frustum.coarse_depth_tile_count == 0u || range.max_x < range.min_x || range.max_y < range.min_y {
        return;
    }

    let width = range.max_x - range.min_x + 1u;
    let height = range.max_y - range.min_y + 1u;
    let covered = width * height;
    var covered_idx = local_id.x;
    while covered_idx < covered {
        let dx = covered_idx % width;
        let dy = covered_idx / width;
        let cx = range.min_x + dx;
        let cy = range.min_y + dy;
        let local_tile = cy * frustum.coarse_tiles_x + cx;
        if local_tile < frustum.coarse_depth_tile_count {
            let tile_idx = frustum.coarse_depth_tile_base + local_tile;
            if tile_idx < arrayLength(&coarse_interval_heads) {
                let ref_idx = atomicAdd(&coarse_interval_refs.tail, 1u);
                if ref_idx < arrayLength(&coarse_interval_refs.refs) {
                    let prev = atomicExchange(&coarse_interval_heads[tile_idx], ref_idx);
                    coarse_interval_refs.refs[ref_idx] = CoarseIntervalRef(
                        prev,
                        range_idx,
                        range.z_min_q,
                        range.z_max_q,
                    );
                }
            }
        }
        covered_idx = covered_idx + WORKGROUP_SIZE;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn build_depth_warp_lut(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_idx = gid.x;
    if tile_idx >= pc.num_elements || tile_idx >= arrayLength(&coarse_interval_heads) {
        return;
    }

    let base = tile_idx * COARSE_DEPTH_SLICES;
    if base + COARSE_DEPTH_SLICES > arrayLength(&coarse_depth_lut) {
        return;
    }

    var interval_mins: array<u32, COARSE_DEPTH_SLICES>;
    var interval_maxs: array<u32, COARSE_DEPTH_SLICES>;
    var interval_count = 0u;
    var ref_idx = atomicLoad(&coarse_interval_heads[tile_idx]);
    var guard = 0u;
    loop {
        if ref_idx == INVALID_PTR || ref_idx >= arrayLength(&coarse_interval_refs.refs) || guard >= 256u {
            break;
        }
        let interval_ref = coarse_interval_refs.refs[ref_idx];
        var z_min = min(interval_ref.z_min_q, interval_ref.z_max_q);
        var z_max = max(interval_ref.z_min_q, interval_ref.z_max_q);
        var merged = false;
        for (var i = 0u; i < COARSE_DEPTH_SLICES; i = i + 1u) {
            if i >= interval_count {
                break;
            }
            if z_min <= interval_maxs[i] + 1u && z_max + 1u >= interval_mins[i] {
                interval_mins[i] = min(interval_mins[i], z_min);
                interval_maxs[i] = max(interval_maxs[i], z_max);
                merged = true;
            }
        }
        if !merged {
            if interval_count < COARSE_DEPTH_SLICES {
                interval_mins[interval_count] = z_min;
                interval_maxs[interval_count] = z_max;
                interval_count = interval_count + 1u;
            } else {
                var best = 0u;
                var best_gap = 0xFFFFFFFFu;
                for (var i = 0u; i < COARSE_DEPTH_SLICES; i = i + 1u) {
                    let gap = select(
                        interval_mins[i] - z_max,
                        z_min - interval_maxs[i],
                        interval_maxs[i] < z_min,
                    );
                    if gap < best_gap {
                        best_gap = gap;
                        best = i;
                    }
                }
                interval_mins[best] = min(interval_mins[best], z_min);
                interval_maxs[best] = max(interval_maxs[best], z_max);
            }
        }
        ref_idx = interval_ref.next;
        guard = guard + 1u;
    }

    if interval_count == 0u {
        for (var i = 0u; i < COARSE_DEPTH_SLICES; i = i + 1u) {
            coarse_depth_lut[base + i] = CoarseDepthLutEntry(0u, 0u, 0u, 0u);
        }
        return;
    }

    for (var i = 1u; i < COARSE_DEPTH_SLICES; i = i + 1u) {
        if i >= interval_count {
            break;
        }
        var j = i;
        let key_min = interval_mins[i];
        let key_max = interval_maxs[i];
        loop {
            if j == 0u || interval_mins[j - 1u] <= key_min {
                break;
            }
            interval_mins[j] = interval_mins[j - 1u];
            interval_maxs[j] = interval_maxs[j - 1u];
            j = j - 1u;
        }
        interval_mins[j] = key_min;
        interval_maxs[j] = key_max;
    }

    // TODO: Replace this equal-share cap with a proportional slice_count_for_interval policy
    // once we have a test scene with wide-depth assets and separated overlapping intervals.
    let capped_slices_per_interval = max(
        1u,
        min(
            COARSE_MAX_SLICES_PER_ASSET_INTERVAL,
            max(1u, COARSE_DEPTH_SLICES / interval_count),
        ),
    );
    var out_slice = 0u;
    for (var interval_idx = 0u; interval_idx < COARSE_DEPTH_SLICES; interval_idx = interval_idx + 1u) {
        if interval_idx >= interval_count {
            break;
        }
        let slice_count = capped_slices_per_interval;
        let z_min = interval_mins[interval_idx];
        let z_max = interval_maxs[interval_idx];
        let span = max(1u, z_max - z_min + 1u);
        for (var local_slice = 0u; local_slice < COARSE_DEPTH_SLICES; local_slice = local_slice + 1u) {
            if local_slice >= slice_count || out_slice >= COARSE_DEPTH_SLICES {
                break;
            }
            let start_q = z_min + (span * local_slice) / slice_count;
            let exclusive_end_q = z_min + (span * (local_slice + 1u)) / slice_count;
            let end_q = min(select(start_q, exclusive_end_q - 1u, exclusive_end_q > start_q), z_max);
            coarse_depth_lut[base + out_slice] = CoarseDepthLutEntry(
                start_q,
                end_q,
                start_q,
                max(1u, end_q - start_q + 1u),
            );
            out_slice = out_slice + 1u;
        }
    }
    while out_slice < COARSE_DEPTH_SLICES {
        let fallback_interval = interval_count - 1u;
        coarse_depth_lut[base + out_slice] = CoarseDepthLutEntry(
            interval_mins[fallback_interval],
            interval_maxs[fallback_interval],
            interval_mins[fallback_interval],
            max(1u, interval_maxs[fallback_interval] - interval_mins[fallback_interval] + 1u),
        );
        out_slice = out_slice + 1u;
    }
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

    let segment_count = strand_meta.count - 1u;
    let frustum_count = min(pc.frustum_count, arrayLength(&frustum_table));

    for (var fi = 0u; fi < frustum_count; fi = fi + 1u) {
        let frustum = frustum_table[fi];
        if frustum.kind > 1u {
            continue;
        }
        let is_shadow = select(0u, 1u, frustum.kind == 1u);
        let packed_field = pack_binning_field(0u, is_shadow, fi);
        for (var i = 0u; i < segment_count; i = i + 1u) {
            let seg_index = strand_meta.offset + i;
            let write_idx = atomicAdd(&binning_queue.tail, 1u);
            if write_idx < arrayLength(&binning_queue.tasks) {
                binning_queue.tasks[write_idx] = BinningTask(
                    task.inst_id,
                    task.strand_local,
                    seg_index,
                    packed_field,
                );
            }
        }
    }
}

fn trace_segment_through_froxels_sparse(
    p0: vec3<f32>,
    p1: vec3<f32>,
    cfg: FroxelConfig,
    frustum_id: u32,
    inst_id: u32,
    strand_idx: u32,
    seg_idx: u32,
) {
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
            let froxel_idx = (fz * num_tiles_y + fy) * num_tiles_x + fx;
            append_sparse_froxel_ref(frustum_id, froxel_idx, inst_id, strand_idx, seg_idx);
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
    if !is_valid_ptr(geo_ptr) {
        return;
    }

    let index_count = index_ptr.size / 4u;
    if task.seg_idx + 1u >= index_count {
        return;
    }

    let clip_from_world = view.unjittered_clip_from_world;
    let viewport = vec4<f32>(0.0, 0.0, f32(frustum_cfg.screen_width), f32(frustum_cfg.screen_height));

    let vi0 = indices[index_ptr.slab].is[task.seg_idx];
    let vi1 = indices[index_ptr.slab].is[task.seg_idx + 1u];
    let p0_world = vec4<f32>(vertices[vertex_ptr.slab].vs[vi0], 1.0);
    let p1_world = vec4<f32>(vertices[vertex_ptr.slab].vs[vi1], 1.0);
    let p0_raw = world_to_screen_raw(
        p0_world,
        clip_from_world,
        viewport,
    );
    let p1_raw = world_to_screen_raw(
        p1_world,
        clip_from_world,
        viewport,
    );
    let p0 = vec3<f32>(p0_raw.xy, to_log_depth(p0_raw.z));
    let p1 = vec3<f32>(p1_raw.xy, to_log_depth(p1_raw.z));

    // Sparse queue binning path:
    // consume BinningTask and append leaf references into froxel bucket chains.
    trace_segment_through_froxels_sparse(
        p0,
        p1,
        frustum_cfg,
        frustum_id,
        inst_id,
        task.chunk_id,
        task.seg_idx,
    );
}
