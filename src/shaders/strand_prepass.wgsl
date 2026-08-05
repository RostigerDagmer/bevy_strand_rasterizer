#import bevy_render::view::View
#import bevy_pbr::mesh_view_types as types

#import "embedded://strand_software_rasterizer/shaders/common.wgsl"::{
    is_valid_ptr,
    world_to_screen_raw,
    canonical_min_mask,
    wang_hash,
    hash_to_unit_float,
    l_and,
    normalize_depth01,
}

#import "embedded://strand_software_rasterizer/shaders/types.wgsl"::{
    Aabb,
    DevicePtr,
    Vertices,
    Indices,
    Geos,
    Meta,
    StrandGeo,
    StrandInstance,
    FroxelConfig,
    PushConstants,
}

#import "embedded://strand_software_rasterizer/shaders/task_contract.wgsl"::{
    FinePrepassTask,
    BinningTask,
    ProjectedSegment,
    RasterWorkItem,
    RasterTileRun,
    FineSegRef,
    FinePageMeta,
    pack_binning_field,
    unpack_binning_frustum,
}

#import bevy_vsms::virtual_surface_types::{
    VirtualSurfaceRequestMetaRow,
    vsms_request_local_tile_index,
}

#import "embedded://strand_software_rasterizer/shaders/queues.wgsl"::{
    FinePrepassQueue,
    BinningQueue,
}

const WORKGROUP_SIZE: u32 = #WORKGROUP_SIZE;
const FINE_WORKGROUP_SIZE: u32 = #FINE_WORKGROUP_SIZE;
const SCAN_SUBGROUP_THREADS: u32 = #NUMBER_OF_THREADS_PER_SUBGROUP;
const PAGE_SCAN_SUBGROUPS: u32 = COARSE_COUNT_PAGE_SIZE / SCAN_SUBGROUP_THREADS;
const SIZEOF_METADATA: u32 = #SIZEOF_METADATA;
const SIZEOF_GEO: u32 = #SIZEOF_GEO;
const POOL_CHUNK_SIZE: u32 = #POOL_CHUNK_SIZE;
const POOL_NUM_HEADS: u32 = #POOL_NUM_HEADS;
const COARSE_FINE_TILE_EXTENT: u32 = #COARSE_FINE_TILE_EXTENT;
const COARSE_DEPTH_SLICES: u32 = #COARSE_DEPTH_SLICES;
const COARSE_MAX_SLICES_PER_ASSET_INTERVAL: u32 = #COARSE_MAX_SLICES_PER_ASSET_INTERVAL;
const COARSE_COUNT_PAGE_SIZE: u32 = #COARSE_COUNT_PAGE_SIZE;
const DOM_PAGE_XY: u32 = #DOM_PAGE_XY;
const DOM_PREFETCH_PAGE_BORDER: u32 = #DOM_PREFETCH_PAGE_BORDER;
const DEPTH_QUANT_MAX: u32 = 16777215u;
const MAX_DISPATCH_WORKGROUPS_PER_DIMENSION: u32 = 65535u;
const CHUNK_WORD_STRIDE: u32 = 2u + POOL_CHUNK_SIZE;
const INVALID_PTR: u32 = 0xFFFFFFFFu;
const MARKED_COUNT_PAGE: u32 = 0xFFFFFFFEu;
const DEBUG_DISABLE_OPAQUE_FINE_CULL: bool = false;
const DEBUG_DISABLE_DEPTH_WARP_LUT: bool = false;
const TELEMETRY_HISTOGRAM_BINS: u32 = 32u;
const TELEMETRY_HISTOGRAM_OFFSET: u32 = 8u;
const TELEMETRY_PAGE_COUNTS_OFFSET: u32 = TELEMETRY_HISTOGRAM_OFFSET + TELEMETRY_HISTOGRAM_BINS;
const FRUSTUM_TELEMETRY_STRIDE: u32 = 11u;
const FRUSTUM_TELEMETRY_VISIBLE_INSTANCES: u32 = 0u;
const FRUSTUM_TELEMETRY_VISIBLE_STRANDS: u32 = 1u;
const FRUSTUM_TELEMETRY_RETAINED_STRANDS: u32 = 2u;
const FRUSTUM_TELEMETRY_EMITTED_SEGMENTS: u32 = 3u;
const FRUSTUM_TELEMETRY_ALLOCATED_PAGES: u32 = 4u;
const FRUSTUM_TELEMETRY_PAGE_CANDIDATES: u32 = 5u;
const FRUSTUM_TELEMETRY_FINE_REFS: u32 = 6u;
const FRUSTUM_TELEMETRY_RASTER_RUNS: u32 = 7u;
const FRUSTUM_TELEMETRY_RASTER_LOAD: u32 = 8u;
const FRUSTUM_TELEMETRY_MAX_PAGE_CANDIDATES: u32 = 9u;
const FRUSTUM_TELEMETRY_MAX_RASTER_LOAD: u32 = 10u;

var<workgroup> page_scan_values: array<u32, COARSE_COUNT_PAGE_SIZE>;
var<workgroup> page_scan_counts: array<u32, COARSE_COUNT_PAGE_SIZE>;
var<workgroup> page_scan_total: u32;
var<workgroup> tile_run_work_base: u32;
var<workgroup> tile_run_idx: u32;
var<workgroup> tile_run_work_count: u32;
var<workgroup> tile_run_load_score: u32;

var<immediate> pc: PushConstants;

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
    cascade_index: u32,
    fine_depth_tile_base: u32,
    pad1: u32,
    pad2: u32,
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

struct CoarseCountPages {
    tail: atomic<u32>,
    counts: array<atomic<u32>>,
}

struct ClipRange {
    ok: u32,
    t_min: f32,
    t_max: f32,
}

struct ClippedSegment {
    ok: u32,
    p0: vec3<f32>,
    p1: vec3<f32>,
}

struct RasterWorkQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    items: array<RasterWorkItem>,
}

struct RasterTileRunQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    items: array<RasterTileRun>,
}

struct ActiveFineTileQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    items: array<u32>,
}

struct FineSegRefBuffer {
    tail: atomic<u32>,
    refs: array<FineSegRef>,
}

struct BroadInstanceMeta {
    strand_count: u32,
    visible: u32,
    camera_keep_probability: f32,
    shadow_keep_probability: f32,
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
@group(#{PREPASS_GROUP}) @binding(#{CLUSTER_INDICES}) var<storage, read> clusterable_object_index_lists: types::ClusterableObjectIndexLists;
@group(#{PREPASS_GROUP}) @binding(#{CLUSTERABLE_OBJECTS}) var<storage, read> clusterable_objects: types::ClusteredLights;
@group(#{PREPASS_GROUP}) @binding(#{CLUSTER_OFFSETS_AND_COUNTS}) var<storage, read> cluster_offsets_and_counts: types::ClusterOffsetsAndCounts;
@group(#{PREPASS_GROUP}) @binding(#{PREPASS_QUEUE}) var<storage, read_write> fine_phase_queue: FinePrepassQueue;
@group(#{PREPASS_GROUP}) @binding(#{BINNING_QUEUE}) var<storage, read_write> binning_queue: BinningQueue;
@group(#{PREPASS_GROUP}) @binding(#{VISIBLE_FLAGS}) var<storage, read_write> visible_flags: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{VISIBLE_GEO}) var<storage, read_write> visible_geos: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{GEO_PREFIX}) var<storage, read_write> geo_prefix: array<u32>;
@group(#{INDIRECT_ARGS_GROUP}) @binding(#{INDIRECT_ARGS}) var<storage, read_write> dispatch_args: array<u32, 3u>;
@group(#{PREPASS_GROUP}) @binding(#{FRUSTUM_TABLE}) var<storage, read> frustum_table: array<FrustumDesc>;
@group(#{PREPASS_GROUP}) @binding(#{FROXEL_BUCKET_HEADS}) var<storage, read_write> froxel_bucket_heads: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{CHUNK_POOL}) var<storage, read_write> chunk_pool_words: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{FREE_HEADS}) var<storage, read_write> free_heads: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{RASTER_WORK_QUEUE}) var<storage, read_write> raster_work_queue: RasterWorkQueue;
@group(#{PREPASS_GROUP}) @binding(#{CAMERA_RASTER_TILE_RUN_QUEUE}) var<storage, read_write> camera_raster_tile_run_queue: RasterTileRunQueue;
@group(#{PREPASS_GROUP}) @binding(#{CAMERA_RASTER_TILE_RUN_DISPATCH_ARGS}) var<storage, read_write> camera_raster_tile_run_dispatch_args: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{SHADOW_RASTER_TILE_RUN_QUEUE}) var<storage, read_write> shadow_raster_tile_run_queue: RasterTileRunQueue;
@group(#{PREPASS_GROUP}) @binding(#{SHADOW_RASTER_TILE_RUN_DISPATCH_ARGS}) var<storage, read_write> shadow_raster_tile_run_dispatch_args: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_DEPTH_LUT}) var<storage, read_write> coarse_depth_lut: array<CoarseDepthLutEntry>;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_RANGE_QUEUE}) var<storage, read_write> coarse_range_queue: CoarseAssetRangeQueue;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_INTERVAL_HEADS}) var<storage, read_write> coarse_interval_heads: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_INTERVAL_REFS}) var<storage, read_write> coarse_interval_refs: CoarseIntervalRefPool;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_RANGE_LOOKUP}) var<storage, read_write> coarse_range_lookup: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_COUNT_PAGE_TABLE}) var<storage, read_write> coarse_count_page_table: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{COARSE_COUNT_PAGES}) var<storage, read_write> coarse_count_pages: CoarseCountPages;
@group(#{PREPASS_GROUP}) @binding(#{STRAND_INSTANCES}) var<storage, read> strand_instances: array<StrandInstance>;
@group(#{PREPASS_GROUP}) @binding(#{FINE_PAGE_META}) var<storage, read_write> fine_page_meta: array<FinePageMeta>;
@group(#{PREPASS_GROUP}) @binding(#{FINE_CELL_OFFSETS}) var<storage, read_write> fine_cell_offsets: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{FINE_CELL_WRITE_CURSORS}) var<storage, read_write> fine_cell_write_cursors: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{FINE_SEG_REFS}) var<storage, read_write> fine_seg_refs: FineSegRefBuffer;
@group(#{PREPASS_GROUP}) @binding(#{VSMS_REQUEST_META}) var<storage, read> vsms_request_meta: array<VirtualSurfaceRequestMetaRow>;
@group(#{PREPASS_GROUP}) @binding(#{VSMS_REQUEST_BITS}) var<storage, read_write> vsms_request_bits: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{SHADOW_DOM_SURFACE_IDS}) var<storage, read> shadow_dom_surface_ids: array<vec2<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{BROAD_INSTANCE_META}) var<storage, read_write> broad_instance_meta: array<BroadInstanceMeta>;
@group(#{PREPASS_GROUP}) @binding(#{OPAQUE_FINE_DEPTH_TILES}) var<storage, read> opaque_fine_depth_tiles: array<vec2<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{TELEMETRY}) var<storage, read_write> prepass_telemetry: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{PROJECTED_SEGMENTS}) var<storage, read_write> projected_segments: array<ProjectedSegment>;
@group(#{PREPASS_GROUP}) @binding(#{PAGE_CANDIDATE_COUNTS}) var<storage, read_write> page_candidate_counts: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{PAGE_CANDIDATE_OFFSETS}) var<storage, read_write> page_candidate_offsets: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{PAGE_CANDIDATE_CURSORS}) var<storage, read_write> page_candidate_cursors: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{PAGE_CANDIDATES}) var<storage, read_write> page_candidates: array<u32>;
@group(#{PREPASS_GROUP}) @binding(#{VIRTUAL_PAGE_CANDIDATE_COUNTS}) var<storage, read_write> virtual_page_candidate_counts: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{FRUSTUM_INSTANCE_KEEP_PROBABILITIES}) var<storage, read_write> frustum_instance_keep_probabilities: array<f32>;
@group(#{PREPASS_GROUP}) @binding(#{ACTIVE_FINE_TILE_QUEUE}) var<storage, read_write> active_fine_tile_queue: ActiveFineTileQueue;
@group(#{PREPASS_GROUP}) @binding(#{ACTIVE_FINE_TILE_FLAGS}) var<storage, read_write> active_fine_tile_flags: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{ACTIVE_FINE_TILE_BLOCK_COUNTS}) var<storage, read_write> active_fine_tile_block_counts: array<atomic<u32>>;
@group(#{PREPASS_GROUP}) @binding(#{ACTIVE_FINE_TILE_BLOCK_OFFSETS}) var<storage, read_write> active_fine_tile_block_offsets: array<u32>;

var<workgroup> candidate_scan_base: u32;
var<workgroup> candidate_scan_block_base: u32;
var<workgroup> candidate_scan_block_total: u32;
var<workgroup> csr_cell_counts: array<atomic<u32>, COARSE_COUNT_PAGE_SIZE>;
var<workgroup> csr_cell_cursors: array<atomic<u32>, COARSE_COUNT_PAGE_SIZE>;
var<workgroup> csr_page_ref_base: u32;
var<workgroup> csr_page_valid: u32;
var<workgroup> active_page_frustum_id: u32;

fn frustum_telemetry_idx(frustum_id: u32, field: u32) -> u32 {
    return pc.scan_save_base + frustum_id * FRUSTUM_TELEMETRY_STRIDE + field;
}

fn frustum_telemetry_add(frustum_id: u32, field: u32, value: u32) {
    if pc.telemetry_enabled == 0u || frustum_id >= pc.frustum_count {
        return;
    }
    let idx = frustum_telemetry_idx(frustum_id, field);
    if idx < arrayLength(&prepass_telemetry) {
        atomicAdd(&prepass_telemetry[idx], value);
    }
}

fn frustum_telemetry_max(frustum_id: u32, field: u32, value: u32) {
    if pc.telemetry_enabled == 0u || frustum_id >= pc.frustum_count {
        return;
    }
    let idx = frustum_telemetry_idx(frustum_id, field);
    if idx < arrayLength(&prepass_telemetry) {
        atomicMax(&prepass_telemetry[idx], value);
    }
}

fn aabb_projected_screen_area(frustum_id: u32, aabb: Aabb, world_from_local: mat4x4<f32>) -> f32 {
    if frustum_id >= arrayLength(&frustum_table) {
        return -1.0;
    }
    let frustum = frustum_table[frustum_id];
    let viewport = vec4<f32>(0.0, 0.0, f32(frustum.screen_width), f32(frustum.screen_height));
    let clip_from_world = clip_from_world_for_frustum(frustum_id);
    let corners = array<vec4<f32>, 8>(
        world_from_local * vec4<f32>(aabb.min, 1.0),
        world_from_local * vec4<f32>(aabb.min.x, aabb.min.y, aabb.max.z, 1.0),
        world_from_local * vec4<f32>(aabb.min.x, aabb.max.y, aabb.min.z, 1.0),
        world_from_local * vec4<f32>(aabb.min.x, aabb.max.y, aabb.max.z, 1.0),
        world_from_local * vec4<f32>(aabb.max.x, aabb.min.y, aabb.min.z, 1.0),
        world_from_local * vec4<f32>(aabb.max.x, aabb.min.y, aabb.max.z, 1.0),
        world_from_local * vec4<f32>(aabb.max.x, aabb.max.y, aabb.min.z, 1.0),
        world_from_local * vec4<f32>(aabb.max, 1.0),
    );

    var screen_min = vec2<f32>(1e30, 1e30);
    var screen_max = vec2<f32>(-1e30, -1e30);
    var z_min = 1.0;
    var z_max = 0.0;
    var any_corner = false;
    var clipped_by_near = false;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let clip = clip_from_world * corners[i];
        if clip.w <= 1e-6 {
            clipped_by_near = true;
            continue;
        }

        let ndc = clip.xyz / clip.w;
        if ndc.z < 0.0 {
            clipped_by_near = true;
        }

        let raw = vec3<f32>(
            viewport.x + (ndc.x * 0.5 + 0.5) * viewport.z,
            viewport.y + (ndc.y * -0.5 + 0.5) * viewport.w,
            ndc.z,
        );
        any_corner = true;
        screen_min = min(screen_min, raw.xy);
        screen_max = max(screen_max, raw.xy);
        let z = depth_key_for_frustum(raw.z, frustum);
        z_min = min(z_min, z);
        z_max = max(z_max, z);
    }

    if clipped_by_near {
        return viewport.z * viewport.w;
    }
    if !any_corner {
        return -1.0;
    }
    if screen_max.x < 0.0 || screen_max.y < 0.0 || screen_min.x >= viewport.z || screen_min.y >= viewport.w {
        return -1.0;
    }
    if z_max < 0.0 || z_min > 1.0 {
        return -1.0;
    }

    let clamped_min = clamp(screen_min, vec2<f32>(0.0, 0.0), vec2<f32>(viewport.z - 1.0, viewport.w - 1.0));
    let clamped_max = clamp(screen_max, vec2<f32>(0.0, 0.0), vec2<f32>(viewport.z - 1.0, viewport.w - 1.0));
    let extent = max(clamped_max - clamped_min, vec2<f32>(1.0, 1.0));
    return extent.x * extent.y;
}

fn stochastic_camera_keep_probability(strand_count: u32, screen_area_px: f32) -> f32 {
    if pc.stochastic_cull_enabled == 0u {
        return 1.0;
    }
    if screen_area_px < 0.0 {
        return 0.0;
    }
    let target_density = max(pc.target_strands_per_pixel, 1e-5);
    let min_keep = clamp(pc.min_keep_probability, 0.0, 1.0);
    return clamp(screen_area_px * target_density / max(f32(strand_count), 1.0), min_keep, 1.0);
    // let area_keep = clamp(screen_area_px * target_density / max(f32(strand_count), 1.0), 0.0, 1.0);
    // return clamp(area_keep * area_keep, min_keep, 1.0);
}

fn stochastic_shadow_keep_probability(strand_count: u32, screen_area_px: f32, frustum: FrustumDesc) -> f32 {
    if pc.stochastic_cull_enabled == 0u {
        return 1.0;
    }
    if screen_area_px < 0.0 {
        return 0.0;
    }
    let target_density = max(pc.target_strands_per_pixel, 1e-5);
    let min_keep = clamp(pc.min_keep_probability, 0.0, 1.0);
    let slider_cap = clamp(pc.shadow_keep_probability, 0.0, 1.0);
    let area_keep = clamp(screen_area_px * target_density / max(f32(strand_count), 1.0), min_keep, 1.0);
    let curved_keep = area_keep * area_keep;
    let cascade_keep_scale = 1.0 / exp2(f32(frustum.cascade_index));
    return clamp(curved_keep * cascade_keep_scale * slider_cap, min_keep, 1.0);
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

fn light_layer_from_frustum(frustum_id: u32) -> u32 {
    if frustum_id >= arrayLength(&frustum_table) {
        return INVALID_PTR;
    }
    if frustum_table[frustum_id].kind != 1u {
        return INVALID_PTR;
    }
    var layer = 0u;
    for (var i = 0u; i < frustum_id; i = i + 1u) {
        if frustum_table[i].kind == 1u {
            layer = layer + 1u;
        }
    }
    if layer >= lights.n_directional_lights {
        return INVALID_PTR;
    }
    return layer;
}

fn cascade_index_for_frustum(frustum_id: u32, light_layer: u32) -> u32 {
    if frustum_id >= arrayLength(&frustum_table) || light_layer >= lights.n_directional_lights {
        return 0u;
    }
    let num_cascades = lights.directional_lights[light_layer].num_cascades;
    if num_cascades == 0u {
        return 0u;
    }
    return min(frustum_table[frustum_id].cascade_index, num_cascades - 1u);
}

fn clip_from_world_for_frustum(frustum_id: u32) -> mat4x4<f32> {
    if frustum_id < arrayLength(&frustum_table) && frustum_table[frustum_id].kind == 1u {
        let light_layer = light_layer_from_frustum(frustum_id);
        if light_layer < lights.n_directional_lights {
            let cascade_index = cascade_index_for_frustum(frustum_id, light_layer);
            return lights.directional_lights[light_layer].cascades[cascade_index].clip_from_world;
        }
    }
    return view.unjittered_clip_from_world;
}

fn depth_key_for_frustum(raw_z: f32, frustum: FrustumDesc) -> f32 {
    // Internal binning/raster-order convention: linear reverse-Z.
    // Higher values are nearer for the Bevy/wgpu projections we target.
    return normalize_depth01(raw_z);
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
    raster_work_queue.items[work_idx] = RasterWorkItem(0u, 1u, frustum_id, local_froxel_idx, 0u, 1u, INVALID_PTR, 0u);

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

fn dequantize_depth01(z_q: u32) -> f32 {
    return f32(z_q) / f32(DEPTH_QUANT_MAX);
}

fn disjoint_interval_gap(a_min: u32, a_max: u32, b_min: u32, b_max: u32) -> u32 {
    if a_max < b_min {
        return b_min - a_max;
    }
    if b_max < a_min {
        return a_min - b_max;
    }
    return 0u;
}

fn fine_tile_occluded_by_opaque(frustum: FrustumDesc, fine_x: u32, fine_y: u32, nearest_depth: f32) -> bool {
    if DEBUG_DISABLE_OPAQUE_FINE_CULL {
        return false;
    }
    if frustum.kind != 0u {
        return false;
    }
    let fine_tiles_x = ceil_div_u32(frustum.screen_width, frustum.froxel_size_x);
    let fine_tiles_y = ceil_div_u32(frustum.screen_height, frustum.froxel_size_y);
    if fine_x >= fine_tiles_x || fine_y >= fine_tiles_y {
        return false;
    }
    let tile_idx = frustum.fine_depth_tile_base + fine_y * fine_tiles_x + fine_x;
    if tile_idx >= arrayLength(&opaque_fine_depth_tiles) {
        return false;
    }
    let opaque = opaque_fine_depth_tiles[tile_idx];
    if opaque.x == 0u {
        return false;
    }
    let farthest_opaque_depth = dequantize_depth01(opaque.y);
    return nearest_depth <= farthest_opaque_depth - 1e-4;
}

fn emit_coarse_asset_range_for_aabb(inst_id: u32, frustum_id: u32, aabb: Aabb, world_from_local: mat4x4<f32>) -> bool {
    if frustum_id >= arrayLength(&frustum_table) {
        return false;
    }
    let frustum = frustum_table[frustum_id];
    if frustum.coarse_depth_tile_count == 0u {
        return false;
    }
    if frustum.kind == 1u && light_layer_from_frustum(frustum_id) == INVALID_PTR {
        return false;
    }

    let viewport = vec4<f32>(0.0, 0.0, f32(frustum.screen_width), f32(frustum.screen_height));
    let clip_from_world = clip_from_world_for_frustum(frustum_id);
    let corners = array<vec4<f32>, 8>(
        world_from_local * vec4<f32>(aabb.min, 1.0),
        world_from_local * vec4<f32>(aabb.min.x, aabb.min.y, aabb.max.z, 1.0),
        world_from_local * vec4<f32>(aabb.min.x, aabb.max.y, aabb.min.z, 1.0),
        world_from_local * vec4<f32>(aabb.min.x, aabb.max.y, aabb.max.z, 1.0),
        world_from_local * vec4<f32>(aabb.max.x, aabb.min.y, aabb.min.z, 1.0),
        world_from_local * vec4<f32>(aabb.max.x, aabb.min.y, aabb.max.z, 1.0),
        world_from_local * vec4<f32>(aabb.max.x, aabb.max.y, aabb.min.z, 1.0),
        world_from_local * vec4<f32>(aabb.max, 1.0),
    );

    var screen_min = vec2<f32>(1e30, 1e30);
    var screen_max = vec2<f32>(-1e30, -1e30);
    var z_min = 1.0;
    var z_max = 0.0;
    var any_corner = false;
    var clipped_by_near = false;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let clip = clip_from_world * corners[i];
        if clip.w <= 1e-6 {
            clipped_by_near = true;
            continue;
        }

        let ndc = clip.xyz / clip.w;
        if ndc.z < 0.0 {
            clipped_by_near = true;
        }

        let raw = vec3<f32>(
            viewport.x + (ndc.x * 0.5 + 0.5) * viewport.z,
            viewport.y + (ndc.y * -0.5 + 0.5) * viewport.w,
            ndc.z,
        );
        any_corner = true;
        screen_min = min(screen_min, raw.xy);
        screen_max = max(screen_max, raw.xy);
        let z = depth_key_for_frustum(raw.z, frustum);
        z_min = min(z_min, z);
        z_max = max(z_max, z);
    }

    if clipped_by_near {
        screen_min = vec2<f32>(0.0, 0.0);
        screen_max = vec2<f32>(viewport.z - 1.0, viewport.w - 1.0);
        z_max = 1.0;
        if !any_corner {
            z_min = 0.0;
        }
        any_corner = true;
    }

    if !any_corner {
        return false;
    }
    if screen_max.x < 0.0 || screen_max.y < 0.0 || screen_min.x >= viewport.z || screen_min.y >= viewport.w {
        return false;
    }
    if z_max < 0.0 || z_min > 1.0 {
        return false;
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
        return false;
    }
    let lookup_idx = inst_id * pc.frustum_count + frustum_id;
    if lookup_idx < arrayLength(&coarse_range_lookup) {
        coarse_range_lookup[lookup_idx] = range_idx;
    }
    coarse_range_queue.ranges[range_idx] = CoarseAssetRange(
        inst_id,
        frustum_id,
        min_cx,
        max_cx,
        min_cy,
        max_cy,
        min_q,
        max_q,
    );
    return true;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn broad_prepass(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let global_invocation = ((workgroup_id.z * num_workgroups.y + workgroup_id.y) * num_workgroups.x + workgroup_id.x) * WORKGROUP_SIZE + local_id.x;
    let total_invocations = num_workgroups.x * num_workgroups.y * num_workgroups.z * WORKGROUP_SIZE;

    let instance_count = min(pc.num_elements, arrayLength(&strand_instances));
    var inst_idx = global_invocation;

    while inst_idx < instance_count {
        let instance = strand_instances[inst_idx];
        if instance.geo_id >= arrayLength(&t_geos) || instance.meta_id >= arrayLength(&t_strand_metadata) {
            inst_idx += total_invocations;
            continue;
        }

        let geo_ptr = t_geos[instance.geo_id];
        let meta_ptr = t_strand_metadata[instance.meta_id];

        if !is_valid_ptr(geo_ptr) || !is_valid_ptr(meta_ptr) {
            inst_idx += total_invocations;
            continue;
        }

        let geo = geos[geo_ptr.slab].gs[geo_ptr.offset / SIZEOF_GEO];
        let strand_count_from_meta = meta_ptr.size / SIZEOF_METADATA;
        let strand_count = min(strand_count_from_meta, geo.strand_count);

        var geo_visible = false;
        var camera_keep_probability = 0.0;
        var shadow_keep_probability = 0.0;
        let frustum_count = min(pc.frustum_count, arrayLength(&frustum_table));
        for (var fi = 0u; fi < frustum_count; fi = fi + 1u) {
            let frustum = frustum_table[fi];
            let keep_idx = inst_idx * frustum_count + fi;
            var frustum_keep_probability = 0.0;
            if frustum.kind <= 1u {
                let visible_in_frustum = emit_coarse_asset_range_for_aabb(inst_idx, fi, geo.aabb, instance.world_from_local);
                geo_visible = visible_in_frustum || geo_visible;
                if visible_in_frustum && frustum.kind == 0u {
                    let area = aabb_projected_screen_area(fi, geo.aabb, instance.world_from_local);
                    frustum_keep_probability = stochastic_camera_keep_probability(strand_count, area);
                    camera_keep_probability = max(camera_keep_probability, frustum_keep_probability);
                } else if visible_in_frustum && frustum.kind == 1u {
                    let area = aabb_projected_screen_area(fi, geo.aabb, instance.world_from_local);
                    frustum_keep_probability = stochastic_shadow_keep_probability(strand_count, area, frustum);
                    shadow_keep_probability = max(shadow_keep_probability, frustum_keep_probability);
                }
                if visible_in_frustum {
                    frustum_telemetry_add(fi, FRUSTUM_TELEMETRY_VISIBLE_INSTANCES, 1u);
                    frustum_telemetry_add(fi, FRUSTUM_TELEMETRY_VISIBLE_STRANDS, strand_count);
                }
            }
            if keep_idx < arrayLength(&frustum_instance_keep_probabilities) {
                frustum_instance_keep_probabilities[keep_idx] = frustum_keep_probability;
            }
        }

        if inst_idx < arrayLength(&visible_flags) {
            visible_flags[inst_idx] = u32(geo_visible);
        }

        if inst_idx < arrayLength(&broad_instance_meta) {
            broad_instance_meta[inst_idx] = BroadInstanceMeta(
                strand_count,
                u32(geo_visible),
                camera_keep_probability,
                shadow_keep_probability,
            );
        }

        inst_idx += total_invocations;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn broad_strand_prepass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let strand_local = gid.x;
    let inst_idx = gid.y;
    if strand_local >= pc.num_elements || inst_idx >= pc.scan_load_base || inst_idx >= arrayLength(&broad_instance_meta) {
        return;
    }
    let broad = broad_instance_meta[inst_idx];
    if broad.visible == 0u || strand_local >= broad.strand_count {
        return;
    }
    let strand_hash = wang_hash((inst_idx * 16777619u) ^ strand_local);
    let keep_probability = max(broad.camera_keep_probability, broad.shadow_keep_probability);
    if hash_to_unit_float(strand_hash) > keep_probability {
        return;
    }
    let task_index = atomicAdd(&fine_phase_queue.tail, 1u);
    if task_index < arrayLength(&fine_phase_queue.tasks) {
        fine_phase_queue.tasks[task_index] = FinePrepassTask(inst_idx, strand_local);
    }
}

@compute @workgroup_size(1, 1, 1)
fn finalize_prepass() {
    let fine_task_count = atomicLoad(&fine_phase_queue.tail);
    let group_count = ceil_div_u32(fine_task_count, FINE_WORKGROUP_SIZE);
    if group_count == 0u {
        dispatch_args[0] = 0u;
        dispatch_args[1] = 0u;
        dispatch_args[2] = 0u;
        return;
    }
    let x = min(group_count, MAX_DISPATCH_WORKGROUPS_PER_DIMENSION);
    dispatch_args[0] = x;
    dispatch_args[1] = ceil_div_u32(group_count, x);
    dispatch_args[2] = 1u;
}

@compute @workgroup_size(1, 1, 1)
fn finalize_binning() {
    let binning_task_count = atomicLoad(&binning_queue.tail);
    let group_count = ceil_div_u32(binning_task_count, FINE_WORKGROUP_SIZE);
    if group_count == 0u {
        dispatch_args[0] = 0u;
        dispatch_args[1] = 0u;
        dispatch_args[2] = 0u;
        return;
    }
    let x = min(group_count, MAX_DISPATCH_WORKGROUPS_PER_DIMENSION);
    dispatch_args[0] = x;
    dispatch_args[1] = ceil_div_u32(group_count, x);
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
    if DEBUG_DISABLE_DEPTH_WARP_LUT {
        for (var i = 0u; i < COARSE_DEPTH_SLICES; i = i + 1u) {
            let start_q = (DEPTH_QUANT_MAX * i) / COARSE_DEPTH_SLICES;
            let exclusive_end_q = (DEPTH_QUANT_MAX * (i + 1u)) / COARSE_DEPTH_SLICES;
            let end_q = min(select(start_q, exclusive_end_q - 1u, exclusive_end_q > start_q), DEPTH_QUANT_MAX);
            coarse_depth_lut[base + i] = CoarseDepthLutEntry(
                start_q,
                end_q,
                start_q,
                max(1u, end_q - start_q + 1u),
            );
        }
        return;
    }

    var interval_mins: array<u32, COARSE_DEPTH_SLICES>;
    var interval_maxs: array<u32, COARSE_DEPTH_SLICES>;
    var interval_count = 0u;
    var ref_idx = atomicLoad(&coarse_interval_heads[tile_idx]);
    var guard = 0u;
    let max_ref_walk = arrayLength(&coarse_interval_refs.refs);
    loop {
        if ref_idx == INVALID_PTR || ref_idx >= max_ref_walk || guard >= max_ref_walk {
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
                    let gap = disjoint_interval_gap(interval_mins[i], interval_maxs[i], z_min, z_max);
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

    var out_slice = 0u;
    for (var interval_idx = 0u; interval_idx < COARSE_DEPTH_SLICES; interval_idx = interval_idx + 1u) {
        if interval_idx >= interval_count {
            break;
        }
        let base_slice_count = max(1u, COARSE_DEPTH_SLICES / interval_count);
        let remainder = COARSE_DEPTH_SLICES % interval_count;
        let slice_count = base_slice_count + select(0u, 1u, interval_idx < remainder);
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
        coarse_depth_lut[base + out_slice] = CoarseDepthLutEntry(0u, 0u, 0u, 0u);
        out_slice = out_slice + 1u;
    }
}

@compute @workgroup_size(FINE_WORKGROUP_SIZE, 1, 1)
fn fine_prepass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let task_idx = gid.y * pc.workgroup_offset + gid.x;
    let fine_task_count = atomicLoad(&fine_phase_queue.tail);

    if task_idx >= fine_task_count {
        return;
    }

    let task = fine_phase_queue.tasks[task_idx];

    if task.inst_id >= arrayLength(&strand_instances) {
        return;
    }

    let instance = strand_instances[task.inst_id];
    if instance.meta_id >= arrayLength(&t_strand_metadata) || instance.index_id >= arrayLength(&t_indices) || instance.vertex_id >= arrayLength(&t_vertices) {
        return;
    }

    let meta_ptr = t_strand_metadata[instance.meta_id];
    let index_ptr = t_indices[instance.index_id];
    let vertex_ptr = t_vertices[instance.vertex_id];

    if !is_valid_ptr(meta_ptr) || !is_valid_ptr(index_ptr) || !is_valid_ptr(vertex_ptr) {
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
    let index_count = index_ptr.size / 4u;
    let index_base = index_ptr.offset / 4u;
    let vertex_base = vertex_ptr.offset / 16u;
    let frustum_count = min(pc.frustum_count, arrayLength(&frustum_table));
    let broad = broad_instance_meta[task.inst_id];
    let strand_hash = wang_hash((task.inst_id * 16777619u) ^ task.strand_local);
    let strand_random = hash_to_unit_float(strand_hash);

    for (var fi = 0u; fi < frustum_count; fi = fi + 1u) {
        let frustum = frustum_table[fi];
        if frustum.kind > 1u {
            continue;
        }
        var keep_probability = select(
            broad.camera_keep_probability,
            broad.shadow_keep_probability,
            frustum.kind == 1u,
        );
        let keep_idx = task.inst_id * frustum_count + fi;
        if keep_idx < arrayLength(&frustum_instance_keep_probabilities) {
            keep_probability = frustum_instance_keep_probabilities[keep_idx];
        }
        if strand_random > keep_probability {
            continue;
        }
        if frustum.kind == 1u && light_layer_from_frustum(fi) == INVALID_PTR {
            continue;
        }
        let is_shadow = select(0u, 1u, frustum.kind == 1u);
        frustum_telemetry_add(fi, FRUSTUM_TELEMETRY_RETAINED_STRANDS, 1u);
        let packed_field = pack_binning_field(0u, is_shadow, fi);
        let clip_from_world = clip_from_world_for_frustum(fi);
        let viewport = vec4<f32>(0.0, 0.0, f32(frustum.screen_width), f32(frustum.screen_height));
        let screen_max = vec2<f32>(
            max(0.0, f32(frustum.screen_width) - 1e-3),
            max(0.0, f32(frustum.screen_height) - 1e-3),
        );
        for (var i = 0u; i < segment_count; i = i + 1u) {
            let seg_index = strand_meta.offset + i;
            if seg_index + 1u >= index_count {
                continue;
            }
            let vi0 = indices[index_ptr.slab].is[index_base + seg_index];
            let vi1 = indices[index_ptr.slab].is[index_base + seg_index + 1u];
            let p0_world = instance.world_from_local * vec4<f32>(vertices[vertex_ptr.slab].vs[vertex_base + vi0], 1.0);
            let p1_world = instance.world_from_local * vec4<f32>(vertices[vertex_ptr.slab].vs[vertex_base + vi1], 1.0);
            let p0_raw = world_to_screen_raw(p0_world, clip_from_world, viewport);
            let p1_raw = world_to_screen_raw(p1_world, clip_from_world, viewport);
            let p0 = vec3<f32>(p0_raw.xy, depth_key_for_frustum(p0_raw.z, frustum));
            let p1 = vec3<f32>(p1_raw.xy, depth_key_for_frustum(p1_raw.z, frustum));
            let clipped = clip_segment_to_box(
                p0,
                p1,
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(screen_max, 1.0),
            );
            if clipped.ok == 0u {
                continue;
            }
            let write_idx = atomicAdd(&binning_queue.tail, 1u);
            if write_idx < arrayLength(&binning_queue.tasks) && write_idx < arrayLength(&projected_segments) {
                binning_queue.tasks[write_idx] = BinningTask(
                    task.inst_id,
                    task.strand_local,
                    seg_index,
                    packed_field,
                );
                let screen_size = max(vec2<f32>(1.0), vec2<f32>(f32(frustum.screen_width), f32(frustum.screen_height)));
                projected_segments[write_idx] = ProjectedSegment(
                    pack2x16unorm(clamp(clipped.p0.xy / screen_size, vec2<f32>(0.0), vec2<f32>(1.0))),
                    pack2x16unorm(clamp(clipped.p1.xy / screen_size, vec2<f32>(0.0), vec2<f32>(1.0))),
                    pack2x16unorm(clamp(vec2<f32>(clipped.p0.z, clipped.p1.z), vec2<f32>(0.0), vec2<f32>(1.0))),
                );
                frustum_telemetry_add(fi, FRUSTUM_TELEMETRY_EMITTED_SEGMENTS, 1u);
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

fn coarse_lut_slice_for_depth(tile_idx: u32, z: f32) -> u32 {
    let z_q = quantize_depth01(z);
    let lut_base = tile_idx * COARSE_DEPTH_SLICES;
    if lut_base + COARSE_DEPTH_SLICES > arrayLength(&coarse_depth_lut) {
        return INVALID_PTR;
    }
    var best_slice = INVALID_PTR;
    var best_distance = 0xFFFFFFFFu;
    for (var i = 0u; i < COARSE_DEPTH_SLICES; i = i + 1u) {
        let entry = coarse_depth_lut[lut_base + i];
        if entry.virtual_count_q == 0u {
            continue;
        }
        if z_q >= entry.z_min_q && z_q <= entry.z_max_q {
            return i;
        }
        let distance = select(entry.z_min_q - z_q, z_q - entry.z_max_q, z_q > entry.z_max_q);
        if distance < best_distance {
            best_distance = distance;
            best_slice = i;
        }
    }
    return best_slice;
}

fn coarse_count_page_table_idx(tile_idx: u32, coarse_z: u32) -> u32 {
    return tile_idx * COARSE_DEPTH_SLICES + coarse_z;
}

fn mark_coarse_count_page(page_table_idx: u32) {
    if page_table_idx >= arrayLength(&coarse_count_page_table) {
        return;
    }
    atomicStore(&coarse_count_page_table[page_table_idx], MARKED_COUNT_PAGE);
}

fn get_coarse_count_page(page_table_idx: u32) -> u32 {
    if page_table_idx >= arrayLength(&coarse_count_page_table) {
        return INVALID_PTR;
    }
    let page_handle = atomicLoad(&coarse_count_page_table[page_table_idx]);
    if page_handle == 0u || page_handle == MARKED_COUNT_PAGE {
        return INVALID_PTR;
    }
    return page_handle - 1u;
}

fn increment_coarse_count_page_cell(page_idx: u32, local_x: u32, local_y: u32, local_z: u32) {
    if local_x >= COARSE_FINE_TILE_EXTENT || local_y >= COARSE_FINE_TILE_EXTENT || local_z >= COARSE_DEPTH_SLICES {
        return;
    }
    let cell_idx = (local_z * COARSE_FINE_TILE_EXTENT + local_y) * COARSE_FINE_TILE_EXTENT + local_x;
    let count_idx = page_idx * COARSE_COUNT_PAGE_SIZE + cell_idx;
    if count_idx < arrayLength(&coarse_count_pages.counts) {
        atomicAdd(&coarse_count_pages.counts[count_idx], 1u);
    }
}

fn fine_cell_idx(local_x: u32, local_y: u32, local_z: u32) -> u32 {
    return (local_z * COARSE_FINE_TILE_EXTENT + local_y) * COARSE_FINE_TILE_EXTENT + local_x;
}

fn make_fine_seg_ref(task: BinningTask, meta_id: u32) -> FineSegRef {
    var material_idx = 0u;
    if meta_id < arrayLength(&t_strand_metadata) {
        let meta_ptr = t_strand_metadata[meta_id];
        if is_valid_ptr(meta_ptr) {
            let meta_base = meta_ptr.offset / SIZEOF_METADATA;
            let meta_count = meta_ptr.size / SIZEOF_METADATA;
            if task.chunk_id < meta_count {
                let strand_meta = strand_metadata[meta_ptr.slab].ms[meta_base + task.chunk_id];
                material_idx = strand_meta.material_idx;
            }
        }
    }
    return FineSegRef(task.id_info, task.seg_idx, material_idx);
}

fn write_fine_seg_ref(page_idx: u32, local_x: u32, local_y: u32, local_z: u32, seg_ref: FineSegRef) {
    if local_x >= COARSE_FINE_TILE_EXTENT || local_y >= COARSE_FINE_TILE_EXTENT || local_z >= COARSE_DEPTH_SLICES {
        return;
    }
    if page_idx >= arrayLength(&fine_page_meta) {
        return;
    }
    let cell_idx = fine_cell_idx(local_x, local_y, local_z);
    let global_cell_idx = page_idx * COARSE_COUNT_PAGE_SIZE + cell_idx;
    if global_cell_idx >= arrayLength(&fine_cell_offsets) || global_cell_idx >= arrayLength(&fine_cell_write_cursors) {
        return;
    }
    let page_meta = fine_page_meta[page_idx];
    if page_meta.seg_ref_count == 0u {
        return;
    }
    // The count and fill passes execute the same deterministic traversal.
    // Prefixing clears this cursor and reserves exactly the counted range, so
    // reloading the atomic cell count for every emitted reference is redundant.
    let local_write = atomicAdd(&fine_cell_write_cursors[global_cell_idx], 1u);
    let dst = page_meta.seg_ref_base + fine_cell_offsets[global_cell_idx] + local_write;
    if dst >= arrayLength(&fine_seg_refs.refs) {
        return;
    }
    fine_seg_refs.refs[dst] = seg_ref;
}

fn clip_axis_to_range(p: f32, d: f32, min_v: f32, max_v: f32, t_min: f32, t_max: f32) -> ClipRange {
    var out_min = t_min;
    var out_max = t_max;
    if abs(d) < 1e-6 {
        if p < min_v || p > max_v {
            return ClipRange(0u, out_min, out_max);
        }
        return ClipRange(1u, out_min, out_max);
    }

    let inv_d = 1.0 / d;
    var a = (min_v - p) * inv_d;
    var b = (max_v - p) * inv_d;
    if a > b {
        let tmp = a;
        a = b;
        b = tmp;
    }

    out_min = max(out_min, a);
    out_max = min(out_max, b);
    if out_min > out_max {
        return ClipRange(0u, out_min, out_max);
    }
    return ClipRange(1u, out_min, out_max);
}

fn clip_segment_to_box(p0: vec3<f32>, p1: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> ClippedSegment {
    let d = p1 - p0;
    var t_min = 0.0;
    var t_max = 1.0;

    var range = clip_axis_to_range(p0.x, d.x, box_min.x, box_max.x, t_min, t_max);
    if range.ok == 0u {
        return ClippedSegment(0u, p0, p1);
    }
    t_min = range.t_min;
    t_max = range.t_max;

    range = clip_axis_to_range(p0.y, d.y, box_min.y, box_max.y, t_min, t_max);
    if range.ok == 0u {
        return ClippedSegment(0u, p0, p1);
    }
    t_min = range.t_min;
    t_max = range.t_max;

    range = clip_axis_to_range(p0.z, d.z, box_min.z, box_max.z, t_min, t_max);
    if range.ok == 0u {
        return ClippedSegment(0u, p0, p1);
    }
    t_min = range.t_min;
    t_max = range.t_max;

    return ClippedSegment(1u, p0 + d * t_min, p0 + d * t_max);
}

fn trace_segment_into_coarse_count_page(
    p0: vec3<f32>,
    p1: vec3<f32>,
    frustum: FrustumDesc,
    tile_idx: u32,
    local_coarse_tile: u32,
    mark_only: bool,
    fill_refs: bool,
    seg_ref: FineSegRef,
) {
    let coarse_x = local_coarse_tile % frustum.coarse_tiles_x;
    let coarse_y = local_coarse_tile / frustum.coarse_tiles_x;
    let coarse_tile_px_x = max(1u, frustum.froxel_size_x * COARSE_FINE_TILE_EXTENT);
    let coarse_tile_px_y = max(1u, frustum.froxel_size_y * COARSE_FINE_TILE_EXTENT);
    let origin = vec2<f32>(f32(coarse_x * coarse_tile_px_x), f32(coarse_y * coarse_tile_px_y));
    let tile_max = vec2<f32>(
        min(origin.x + f32(coarse_tile_px_x), f32(frustum.screen_width)),
        min(origin.y + f32(coarse_tile_px_y), f32(frustum.screen_height)),
    ) - vec2<f32>(1e-3, 1e-3);
    let clipped = clip_segment_to_box(
        p0,
        p1,
        vec3<f32>(origin, 0.0),
        vec3<f32>(max(origin, tile_max), 1.0),
    );
    if clipped.ok == 0u {
        return;
    }

    let z0_slice = coarse_lut_slice_for_depth(tile_idx, clipped.p0.z);
    let z1_slice = coarse_lut_slice_for_depth(tile_idx, clipped.p1.z);
    if z0_slice == INVALID_PTR || z1_slice == INVALID_PTR {
        return;
    }

    let min_z_slice = min(z0_slice, z1_slice);
    let max_z_slice = max(z0_slice, z1_slice);
    let lut_base = tile_idx * COARSE_DEPTH_SLICES;
    for (var coarse_z = min_z_slice; coarse_z <= max_z_slice; coarse_z = coarse_z + 1u) {
        let lut_idx = lut_base + coarse_z;
        if lut_idx >= arrayLength(&coarse_depth_lut) {
            continue;
        }
        let lut_entry = coarse_depth_lut[lut_idx];
        if lut_entry.virtual_count_q == 0u {
            continue;
        }

        let z_min = dequantize_depth01(lut_entry.z_min_q);
        let z_max = dequantize_depth01(lut_entry.z_max_q);
        let depth_clipped = clip_segment_to_box(
            clipped.p0,
            clipped.p1,
            vec3<f32>(origin, z_min),
            vec3<f32>(max(origin, tile_max), z_max),
        );
        if depth_clipped.ok == 0u {
            continue;
        }

        let page_table_idx = coarse_count_page_table_idx(tile_idx, coarse_z);
        if mark_only {
            mark_coarse_count_page(page_table_idx);
            if pc.fine_binning_backend != 0u && page_table_idx < arrayLength(&virtual_page_candidate_counts) {
                atomicAdd(&virtual_page_candidate_counts[page_table_idx], 1u);
            }
            continue;
        }

        let page_idx = get_coarse_count_page(page_table_idx);
        if page_idx == INVALID_PTR {
            continue;
        }
        if pc.telemetry_enabled != 0u && !fill_refs {
            let telemetry_idx = TELEMETRY_PAGE_COUNTS_OFFSET + page_idx;
            if telemetry_idx < arrayLength(&prepass_telemetry) {
                atomicAdd(&prepass_telemetry[telemetry_idx], 1u);
            }
        }

        let z_span = max(z_max - z_min, 1.0 / f32(DEPTH_QUANT_MAX));
        let p0_local = vec3<f32>(
            (depth_clipped.p0.x - origin.x) / f32(frustum.froxel_size_x),
            (depth_clipped.p0.y - origin.y) / f32(frustum.froxel_size_y),
            clamp((depth_clipped.p0.z - z_min) / z_span, 0.0, 0.999999) * f32(COARSE_DEPTH_SLICES),
        );
        let p1_local = vec3<f32>(
            (depth_clipped.p1.x - origin.x) / f32(frustum.froxel_size_x),
            (depth_clipped.p1.y - origin.y) / f32(frustum.froxel_size_y),
            clamp((depth_clipped.p1.z - z_min) / z_span, 0.0, 0.999999) * f32(COARSE_DEPTH_SLICES),
        );

        let f0 = vec3<i32>(
            clamp(i32(floor(p0_local.x)), 0, i32(COARSE_FINE_TILE_EXTENT) - 1),
            clamp(i32(floor(p0_local.y)), 0, i32(COARSE_FINE_TILE_EXTENT) - 1),
            clamp(i32(floor(p0_local.z)), 0, i32(COARSE_DEPTH_SLICES) - 1),
        );
        let f1 = vec3<i32>(
            clamp(i32(floor(p1_local.x)), 0, i32(COARSE_FINE_TILE_EXTENT) - 1),
            clamp(i32(floor(p1_local.y)), 0, i32(COARSE_FINE_TILE_EXTENT) - 1),
            clamp(i32(floor(p1_local.z)), 0, i32(COARSE_DEPTH_SLICES) - 1),
        );
        let dir = p1_local - p0_local;
        let step = vec3<i32>(sgn_i32(dir.x), sgn_i32(dir.y), sgn_i32(dir.z));
        let safe_dir = select(
            dir,
            vec3<f32>(1e-6, 1e-6, 1e-6),
            abs(dir) < vec3<f32>(1e-6, 1e-6, 1e-6),
        );
        let delta_dist = abs(vec3<f32>(1.0, 1.0, 1.0) / safe_dir);
        var f = f0;
        var t_max = select(
            (floor(p0_local) - p0_local) / safe_dir,
            ((floor(p0_local) + 1.0) - p0_local) / safe_dir,
            step > vec3(0),
        );
        t_max = select(
            t_max,
            vec3<f32>(1e38, 1e38, 1e38),
            abs(dir) < vec3<f32>(1e-6, 1e-6, 1e-6),
        );

        var safety = 0u;
        loop {
            if safety > COARSE_FINE_TILE_EXTENT + COARSE_FINE_TILE_EXTENT + COARSE_DEPTH_SLICES + 3u {
                break;
            }
            let fine_x = coarse_x * COARSE_FINE_TILE_EXTENT + u32(f.x);
            let fine_y = coarse_y * COARSE_FINE_TILE_EXTENT + u32(f.y);
            if fine_tile_occluded_by_opaque(frustum, fine_x, fine_y, z_max) {
                if all(f == f1) {
                    break;
                }
                let a_min = canonical_min_mask(t_max);
                let fd = select(vec3<i32>(0), step, a_min);
                let dist = select(vec3<f32>(0), delta_dist, a_min);
                f += fd;
                t_max += dist;
                if any(f < vec3<i32>(0)) || f.x >= i32(COARSE_FINE_TILE_EXTENT) || f.y >= i32(COARSE_FINE_TILE_EXTENT) || f.z >= i32(COARSE_DEPTH_SLICES) {
                    break;
                }
                safety = safety + 1u;
                continue;
            }
            if fill_refs {
                write_fine_seg_ref(page_idx, u32(f.x), u32(f.y), u32(f.z), seg_ref);
            } else {
                increment_coarse_count_page_cell(page_idx, u32(f.x), u32(f.y), u32(f.z));
            }
            if all(f == f1) {
                break;
            }
            let a_min = canonical_min_mask(t_max);
            let fd = select(vec3<i32>(0), step, a_min);
            let dist = select(vec3<f32>(0), delta_dist, a_min);
            f += fd;
            t_max += dist;
            if any(f < vec3<i32>(0)) || f.x >= i32(COARSE_FINE_TILE_EXTENT) || f.y >= i32(COARSE_FINE_TILE_EXTENT) || f.z >= i32(COARSE_DEPTH_SLICES) {
                break;
            }
            safety = safety + 1u;
        }
    }
}

fn find_coarse_asset_range(inst_id: u32, frustum_id: u32) -> u32 {
    let lookup_idx = inst_id * pc.frustum_count + frustum_id;
    if lookup_idx < arrayLength(&coarse_range_lookup) {
        let range_idx = coarse_range_lookup[lookup_idx];
        if range_idx != INVALID_PTR && range_idx < arrayLength(&coarse_range_queue.ranges) {
            let range = coarse_range_queue.ranges[range_idx];
            if range.geo_id == inst_id && range.frustum_id == frustum_id {
                return range_idx;
            }
        }
    }
    return INVALID_PTR;
}

fn visit_segment_asset_coarse_tiles(p0: vec3<f32>, p1: vec3<f32>, frustum: FrustumDesc, frustum_id: u32, inst_id: u32, mark_only: bool, fill_refs: bool, seg_ref: FineSegRef) {
    let range_idx = find_coarse_asset_range(inst_id, frustum_id);
    if range_idx == INVALID_PTR || range_idx >= arrayLength(&coarse_range_queue.ranges) {
        return;
    }
    let range = coarse_range_queue.ranges[range_idx];
    if range.frustum_id != frustum_id {
        return;
    }

    let coarse_tile_px_x = max(1u, frustum.froxel_size_x * COARSE_FINE_TILE_EXTENT);
    let coarse_tile_px_y = max(1u, frustum.froxel_size_y * COARSE_FINE_TILE_EXTENT);
    let seg_min = min(p0.xy, p1.xy);
    let seg_max = max(p0.xy, p1.xy);
    let min_cx = max(range.min_x, min(u32(max(seg_min.x, 0.0)) / coarse_tile_px_x, frustum.coarse_tiles_x - 1u));
    let max_cx = min(range.max_x, min(u32(max(seg_max.x, 0.0)) / coarse_tile_px_x, frustum.coarse_tiles_x - 1u));
    let min_cy = max(range.min_y, min(u32(max(seg_min.y, 0.0)) / coarse_tile_px_y, frustum.coarse_tiles_y - 1u));
    let max_cy = min(range.max_y, min(u32(max(seg_max.y, 0.0)) / coarse_tile_px_y, frustum.coarse_tiles_y - 1u));
    if max_cx < min_cx || max_cy < min_cy {
        return;
    }

    let coarse_box_min = vec3<f32>(
        f32(min_cx * coarse_tile_px_x),
        f32(min_cy * coarse_tile_px_y),
        0.0,
    );
    let coarse_box_max = vec3<f32>(
        min(f32((max_cx + 1u) * coarse_tile_px_x), f32(frustum.screen_width)) - 1e-3,
        min(f32((max_cy + 1u) * coarse_tile_px_y), f32(frustum.screen_height)) - 1e-3,
        1.0,
    );
    let clipped = clip_segment_to_box(p0, p1, coarse_box_min, coarse_box_max);
    if clipped.ok == 0u {
        return;
    }

    let p0_coarse = vec2<f32>(
        clipped.p0.x / f32(coarse_tile_px_x),
        clipped.p0.y / f32(coarse_tile_px_y),
    );
    let p1_coarse = vec2<f32>(
        clipped.p1.x / f32(coarse_tile_px_x),
        clipped.p1.y / f32(coarse_tile_px_y),
    );
    let f0 = vec2<i32>(
        clamp(i32(floor(p0_coarse.x)), i32(min_cx), i32(max_cx)),
        clamp(i32(floor(p0_coarse.y)), i32(min_cy), i32(max_cy)),
    );
    let f1 = vec2<i32>(
        clamp(i32(floor(p1_coarse.x)), i32(min_cx), i32(max_cx)),
        clamp(i32(floor(p1_coarse.y)), i32(min_cy), i32(max_cy)),
    );

    let dir = p1_coarse - p0_coarse;
    let step = vec2<i32>(sgn_i32(dir.x), sgn_i32(dir.y));
    let safe_dir = select(
        dir,
        vec2<f32>(1e-6, 1e-6),
        abs(dir) < vec2<f32>(1e-6, 1e-6),
    );
    let delta_dist = abs(vec2<f32>(1.0, 1.0) / safe_dir);
    var f = f0;
    var t_max = select(
        (floor(p0_coarse) - p0_coarse) / safe_dir,
        ((floor(p0_coarse) + 1.0) - p0_coarse) / safe_dir,
        step > vec2<i32>(0, 0),
    );
    t_max = select(
        t_max,
        vec2<f32>(1e38, 1e38),
        abs(dir) < vec2<f32>(1e-6, 1e-6),
    );

    var safety = 0u;
    let max_steps = (max_cx - min_cx + 1u) + (max_cy - min_cy + 1u) + 2u;
    loop {
        if safety > max_steps {
            break;
        }
        if f.x < i32(min_cx) || f.x > i32(max_cx) || f.y < i32(min_cy) || f.y > i32(max_cy) {
            break;
        }

        let cx = u32(f.x);
        let cy = u32(f.y);
        let local_tile = cy * frustum.coarse_tiles_x + cx;
        if local_tile < frustum.coarse_depth_tile_count {
            let tile_idx = frustum.coarse_depth_tile_base + local_tile;
            trace_segment_into_coarse_count_page(p0, p1, frustum, tile_idx, local_tile, mark_only, fill_refs, seg_ref);
        }

        if all(f == f1) {
            break;
        }
        let step_x = t_max.x <= t_max.y;
        if step_x {
            f.x = f.x + step.x;
            t_max.x = t_max.x + delta_dist.x;
        } else {
            f.y = f.y + step.y;
            t_max.y = t_max.y + delta_dist.y;
        }
        safety = safety + 1u;
    }
}

fn scatter_page_candidate(page_idx: u32, task_idx: u32) {
    if page_idx >= arrayLength(&page_candidate_counts) {
        return;
    }
    if page_idx >= arrayLength(&page_candidate_cursors) || page_idx >= arrayLength(&page_candidate_offsets) {
        return;
    }
    let local_idx = atomicAdd(&page_candidate_cursors[page_idx], 1u);
    let candidate_count = atomicLoad(&page_candidate_counts[page_idx]);
    if local_idx >= candidate_count {
        return;
    }
    let dst = page_candidate_offsets[page_idx] + local_idx;
    if dst < arrayLength(&page_candidates) {
        page_candidates[dst] = task_idx;
    }
}

fn trace_segment_into_candidate_pages(
    task_idx: u32,
    p0: vec3<f32>,
    p1: vec3<f32>,
    frustum: FrustumDesc,
    tile_idx: u32,
    local_coarse_tile: u32,
) {
    let coarse_x = local_coarse_tile % frustum.coarse_tiles_x;
    let coarse_y = local_coarse_tile / frustum.coarse_tiles_x;
    let coarse_tile_px_x = max(1u, frustum.froxel_size_x * COARSE_FINE_TILE_EXTENT);
    let coarse_tile_px_y = max(1u, frustum.froxel_size_y * COARSE_FINE_TILE_EXTENT);
    let origin = vec2<f32>(f32(coarse_x * coarse_tile_px_x), f32(coarse_y * coarse_tile_px_y));
    let tile_max = vec2<f32>(
        min(origin.x + f32(coarse_tile_px_x), f32(frustum.screen_width)),
        min(origin.y + f32(coarse_tile_px_y), f32(frustum.screen_height)),
    ) - vec2<f32>(1e-3, 1e-3);
    let clipped = clip_segment_to_box(
        p0,
        p1,
        vec3<f32>(origin, 0.0),
        vec3<f32>(max(origin, tile_max), 1.0),
    );
    if clipped.ok == 0u {
        return;
    }

    let z0_slice = coarse_lut_slice_for_depth(tile_idx, clipped.p0.z);
    let z1_slice = coarse_lut_slice_for_depth(tile_idx, clipped.p1.z);
    if z0_slice == INVALID_PTR || z1_slice == INVALID_PTR {
        return;
    }

    let lut_base = tile_idx * COARSE_DEPTH_SLICES;
    for (var coarse_z = min(z0_slice, z1_slice); coarse_z <= max(z0_slice, z1_slice); coarse_z = coarse_z + 1u) {
        let lut_idx = lut_base + coarse_z;
        if lut_idx >= arrayLength(&coarse_depth_lut) {
            continue;
        }
        let lut_entry = coarse_depth_lut[lut_idx];
        if lut_entry.virtual_count_q == 0u {
            continue;
        }
        let depth_clipped = clip_segment_to_box(
            clipped.p0,
            clipped.p1,
            vec3<f32>(origin, dequantize_depth01(lut_entry.z_min_q)),
            vec3<f32>(max(origin, tile_max), dequantize_depth01(lut_entry.z_max_q)),
        );
        if depth_clipped.ok == 0u {
            continue;
        }
        let page_idx = get_coarse_count_page(coarse_count_page_table_idx(tile_idx, coarse_z));
        if page_idx != INVALID_PTR {
            scatter_page_candidate(page_idx, task_idx);
        }
    }
}

fn visit_segment_candidate_pages(
    task_idx: u32,
    p0: vec3<f32>,
    p1: vec3<f32>,
    frustum: FrustumDesc,
    frustum_id: u32,
    inst_id: u32,
) {
    let range_idx = find_coarse_asset_range(inst_id, frustum_id);
    if range_idx == INVALID_PTR || range_idx >= arrayLength(&coarse_range_queue.ranges) {
        return;
    }
    let range = coarse_range_queue.ranges[range_idx];
    if range.frustum_id != frustum_id {
        return;
    }

    let coarse_tile_px_x = max(1u, frustum.froxel_size_x * COARSE_FINE_TILE_EXTENT);
    let coarse_tile_px_y = max(1u, frustum.froxel_size_y * COARSE_FINE_TILE_EXTENT);
    let seg_min = min(p0.xy, p1.xy);
    let seg_max = max(p0.xy, p1.xy);
    let min_cx = max(range.min_x, min(u32(max(seg_min.x, 0.0)) / coarse_tile_px_x, frustum.coarse_tiles_x - 1u));
    let max_cx = min(range.max_x, min(u32(max(seg_max.x, 0.0)) / coarse_tile_px_x, frustum.coarse_tiles_x - 1u));
    let min_cy = max(range.min_y, min(u32(max(seg_min.y, 0.0)) / coarse_tile_px_y, frustum.coarse_tiles_y - 1u));
    let max_cy = min(range.max_y, min(u32(max(seg_max.y, 0.0)) / coarse_tile_px_y, frustum.coarse_tiles_y - 1u));
    if max_cx < min_cx || max_cy < min_cy {
        return;
    }

    let coarse_box_min = vec3<f32>(f32(min_cx * coarse_tile_px_x), f32(min_cy * coarse_tile_px_y), 0.0);
    let coarse_box_max = vec3<f32>(
        min(f32((max_cx + 1u) * coarse_tile_px_x), f32(frustum.screen_width)) - 1e-3,
        min(f32((max_cy + 1u) * coarse_tile_px_y), f32(frustum.screen_height)) - 1e-3,
        1.0,
    );
    let clipped = clip_segment_to_box(p0, p1, coarse_box_min, coarse_box_max);
    if clipped.ok == 0u {
        return;
    }

    let p0_coarse = clipped.p0.xy / vec2<f32>(f32(coarse_tile_px_x), f32(coarse_tile_px_y));
    let p1_coarse = clipped.p1.xy / vec2<f32>(f32(coarse_tile_px_x), f32(coarse_tile_px_y));
    let f0 = vec2<i32>(
        clamp(i32(floor(p0_coarse.x)), i32(min_cx), i32(max_cx)),
        clamp(i32(floor(p0_coarse.y)), i32(min_cy), i32(max_cy)),
    );
    let f1 = vec2<i32>(
        clamp(i32(floor(p1_coarse.x)), i32(min_cx), i32(max_cx)),
        clamp(i32(floor(p1_coarse.y)), i32(min_cy), i32(max_cy)),
    );
    let dir = p1_coarse - p0_coarse;
    let step = vec2<i32>(sgn_i32(dir.x), sgn_i32(dir.y));
    let safe_dir = select(dir, vec2<f32>(1e-6), abs(dir) < vec2<f32>(1e-6));
    let delta_dist = abs(vec2<f32>(1.0) / safe_dir);
    var f = f0;
    var t_max = select(
        (floor(p0_coarse) - p0_coarse) / safe_dir,
        ((floor(p0_coarse) + 1.0) - p0_coarse) / safe_dir,
        step > vec2<i32>(0),
    );
    t_max = select(t_max, vec2<f32>(1e38), abs(dir) < vec2<f32>(1e-6));
    var safety = 0u;
    let max_steps = (max_cx - min_cx + 1u) + (max_cy - min_cy + 1u) + 2u;
    loop {
        if safety > max_steps || f.x < i32(min_cx) || f.x > i32(max_cx) || f.y < i32(min_cy) || f.y > i32(max_cy) {
            break;
        }
        let local_tile = u32(f.y) * frustum.coarse_tiles_x + u32(f.x);
        if local_tile < frustum.coarse_depth_tile_count {
            trace_segment_into_candidate_pages(
                task_idx,
                p0,
                p1,
                frustum,
                frustum.coarse_depth_tile_base + local_tile,
                local_tile,
            );
        }
        if all(f == f1) {
            break;
        }
        if t_max.x <= t_max.y {
            f.x += step.x;
            t_max.x += delta_dist.x;
        } else {
            f.y += step.y;
            t_max.y += delta_dist.y;
        }
        safety += 1u;
    }
}

fn scatter_page_candidate_task(task_idx: u32) {
    let binning_task_count = atomicLoad(&binning_queue.tail);
    if task_idx >= binning_task_count || task_idx >= arrayLength(&projected_segments) {
        return;
    }
    let task = binning_queue.tasks[task_idx];
    let frustum_id = unpack_binning_frustum(task.packed_field);
    if frustum_id >= arrayLength(&frustum_table) {
        return;
    }
    let frustum = frustum_table[frustum_id];
    if task.id_info >= arrayLength(&strand_instances) {
        return;
    }
    let projected = projected_segments[task_idx];
    let screen_size = vec2<f32>(f32(frustum.screen_width), f32(frustum.screen_height));
    let depths = unpack2x16unorm(projected.depths);
    visit_segment_candidate_pages(
        task_idx,
        vec3<f32>(unpack2x16unorm(projected.p0_xy) * screen_size, depths.x),
        vec3<f32>(unpack2x16unorm(projected.p1_xy) * screen_size, depths.y),
        frustum,
        frustum_id,
        task.id_info,
    );
}

fn process_binning_task(task_idx: u32, mark_only: bool, fill_refs: bool) {
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
    let inst_id = task.id_info;
    if inst_id >= arrayLength(&strand_instances) || task_idx >= arrayLength(&projected_segments) {
        return;
    }

    let instance = strand_instances[inst_id];
    let projected = projected_segments[task_idx];
    let p0_xy = unpack2x16unorm(projected.p0_xy)
        * vec2<f32>(f32(frustum.screen_width), f32(frustum.screen_height));
    let p1_xy = unpack2x16unorm(projected.p1_xy)
        * vec2<f32>(f32(frustum.screen_width), f32(frustum.screen_height));
    let depths = unpack2x16unorm(projected.depths);
    let p0 = vec3<f32>(p0_xy, depths.x);
    let p1 = vec3<f32>(p1_xy, depths.y);
    var seg_ref = FineSegRef(task.id_info, task.seg_idx, 0u);
    if fill_refs {
        seg_ref = make_fine_seg_ref(task, instance.meta_id);
    }

    visit_segment_asset_coarse_tiles(p0, p1, frustum, frustum_id, inst_id, mark_only, fill_refs, seg_ref);
}

@compute @workgroup_size(FINE_WORKGROUP_SIZE, 1, 1)
fn mark_coarse_count_pages_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let task_idx = gid.y * pc.workgroup_offset + gid.x;
    process_binning_task(task_idx, true, false);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn allocate_coarse_count_pages(@builtin(global_invocation_id) gid: vec3<u32>) {
    let page_table_idx = gid.x;
    if page_table_idx >= pc.num_elements || page_table_idx >= arrayLength(&coarse_count_page_table) {
        return;
    }
    if atomicLoad(&coarse_count_page_table[page_table_idx]) != MARKED_COUNT_PAGE {
        return;
    }

    let page_count_capacity = arrayLength(&coarse_count_pages.counts) / COARSE_COUNT_PAGE_SIZE;
    let page_idx = atomicAdd(&coarse_count_pages.tail, 1u);
    if page_idx >= page_count_capacity {
        atomicStore(&coarse_count_page_table[page_table_idx], 0u);
        return;
    }
    // Keep the reverse mapping next to the compact page allocation. Prefixing can
    // then iterate allocated pages instead of probing the sparse page table.
    fine_page_meta[page_idx] = FinePageMeta(
        0u,
        0u,
        0u,
        0u,
        page_table_idx / COARSE_DEPTH_SLICES,
        page_table_idx % COARSE_DEPTH_SLICES,
        0u,
        0u,
    );
    if pc.fine_binning_backend != 0u && page_idx < arrayLength(&page_candidate_counts) && page_table_idx < arrayLength(&virtual_page_candidate_counts) {
        atomicStore(
            &page_candidate_counts[page_idx],
            atomicLoad(&virtual_page_candidate_counts[page_table_idx]),
        );
    }
    atomicStore(&coarse_count_page_table[page_table_idx], page_idx + 1u);
}

@compute @workgroup_size(1, 1, 1)
fn finalize_prefix_dispatch() {
    let page_count = min(atomicLoad(&coarse_count_pages.tail), arrayLength(&fine_page_meta));
    if page_count == 0u {
        dispatch_args[0] = 0u;
        dispatch_args[1] = 0u;
        dispatch_args[2] = 0u;
        return;
    }
    let x = min(page_count, MAX_DISPATCH_WORKGROUPS_PER_DIMENSION);
    dispatch_args[0] = x;
    dispatch_args[1] = ceil_div_u32(page_count, x);
    dispatch_args[2] = 1u;
}

@compute @workgroup_size(FINE_WORKGROUP_SIZE, 1, 1)
fn binning_queue_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let task_idx = gid.y * pc.workgroup_offset + gid.x;
    process_binning_task(task_idx, false, false);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn prefix_page_candidates(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
) {
    let lane = local_id.x;
    let page_count = min(atomicLoad(&coarse_count_pages.tail), arrayLength(&page_candidate_counts));
    if lane == 0u {
        candidate_scan_base = 0u;
    }
    workgroupBarrier();

    let block_count = ceil_div_u32(page_count, WORKGROUP_SIZE);
    for (var block = 0u; block < block_count; block += 1u) {
        let page_idx = block * WORKGROUP_SIZE + lane;
        var count = 0u;
        if page_idx < page_count {
            count = atomicLoad(&page_candidate_counts[page_idx]);
        }
        let subgroup_exclusive = subgroupExclusiveAdd(count);
        let subgroup_inclusive = subgroupInclusiveAdd(count);
        if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
            page_scan_values[subgroup_id] = subgroup_inclusive;
        }
        workgroupBarrier();
        if lane == 0u {
            var block_total = 0u;
            for (var i = 0u; i < PAGE_SCAN_SUBGROUPS; i += 1u) {
                let subtotal = page_scan_values[i];
                page_scan_values[i] = block_total;
                block_total += subtotal;
            }
            candidate_scan_block_base = candidate_scan_base;
            candidate_scan_block_total = block_total;
        }
        workgroupBarrier();
        if page_idx < page_count && page_idx < arrayLength(&page_candidate_offsets) {
            page_candidate_offsets[page_idx] = candidate_scan_block_base
                + page_scan_values[subgroup_id]
                + subgroup_exclusive;
        }
        workgroupBarrier();
        if lane == 0u {
            candidate_scan_base += candidate_scan_block_total;
        }
        workgroupBarrier();
    }
    if lane == 0u && page_count < arrayLength(&page_candidate_offsets) {
        page_candidate_offsets[page_count] = candidate_scan_base;
    }
}

@compute @workgroup_size(FINE_WORKGROUP_SIZE, 1, 1)
fn scatter_page_candidates(@builtin(global_invocation_id) gid: vec3<u32>) {
    let task_idx = gid.y * pc.workgroup_offset + gid.x;
    scatter_page_candidate_task(task_idx);
}

@compute @workgroup_size(COARSE_COUNT_PAGE_SIZE, 1, 1)
fn prefix_fine_pages(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let page_idx = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    let cell_idx = local_id.x;
    // A single-row dispatch is exact. Only a two-dimensional dispatch can have
    // padding in its final row, so keep the page-tail load out of the common path.
    if num_workgroups.y > 1u && page_idx >= min(atomicLoad(&coarse_count_pages.tail), arrayLength(&fine_page_meta)) {
        return;
    }
    let page_desc = fine_page_meta[page_idx];

    let count_idx = page_idx * COARSE_COUNT_PAGE_SIZE + cell_idx;
    var count = 0u;
    if count_idx < arrayLength(&coarse_count_pages.counts) {
        count = atomicLoad(&coarse_count_pages.counts[count_idx]);
    }
    page_scan_counts[cell_idx] = count;
    page_scan_values[cell_idx] = 0u;
    workgroupBarrier();

    let seg_ref_exclusive = subgroupExclusiveAdd(count);
    let seg_ref_inclusive = subgroupInclusiveAdd(count);
    if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
        page_scan_values[subgroup_id] = seg_ref_inclusive;
    }
    workgroupBarrier();
    if cell_idx == 0u {
        var accumulator = 0u;
        for (var i = 0u; i < PAGE_SCAN_SUBGROUPS; i = i + 1u) {
            let subtotal = page_scan_values[i];
            page_scan_values[i] = accumulator;
            accumulator = accumulator + subtotal;
        }
    }
    workgroupBarrier();

    let exclusive_offset = seg_ref_exclusive + page_scan_values[subgroup_id];
    let global_cell_idx = page_idx * COARSE_COUNT_PAGE_SIZE + cell_idx;
    if global_cell_idx < arrayLength(&fine_cell_offsets) {
        fine_cell_offsets[global_cell_idx] = exclusive_offset;
    }
    if global_cell_idx < arrayLength(&fine_cell_write_cursors) {
        atomicStore(&fine_cell_write_cursors[global_cell_idx], 0u);
    }

    if cell_idx == COARSE_COUNT_PAGE_SIZE - 1u {
        page_scan_total = exclusive_offset + page_scan_counts[cell_idx];
    }
    workgroupBarrier();

    if cell_idx == 0u {
        let seg_ref_count = page_scan_total;
        var seg_ref_base = 0u;
        var flags = 0u;
        if seg_ref_count > 0u {
            seg_ref_base = atomicAdd(&fine_seg_refs.tail, seg_ref_count);
            if seg_ref_base + seg_ref_count > arrayLength(&fine_seg_refs.refs) {
                flags = 1u;
            }
        }
        fine_page_meta[page_idx] = FinePageMeta(
            seg_ref_base,
            select(seg_ref_count, 0u, flags != 0u),
            0u,
            0u,
            page_desc.coarse_tile_id,
            page_desc.coarse_z,
            0u,
            flags,
        );
        active_page_frustum_id = frustum_for_coarse_tile(page_desc.coarse_tile_id);
    }
    workgroupBarrier();

    if cell_idx < COARSE_FINE_TILE_EXTENT * COARSE_FINE_TILE_EXTENT
        && fine_page_meta[page_idx].flags == 0u
        && active_page_frustum_id != INVALID_PTR
    {
        var tile_has_work = false;
        for (var z = 0u; z < COARSE_DEPTH_SLICES; z += 1u) {
            tile_has_work = tile_has_work || page_scan_counts[z * COARSE_FINE_TILE_EXTENT * COARSE_FINE_TILE_EXTENT + cell_idx] > 0u;
        }
        if tile_has_work {
            append_active_fine_tile(active_page_frustum_id, page_desc.coarse_tile_id, cell_idx);
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn finalize_telemetry(@builtin(global_invocation_id) gid: vec3<u32>) {
    let page_count = min(atomicLoad(&coarse_count_pages.tail), arrayLength(&fine_page_meta));
    let page_idx = gid.x;
    if page_idx == 0u {
        atomicStore(&prepass_telemetry[0], page_count);
        atomicStore(&prepass_telemetry[1], atomicLoad(&binning_queue.tail));
        atomicStore(&prepass_telemetry[3], atomicLoad(&fine_seg_refs.tail));
    }
    if page_idx >= page_count {
        return;
    }
    let page_desc = fine_page_meta[page_idx];
    let frustum_id = frustum_for_coarse_tile(page_desc.coarse_tile_id);
    if frustum_id != INVALID_PTR {
        frustum_telemetry_add(frustum_id, FRUSTUM_TELEMETRY_ALLOCATED_PAGES, 1u);
    }
    let telemetry_idx = TELEMETRY_PAGE_COUNTS_OFFSET + page_idx;
    if telemetry_idx >= arrayLength(&prepass_telemetry) {
        return;
    }
    var candidate_count = atomicLoad(&prepass_telemetry[telemetry_idx]);
    if pc.fine_binning_backend != 0u && page_idx < arrayLength(&page_candidate_counts) {
        candidate_count = atomicLoad(&page_candidate_counts[page_idx]);
    }
    if candidate_count == 0u {
        return;
    }
    atomicAdd(&prepass_telemetry[2], candidate_count);
    atomicAdd(&prepass_telemetry[4], 1u);
    atomicMax(&prepass_telemetry[5], candidate_count);
    if frustum_id != INVALID_PTR {
        frustum_telemetry_add(frustum_id, FRUSTUM_TELEMETRY_PAGE_CANDIDATES, candidate_count);
        frustum_telemetry_add(frustum_id, FRUSTUM_TELEMETRY_FINE_REFS, page_desc.seg_ref_count);
        frustum_telemetry_max(frustum_id, FRUSTUM_TELEMETRY_MAX_PAGE_CANDIDATES, candidate_count);
    }
    var histogram_bin = 0u;
    var remaining = candidate_count;
    while remaining > 1u && histogram_bin + 1u < TELEMETRY_HISTOGRAM_BINS {
        remaining = remaining >> 1u;
        histogram_bin = histogram_bin + 1u;
    }
    atomicAdd(&prepass_telemetry[TELEMETRY_HISTOGRAM_OFFSET + histogram_bin], 1u);
}

@compute @workgroup_size(FINE_WORKGROUP_SIZE, 1, 1)
fn fill_fine_seg_refs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let task_idx = gid.y * pc.workgroup_offset + gid.x;
    process_binning_task(task_idx, false, true);
}

fn frustum_for_coarse_tile(tile_idx: u32) -> u32 {
    let frustum_count = min(pc.frustum_count, arrayLength(&frustum_table));
    for (var frustum_id = 0u; frustum_id < frustum_count; frustum_id = frustum_id + 1u) {
        let frustum = frustum_table[frustum_id];
        if tile_idx >= frustum.coarse_depth_tile_base && tile_idx < frustum.coarse_depth_tile_base + frustum.coarse_depth_tile_count {
            return frustum_id;
        }
    }
    return INVALID_PTR;
}

fn append_active_fine_tile(frustum_id: u32, coarse_tile_id: u32, local_fine_tile: u32) {
    if frustum_id >= arrayLength(&frustum_table) {
        return;
    }
    let frustum = frustum_table[frustum_id];
    let local_coarse_tile = coarse_tile_id - frustum.coarse_depth_tile_base;
    let coarse_x = local_coarse_tile % frustum.coarse_tiles_x;
    let coarse_y = local_coarse_tile / frustum.coarse_tiles_x;
    let local_x = local_fine_tile % COARSE_FINE_TILE_EXTENT;
    let local_y = local_fine_tile / COARSE_FINE_TILE_EXTENT;
    let fine_x = coarse_x * COARSE_FINE_TILE_EXTENT + local_x;
    let fine_y = coarse_y * COARSE_FINE_TILE_EXTENT + local_y;
    let fine_tiles_x = ceil_div_u32(frustum.screen_width, frustum.froxel_size_x);
    let fine_tiles_y = ceil_div_u32(frustum.screen_height, frustum.froxel_size_y);
    if fine_x >= fine_tiles_x || fine_y >= fine_tiles_y {
        return;
    }
    // Match the former dense emitter's coarse-tile-major traversal so the
    // compact queue retains its established cache and raster work ordering.
    let flag_idx = coarse_tile_id * (COARSE_FINE_TILE_EXTENT * COARSE_FINE_TILE_EXTENT)
        + local_fine_tile;
    if flag_idx >= arrayLength(&active_fine_tile_flags) {
        return;
    }
    if atomicExchange(&active_fine_tile_flags[flag_idx], 1u) != 0u {
        return;
    }
    let block_idx = flag_idx / COARSE_COUNT_PAGE_SIZE;
    if block_idx < arrayLength(&active_fine_tile_block_counts) {
        atomicAdd(&active_fine_tile_block_counts[block_idx], 1u);
    }
}

fn trace_candidate_inside_page(
    p0: vec3<f32>,
    p1: vec3<f32>,
    frustum: FrustumDesc,
    page_idx: u32,
    page_desc: FinePageMeta,
    fill_refs: bool,
    seg_ref: FineSegRef,
) {
    let local_coarse_tile = page_desc.coarse_tile_id - frustum.coarse_depth_tile_base;
    let coarse_x = local_coarse_tile % frustum.coarse_tiles_x;
    let coarse_y = local_coarse_tile / frustum.coarse_tiles_x;
    let coarse_tile_px_x = max(1u, frustum.froxel_size_x * COARSE_FINE_TILE_EXTENT);
    let coarse_tile_px_y = max(1u, frustum.froxel_size_y * COARSE_FINE_TILE_EXTENT);
    let origin = vec2<f32>(f32(coarse_x * coarse_tile_px_x), f32(coarse_y * coarse_tile_px_y));
    let tile_max = vec2<f32>(
        min(origin.x + f32(coarse_tile_px_x), f32(frustum.screen_width)),
        min(origin.y + f32(coarse_tile_px_y), f32(frustum.screen_height)),
    ) - vec2<f32>(1e-3);
    let lut_idx = page_desc.coarse_tile_id * COARSE_DEPTH_SLICES + page_desc.coarse_z;
    if lut_idx >= arrayLength(&coarse_depth_lut) {
        return;
    }
    let lut_entry = coarse_depth_lut[lut_idx];
    if lut_entry.virtual_count_q == 0u {
        return;
    }
    let z_min = dequantize_depth01(lut_entry.z_min_q);
    let z_max = dequantize_depth01(lut_entry.z_max_q);
    let coarse_clipped = clip_segment_to_box(
        p0,
        p1,
        vec3<f32>(origin, 0.0),
        vec3<f32>(max(origin, tile_max), 1.0),
    );
    if coarse_clipped.ok == 0u {
        return;
    }
    let clipped = clip_segment_to_box(
        coarse_clipped.p0,
        coarse_clipped.p1,
        vec3<f32>(origin, z_min),
        vec3<f32>(max(origin, tile_max), z_max),
    );
    if clipped.ok == 0u {
        return;
    }

    let z_span = max(z_max - z_min, 1.0 / f32(DEPTH_QUANT_MAX));
    let p0_local = vec3<f32>(
        (clipped.p0.x - origin.x) / f32(frustum.froxel_size_x),
        (clipped.p0.y - origin.y) / f32(frustum.froxel_size_y),
        clamp((clipped.p0.z - z_min) / z_span, 0.0, 0.999999) * f32(COARSE_DEPTH_SLICES),
    );
    let p1_local = vec3<f32>(
        (clipped.p1.x - origin.x) / f32(frustum.froxel_size_x),
        (clipped.p1.y - origin.y) / f32(frustum.froxel_size_y),
        clamp((clipped.p1.z - z_min) / z_span, 0.0, 0.999999) * f32(COARSE_DEPTH_SLICES),
    );
    let f0 = vec3<i32>(
        clamp(i32(floor(p0_local.x)), 0, i32(COARSE_FINE_TILE_EXTENT) - 1),
        clamp(i32(floor(p0_local.y)), 0, i32(COARSE_FINE_TILE_EXTENT) - 1),
        clamp(i32(floor(p0_local.z)), 0, i32(COARSE_DEPTH_SLICES) - 1),
    );
    let f1 = vec3<i32>(
        clamp(i32(floor(p1_local.x)), 0, i32(COARSE_FINE_TILE_EXTENT) - 1),
        clamp(i32(floor(p1_local.y)), 0, i32(COARSE_FINE_TILE_EXTENT) - 1),
        clamp(i32(floor(p1_local.z)), 0, i32(COARSE_DEPTH_SLICES) - 1),
    );
    let dir = p1_local - p0_local;
    let step = vec3<i32>(sgn_i32(dir.x), sgn_i32(dir.y), sgn_i32(dir.z));
    let safe_dir = select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    let delta_dist = abs(vec3<f32>(1.0) / safe_dir);
    var f = f0;
    var t_max = select(
        (floor(p0_local) - p0_local) / safe_dir,
        ((floor(p0_local) + 1.0) - p0_local) / safe_dir,
        step > vec3<i32>(0),
    );
    t_max = select(t_max, vec3<f32>(1e38), abs(dir) < vec3<f32>(1e-6));
    var safety = 0u;
    loop {
        if safety > COARSE_FINE_TILE_EXTENT + COARSE_FINE_TILE_EXTENT + COARSE_DEPTH_SLICES + 3u {
            break;
        }
        let fine_x = coarse_x * COARSE_FINE_TILE_EXTENT + u32(f.x);
        let fine_y = coarse_y * COARSE_FINE_TILE_EXTENT + u32(f.y);
        if !fine_tile_occluded_by_opaque(frustum, fine_x, fine_y, z_max) {
            let cell_idx = fine_cell_idx(u32(f.x), u32(f.y), u32(f.z));
            if fill_refs {
                let local_write = atomicAdd(&csr_cell_cursors[cell_idx], 1u);
                let global_cell_idx = page_idx * COARSE_COUNT_PAGE_SIZE + cell_idx;
                let dst = csr_page_ref_base + fine_cell_offsets[global_cell_idx] + local_write;
                if dst < arrayLength(&fine_seg_refs.refs) {
                    fine_seg_refs.refs[dst] = seg_ref;
                }
            } else {
                atomicAdd(&csr_cell_counts[cell_idx], 1u);
            }
        }
        if all(f == f1) {
            break;
        }
        let a_min = canonical_min_mask(t_max);
        f += select(vec3<i32>(0), step, a_min);
        t_max += select(vec3<f32>(0), delta_dist, a_min);
        if any(f < vec3<i32>(0)) || f.x >= i32(COARSE_FINE_TILE_EXTENT) || f.y >= i32(COARSE_FINE_TILE_EXTENT) || f.z >= i32(COARSE_DEPTH_SLICES) {
            break;
        }
        safety += 1u;
    }
}

fn process_candidate_for_page(
    task_idx: u32,
    page_idx: u32,
    page_desc: FinePageMeta,
    frustum_id: u32,
    frustum: FrustumDesc,
    fill_refs: bool,
) {
    if task_idx >= atomicLoad(&binning_queue.tail) || task_idx >= arrayLength(&projected_segments) {
        return;
    }
    let task = binning_queue.tasks[task_idx];
    if unpack_binning_frustum(task.packed_field) != frustum_id || task.id_info >= arrayLength(&strand_instances) {
        return;
    }
    let projected = projected_segments[task_idx];
    let screen_size = vec2<f32>(f32(frustum.screen_width), f32(frustum.screen_height));
    let depths = unpack2x16unorm(projected.depths);
    var seg_ref = FineSegRef(task.id_info, task.seg_idx, 0u);
    if fill_refs {
        let instance = strand_instances[task.id_info];
        seg_ref = make_fine_seg_ref(task, instance.meta_id);
    }
    trace_candidate_inside_page(
        vec3<f32>(unpack2x16unorm(projected.p0_xy) * screen_size, depths.x),
        vec3<f32>(unpack2x16unorm(projected.p1_xy) * screen_size, depths.y),
        frustum,
        page_idx,
        page_desc,
        fill_refs,
        seg_ref,
    );
}

@compute @workgroup_size(COARSE_COUNT_PAGE_SIZE, 1, 1)
fn build_fine_pages_csr(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let page_idx = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    let page_count = min(atomicLoad(&coarse_count_pages.tail), arrayLength(&fine_page_meta));
    if page_idx >= page_count {
        return;
    }
    let lane = local_id.x;
    atomicStore(&csr_cell_counts[lane], 0u);
    atomicStore(&csr_cell_cursors[lane], 0u);
    if lane == 0u {
        csr_page_ref_base = 0u;
        csr_page_valid = 0u;
    }
    workgroupBarrier();

    let page_desc = fine_page_meta[page_idx];
    let frustum_id = frustum_for_coarse_tile(page_desc.coarse_tile_id);
    if frustum_id == INVALID_PTR || page_idx >= arrayLength(&page_candidate_offsets) || page_idx >= arrayLength(&page_candidate_counts) {
        return;
    }
    let frustum = frustum_table[frustum_id];
    let candidate_base = page_candidate_offsets[page_idx];
    let candidate_count = atomicLoad(&page_candidate_counts[page_idx]);
    let candidate_range_valid = candidate_base <= arrayLength(&page_candidates)
        && candidate_count <= arrayLength(&page_candidates) - candidate_base;

    if candidate_range_valid {
        for (var candidate_local = lane; candidate_local < candidate_count; candidate_local += COARSE_COUNT_PAGE_SIZE) {
            process_candidate_for_page(
                page_candidates[candidate_base + candidate_local],
                page_idx,
                page_desc,
                frustum_id,
                frustum,
                false,
            );
        }
    }
    workgroupBarrier();

    let count = atomicLoad(&csr_cell_counts[lane]);
    let subgroup_exclusive = subgroupExclusiveAdd(count);
    let subgroup_inclusive = subgroupInclusiveAdd(count);
    if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
        page_scan_values[subgroup_id] = subgroup_inclusive;
    }
    workgroupBarrier();
    if lane == 0u {
        var accumulator = 0u;
        for (var i = 0u; i < PAGE_SCAN_SUBGROUPS; i += 1u) {
            let subtotal = page_scan_values[i];
            page_scan_values[i] = accumulator;
            accumulator += subtotal;
        }
        page_scan_total = accumulator;
    }
    workgroupBarrier();

    let exclusive_offset = subgroup_exclusive + page_scan_values[subgroup_id];
    let global_cell_idx = page_idx * COARSE_COUNT_PAGE_SIZE + lane;
    if global_cell_idx < arrayLength(&fine_cell_offsets) {
        fine_cell_offsets[global_cell_idx] = exclusive_offset;
    }
    if global_cell_idx < arrayLength(&coarse_count_pages.counts) {
        atomicStore(&coarse_count_pages.counts[global_cell_idx], count);
    }
    if lane == 0u {
        var flags = select(1u, 0u, candidate_range_valid);
        var seg_ref_base = 0u;
        if page_scan_total > 0u && flags == 0u {
            seg_ref_base = atomicAdd(&fine_seg_refs.tail, page_scan_total);
            if seg_ref_base + page_scan_total > arrayLength(&fine_seg_refs.refs) {
                flags = 1u;
            }
        }
        csr_page_ref_base = seg_ref_base;
        csr_page_valid = select(1u, 0u, flags != 0u);
        fine_page_meta[page_idx] = FinePageMeta(
            seg_ref_base,
            select(page_scan_total, 0u, flags != 0u),
            0u,
            0u,
            page_desc.coarse_tile_id,
            page_desc.coarse_z,
            0u,
            flags,
        );
        active_page_frustum_id = frustum_id;
    }
    workgroupBarrier();

    if lane < COARSE_FINE_TILE_EXTENT * COARSE_FINE_TILE_EXTENT && csr_page_valid != 0u {
        var tile_has_work = false;
        for (var z = 0u; z < COARSE_DEPTH_SLICES; z += 1u) {
            tile_has_work = tile_has_work || atomicLoad(&csr_cell_counts[z * COARSE_FINE_TILE_EXTENT * COARSE_FINE_TILE_EXTENT + lane]) > 0u;
        }
        if tile_has_work {
            append_active_fine_tile(active_page_frustum_id, page_desc.coarse_tile_id, lane);
        }
    }

    if csr_page_valid != 0u {
        for (var candidate_local = lane; candidate_local < candidate_count; candidate_local += COARSE_COUNT_PAGE_SIZE) {
            process_candidate_for_page(
                page_candidates[candidate_base + candidate_local],
                page_idx,
                page_desc,
                frustum_id,
                frustum,
                true,
            );
        }
    }
}

fn raster_depth_key(page_table_idx: u32, fine_z: u32) -> u32 {
    if page_table_idx >= arrayLength(&coarse_depth_lut) {
        return fine_z;
    }
    let lut = coarse_depth_lut[page_table_idx];
    if lut.virtual_count_q == 0u {
        return fine_z;
    }
    return lut.virtual_start_q + (lut.virtual_count_q * fine_z) / COARSE_DEPTH_SLICES;
}

fn mark_vsms_request(surface_id: u32, page_tile: vec3<u32>) {
    if surface_id == INVALID_PTR || surface_id >= arrayLength(&vsms_request_meta) {
        return;
    }
    let request_row = vsms_request_meta[surface_id];
    let addr = vsms_request_local_tile_index(request_row, page_tile, 0u, 0u);
    if addr.valid == 0u || addr.word_index >= arrayLength(&vsms_request_bits) {
        return;
    }
    atomicOr(&vsms_request_bits[addr.word_index], addr.bit_mask);
}

fn mark_shadow_dom_page_request(surface_ids: vec2<u32>, page_tile: vec3<u32>) {
    mark_vsms_request(surface_ids.x, page_tile);
    mark_vsms_request(surface_ids.y, page_tile);
}

fn mark_shadow_dom_page_request_with_border(surface_ids: vec2<u32>, page_tile: vec3<u32>) {
    if surface_ids.x == INVALID_PTR || surface_ids.x >= arrayLength(&vsms_request_meta) {
        return;
    }
    let request_row = vsms_request_meta[surface_ids.x];
    if request_row.base_tiles_x == 0u || request_row.base_tiles_y == 0u {
        return;
    }

    let min_x = page_tile.x - min(page_tile.x, DOM_PREFETCH_PAGE_BORDER);
    let min_y = page_tile.y - min(page_tile.y, DOM_PREFETCH_PAGE_BORDER);
    let max_x = min(page_tile.x + DOM_PREFETCH_PAGE_BORDER, request_row.base_tiles_x - 1u);
    let max_y = min(page_tile.y + DOM_PREFETCH_PAGE_BORDER, request_row.base_tiles_y - 1u);

    for (var py = min_y; py <= max_y; py = py + 1u) {
        for (var px = min_x; px <= max_x; px = px + 1u) {
            mark_shadow_dom_page_request(surface_ids, vec3<u32>(px, py, page_tile.z));
        }
    }
}

fn request_shadow_dom_pages(frustum_id: u32, frustum: FrustumDesc, fine_x: u32, fine_y: u32) {
    if frustum.kind != 1u || frustum_id >= arrayLength(&shadow_dom_surface_ids) {
        return;
    }
    let surface_ids = shadow_dom_surface_ids[frustum_id];
    if surface_ids.x == INVALID_PTR || surface_ids.y == INVALID_PTR {
        return;
    }

    let pixel_min_x = fine_x * frustum.froxel_size_x;
    let pixel_min_y = fine_y * frustum.froxel_size_y;
    let pixel_max_x = min(pixel_min_x + frustum.froxel_size_x - 1u, frustum.screen_width - 1u);
    let pixel_max_y = min(pixel_min_y + frustum.froxel_size_y - 1u, frustum.screen_height - 1u);
    let page_min = vec2<u32>(pixel_min_x / DOM_PAGE_XY, pixel_min_y / DOM_PAGE_XY);
    let page_max = vec2<u32>(pixel_max_x / DOM_PAGE_XY, pixel_max_y / DOM_PAGE_XY);

    for (var py = page_min.y; py <= page_max.y; py = py + 1u) {
        for (var px = page_min.x; px <= page_max.x; px = px + 1u) {
            mark_shadow_dom_page_request_with_border(surface_ids, vec3<u32>(px, py, 0u));
        }
    }
}

@compute @workgroup_size(COARSE_COUNT_PAGE_SIZE, 1, 1)
fn prefix_active_tile_blocks(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
) {
    let lane = local_id.x;
    if lane == 0u {
        candidate_scan_base = 0u;
    }
    workgroupBarrier();

    let block_count = arrayLength(&active_fine_tile_block_counts);
    let chunk_count = ceil_div_u32(block_count, COARSE_COUNT_PAGE_SIZE);
    for (var chunk = 0u; chunk < chunk_count; chunk += 1u) {
        let block_idx = chunk * COARSE_COUNT_PAGE_SIZE + lane;
        var count = 0u;
        if block_idx < block_count {
            count = atomicLoad(&active_fine_tile_block_counts[block_idx]);
        }
        let subgroup_exclusive = subgroupExclusiveAdd(count);
        let subgroup_inclusive = subgroupInclusiveAdd(count);
        if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
            page_scan_values[subgroup_id] = subgroup_inclusive;
        }
        workgroupBarrier();
        if lane == 0u {
            var chunk_total = 0u;
            for (var i = 0u; i < PAGE_SCAN_SUBGROUPS; i += 1u) {
                let subtotal = page_scan_values[i];
                page_scan_values[i] = chunk_total;
                chunk_total += subtotal;
            }
            candidate_scan_block_base = candidate_scan_base;
            candidate_scan_block_total = chunk_total;
        }
        workgroupBarrier();
        if block_idx < block_count {
            active_fine_tile_block_offsets[block_idx] = candidate_scan_block_base
                + page_scan_values[subgroup_id]
                + subgroup_exclusive;
        }
        workgroupBarrier();
        if lane == 0u {
            candidate_scan_base += candidate_scan_block_total;
        }
        workgroupBarrier();
    }
    if lane == 0u {
        atomicStore(&active_fine_tile_queue.tail, candidate_scan_base);
    }
}

@compute @workgroup_size(COARSE_COUNT_PAGE_SIZE, 1, 1)
fn compact_active_tiles(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
) {
    let block_idx = workgroup_id.x;
    if block_idx >= arrayLength(&active_fine_tile_block_counts) {
        return;
    }
    let lane = local_id.x;
    let dense_tile_idx = block_idx * COARSE_COUNT_PAGE_SIZE + lane;
    var active_flag = 0u;
    if dense_tile_idx < arrayLength(&active_fine_tile_flags) {
        active_flag = atomicLoad(&active_fine_tile_flags[dense_tile_idx]);
    }
    let subgroup_exclusive = subgroupExclusiveAdd(active_flag);
    let subgroup_inclusive = subgroupInclusiveAdd(active_flag);
    if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
        page_scan_values[subgroup_id] = subgroup_inclusive;
    }
    workgroupBarrier();
    if lane == 0u {
        var accumulator = 0u;
        for (var i = 0u; i < PAGE_SCAN_SUBGROUPS; i += 1u) {
            let subtotal = page_scan_values[i];
            page_scan_values[i] = accumulator;
            accumulator += subtotal;
        }
    }
    workgroupBarrier();

    if active_flag == 0u {
        return;
    }
    let queue_idx = active_fine_tile_block_offsets[block_idx]
        + page_scan_values[subgroup_id]
        + subgroup_exclusive;
    if queue_idx < arrayLength(&active_fine_tile_queue.items) {
        active_fine_tile_queue.items[queue_idx] = dense_tile_idx;
    }
}

@compute @workgroup_size(COARSE_COUNT_PAGE_SIZE, 1, 1)
fn emit_raster_work(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_local_id: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let active_tile_idx = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    let active_tile_count = min(
        atomicLoad(&active_fine_tile_queue.tail),
        arrayLength(&active_fine_tile_queue.items),
    );
    if active_tile_idx >= active_tile_count {
        return;
    }

    let fine_stack_idx = active_fine_tile_queue.items[active_tile_idx];
    let local_fine_tile = fine_stack_idx
        % (COARSE_FINE_TILE_EXTENT * COARSE_FINE_TILE_EXTENT);
    let tile_idx = fine_stack_idx
        / (COARSE_FINE_TILE_EXTENT * COARSE_FINE_TILE_EXTENT);
    let frustum_id = frustum_for_coarse_tile(tile_idx);
    if frustum_id == INVALID_PTR {
        return;
    }
    let frustum = frustum_table[frustum_id];
    let local_coarse_tile = tile_idx - frustum.coarse_depth_tile_base;
    let coarse_x = local_coarse_tile % frustum.coarse_tiles_x;
    let coarse_y = local_coarse_tile / frustum.coarse_tiles_x;
    let fine_tiles_x = ceil_div_u32(frustum.screen_width, frustum.froxel_size_x);
    let local_x = local_fine_tile % COARSE_FINE_TILE_EXTENT;
    let local_y = local_fine_tile / COARSE_FINE_TILE_EXTENT;
    let fine_x = coarse_x * COARSE_FINE_TILE_EXTENT + local_x;
    let fine_y = coarse_y * COARSE_FINE_TILE_EXTENT + local_y;
    let screen_tile_id = fine_y * fine_tiles_x + fine_x;

    let cell_lane = local_id.x;
    // Raster consumers iterate tile runs forward and expect front-to-back order.
    // Depth keys use linear reverse-Z, so larger keys are nearer. LUT slices are
    // stored in ascending key order; reverse the depth lane mapping during
    // compaction so each tile run is emitted near-to-far.
    let depth_lane = (COARSE_COUNT_PAGE_SIZE - 1u) - cell_lane;
    let coarse_z = depth_lane / COARSE_DEPTH_SLICES;
    let fine_z = depth_lane % COARSE_DEPTH_SLICES;
    var page_table_idx = 0u;
    var page_idx = INVALID_PTR;
    var cell_idx = 0u;
    var seg_ref_count = 0u;
    var seg_ref_base = 0u;
    var page_flags = 1u;
    if coarse_z < COARSE_DEPTH_SLICES && fine_z < COARSE_DEPTH_SLICES {
        page_table_idx = coarse_count_page_table_idx(tile_idx, coarse_z);
        if page_table_idx < arrayLength(&coarse_count_page_table) {
            let page_handle = atomicLoad(&coarse_count_page_table[page_table_idx]);
            if page_handle != 0u && page_handle != MARKED_COUNT_PAGE {
                page_idx = page_handle - 1u;
                if page_idx < arrayLength(&fine_page_meta) {
                    let page_desc = fine_page_meta[page_idx];
                    page_flags = page_desc.flags;
                    cell_idx = fine_cell_idx(local_x, local_y, fine_z);
                    let count_idx = page_idx * COARSE_COUNT_PAGE_SIZE + cell_idx;
                    let global_cell_idx = page_idx * COARSE_COUNT_PAGE_SIZE + cell_idx;
                    if page_desc.seg_ref_count > 0u && page_flags == 0u && count_idx < arrayLength(&coarse_count_pages.counts) && global_cell_idx < arrayLength(&fine_cell_offsets) {
                        seg_ref_count = atomicLoad(&coarse_count_pages.counts[count_idx]);
                        if seg_ref_count > 0u {
                            seg_ref_base = page_desc.seg_ref_base + fine_cell_offsets[global_cell_idx];
                        }
                    }
                }
            }
        }
    }

    let has_work = select(0u, 1u, seg_ref_count > 0u);
    let work_exclusive = subgroupExclusiveAdd(has_work);
    let work_inclusive = subgroupInclusiveAdd(has_work);
    let load_inclusive = subgroupInclusiveAdd(seg_ref_count);
    if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
        page_scan_values[subgroup_id] = work_inclusive;
        page_scan_counts[subgroup_id] = load_inclusive;
    }
    workgroupBarrier();

    if cell_lane == 0u {
        var work_acc = 0u;
        var load_acc = 0u;
        for (var i = 0u; i < PAGE_SCAN_SUBGROUPS; i = i + 1u) {
            let work_subtotal = page_scan_values[i];
            let load_subtotal = page_scan_counts[i];
            page_scan_values[i] = work_acc;
            work_acc = work_acc + work_subtotal;
            load_acc = load_acc + load_subtotal;
        }
        tile_run_work_count = work_acc;
        tile_run_load_score = load_acc;
        tile_run_idx = INVALID_PTR;
        tile_run_work_base = INVALID_PTR;
        if work_acc > 0u {
            if frustum.kind == 1u {
                let run_idx = atomicAdd(&shadow_raster_tile_run_queue.tail, 1u);
                if run_idx < arrayLength(&shadow_raster_tile_run_queue.items) {
                    tile_run_idx = run_idx;
                }
            } else {
                let run_idx = atomicAdd(&camera_raster_tile_run_queue.tail, 1u);
                if run_idx < arrayLength(&camera_raster_tile_run_queue.items) {
                    tile_run_idx = run_idx;
                }
            }
            if tile_run_idx != INVALID_PTR {
                tile_run_work_base = atomicAdd(&raster_work_queue.tail, work_acc);
                request_shadow_dom_pages(frustum_id, frustum, fine_x, fine_y);
            }
        }
    }
    workgroupBarrier();

    let work_offset = work_exclusive + page_scan_values[subgroup_id];
    if has_work != 0u && tile_run_work_base != INVALID_PTR {
        let write_idx = tile_run_work_base + work_offset;
        if write_idx < arrayLength(&raster_work_queue.items) {
            raster_work_queue.items[write_idx] = RasterWorkItem(
                seg_ref_base,
                seg_ref_count,
                frustum_id,
                screen_tile_id,
                raster_depth_key(page_table_idx, fine_z),
                seg_ref_count,
                page_idx,
                cell_idx,
            );
        }
    }
    workgroupBarrier();

    if cell_lane == 0u && tile_run_idx != INVALID_PTR {
        let run = RasterTileRun(
            tile_run_work_base,
            tile_run_work_count,
            frustum_id,
            screen_tile_id,
            tile_run_load_score,
            0u,
            0u,
            0u,
        );
        if frustum.kind == 1u {
            shadow_raster_tile_run_queue.items[tile_run_idx] = run;
        } else {
            camera_raster_tile_run_queue.items[tile_run_idx] = run;
        }
        frustum_telemetry_add(frustum_id, FRUSTUM_TELEMETRY_RASTER_RUNS, 1u);
        frustum_telemetry_add(frustum_id, FRUSTUM_TELEMETRY_RASTER_LOAD, tile_run_load_score);
        frustum_telemetry_max(frustum_id, FRUSTUM_TELEMETRY_MAX_RASTER_LOAD, tile_run_load_score);
    }
}

@compute @workgroup_size(1, 1, 1)
fn finalize_active_tile_dispatch() {
    let active_tile_count = min(
        atomicLoad(&active_fine_tile_queue.tail),
        arrayLength(&active_fine_tile_queue.items),
    );
    if active_tile_count == 0u {
        dispatch_args[0] = 0u;
        dispatch_args[1] = 0u;
        dispatch_args[2] = 0u;
        return;
    }
    let x = min(active_tile_count, MAX_DISPATCH_WORKGROUPS_PER_DIMENSION);
    dispatch_args[0] = x;
    dispatch_args[1] = ceil_div_u32(active_tile_count, x);
    dispatch_args[2] = 1u;
}

@compute @workgroup_size(1, 1, 1)
fn finalize_raster_dispatch() {
    let camera_run_count = min(atomicLoad(&camera_raster_tile_run_queue.tail), arrayLength(&camera_raster_tile_run_queue.items));
    if camera_run_count == 0u {
        camera_raster_tile_run_dispatch_args[0] = 0u;
        camera_raster_tile_run_dispatch_args[1] = 0u;
        camera_raster_tile_run_dispatch_args[2] = 0u;
    } else {
        let x = min(camera_run_count, 65535u);
        camera_raster_tile_run_dispatch_args[0] = x;
        camera_raster_tile_run_dispatch_args[1] = ceil_div_u32(camera_run_count, x);
        camera_raster_tile_run_dispatch_args[2] = 1u;
    }

    let shadow_run_count = min(atomicLoad(&shadow_raster_tile_run_queue.tail), arrayLength(&shadow_raster_tile_run_queue.items));
    if shadow_run_count == 0u {
        shadow_raster_tile_run_dispatch_args[0] = 0u;
        shadow_raster_tile_run_dispatch_args[1] = 0u;
        shadow_raster_tile_run_dispatch_args[2] = 0u;
    } else {
        let x = min(shadow_run_count, 65535u);
        shadow_raster_tile_run_dispatch_args[0] = x;
        shadow_raster_tile_run_dispatch_args[1] = ceil_div_u32(shadow_run_count, x);
        shadow_raster_tile_run_dispatch_args[2] = 1u;
    }
}
