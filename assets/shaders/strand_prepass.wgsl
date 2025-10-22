#import bevy_pbr::{
    mesh_view_types::POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE,
    mesh_view_bindings as view_bindings,
}

#import "shaders/common.wgsl"::{
    world_to_screen_raw
}

#import "shaders/types.wgsl"::{
    Aabb,
    StrandGeo,
    StrandMeta,
    FinePrepassTask,
    BinningTask,
    unpackU8,
}

#import "shaders/queues.wgsl"

const WORKGROUP_SIZE: u32 = #WORKGROUP_SIZE;
const VISIBILITY_CULL_LIMIT_PX_LEN: f32 = 0.5; // TODO: can be a runtime param.

@group(0) @binding(#{VERTEX_BUFFER}) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(#{INDEX_BUFFER}) var<storage, read> indices: array<u32>;
@group(0) @binding(#{META_BUFFER}) var<storage, read> strand_metadata: array<StrandMeta>;
@group(0) @binding(#{GEO_BUFFER}) var<storage, read> geos: array<StrandGeo>;
@group(0) @binding(#{VIEW_UNIFORM}) var<uniform> view: View;
@group(0) @binding(#{LIGHT_UNIFORM}) var<uniform> lights: types::Lights;
@group(0) @binding(#{CLUSTER_INDICES}) var<storage> clusterable_object_index_lists: types::ClusterLightIndexLists;
@group(0) @binding(#{CLUSTERABLE_OBJECTS}) var<storage> clusterable_objects: types::ClusterableObjects;
@group(0) @binding(#{CLUSTER_OFFSETS_AND_COUNTS}) var<storage> cluster_offsets_and_counts: types::ClusterOffsetsAndCounts;
@group(0) @binding(#{FINE_QUEUE}) var<storage, write> fine_phase_queue: FinePrepassQueue;
@group(0) @binding(#{VISIBLE_FLAGS}) var<storage, write> visible_flag: array<u32>;
@group(0) @binding(#{VISIBLE_GEO}) var<storage, write> visible_geos: array<u32>; // prefix
@group(0) @binding(#{GEO_PREFIX}) var<storage, write> geo_prefix: array<u32>;
@group(0) @binding(#{INDIRECT_BUFFER}) var<storage, write> dispatch_args: array<u32, 3u>;

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

fn frustrum_check(geo: StrandGeo) -> bool {
    // TODO build a view camera visibility mask

}

fn push_fine_task(task: FinePrepassTask) {
    let write_index = atomicAdd(&fine_phase_queue.tail, 1u);
    fine_phase_queue.tasks[write_index] = task;
}

fn pop_task() -> FinePrepassTask {
    let read_idx = atomicAdd(&fine_phase_queue.head, 1u);
    let task = fine_phase_queue.tasks[read_idx];
    return task;
}

fn process_broad(geo: StrandGeo, strand_idx: u32, subgroup_id: u32, subgroup_local_id: u32) {
    if (strand_idx >= geo.strand_count) { return; }

    // Example stochastic cull
    let strand_hash = wang_hash(strand_idx);
    let sample_threshold = hash_to_unit_float(strand_hash);
    let strand_visible = stochastic_cull_camera(view, geo.aabb, sample_threshold);

    let flag = u32(strand_visible);

    // ---- Workgroup prefix scan ----
    let wg_total = workgroup_exclusive_scan(local_id, subgroup_id, subgroup_local_id, flag);
    workgroupBarrier();

    // Only one thread atomically reserves space
    var base : u32 = 0u;
    if (lid == SCAN_THREADS - 1u) {
        base = atomicAdd(&fine_phase_queue.tail, wg_total);
    }
    // Broadcast base to all threads
    base = workgroupBroadcastFirst(base);
    workgroupBarrier();

    // ---- Write visible strands to global queue ----
    if (strand_visible) {
        let local_offset = wg_scan_storage[lid]; // written by scan
        let write_idx = base + local_offset;
        fine_phase_queue.tasks[write_idx] = FinePrepassTask(
            geo_id,
            strand_idx,
            geo.offset + strand_idx,
            geo.offset + geo.strand_count
        );
    }
}

const Lx = SCAN_THREADS;
const Ly = 1;
const Lz = 1;

@compute @workgroup_size(SCAN_THREADS, 1, 1)
fn broad_prepass(
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(subgroup_id) subgroup_id : u32,
    @builtin(subgroup_invocation_id) subgroup_local_id : u32
) {
    // Produces:
    // frustum intersection mask for lights and view
    let local_flat = local_id.z * Lx * Ly +
                            local_id.y * Lx +
                            local_id.x;

    let wg_flat = workgroup_id.z * num_workgroups.x * num_workgroups.y +
                     workgroup_id.y * num_workgroups.x +
                     workgroup_id.x;

    let global_flat = wg_flat * (Lx * Ly * Lz) + local_flat;
    let total_wgs = num_workgroups.x * num_workgroups.y;

    var geo_idx = wg_flat;
    var strand_idx = global_flat;
    let N_STRANDS = arrayLength(&strand_metadata);
    let N_GEOS = arrayLength(&geos);
    // grid-stride over geos
    while (geo_idx < N_GEOS) {
        let geo = geos[geo_idx];
        let strand_base = geo.offset;
        let strand_count = geo.strand_count;
        var strand_local = local_flat;
        // grid-stride over strands in geos
        while (strand_idx < N_STRANDS) {
            let strand_idx = strand_base + strand_local;
            process_broad(geo_idx, strand_idx, subgroup_id, subgroup_local_id);
            strand_local += Lx * Ly * Lz; // stride by workgroup
        }
        geo_idx += total_wgs; // next geo handled by this workgroup
    }
}

fn stochastic_cull_camera(view: View, aabb: Aabb, sample_threshold: f32) -> bool {
    let aabb_center = (aabb.max + aabb.min) / 2.0;
    let distance = length(view.world_position - aabb_center);
    let norm_distance = max((distance - CULL_MIN_DIST), 0.0) / CULL_MAX_DIST;
    if sample_threshold <= pow(norm_distance, 0.5) {
        return true;
    }
    return false;
}

fn local_light_index(task: ptr<function, FinePrepassTask>, i: u32) -> u32 {
    let word = (*task).local_light_indices[i >> 2u];
    return unpackU8(word, i & 3u); // index into ClusterableObjects
}

fn is_active(task: ptr<function, FinePrepassTask>, i: u32) -> bool {
    return (((*task).active_mask >> i) & 1u) != 0u;
}

fn is_shadow(task: ptr<function, FinePrepassTask>, i: u32) -> bool {
    return (((*task).shadow_mask >> i) & 1u) != 0u;
}
fn is_hq(task: ptr<function, FinePrepassTask>, i: u32) -> bool {
    return (((*task).hq_shadow_mask >> i) & 1u) != 0u;
}

fn importance(task: ptr<function, FinePrepassTask>, i: u32) -> u32 {
    let word = (*task).importance4[i >> 2u];
    return unpackU8(word, i & 3u);
}

fn seg_camera_visible(view: View, seg_v1: vec4<f32>, seg_v2: vec4<f32>) -> bool {
    let p1 = world_to_screen_raw(seg_v1, view.unjittered_clip_from_world, view.viewport);
    let p2 = world_to_screen_raw(seg_v2, view.unjittered_clip_from_world, view.viewport);
    if (distance(p1.xy, p2.xy) < VISIBILITY_CULL_LIMIT_PX_LEN) { return false; }
    return true;
}

fn seg_intersects_view(clip_from_world: mat4x4<f32>, viewport: vec4<f32>, t_view: u32, seg_v1: vec4<f32>, seg_v2: vec4<f32>) -> bool {
    /*
    clip_from_world: projection matrix of the view
    viewport: in the case of t_view = 0 this is the view.viewport
        in case of t_view = 1 this is ignored
        in case of t_view > 1 this is:
        light_custom_data: vec4<f32>,
        For point lights: the lower-right 2x2 values of the projection matrix [2][2] [2][3] [3][2] [3][3]
        For spot lights: the direction (x,z), spot_scale and spot_offset
    t_view: 0 = camera, 1 = directional light, 2 = point light, 3 = spot light
    */
    // TODO
    return true;
}

fn process_fine(task: FinePrepassTask, lid: u32) {
    let NUM_VIEWS = task.cli_base24_count8 + 1u; // TODO: multiview
    for (var v=0u; v < NUM_VIEWS; v++) {
        var clip_matrix: mat4x4<f32>;
        var viewport_info: vec4<f32>;
        var t_view: u32;
        if v == 0u {
            // camera view
            clip_matrix = view.unjittered_clip_from_world;
            viewport_info = view.viewport;
            t_view = 0u;
        }
        else if !is_active(task, v - 1u) {
            continue;
        } else {
            // TODO: look up the necessary lighting info for intersection test
        }

        // Now all threads handle this view in parallel
        // grid-stride over segments
        var seg_idx = lid;
        while (seg_idx < strand_meta[task.strand_id].count - 1u) {
            // build segment, test intersection with view frustum
            let idx = indices[seg_idx];
            let v1 = vertices[idx];
            let v2 = vertices[idx + 1u];
            // TODO seg_intersects_view
            if (seg_camera_visible(view, v1, v2) && seg_intersects_view(clip_matrix, viewport_info, t_view, v1, v2)) {
                local_visible_flags[seg_idx] = 1u;
            } else {
                local_visible_flags[seg_idx] = 0u;
            }
            seg_idx += WG_SIZE;
        }

        workgroupBarrier();

        // Compact visible segments
        let local_sum = workgroup_exclusive_scan(lid, subgroup_id, subgroup_local_id, local_visible_flags[lid]);
        let total = workgroupVisibleTotal; // returned by scan

        var base:u32 = 0u;
        if (lid == WG_SIZE - 1u) {
            base = atomicAdd(&binning_queue.tail, total);
        }
        base = workgroupBroadcastFirst(base);
        workgroupBarrier();

        if (local_visible_flags[lid] == 1u) {
            let write_idx = base + local_sum;
            let frustrum_id = (v << 1u) && (v != 0u);
            binning_queue.tasks[write_idx] = BinningTask(task.geo_id, task.strand_id, seg_idx_global, frustrum_id);
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn fine_prepass(@builtin(local_invocation_index) lid:u32)
{
    loop {
        // one atomic per workgroup, not per thread
        var task_idx:u32;
        if (lid==0u) {
            task_idx = atomicAdd(&fine_phase_queue.head, 1u);
        }
        task_idx = workgroupBroadcastFirst(task_idx);
        if (task_idx >= fine_phase_queue.tail) { break; }

        let task = fine_phase_queue.tasks[task_idx];
        process_fine(task, lid);
        // workgroupBarrier();      // optional if process_task uses shared mem
    }
}
