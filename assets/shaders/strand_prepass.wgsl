#import bevy_render::view::View
#import bevy_pbr::mesh_view_types as types

#import "shaders/common.wgsl"::{
    wang_hash,
    hash_to_unit_float,
}

#import "shaders/types.wgsl"::{
    Aabb,
    DevicePtr,
    Vertices,
    Indices,
    Geos,
    Meta,
    StrandGeo,
    FinePrepassTask,
    BinningTask,
}

#import "shaders/queues.wgsl"::{
    FinePrepassQueue,
    BinningQueue,
}

const WORKGROUP_SIZE: u32 = #WORKGROUP_SIZE;
const FINE_WORKGROUP_SIZE: u32 = #FINE_WORKGROUP_SIZE;
const CULL_MAX_DIST: f32 = 100.0;
const CULL_MIN_DIST: f32 = 12.0;

const SIZEOF_METADATA: u32 = #SIZEOF_METADATA;
const SIZEOF_GEO: u32 = #SIZEOF_GEO;

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

fn is_valid_ptr(_ptr: DevicePtr) -> bool {
    return _ptr.slab != 0xFFFFFFFFu;
}

fn stochastic_cull_camera(cam: View, aabb: Aabb, sample_threshold: f32) -> bool {
    let aabb_center = (aabb.max + aabb.min) * 0.5;
    let distance_to_cam = length(cam.world_position - aabb_center);
    let norm_distance = max(distance_to_cam - CULL_MIN_DIST, 0.0) / CULL_MAX_DIST;
    return sample_threshold <= pow(norm_distance, 0.15);
}

fn ceil_div_u32(x: u32, y: u32) -> u32 {
    return (x + y - 1u) / y;
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
                let task_index = atomicAdd(&fine_phase_queue.tail, 1u);
                let strand_hash = wang_hash(geo_idx + strand_local);
                let strand_visible = !stochastic_cull_camera(view, geo.aabb, hash_to_unit_float(strand_hash));
                if strand_visible && task_index < arrayLength(&fine_phase_queue.tasks) {
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

    // Camera frustum only for now.
    let packed_field_camera = 0u;
    let segment_count = strand_meta.count - 1u;

    for (var i = 0u; i < segment_count; i = i + 1u) {
        let seg_index = strand_meta.offset + i;
        let idx = indices[index_ptr.slab].is[seg_index];
        let write_idx = atomicAdd(&binning_queue.tail, 1u);
        if write_idx < arrayLength(&binning_queue.tasks) {
            binning_queue.tasks[write_idx] = BinningTask(
                task.inst_id,
                0u,
                seg_index,
                packed_field_camera,
            );
        }
    }
}
