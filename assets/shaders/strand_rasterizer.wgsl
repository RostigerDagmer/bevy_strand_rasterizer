#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::mesh_view_types as types
// #import "shaders/spline.wgsl"::{ intersect_catmull_rom_spline_3d, closest_point };
#import "shaders/common.wgsl"::{ DOM_GAMMA, is_valid_ptr, find_clip_bounds, world_to_screen, world_to_screen_aabbnorm, world_to_screen_raw, screen_to_world_raw, screen_to_world, calculate_froxel_index, normalize_depth01, to_log_depth, }
#import "shaders/types.wgsl"::{
    Aabb,
    DevicePtr,
    FroxelConfig,
    Geos,
    Indices,
    Materials,
    Meta,
    SegmentRef,
    StrandGeo,
    StrandInstance,
    StrandMeta,
    StrandMaterial,
    Vertices,
    PushConstants,
}
#import "shaders/task_contract.wgsl"::{
    RasterWorkItem,
    FineSegRef,
}

const MAX_TEXTURE_EXT: u32 = #MAX_TEXTURE_EXTENT;
const SIZEOF_METADATA: u32 = #SIZEOF_METADATA;
const SIZEOF_MATERIAL: u32 = #SIZEOF_MATERIAL;
const SIZEOF_GEO: u32 = #SIZEOF_GEO;
const WORKGROUP_SIZE: u32 = #WORKGROUP_SIZE;
const LIGHT_INDEX: u32 = 0u; // Example constant for light index TODO: compute prepass -> indirect dispatch -> light index from uniforms
const MIN_HAIR_RADIUS_PIXELS : f32 = 0.4; // Example: Thickness in pixels
const MAX_HAIR_RADIUS_PIXELS : f32 = 2.0; // Example: Thickness in pixels
const POOL_CHUNK_SIZE: u32 = #POOL_CHUNK_SIZE;
const COARSE_FINE_TILE_EXTENT: u32 = #COARSE_FINE_TILE_EXTENT;
const CHUNK_WORD_STRIDE: u32 = 2u + POOL_CHUNK_SIZE;
const DEBUG_FORCE_SINGLE_LIGHT: bool = false;
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

struct RasterWorkQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    items: array<RasterWorkItem>,
}

struct FineSegRefBuffer {
    tail: atomic<u32>,
    refs: array<FineSegRef>,
}

@group(#{BIND_ARRAYS}) @binding(#{VERTICES}) var<storage, read_write> vertices: binding_array<Vertices>;
@group(#{BIND_ARRAYS}) @binding(#{INDICES}) var<storage, read_write> indices: binding_array<Indices>;
@group(#{BIND_ARRAYS}) @binding(#{STRAND_METADATA}) var<storage, read_write> strand_metadata: binding_array<Meta>;
@group(#{BIND_ARRAYS}) @binding(#{STRAND_MATERIALS}) var<storage, read_write> materials: binding_array<Materials>;
@group(#{BIND_ARRAYS}) @binding(#{STRAND_GEOS}) var<storage, read_write> geos: binding_array<Geos>;

@group(#{PAGE_TABLES}) @binding(#{VERTICES}) var<storage, read_write> t_vertices: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{INDICES}) var<storage, read_write> t_indices: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{STRAND_METADATA}) var<storage, read_write> t_strand_metadata: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{STRAND_MATERIALS}) var<storage, read_write> t_materials: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{STRAND_GEOS}) var<storage, read_write> t_geos: array<DevicePtr>;

#ifndef SHADOWS
@group(#{RASTER_GROUP}) @binding(#{OUTPUT_TEXTURE}) var render_target: texture_storage_2d<rgba8unorm, write>;
@group(#{RASTER_GROUP}) @binding(#{OUTPUT_DEPTH}) var depth_target: texture_storage_2d<r32float, write>;
@group(#{RASTER_GROUP}) @binding(#{SHADING_BUFFER}) var shading_buffer: texture_2d_array<f32>;
#endif
@group(#{RASTER_GROUP}) @binding(#{VIEW_UNIFORM}) var<uniform> view: View;
@group(#{RASTER_GROUP}) @binding(#{LIGHT_UNIFORM}) var<uniform> lights: types::Lights;
@group(#{RASTER_GROUP}) @binding(#{FRUSTUM_TABLE}) var<storage, read> frustum_table: array<FrustumDesc>;
@group(#{RASTER_GROUP}) @binding(#{FROXEL_BUCKET_HEADS}) var<storage, read> froxel_bucket_heads: array<atomic<u32>>;
@group(#{RASTER_GROUP}) @binding(#{CHUNK_POOL}) var<storage, read> chunk_pool_words: array<u32>;
@group(#{RASTER_GROUP}) @binding(#{RASTER_WORK_QUEUE}) var<storage, read> raster_work_queue: RasterWorkQueue;
@group(#{RASTER_GROUP}) @binding(#{FINE_SEG_REFS}) var<storage, read> fine_seg_refs: FineSegRefBuffer;
@group(#{RASTER_GROUP}) @binding(#{STRAND_INSTANCES}) var<storage, read> strand_instances: array<StrandInstance>;
@group(#{RASTER_GROUP}) @binding(#{COARSE_TILE_WORK_COUNTS}) var<storage, read> coarse_tile_work_counts: array<u32>;
@group(#{RASTER_GROUP}) @binding(#{COARSE_TILE_WORK_OFFSETS}) var<storage, read> coarse_tile_work_offsets: array<u32>;

// Helper: Signed distance from point `p` to line segment `a` -> `b`
// Returns distance. Clamps distance calc to the segment endpoints.
fn point_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, t: f32) -> f32 {
    // Clamp t to [0, 1] to stay within the segment
    let t_clamped = clamp(t, 0.0, 1.0);

    // Calculate the closest point on the segment to p
    let closest_point = a + t_clamped * (b - a);

    return distance(p, closest_point);
}

fn fragment_position_line_relative(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let l2 = distance(a, b);
    if l2 == 0.0 { return 0.0; } // Segment is a point
    let l2_sq = l2 * l2;

    // Project p onto the line defined by a, b. t is the projection parameter.
    let t = dot(p - a, b - a) / l2_sq;

    return t;
}

// Helper: Basic alpha blending (foreground "over" background)
fn blend_over(foreground: vec4<f32>, background: vec4<f32>) -> vec4<f32> {
    // Premultiply alpha for foreground
    let fg_rgb = foreground.rgb * foreground.a;
    let final_alpha = foreground.a + background.a * (1.0 - foreground.a);
    if final_alpha < 1e-6 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let final_rgb = (fg_rgb + background.rgb * background.a * (1.0 - foreground.a)) / final_alpha;
    return vec4<f32>(final_rgb, final_alpha);
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

fn get_segment_material(asset_id: u32, segment_ref: SegmentRef) -> StrandMaterial {
    if arrayLength(&t_strand_metadata) == 0u || arrayLength(&t_materials) == 0u || asset_id >= arrayLength(&t_strand_metadata) || asset_id >= arrayLength(&t_materials) {
        return StrandMaterial(vec4<f32>(1.0), vec4<f32>(1.0), 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0u, 0u);
    }
    let meta_ptr = t_strand_metadata[asset_id];
    let material_ptr = t_materials[asset_id];
    if !is_valid_ptr(meta_ptr) || !is_valid_ptr(material_ptr) {
        return StrandMaterial(vec4<f32>(1.0), vec4<f32>(1.0), 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0u, 0u);
    }

    let strand_local = segment_ref.strand_idx;
    let meta_base = meta_ptr.offset / SIZEOF_METADATA;
    let meta_count = meta_ptr.size / SIZEOF_METADATA;
    if strand_local >= meta_count {
        return StrandMaterial(vec4<f32>(1.0), vec4<f32>(1.0), 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0u, 0u);
    }

    let strand_meta = strand_metadata[meta_ptr.slab].ms[meta_base + strand_local];
    let material_base = material_ptr.offset / SIZEOF_MATERIAL;
    let material_count = material_ptr.size / SIZEOF_MATERIAL;
    if strand_meta.material_idx >= material_count {
        return StrandMaterial(vec4<f32>(1.0), vec4<f32>(1.0), 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0u, 0u);
    }
    return materials[material_ptr.slab].mats[material_base + strand_meta.material_idx];
}

#ifndef SHADOWS

fn get_segment_meta(asset_id: u32, segment_ref: SegmentRef) -> StrandMeta {
    if arrayLength(&t_strand_metadata) == 0u || asset_id >= arrayLength(&t_strand_metadata) {
        return StrandMeta(0u, 0u, 0u, 0u);
    }
    let meta_ptr = t_strand_metadata[asset_id];
    if !is_valid_ptr(meta_ptr) {
        return StrandMeta(0u, 0u, 0u, 0u);
    }
    let strand_local = segment_ref.strand_idx;
    let meta_base = meta_ptr.offset / SIZEOF_METADATA;
    let meta_count = meta_ptr.size / SIZEOF_METADATA;
    if strand_local >= meta_count {
        return StrandMeta(0u, 0u, 0u, 0u);
    }
    return strand_metadata[meta_ptr.slab].ms[meta_base + strand_local];
}

#endif

fn get_segment_vertices(inst_id: u32, asset_id: u32, segment_ref: SegmentRef) -> mat2x4<f32> {
    if arrayLength(&t_strand_metadata) == 0u || arrayLength(&t_indices) == 0u || arrayLength(&t_vertices) == 0u || asset_id >= arrayLength(&t_strand_metadata) || asset_id >= arrayLength(&t_indices) || asset_id >= arrayLength(&t_vertices) || inst_id >= arrayLength(&strand_instances) {
        return mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    let meta_ptr = t_strand_metadata[asset_id];
    let index_ptr = t_indices[asset_id];
    let vertex_ptr = t_vertices[asset_id];
    if !is_valid_ptr(meta_ptr) || !is_valid_ptr(index_ptr) || !is_valid_ptr(vertex_ptr) {
        return mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0));
    }

    let strand_local = segment_ref.strand_idx;
    let meta_base = meta_ptr.offset / SIZEOF_METADATA;
    let meta_count = meta_ptr.size / SIZEOF_METADATA;
    if strand_local >= meta_count {
        return mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    let strand_meta = strand_metadata[meta_ptr.slab].ms[meta_base + strand_local];

    let index_base = index_ptr.offset / 4u;
    let vertex_base = vertex_ptr.offset / 16u;
    let v0_idx = index_base + segment_ref.segment_start_idx;
    let v1_idx = index_base + segment_ref.segment_start_idx + 1u;
    let index_count = index_ptr.size / 4u;
    if segment_ref.segment_start_idx + 1u >= index_count || (segment_ref.segment_start_idx + 1u - strand_meta.offset) >= strand_meta.count - 1u {
        return mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0));
    }

    let v0_strand_idx = indices[index_ptr.slab].is[v0_idx];
    let v1_strand_idx = indices[index_ptr.slab].is[v1_idx];
    let vertex_count = vertex_ptr.size / 16u;
    if v0_strand_idx >= vertex_count || v1_strand_idx >= vertex_count {
        return mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    let world_from_local = strand_instances[inst_id].world_from_local;

    return mat2x4<f32>(
        world_from_local * vec4<f32>(vertices[vertex_ptr.slab].vs[vertex_base + v0_strand_idx], 1.0),
        world_from_local * vec4<f32>(vertices[vertex_ptr.slab].vs[vertex_base + v1_strand_idx], 1.0),
    );
}

// #define DEBUG

#ifdef SHADOWS
const DOM_SLICES: u32 = #{NUM_DOM_SLICES};
@group(#{VSMS_OPACITY_WRITE_GROUP}) @binding(#{VSMS_STORAGE_BINDING}) var deep_opacity_maps: texture_storage_3d<r32float, write>; // TODO: maybe find a more compact format
@group(#{VSMS_DEPTH_WRITE_GROUP}) @binding(#{VSMS_STORAGE_BINDING}) var deep_opacity_maps_depth: texture_storage_2d_array<r32float, write>; // TODO: maybe find a more compact format

fn light_layer_from_frustum(frustum_id: u32) -> u32 {
    if frustum_id >= arrayLength(&frustum_table) {
        return 0xFFFFFFFFu;
    }
    if frustum_table[frustum_id].kind != 1u {
        return 0xFFFFFFFFu;
    }
    var layer = 0u;
    for (var i = 0u; i < frustum_id; i = i + 1u) {
        if frustum_table[i].kind == 1u {
            layer = layer + 1u;
        }
    }
    return layer;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rasterize_strands(
    @builtin(workgroup_id) wg: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(num_workgroups) num_wg: vec3u,
) {
    let active_frustum_id = pc.scan_load_base;
    if active_frustum_id >= arrayLength(&frustum_table) {
        return;
    }
    let active_desc = frustum_table[active_frustum_id];
    if active_desc.kind != 1u {
        return;
    }
    var light_layer = light_layer_from_frustum(active_frustum_id);
    if light_layer == 0xFFFFFFFFu {
        return;
    }
    if DEBUG_FORCE_SINGLE_LIGHT {
        light_layer = 0u;
    }
    if light_layer >= lights.n_directional_lights {
        return;
    }
    let light_clip_from_world = lights.directional_lights[light_layer].cascades[0].clip_from_world;
    let active_config = frustum_to_config(active_desc);
    let total_pixels = active_config.screen_width * active_config.screen_height;
    let linear_local = local_id.x;
    let linear_wg = (wg.z * num_wg.y + wg.y) * num_wg.x + wg.x;
    let thread_idx = linear_wg * WORKGROUP_SIZE + linear_local;
    let thread_stride = num_wg.x * num_wg.y * num_wg.z * WORKGROUP_SIZE;

    for (var pixel_idx = thread_idx; pixel_idx < total_pixels; pixel_idx = pixel_idx + thread_stride) {
        let px_u = vec2<u32>(
            pixel_idx % active_config.screen_width,
            pixel_idx / active_config.screen_width,
        );
        let px_i = vec2<i32>(px_u);
        let px_f = vec2<f32>(px_u) + vec2(0.5, 0.5);
        let tile_coord_x = px_u.x / active_config.froxel_size_x;
        let tile_coord_y = px_u.y / active_config.froxel_size_y;

        // Pass 1: determine nearest depth at this pixel (z0).
        var z0 = 1e9;
        for (var dz = 0u; dz < active_config.depth_slices; dz = dz + 1u) {
            let local_froxel_idx = calculate_froxel_index(tile_coord_x, tile_coord_y, dz, active_config);
            if local_froxel_idx >= active_desc.bucket_count {
                continue;
            }
            let bucket_idx = active_desc.bucket_base + local_froxel_idx;
            if bucket_idx >= arrayLength(&froxel_bucket_heads) {
                continue;
            }
            var chunk_idx = atomicLoad(&froxel_bucket_heads[bucket_idx]);
            loop {
                if chunk_idx == 0xFFFFFFFFu {
                    break;
                }
                let chunk_word_base = chunk_idx * CHUNK_WORD_STRIDE;
                if chunk_word_base + 1u >= arrayLength(&chunk_pool_words) {
                    break;
                }
                let next_chunk = chunk_pool_words[chunk_word_base];
                let item_count = min(chunk_pool_words[chunk_word_base + 1u], POOL_CHUNK_SIZE);
                for (var ci = 0u; ci < item_count; ci = ci + 1u) {
                    let payload_idx = chunk_word_base + 2u + ci;
                    if payload_idx >= arrayLength(&chunk_pool_words) {
                        continue;
                    }
                    let work_idx = chunk_pool_words[payload_idx];
                    if work_idx >= arrayLength(&raster_work_queue.items) {
                        continue;
                    }
                    let work_item = raster_work_queue.items[work_idx];
                    if work_item.frustum_id != active_frustum_id {
                        continue;
                    }
                    let inst_id = work_item.inst_id;
                    let asset_id = strand_instances[inst_id].asset_id;
                    let segment_ref = SegmentRef(work_item.strand_id, work_item.seg_id);
                    let V = get_segment_vertices(inst_id, asset_id, segment_ref);
                    let v0 = V[0];
                    let v1 = V[1];
                    if all(v0 == vec4<f32>(0.0)) && all(v1 == vec4<f32>(0.0)) {
                        continue;
                    }
                    let p0_raw = world_to_screen_raw(v0, light_clip_from_world, vec4<f32>(0.0, 0.0, f32(active_config.screen_width), f32(active_config.screen_height)));
                    let p1_raw = world_to_screen_raw(v1, light_clip_from_world, vec4<f32>(0.0, 0.0, f32(active_config.screen_width), f32(active_config.screen_height)));
                    let p0 = vec3<f32>(p0_raw.xy, normalize_depth01(p0_raw.z));
                    let p1 = vec3<f32>(p1_raw.xy, normalize_depth01(p1_raw.z));
                    let t = fragment_position_line_relative(px_f, p0.xy, p1.xy);
                    if t < 0.0 || t > 1.0 {
                        continue;
                    }
                    let p = mix(p0, p1, t);
                    let r = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, clamp(p.z, 0.0, 1.0));
                    let cov = clamp(1.0 - distance(px_f, p.xy) / r, 0.0, 1.0);
                    if cov <= 0.0 {
                        continue;
                    }
                    z0 = min(z0, p.z);
                }
                chunk_idx = next_chunk;
            }
        }

        let z_base = light_layer * DOM_SLICES;
        if z0 == 1e9 {
            textureStore(deep_opacity_maps_depth, px_i, i32(light_layer), vec4<f32>(1.0, 0.0, 0.0, 0.0));
            for (var i = 0u; i < DOM_SLICES; i = i + 1u) {
                textureStore(deep_opacity_maps, vec3<i32>(px_i, i32(z_base + i)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
            }
            continue;
        }

        // Pass 2: accumulate opacity slices relative to z0.
        var alpha: array<f32, DOM_SLICES>;
        for (var i = 0u; i < DOM_SLICES; i = i + 1u) {
            alpha[i] = 0.0;
        }
        let span = max(1e-6, 1.0 - z0);
        let inv_span = 1.0 / span;

        for (var dz = 0u; dz < active_config.depth_slices; dz = dz + 1u) {
            let local_froxel_idx = calculate_froxel_index(tile_coord_x, tile_coord_y, dz, active_config);
            if local_froxel_idx >= active_desc.bucket_count {
                continue;
            }
            let bucket_idx = active_desc.bucket_base + local_froxel_idx;
            if bucket_idx >= arrayLength(&froxel_bucket_heads) {
                continue;
            }
            var chunk_idx = atomicLoad(&froxel_bucket_heads[bucket_idx]);
            loop {
                if chunk_idx == 0xFFFFFFFFu {
                    break;
                }
                let chunk_word_base = chunk_idx * CHUNK_WORD_STRIDE;
                if chunk_word_base + 1u >= arrayLength(&chunk_pool_words) {
                    break;
                }
                let next_chunk = chunk_pool_words[chunk_word_base];
                let item_count = min(chunk_pool_words[chunk_word_base + 1u], POOL_CHUNK_SIZE);
                for (var ci = 0u; ci < item_count; ci = ci + 1u) {
                    let payload_idx = chunk_word_base + 2u + ci;
                    if payload_idx >= arrayLength(&chunk_pool_words) {
                        continue;
                    }
                    let work_idx = chunk_pool_words[payload_idx];
                    if work_idx >= arrayLength(&raster_work_queue.items) {
                        continue;
                    }
                    let work_item = raster_work_queue.items[work_idx];
                    if work_item.frustum_id != active_frustum_id {
                        continue;
                    }
                    let inst_id = work_item.inst_id;
                    let asset_id = strand_instances[inst_id].asset_id;
                    let segment_ref = SegmentRef(work_item.strand_id, work_item.seg_id);
                    let V = get_segment_vertices(inst_id, asset_id, segment_ref);
                    let v0 = V[0];
                    let v1 = V[1];
                    if all(v0 == vec4<f32>(0.0)) && all(v1 == vec4<f32>(0.0)) {
                        continue;
                    }
                    let p0_raw = world_to_screen_raw(v0, light_clip_from_world, vec4<f32>(0.0, 0.0, f32(active_config.screen_width), f32(active_config.screen_height)));
                    let p1_raw = world_to_screen_raw(v1, light_clip_from_world, vec4<f32>(0.0, 0.0, f32(active_config.screen_width), f32(active_config.screen_height)));
                    let p0 = vec3<f32>(p0_raw.xy, normalize_depth01(p0_raw.z));
                    let p1 = vec3<f32>(p1_raw.xy, normalize_depth01(p1_raw.z));
                    let t = fragment_position_line_relative(px_f, p0.xy, p1.xy);
                    if t < 0.0 || t > 1.0 {
                        continue;
                    }
                    let p = mix(p0, p1, t);
                    let r = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, clamp(p.z, 0.0, 1.0));
                    let cov = clamp(1.0 - distance(px_f, p.xy) / r, 0.0, 1.0);
                    if cov <= 0.0 {
                        continue;
                    }

                    let mat = get_segment_material(asset_id, segment_ref);
                    let dzp = max(0.0, p.z - z0) * inv_span;
                    let u = pow(clamp(dzp, 0.0, 1.0), DOM_GAMMA);
                    let tL = u * f32(DOM_SLICES);
                    let si = min(u32(floor(tL)), DOM_SLICES - 1u);
                    let w = fract(tL);
                    let a = cov * mat.absorption_color.w * 0.5;

                    let a0 = alpha[si];
                    alpha[si] = clamp(a0 + (1.0 - a0) * (1.0 - w) * a, 0.0, 1.0);
                    if si + 1u < DOM_SLICES {
                        let a1 = alpha[si + 1u];
                        alpha[si + 1u] = clamp(a1 + (1.0 - a1) * w * a, 0.0, 1.0);
                    }
                }
                chunk_idx = next_chunk;
            }
        }

        textureStore(deep_opacity_maps_depth, px_i, i32(light_layer), vec4<f32>(z0, 0.0, 0.0, 0.0));

        var acc = 0.0;
        for (var i = 0u; i < DOM_SLICES; i = i + 1u) {
            let a = alpha[i];
            acc = acc + (1.0 - acc) * a;
            textureStore(deep_opacity_maps, vec3<i32>(px_i, i32(z_base + i)), vec4<f32>(acc, 0.0, 0.0, 0.0));
        }
    }
}
#endif

#ifdef LINEAR
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rasterize_strands(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(num_workgroups) num_wg: vec3u,
) {
    let active_frustum_id = pc.scan_load_base;
    let linear_local = local_id.x;
    let linear_wg = (workgroup_id.z * num_wg.y + workgroup_id.y) * num_wg.x + workgroup_id.x;
    let thread_idx = linear_wg * WORKGROUP_SIZE + linear_local;
    let thread_stride = num_wg.x * num_wg.y * num_wg.z * WORKGROUP_SIZE;

    if active_frustum_id >= arrayLength(&frustum_table) {
        return;
    }
    let active_desc = frustum_table[active_frustum_id];
    if active_desc.kind != 0u {
        return;
    }
    let active_config = frustum_to_config(active_desc);
    let total_pixels = active_config.screen_width * active_config.screen_height;
    let camera_viewport = vec4<f32>(0.0, 0.0, f32(active_config.screen_width), f32(active_config.screen_height));
    let fine_tiles_x = (active_config.screen_width + active_config.froxel_size_x - 1u) / active_config.froxel_size_x;

    for (var pixel_idx = thread_idx; pixel_idx < total_pixels; pixel_idx = pixel_idx + thread_stride) {
        let pixel_u = vec2<u32>(
            pixel_idx % active_config.screen_width,
            pixel_idx / active_config.screen_width,
        );
        let pixel_coord_int = vec2<i32>(pixel_u);
        let pixel_center = vec2<f32>(pixel_u) + vec2<f32>(0.5, 0.5);
        let tile_coord_x = pixel_u.x / active_config.froxel_size_x;
        let tile_coord_y = pixel_u.y / active_config.froxel_size_y;
        let screen_tile_id = tile_coord_y * fine_tiles_x + tile_coord_x;
        let coarse_x = tile_coord_x / COARSE_FINE_TILE_EXTENT;
        let coarse_y = tile_coord_y / COARSE_FINE_TILE_EXTENT;
        if coarse_x >= active_desc.coarse_tiles_x || coarse_y >= active_desc.coarse_tiles_y {
            continue;
        }
        let coarse_tile_idx = active_desc.coarse_depth_tile_base + coarse_y * active_desc.coarse_tiles_x + coarse_x;
        if coarse_tile_idx >= arrayLength(&coarse_tile_work_counts) || coarse_tile_idx >= arrayLength(&coarse_tile_work_offsets) {
            continue;
        }
        let work_base = coarse_tile_work_offsets[coarse_tile_idx];
        let work_count = coarse_tile_work_counts[coarse_tile_idx];
        let work_end = min(work_base + work_count, arrayLength(&raster_work_queue.items));

        var final_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var g_min_depth: f32 = 0.0;

        for (var work_idx = work_base; work_idx < work_end; work_idx = work_idx + 1u) {
            let work_item = raster_work_queue.items[work_idx];
            if work_item.frustum_id != active_frustum_id || work_item.screen_tile_id != screen_tile_id {
                continue;
            }

            var froxel_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            let ref_end = min(work_item.seg_ref_base + work_item.seg_ref_count, arrayLength(&fine_seg_refs.refs));
            for (var ref_idx = work_item.seg_ref_base; ref_idx < ref_end; ref_idx = ref_idx + 1u) {
                    let seg_ref = fine_seg_refs.refs[ref_idx];
                    let inst_id = seg_ref.inst_id;
                    if inst_id >= arrayLength(&strand_instances) {
                        continue;
                    }
                    let asset_id = strand_instances[inst_id].asset_id;
                    let segment_ref = SegmentRef(seg_ref.strand_id, seg_ref.seg_id);

                    let strand_meta = get_segment_meta(asset_id, segment_ref);
                    if strand_meta.count < 2u { continue; }

                    let V = get_segment_vertices(inst_id, asset_id, segment_ref);
                    let v0_world = V[0];
                    let v1_world = V[1];

                    let p0_screen = world_to_screen_raw(v0_world, view.unjittered_clip_from_world, camera_viewport);
                    let p1_screen = world_to_screen_raw(v1_world, view.unjittered_clip_from_world, camera_viewport);

                    if p0_screen.x < 0.0 && p1_screen.x < 0.0 { continue; }

                    let t = fragment_position_line_relative(pixel_center, p0_screen.xy, p1_screen.xy);
                    if t < 0.0 || t > 1.0 { continue; }
                    let p_frag = mix(p0_screen, p1_screen, t);
                    let dist = distance(pixel_center, p_frag.xy);

                    let z_cam = to_log_depth(p_frag.z);
                    let r = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, z_cam);
                    let coverage = clamp(1.0 - dist / r, 0.0, 1.0);

                    if coverage > 0.0 {
                        if segment_ref.segment_start_idx < strand_meta.offset {
                            continue;
                        }
                        let seg_local = segment_ref.segment_start_idx - strand_meta.offset;
                        if seg_local >= (strand_meta.count - 1u) {
                            continue;
                        }
                        let layer = inst_id;
                        if layer >= textureNumLayers(shading_buffer) {
                            continue;
                        }
                        let dims = textureDimensions(shading_buffer, 0);
                        if seg_local >= dims.x || segment_ref.strand_idx >= dims.y {
                            continue;
                        }
                        let shaded = textureLoad(
                            shading_buffer,
                            vec2<i32>(i32(seg_local), i32(segment_ref.strand_idx)),
                            i32(layer),
                            0,
                        );
                        let hair_fragment = vec4<f32>(shaded.rgb, shaded.a * coverage);
                        froxel_color = blend_over(froxel_color, hair_fragment);
                        g_min_depth = max(g_min_depth, p_frag.z);
                    }
                    if froxel_color.a > 0.9995 {
                        break;
                    }
            }

            final_color = blend_over(final_color, froxel_color);
            if final_color.a > 0.999 {
                break;
            }
        }

        textureStore(render_target, pixel_coord_int, final_color);
        textureStore(depth_target, pixel_coord_int, vec4<f32>(g_min_depth, 0.0, 0.0, 0.0));
    }
}
#endif

// #endif // SHADOWS vs camera
