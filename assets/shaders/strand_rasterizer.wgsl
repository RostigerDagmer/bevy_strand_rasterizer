#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::mesh_view_types as types
// #import "shaders/spline.wgsl"::{ intersect_catmull_rom_spline_3d, closest_point };
#import "shaders/common.wgsl"::{ DOM_GAMMA, is_valid_ptr, find_clip_bounds, world_to_screen, world_to_screen_aabbnorm, world_to_screen_raw, screen_to_world_raw, screen_to_world, calculate_froxel_index, normalize_depth01, to_log_depth, to_reverse_log_depth, }
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
    RasterTileRun,
    FineSegRef,
}
#import bevy_vsms::virtual_surface_types::{
    VirtualPageTableEntry,
    VirtualPageTableMetaRow,
    vsms_virtual_page_table_address,
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

struct RasterTileRunQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    items: array<RasterTileRun>,
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
@group(#{RASTER_GROUP}) @binding(#{RASTER_TILE_RUN_QUEUE}) var<storage, read> raster_tile_run_queue: RasterTileRunQueue;
@group(#{RASTER_GROUP}) @binding(#{FINE_SEG_REFS}) var<storage, read> fine_seg_refs: FineSegRefBuffer;
@group(#{RASTER_GROUP}) @binding(#{STRAND_INSTANCES}) var<storage, read> strand_instances: array<StrandInstance>;
#ifdef LINEAR
@group(#{RASTER_GROUP}) @binding(#{SHADOW_DOM_SURFACE_IDS}) var<storage, read> shadow_dom_surface_ids: array<vec2<u32>>;
#endif

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

fn perspective_correct_line_t(t_screen: f32, w0: f32, w1: f32) -> f32 {
    let t = clamp(t_screen, 0.0, 1.0);
    let inv_w0 = 1.0 / max(abs(w0), 1e-6);
    let inv_w1 = 1.0 / max(abs(w1), 1e-6);
    let denom = (1.0 - t) * inv_w0 + t * inv_w1;
    if denom <= 1e-6 {
        return t;
    }
    return (t * inv_w1) / denom;
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

fn get_segment_material(asset_id: u32, material_id: u32, segment_ref: SegmentRef) -> StrandMaterial {
    if arrayLength(&t_strand_metadata) == 0u || arrayLength(&t_materials) == 0u || asset_id >= arrayLength(&t_strand_metadata) || material_id >= arrayLength(&t_materials) {
        return StrandMaterial(vec4<f32>(1.0), vec4<f32>(1.0), 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0u, 0u);
    }
    let meta_ptr = t_strand_metadata[asset_id];
    let material_ptr = t_materials[material_id];
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
@group(#{VSMS_OPACITY_WRITE_GROUP}) @binding(#{VSMS_STORAGE_BINDING}) var deep_opacity_maps: binding_array<texture_storage_3d<r32float, write> >;
@group(#{VSMS_DEPTH_WRITE_GROUP}) @binding(#{VSMS_STORAGE_BINDING}) var deep_opacity_maps_depth: binding_array<texture_storage_2d_array<r32float, write> >;
@group(#{VSMS_OPACITY_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_META_BINDING}) var<storage, read> opacity_virtual_meta: array<VirtualPageTableMetaRow>;
@group(#{VSMS_OPACITY_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_PAGE_TABLE_BINDING}) var<storage, read> opacity_virtual_pages: array<VirtualPageTableEntry>;
@group(#{VSMS_DEPTH_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_META_BINDING}) var<storage, read> depth_virtual_meta: array<VirtualPageTableMetaRow>;
@group(#{VSMS_DEPTH_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_PAGE_TABLE_BINDING}) var<storage, read> depth_virtual_pages: array<VirtualPageTableEntry>;

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
    let opacity_surface_id = pc.scan_save_base;
    let depth_surface_id = pc.workgroup_offset;
    if opacity_surface_id >= arrayLength(&opacity_virtual_meta) || depth_surface_id >= arrayLength(&depth_virtual_meta) {
        return;
    }
    let opacity_meta = opacity_virtual_meta[opacity_surface_id];
    let depth_meta = depth_virtual_meta[depth_surface_id];
    if opacity_meta.entry_count == 0u || depth_meta.entry_count == 0u {
        return;
    }
    let run_idx = (wg.z * num_wg.y + wg.y) * num_wg.x + wg.x;
    if run_idx >= atomicLoad(&raster_tile_run_queue.tail) || run_idx >= arrayLength(&raster_tile_run_queue.items) {
        return;
    }
    let run = raster_tile_run_queue.items[run_idx];
    if run.frustum_id != active_frustum_id || run.work_count == 0u {
        return;
    }
    let light_clip_from_world = lights.directional_lights[light_layer].cascades[0].clip_from_world;
    let active_config = frustum_to_config(active_desc);
    let light_viewport = vec4<f32>(0.0, 0.0, f32(active_config.screen_width), f32(active_config.screen_height));
    let fine_tiles_x = (active_config.screen_width + active_config.froxel_size_x - 1u) / active_config.froxel_size_x;
    let tile_x = run.screen_tile_id % fine_tiles_x;
    let tile_y = run.screen_tile_id / fine_tiles_x;
    let tile_pixel_count = active_config.froxel_size_x * active_config.froxel_size_y;
    let work_base = run.work_base;
    let work_end = min(work_base + run.work_count, arrayLength(&raster_work_queue.items));

    // TODO: use workgroup local/subgroup cooperation across pixels in the tile.
    // This first migration keeps one lane responsible for one or more pixels.
    for (var tile_pixel_idx = local_id.x; tile_pixel_idx < tile_pixel_count; tile_pixel_idx = tile_pixel_idx + WORKGROUP_SIZE) {
        let px_u = vec2<u32>(
            tile_x * active_config.froxel_size_x + (tile_pixel_idx % active_config.froxel_size_x),
            tile_y * active_config.froxel_size_y + (tile_pixel_idx / active_config.froxel_size_x),
        );
        if px_u.x >= active_config.screen_width || px_u.y >= active_config.screen_height {
            continue;
        }
        let px_f = vec2<f32>(px_u) + vec2(0.5, 0.5);
        let opacity_page_size = vec3<u32>(
            max(opacity_meta.page_size_x, 1u),
            max(opacity_meta.page_size_y, 1u),
            max(opacity_meta.page_size_z, 1u),
        );
        let depth_page_size = vec2<u32>(
            max(depth_meta.page_size_x, 1u),
            max(depth_meta.page_size_y, 1u),
        );
        let opacity_tile = vec3<u32>(px_u.x / opacity_page_size.x, px_u.y / opacity_page_size.y, 0u);
        let depth_tile = vec3<u32>(px_u.x / depth_page_size.x, px_u.y / depth_page_size.y, 0u);
        let opacity_addr = vsms_virtual_page_table_address(opacity_meta, opacity_tile, 0u, 0u);
        let depth_addr = vsms_virtual_page_table_address(depth_meta, depth_tile, 0u, 0u);
        if opacity_addr.valid == 0u || depth_addr.valid == 0u {
            continue;
        }
        if opacity_addr.entry_index >= arrayLength(&opacity_virtual_pages) || depth_addr.entry_index >= arrayLength(&depth_virtual_pages) {
            continue;
        }
        let opacity_entry = opacity_virtual_pages[opacity_addr.entry_index];
        let depth_entry = depth_virtual_pages[depth_addr.entry_index];
        if opacity_entry.valid == 0u || depth_entry.valid == 0u {
            continue;
        }
        let opacity_px = vec2<i32>(
            i32(px_u.x % opacity_page_size.x),
            i32(px_u.y % opacity_page_size.y),
        );
        let depth_px = vec2<i32>(
            i32(px_u.x % depth_page_size.x),
            i32(px_u.y % depth_page_size.y),
        );

        // Pass 1: determine nearest depth at this pixel (z0). Bevy directional
        // light projections use reverse-Z, where larger values are closer.
        var z0 = -1.0;
        for (var work_idx = work_base; work_idx < work_end; work_idx = work_idx + 1u) {
            let work_item = raster_work_queue.items[work_idx];
            if work_item.frustum_id != active_frustum_id || work_item.screen_tile_id != run.screen_tile_id {
                continue;
            }
            let ref_end = min(work_item.seg_ref_base + work_item.seg_ref_count, arrayLength(&fine_seg_refs.refs));
            for (var ref_idx = work_item.seg_ref_base; ref_idx < ref_end; ref_idx = ref_idx + 1u) {
                    let seg_ref = fine_seg_refs.refs[ref_idx];
                    let inst_id = seg_ref.inst_id;
                    if inst_id >= arrayLength(&strand_instances) {
                        continue;
                    }
                    let asset_id = strand_instances[inst_id].asset_id;
                    let segment_ref = SegmentRef(seg_ref.strand_id, seg_ref.seg_id);
                    let V = get_segment_vertices(inst_id, asset_id, segment_ref);
                    let v0 = V[0];
                    let v1 = V[1];
                    if all(v0 == vec4<f32>(0.0)) && all(v1 == vec4<f32>(0.0)) {
                        continue;
                    }
                    let p0_raw = world_to_screen_raw(v0, light_clip_from_world, light_viewport);
                    let p1_raw = world_to_screen_raw(v1, light_clip_from_world, light_viewport);
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
                    z0 = max(z0, p.z);
            }
        }

        if z0 < 0.0 {
            textureStore(deep_opacity_maps_depth[i32(depth_entry.physical_index)], depth_px, 0, vec4<f32>(0.0, 0.0, 0.0, 0.0));
            for (var i = 0u; i < DOM_SLICES; i = i + 1u) {
                if i < opacity_page_size.z {
                    textureStore(deep_opacity_maps[i32(opacity_entry.physical_index)], vec3<i32>(opacity_px, i32(i)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
                }
            }
            continue;
        }

        // Pass 2: accumulate opacity slices relative to z0.
        var alpha: array<f32, DOM_SLICES>;
        for (var i = 0u; i < DOM_SLICES; i = i + 1u) {
            alpha[i] = 0.0;
        }
        let span = max(1e-6, z0);
        let inv_span = 1.0 / span;

        for (var work_idx = work_base; work_idx < work_end; work_idx = work_idx + 1u) {
            let work_item = raster_work_queue.items[work_idx];
            if work_item.frustum_id != active_frustum_id || work_item.screen_tile_id != run.screen_tile_id {
                continue;
            }
            let ref_end = min(work_item.seg_ref_base + work_item.seg_ref_count, arrayLength(&fine_seg_refs.refs));
            for (var ref_idx = work_item.seg_ref_base; ref_idx < ref_end; ref_idx = ref_idx + 1u) {
                    let seg_ref = fine_seg_refs.refs[ref_idx];
                    let inst_id = seg_ref.inst_id;
                    if inst_id >= arrayLength(&strand_instances) {
                        continue;
                    }
                    let asset_id = strand_instances[inst_id].asset_id;
                    let segment_ref = SegmentRef(seg_ref.strand_id, seg_ref.seg_id);
                    let V = get_segment_vertices(inst_id, asset_id, segment_ref);
                    let v0 = V[0];
                    let v1 = V[1];
                    if all(v0 == vec4<f32>(0.0)) && all(v1 == vec4<f32>(0.0)) {
                        continue;
                    }
                    let p0_raw = world_to_screen_raw(v0, light_clip_from_world, light_viewport);
                    let p1_raw = world_to_screen_raw(v1, light_clip_from_world, light_viewport);
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

                    let mat = get_segment_material(asset_id, strand_instances[inst_id].material_id, segment_ref);
                    let dzp = max(0.0, z0 - p.z) * inv_span;
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
        }

        textureStore(deep_opacity_maps_depth[i32(depth_entry.physical_index)], depth_px, 0, vec4<f32>(z0, 0.0, 0.0, 0.0));

        var acc = 0.0;
        for (var i = 0u; i < DOM_SLICES; i = i + 1u) {
            let a = alpha[i];
            acc = acc + (1.0 - acc) * a;
            if i < opacity_page_size.z {
                textureStore(deep_opacity_maps[i32(opacity_entry.physical_index)], vec3<i32>(opacity_px, i32(i)), vec4<f32>(acc, 0.0, 0.0, 0.0));
            }
        }
    }
}
#endif

#ifdef LINEAR
const DOM_SLICES: u32 = #{NUM_DOM_SLICES};
const INVALID_PTR: u32 = 0xFFFFFFFFu;
const VSMS_OPACITY_POOL_TEXTURE_COUNT: u32 = #{VSMS_OPACITY_POOL_TEXTURE_COUNT};
const VSMS_DEPTH_POOL_TEXTURE_COUNT: u32 = #{VSMS_DEPTH_POOL_TEXTURE_COUNT};

@group(#{VSMS_OPACITY_WRITE_GROUP}) @binding(#{VSMS_POOL_TEXTURE_BINDING}) var shadow_opacity_maps: binding_array<texture_3d<f32> >;
@group(#{VSMS_OPACITY_WRITE_GROUP}) @binding(#{VSMS_POOL_SAMPLER_BINDING}) var shadow_opacity_sampler: sampler;
@group(#{VSMS_DEPTH_WRITE_GROUP}) @binding(#{VSMS_POOL_TEXTURE_BINDING}) var shadow_depth_maps: binding_array<texture_2d_array<f32> >;
@group(#{VSMS_DEPTH_WRITE_GROUP}) @binding(#{VSMS_POOL_SAMPLER_BINDING}) var shadow_depth_sampler: sampler;

@group(#{VSMS_OPACITY_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_META_BINDING}) var<storage, read> opacity_virtual_meta: array<VirtualPageTableMetaRow>;
@group(#{VSMS_OPACITY_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_PAGE_TABLE_BINDING}) var<storage, read> opacity_virtual_pages: array<VirtualPageTableEntry>;
@group(#{VSMS_DEPTH_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_META_BINDING}) var<storage, read> depth_virtual_meta: array<VirtualPageTableMetaRow>;
@group(#{VSMS_DEPTH_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_PAGE_TABLE_BINDING}) var<storage, read> depth_virtual_pages: array<VirtualPageTableEntry>;

fn light_frustum_from_layer(light_layer: u32) -> u32 {
    var layer = 0u;
    for (var frustum_id = 0u; frustum_id < arrayLength(&frustum_table); frustum_id = frustum_id + 1u) {
        if frustum_table[frustum_id].kind != 1u {
            continue;
        }
        if layer == light_layer {
            return frustum_id;
        }
        layer = layer + 1u;
    }
    return INVALID_PTR;
}

fn sample_shadow_dom_visibility(p_world: vec3<f32>, light_layer: u32) -> f32 {
    let frustum_id = light_frustum_from_layer(light_layer);
    if frustum_id == INVALID_PTR || frustum_id >= arrayLength(&frustum_table) || frustum_id >= arrayLength(&shadow_dom_surface_ids) {
        return 1.0;
    }
    let surface_ids = shadow_dom_surface_ids[frustum_id];
    let opacity_surface_id = surface_ids.x;
    let depth_surface_id = surface_ids.y;
    if opacity_surface_id == INVALID_PTR || depth_surface_id == INVALID_PTR {
        return 1.0;
    }
    if opacity_surface_id >= arrayLength(&opacity_virtual_meta) || depth_surface_id >= arrayLength(&depth_virtual_meta) {
        return 1.0;
    }
    let opacity_row = opacity_virtual_meta[opacity_surface_id];
    let depth_row = depth_virtual_meta[depth_surface_id];
    if opacity_row.entry_count == 0u || depth_row.entry_count == 0u {
        return 1.0;
    }

    let desc = frustum_table[frustum_id];
    let viewport = vec4<f32>(0.0, 0.0, f32(desc.screen_width), f32(desc.screen_height));
    let light_clip_from_world = lights.directional_lights[light_layer].cascades[0].clip_from_world;
    let raw = world_to_screen_raw(vec4<f32>(p_world, 1.0), light_clip_from_world, viewport);
    if raw.x < 0.0 || raw.y < 0.0 || raw.x >= viewport.z || raw.y >= viewport.w || raw.z < 0.0 || raw.z > 1.0 {
        return 1.0;
    }

    let px = vec2<u32>(u32(raw.x), u32(raw.y));
    let opacity_page_size = vec3<u32>(
        max(opacity_row.page_size_x, 1u),
        max(opacity_row.page_size_y, 1u),
        max(opacity_row.page_size_z, 1u),
    );
    let depth_page_size = vec2<u32>(
        max(depth_row.page_size_x, 1u),
        max(depth_row.page_size_y, 1u),
    );
    let opacity_tile = vec3<u32>(px.x / opacity_page_size.x, px.y / opacity_page_size.y, 0u);
    let depth_tile = vec3<u32>(px.x / depth_page_size.x, px.y / depth_page_size.y, 0u);
    let opacity_addr = vsms_virtual_page_table_address(opacity_row, opacity_tile, 0u, 0u);
    let depth_addr = vsms_virtual_page_table_address(depth_row, depth_tile, 0u, 0u);
    if opacity_addr.valid == 0u || depth_addr.valid == 0u {
        return 1.0;
    }
    if opacity_addr.entry_index >= arrayLength(&opacity_virtual_pages) || depth_addr.entry_index >= arrayLength(&depth_virtual_pages) {
        return 1.0;
    }

    let opacity_entry = opacity_virtual_pages[opacity_addr.entry_index];
    let depth_entry = depth_virtual_pages[depth_addr.entry_index];
    if opacity_entry.valid == 0u || depth_entry.valid == 0u {
        return 1.0;
    }
    if opacity_entry.physical_index >= VSMS_OPACITY_POOL_TEXTURE_COUNT || depth_entry.physical_index >= VSMS_DEPTH_POOL_TEXTURE_COUNT {
        return 1.0;
    }

    let opacity_px = vec2<i32>(i32(px.x % opacity_page_size.x), i32(px.y % opacity_page_size.y));
    let depth_px = vec2<i32>(i32(px.x % depth_page_size.x), i32(px.y % depth_page_size.y));
    let z0 = textureLoad(shadow_depth_maps[i32(depth_entry.physical_index)], depth_px, 0, 0).x;
    if z0 <= 0.0 {
        return 1.0;
    }

    let z = normalize_depth01(raw.z);
    if z > z0 - 1e-3 {
        return 1.0;
    }
    let u = pow(clamp(max(0.0, z0 - z) / max(z0, 1e-6), 0.0, 1.0), DOM_GAMMA);
    let slice_f = u * f32(DOM_SLICES);
    let slice = min(u32(floor(slice_f)), min(DOM_SLICES, opacity_page_size.z) - 1u);
    let opacity = textureLoad(
        shadow_opacity_maps[i32(opacity_entry.physical_index)],
        vec3<i32>(opacity_px, i32(slice)),
        0,
    ).x;
    return clamp(1.0 - opacity, 0.08, 1.0);
}

fn sample_shadow_visibility(p_world: vec3<f32>) -> f32 {
    let light_count = min(lights.n_directional_lights, 4u);
    if light_count == 0u {
        return 1.0;
    }
    var visibility = 0.0;
    for (var i = 0u; i < light_count; i = i + 1u) {
        visibility = visibility + sample_shadow_dom_visibility(p_world, i);
    }
    return visibility / f32(light_count);
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
    if active_frustum_id >= arrayLength(&frustum_table) {
        return;
    }
    let active_desc = frustum_table[active_frustum_id];
    if active_desc.kind != 0u {
        return;
    }
    let run_idx = (workgroup_id.z * num_wg.y + workgroup_id.y) * num_wg.x + workgroup_id.x;
    if run_idx >= atomicLoad(&raster_tile_run_queue.tail) || run_idx >= arrayLength(&raster_tile_run_queue.items) {
        return;
    }
    let run = raster_tile_run_queue.items[run_idx];
    if run.frustum_id != active_frustum_id || run.work_count == 0u {
        return;
    }
    let active_config = frustum_to_config(active_desc);
    let camera_viewport = vec4<f32>(0.0, 0.0, f32(active_config.screen_width), f32(active_config.screen_height));
    let fine_tiles_x = (active_config.screen_width + active_config.froxel_size_x - 1u) / active_config.froxel_size_x;
    let tile_x = run.screen_tile_id % fine_tiles_x;
    let tile_y = run.screen_tile_id / fine_tiles_x;
    let tile_pixel_count = active_config.froxel_size_x * active_config.froxel_size_y;
    let work_base = run.work_base;
    let work_end = min(work_base + run.work_count, arrayLength(&raster_work_queue.items));

    // TODO: use tile-local/shared reductions so a subgroup can cooperate on
    // heavy pixels instead of assigning independent pixels to lanes only.
    for (var tile_pixel_idx = local_id.x; tile_pixel_idx < tile_pixel_count; tile_pixel_idx = tile_pixel_idx + WORKGROUP_SIZE) {
        let pixel_u = vec2<u32>(
            tile_x * active_config.froxel_size_x + (tile_pixel_idx % active_config.froxel_size_x),
            tile_y * active_config.froxel_size_y + (tile_pixel_idx / active_config.froxel_size_x),
        );
        if pixel_u.x >= active_config.screen_width || pixel_u.y >= active_config.screen_height {
            continue;
        }
        let pixel_coord_int = vec2<i32>(pixel_u);
        let pixel_center = vec2<f32>(pixel_u) + vec2<f32>(0.5, 0.5);

        var final_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var g_min_depth: f32 = 0.0;

        for (var work_idx = work_base; work_idx < work_end; work_idx = work_idx + 1u) {
            let work_item = raster_work_queue.items[work_idx];
            if work_item.frustum_id != active_frustum_id || work_item.screen_tile_id != run.screen_tile_id {
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
                    let clip0 = view.unjittered_clip_from_world * v0_world;
                    let clip1 = view.unjittered_clip_from_world * v1_world;
                    let t_world = perspective_correct_line_t(t, clip0.w, clip1.w);
                    let p_world = mix(v0_world.xyz, v1_world.xyz, t_world);
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
                        let shadow_visibility = sample_shadow_visibility(p_world);
                        let hair_fragment = vec4<f32>(shaded.rgb * shadow_visibility, shaded.a * coverage);
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
