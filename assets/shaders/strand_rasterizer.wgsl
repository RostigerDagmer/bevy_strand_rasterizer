#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::mesh_view_types as types
// #import "shaders/spline.wgsl"::{ intersect_catmull_rom_spline_3d, closest_point };
#import "shaders/common.wgsl"::{ DOM_GAMMA, find_clip_bounds, world_to_screen, world_to_screen_aabbnorm, world_to_screen_raw, screen_to_world_raw, screen_to_world, calculate_froxel_index }
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
    StrandMeta,
    StrandMaterial,
    Vertices,
    PushConstants,
}

const MAX_TEXTURE_EXT: u32 = #MAX_TEXTURE_EXTENT;
const SIZEOF_METADATA: u32 = #SIZEOF_METADATA;
const SIZEOF_MATERIAL: u32 = #SIZEOF_MATERIAL;
const SIZEOF_GEO: u32 = #SIZEOF_GEO;
const LIGHT_INDEX: u32 = 0u; // Example constant for light index TODO: compute prepass -> indirect dispatch -> light index from uniforms
const MIN_HAIR_RADIUS_PIXELS : f32 = 0.5; // Example: Thickness in pixels
const MAX_HAIR_RADIUS_PIXELS : f32 = 2.0; // Example: Thickness in pixels
var<push_constant> pc: PushConstants;

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

@group(#{RASTER_GROUP}) @binding(#{TILE_OFFSETS_BUFFER}) var<storage, read> tile_offsets_buffer: array<u32>;
@group(#{RASTER_GROUP}) @binding(#{TILE_COUNTS_BUFFER}) var<storage, read> tile_counts_buffer: array<atomic<u32>>;
@group(#{RASTER_GROUP}) @binding(#{FROXEL_TILE_BUFFER}) var<storage, read> packed_segments_buffer: array<SegmentRef>; // Read only
@group(#{RASTER_GROUP}) @binding(#{OUTPUT_TEXTURE}) var render_target: texture_storage_2d<rgba8unorm, write>;
@group(#{RASTER_GROUP}) @binding(#{OUTPUT_DEPTH}) var depth_target: texture_storage_2d<r32float, write>;
@group(#{RASTER_GROUP}) @binding(#{FROXEL_CONFIG}) var<uniform> config: FroxelConfig;
@group(#{RASTER_GROUP}) @binding(#{VIEW_UNIFORM}) var<uniform> view: View;
@group(#{RASTER_GROUP}) @binding(#{LIGHT_UNIFORM}) var<uniform> lights: types::Lights;
// Disabled for correctness-only raster pass:
// @group(#{RASTER_GROUP}) @binding(#{SHADING_BUFFER}) var shading_buffer: texture_storage_2d<rgba8unorm, read>;


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

#ifndef SHADOWS
fn is_valid_ptr(_ptr: DevicePtr) -> bool {
    return _ptr.slab != 0xFFFFFFFFu;
}

fn get_geo0() -> StrandGeo {
    if arrayLength(&t_geos) == 0u {
        return StrandGeo(0u, 0u, 0u, 0u, Aabb(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0));
    }
    let geo_ptr = t_geos[0u];
    if !is_valid_ptr(geo_ptr) {
        return StrandGeo(0u, 0u, 0u, 0u, Aabb(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0));
    }
    let geo_base = geo_ptr.offset / SIZEOF_GEO;
    return geos[geo_ptr.slab].gs[geo_base];
}

fn get_segment_material(segment_ref: SegmentRef) -> StrandMaterial {
    if arrayLength(&t_strand_metadata) == 0u || arrayLength(&t_materials) == 0u {
        return StrandMaterial(vec4<f32>(1.0), vec4<f32>(1.0), 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0u, 0u);
    }
    let meta_ptr = t_strand_metadata[0u];
    let material_ptr = t_materials[0u];
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

fn get_segment_meta(segment_ref: SegmentRef) -> StrandMeta {
    if arrayLength(&t_strand_metadata) == 0u {
        return StrandMeta(0u, 0u, 0u, 0u);
    }
    let meta_ptr = t_strand_metadata[0u];
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

fn get_segment_vertices(segment_ref: SegmentRef) -> mat2x4<f32> {
#ifdef SHADOWS
    // Get strand metadata
    let strand_idx = segment_ref.strand_idx;
    let strand_meta = strand_metadata[strand_idx];

    let v0_idx = segment_ref.segment_start_idx;
    let v1_idx = segment_ref.segment_start_idx + 1u;
    if (v1_idx - strand_meta.offset) >= strand_meta.count - 1 { return mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0)); } // Safety check

    // Get segment vertex indices within the strand
    let v0_strand_idx = indices[v0_idx];
    let v1_strand_idx = indices[v1_idx];

    // Get world-space vertex positions
    return mat2x4<f32>(vertices[v0_strand_idx], vertices[v1_strand_idx]);
#else
    if arrayLength(&t_strand_metadata) == 0u || arrayLength(&t_indices) == 0u || arrayLength(&t_vertices) == 0u {
        return mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    let meta_ptr = t_strand_metadata[0u];
    let index_ptr = t_indices[0u];
    let vertex_ptr = t_vertices[0u];
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

    let v0_idx = segment_ref.segment_start_idx;
    let v1_idx = segment_ref.segment_start_idx + 1u;
    let index_count = index_ptr.size / 4u;
    if v1_idx >= index_count || (v1_idx - strand_meta.offset) >= strand_meta.count - 1u {
        return mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0));
    }

    let v0_strand_idx = indices[index_ptr.slab].is[v0_idx];
    let v1_strand_idx = indices[index_ptr.slab].is[v1_idx];
    let vertex_count = vertex_ptr.size / 12u;
    if v0_strand_idx >= vertex_count || v1_strand_idx >= vertex_count {
        return mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0));
    }

    return mat2x4<f32>(
        vec4<f32>(vertices[vertex_ptr.slab].vs[v0_strand_idx], 1.0),
        vec4<f32>(vertices[vertex_ptr.slab].vs[v1_strand_idx], 1.0),
    );
#endif
}

// #define DEBUG

#ifdef SHADOWS
const DOM_SLICES: u32 = #{NUM_DOM_SLICES};
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_O}) var deep_opacity_maps: texture_storage_3d<r32float, write>; // TODO: maybe find a more compact format
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_D}) var deep_opacity_maps_depth: texture_storage_2d<r32float, write>; // TODO: maybe find a more compact format
@group(0) @binding(#{CLUSTER_INDICES}) var<storage> clusterable_object_index_lists: types::ClusterLightIndexLists;
@group(0) @binding(#{CLUSTERABLE_OBJECTS}) var<storage> clusterable_objects: types::ClusterableObjects;
@group(0) @binding(#{CLUSTER_OFFSETS_AND_COUNTS}) var<storage> cluster_offsets_and_counts: types::ClusterOffsetsAndCounts;
@group(0) @binding(#{POINT_LIGHT_DEPTH_TEXTURE_SAMPLER}) var point_shadow_textures_linear_sampler: sampler;
@group(0) @binding(#{POINT_LIGHT_DEPTH_TEXTURE}) var point_shadow_textures: texture_depth_cube_array;
@group(0) @binding(#{DIRECTIONAL_LIGHT_DEPTH_TEXTURE_SAMPLER}) var directional_shadow_textures_linear_sampler: sampler;
@group(0) @binding(#{DIRECTIONAL_LIGHT_DEPTH_TEXTURE}) var directional_shadow_textures: texture_depth_2d_array;

@compute @workgroup_size(8,8,1)
fn rasterize_strands(@builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id)      wg: vec3u) {
    let px_i = vec2<i32>(gid.xy);
    let px_f = vec2<f32>(gid.xy) + vec2(0.5, 0.5);

    // --- constants / inputs
    let light = lights.directional_lights[LIGHT_INDEX];
    let cascade_index = 0u;
    let cas = light.cascades[cascade_index];
    let M = cas.clip_from_world;
    let geo = geos[0];
    let cb = find_clip_bounds(M, geo.aabb.min, geo.aabb.max);
    let depth_slices = config.depth_slices;

    // froxel tile coords
    let tx = wg.x; let ty = wg.y;

    // -------- pass 0: early reject based on tile counts
    var any_in_tile = false;
    for (var dz: u32 = 0u; dz < depth_slices; dz++) {
        let fc = tile_counts_buffer[calculate_froxel_index(tx, ty, dz, config)];
        any_in_tile = any_in_tile || (fc > 0u);
    }
    if !any_in_tile { return; }

    // -------- pass 1: find per-pixel nearest depth z0 (in your chosen convention)

    let clip_dim = (cb[1] - cb[0]);
    let tm_x = (px_f.x / f32(config.screen_width)) * clip_dim.x + cb[0].x;
    let tm_y = (px_f.y / f32(config.screen_height)) * clip_dim.y + cb[0].y;

    let light_z = 1e9;

    var z0 = light_z; // if near=0..1 use +INF init, otherwise adjust
    for (var dz: u32 = 0u; dz < depth_slices; dz++) {
        let fidx = calculate_froxel_index(tx, ty, dz, config);
        if fidx >= arrayLength(&tile_offsets_buffer) { continue; }

        let start = tile_offsets_buffer[fidx];
        let count = tile_counts_buffer[fidx];

        for (var s: u32 = 0u; s < count; s++) {
            let idx = start + s;
            if idx >= arrayLength(&packed_segments_buffer) { continue; }

            let seg = packed_segments_buffer[idx];
            if seg.strand_idx >= arrayLength(&strand_metadata) { continue; }

            let V = get_segment_vertices(seg);
            let p0 = world_to_screen_aabbnorm(V[0], M, f32(config.screen_width), f32(config.screen_height), cb);
            let p1 = world_to_screen_aabbnorm(V[1], M, f32(config.screen_width), f32(config.screen_height), cb);

            let t = fragment_position_line_relative(px_f, p0.xy, p1.xy);
            if t < 0.0 || t > 1.0 { continue; }
            let p = mix(p0, p1, t);

            // optional radius & coverage test to skip non-overlaps early
            let r = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, clamp(p.z, 0.0, 1.0));
            if clamp(1.0 - distance(px_f, p.xy) / r, 0.0, 1.0) <= 0.0 { continue; }

            // assume near=0, far=1; if reversed, invert consistently everywhere
            z0 = min(z0, p.z);
        }
    }
    if z0 == 1e9 {
        // nothing touched this pixel: optionally clear outputs here
        textureStore(deep_opacity_maps_depth, px_i, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        for (var i: u32 = 0u; i < DOM_SLICES; i++) {
            textureStore(deep_opacity_maps, vec3<i32>(px_i, i32(i)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        }
        return;
    }

    // -------- per-pixel slice params
    let L = DOM_SLICES;
    let span = max(1e-6, 1.0 - z0);
    let invSpan = 1.0 / span;

    // -------- zero the slice bin accumulator
    var alpha: array<f32, DOM_SLICES>;
    for (var i: u32 = 0u; i < L; i++) { alpha[i] = 0.0; }

    // -------- pass 2: bin opacities relative to z0
    var z_nearest = z0;
    for (var dz: u32 = 0u; dz < depth_slices; dz++) {
        let fidx = calculate_froxel_index(tx, ty, dz, config);
        if fidx >= arrayLength(&tile_offsets_buffer) { continue; }

        let start = tile_offsets_buffer[fidx];
        let count = tile_counts_buffer[fidx];

        for (var s: u32 = 0u; s < count; s++) {
            let idx = start + s;
            if idx >= arrayLength(&packed_segments_buffer) { continue; }

            let seg = packed_segments_buffer[idx];
            if seg.strand_idx >= arrayLength(&strand_metadata) { continue; }

            let V = get_segment_vertices(seg);
            let p0 = world_to_screen_aabbnorm(V[0], M, f32(config.screen_width), f32(config.screen_height), cb);
            let p1 = world_to_screen_aabbnorm(V[1], M, f32(config.screen_width), f32(config.screen_height), cb);

            let t = fragment_position_line_relative(px_f, p0.xy, p1.xy);
            if t < 0.0 || t > 1.0 { continue; }
            let p = mix(p0, p1, t);

            // coverage
            let r = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, clamp(p.z, 0.0, 1.0)); // clamp(0.5 * (p0.z + p1.z), 0.0, 1.0));
            let cov = clamp(1.0 - distance(px_f, p.xy) / r, 0.0, 1.0);
            if cov <= 0.0 { continue; }
            // fetch material because we are going to need the alpha/Beer.
            let mat = materials[strand_metadata[seg.strand_idx].material_idx];
            // map to [0,1] behind z0
            let dzp = max(0.0, p.z - z0) * invSpan;

            // optional gamma warp AFTER normalization
            let u = pow(clamp(dzp, 0.0, 1.0), DOM_GAMMA);

            // slice index (safe clamp)
            let tL = u * f32(L);
            let i = min(u32(floor(tL)), L - 1u);

            // optional 2-slice linear distribution to reduce stair-steps
            let w = fract(tL);

            // your per-fragment opacity
            let a = cov * mat.absorption_color.w * 0.5;

            // accumulate (keep α in [0,1])
            let a0 = alpha[i];
            alpha[i] = clamp(a0 + (1.0 - a0) * (1.0 - w) * a, 0.0, 1.0);
            if i + 1u < L {
                let a1 = alpha[i + 1u];
                alpha[i + 1u] = clamp(a1 + (1.0 - a1) * w * a, 0.0, 1.0);
            }

            z_nearest = min(z_nearest, p.z);
        }
    }

    // -------- write z0 (consistent convention; example: near=0..1)
    textureStore(deep_opacity_maps_depth, px_i, vec4<f32>(z_nearest, 0.0, 0.0, 0.0));

    var acc = 0.0;
    for (var i: u32 = 0u; i < L; i++) {
        let a = alpha[i];
        acc = acc + (1.0 - acc) * a;   // acc = 1 - Π(1-a_k)
        textureStore(deep_opacity_maps, vec3<i32>(px_i, i32(i)), vec4<f32>(acc, 0.0, 0.0, 0.0));
    }
}

#endif

// Disabled for correctness-only raster pass:
// @group(0) @binding(#{DEEP_OPACITY_TEXTURE_O}) var deep_opacity_sampler: sampler;
// @group(0) @binding(#{DEEP_OPACITY_TEXTURE_D}) var deep_opacity_depth_sampler: sampler;
// @group(0) @binding(#{DEEP_OPACITY_TEXTURE_O_VIEW}) var deep_opacity_maps: texture_3d<f32>;
// @group(0) @binding(#{DEEP_OPACITY_TEXTURE_D_VIEW}) var deep_opacity_depth_maps: texture_2d<f32>;

#ifdef LINEAR
@compute @workgroup_size(8, 8, 1) // TODO: Should match froxel_size_x, froxel_size_y
fn rasterize_strands(
    @builtin(global_invocation_id) global_id: vec3<u32>,    // Represents the pixel coordinate (x, y, 0)
    @builtin(workgroup_id) workgroup_id: vec3u,             // Represents the tile index (tx, ty, 0)
    @builtin(local_invocation_id) local_id: vec3u           // Represents pixel within tile (lx, ly, 0)
) {
    let pixel_coord_int = vec2<i32>(global_id.xy);

    let pixel_center = vec2<f32>(global_id.xy) + vec2<f32>(0.5, 0.5); // Center of the pixel

    // Initialize final pixel color (start transparent black)
    var final_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let tile_coord_x = workgroup_id.x;
    let tile_coord_y = workgroup_id.y;

    let geo = get_geo0();
    let clip_bounds = find_clip_bounds(view.unjittered_clip_from_world, geo.aabb.min, geo.aabb.max);
    let aabb_znear_zfar = vec2<f32>(clip_bounds[0].z, clip_bounds[1].z);

    // Disabled for correctness-only raster pass:
    // let light: types::DirectionalLight = lights.directional_lights[LIGHT_INDEX];
    // let cascade = light.cascades[0];
    // let light_cascade_clip_from_world = cascade.clip_from_world;
    // let light_clip_bounds = find_clip_bounds(light_cascade_clip_from_world, geo.aabb.min, geo.aabb.max);
    // let light_aabb_znear_zfar = vec2<f32>(light_clip_bounds[0].z, light_clip_bounds[1].z);
    // let texture_dims = textureDimensions(deep_opacity_maps);
    // let shadow_map_dims = vec2<f32>(texture_dims.xy);
    // let depth_texture_slices = texture_dims.z;
    var g_min_depth: f32 = 0.0;

    for (var dz: u32 = 0; dz < config.depth_slices; dz = dz + 1) {

        let froxel_idx = calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config);

        // bounds check
        if froxel_idx >= arrayLength(&tile_offsets_buffer) { continue; }

        let start_segment_offset = tile_offsets_buffer[froxel_idx];
        let segment_count_in_froxel = tile_counts_buffer[froxel_idx];

        var froxel_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        // Process all segments within this froxel
        for (var s: u32 = 0; s < segment_count_in_froxel; s = s + 1u) {
            // if s >= 300 { break; } // Limit number of segments processed per froxel
            let packed_buffer_idx = start_segment_offset + s;
             // Safety check packed buffer bounds
            if packed_buffer_idx >= arrayLength(&packed_segments_buffer) { continue; }

            let segment_ref = packed_segments_buffer[packed_buffer_idx];

            // Get strand metadata
            let strand_idx = segment_ref.strand_idx;
            let strand_meta = get_segment_meta(segment_ref);
            if strand_meta.count < 2u { continue; }

            let v0_idx = segment_ref.segment_start_idx;
            let v1_idx = segment_ref.segment_start_idx + 1u;
            let V = get_segment_vertices(segment_ref);
            // if all(V[0] == vec4<f32>(0.0)) && all(V[1] == vec4<f32>(0.0)) { continue; }
            let v0_world = V[0];
            let v1_world = V[1];

            // Project to screen space (pixels)
            let p0_screen = world_to_screen_raw(v0_world, view.unjittered_clip_from_world, vec4<f32>(0.0, 0.0, f32(config.screen_width), f32(config.screen_height)));
            let p1_screen = world_to_screen_raw(v1_world, view.unjittered_clip_from_world, vec4<f32>(0.0, 0.0, f32(config.screen_width), f32(config.screen_height)));

            // Skip if segment is fully behind camera or off-screen after projection
            if p0_screen.x < 0.0 && p1_screen.x < 0.0 { continue; } // Basic culling

            // Calculate analytical coverage
            let t = fragment_position_line_relative(pixel_center, p0_screen.xy, p1_screen.xy);
            if t < 0.0 || t > 1.0 { continue; } // Skip if outside segment
            let p_frag = mix(p0_screen, p1_screen, t);
            let dist = distance(pixel_center, p_frag.xy);

            let z_cam = clip_bounds[0].z + (clip_bounds[0].z - clip_bounds[1].z) * p_frag.z;

            // blend hair radius from MIN to MAX based on distance to camera
            let r = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, z_cam); // p_frag.z);

            // Simple linear falloff based on distance
            let coverage = clamp(1.0 - dist / r, 0.0, 1.0);

            if coverage > 0.0 {
                // Disabled for correctness-only raster pass:
                // let out_row = strand_idx % MAX_TEXTURE_EXT;
                // let out_col = strand_idx / MAX_TEXTURE_EXT;
                // let y_coord = out_row;
                // let x0_coord = out_col * (pc.workgroup_offset + 1) + (v0_idx - strand_meta.offset);
                // let x1_coord = out_col * (pc.workgroup_offset + 1) + (v1_idx - strand_meta.offset);
                // let shading0 = textureLoad(shading_buffer, vec2<u32>(x0_coord, y_coord));
                // let shading1 = textureLoad(shading_buffer, vec2<u32>(x1_coord, y_coord));
                // let hair_color = mix(shading0, shading1, clamp(t, 0.0, 1.0));
                // let fragment_world_pos = vec4<f32>(screen_to_world_raw(p_frag.xyz, view, vec4<f32>(0.0, 0.0, f32(config.screen_width), f32(config.screen_height))), 1.0);
                // let fragment_light = world_to_screen_aabbnorm(fragment_world_pos, light_cascade_clip_from_world, shadow_map_dims.x, shadow_map_dims.y, light_clip_bounds);
                // var sample_coord = vec2<f32>(fragment_light.xy / shadow_map_dims.xy);
                // let dom_depth = textureSampleLevel(deep_opacity_depth_maps, deep_opacity_depth_sampler, sample_coord, 0.0);
                // let min_depth = dom_depth.x;
                // var occlusion = 0.0;
                // if fragment_light.z > min_depth {
                //     let span = max(1e-6, 1.0 - min_depth);
                //     let invSpan = 1.0 / span;
                //     let dzp = max(0.0, fragment_light.z - min_depth) * invSpan;
                //     let u = pow(clamp(dzp, 0.0, 1.0), DOM_GAMMA);
                //     let tL = u * f32(texture_dims.z);
                //     let d = min(tL, f32(texture_dims.z - 1u));
                //     occlusion = textureSampleLevel(deep_opacity_maps, deep_opacity_sampler, vec3<f32>(sample_coord, d), 0.0).x;
                // }
                // let mat = get_segment_material(segment_ref);
                // var ambient_occlusion = (1.0 - occlusion) + (lights.ambient_color.xyz / 255.0) * 0.5 * mat.ambient_factor + (mat.absorption_color.xyz) * 0.5 * mat.ambient_factor;
                // let hair_fragment = vec4<f32>(hair_color.xyz * ambient_occlusion, hair_color.w * coverage);
                let hair_fragment = vec4<f32>(0.92, 0.78, 0.62, 0.65 * coverage);
                // transmittance accumulation
                froxel_color = blend_over(froxel_color, hair_fragment);
                g_min_depth = max(g_min_depth, p_frag.z);
            }
            if froxel_color.a > 0.9995 {
                break; // Stop processing this segment
            }
        } // End loop over segments in froxel

        // Order dependent transparency (we go front to back)
        final_color = blend_over(final_color, froxel_color);

        // --- Optional Early Exit ---
        // If pixel becomes nearly opaque, we can stop processing deeper Z slices
        if final_color.a > 0.999 {
            break; // Stop Z loop
        }
    } // End loop over depth slices (dz)

    // Write the final accumulated color to the render target
    textureStore(render_target, pixel_coord_int, final_color);
    textureStore(depth_target, pixel_coord_int, vec4<f32>(g_min_depth, 0.0, 0.0, 0.0));
}
#endif

// #endif // SHADOWS vs camera
