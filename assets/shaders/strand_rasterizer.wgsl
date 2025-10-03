#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::mesh_view_types as types
#import "shaders/spline.wgsl"::{ intersect_catmull_rom_spline_3d, closest_point };
#import "shaders/common.wgsl"::{ find_clip_bounds, world_to_screen, world_to_screen_aabbnorm, world_to_screen_raw, screen_to_world_raw, screen_to_world, calculate_froxel_index }
#import "shaders/types.wgsl"::{
    Aabb,
    FroxelConfig,
    SegmentRef,
    StrandGeo,
    StrandMeta,
    PushConstants,
}

const LIGHT_INDEX: u32 = 0u; // Example constant for light index TODO: compute prepass -> indirect dispatch -> light index from uniforms

var<push_constant> pc: PushConstants;

const MAX_TEXTURE_EXT: u32 = #{MAX_TEXTURE_EXTENT};
const MIN_HAIR_RADIUS_PIXELS : f32 = 0.4; // Example: Thickness in pixels
const MAX_HAIR_RADIUS_PIXELS : f32 = 0.8; // Example: Thickness in pixels

@group(0) @binding(#{VERTEX_BUFFER}) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(#{INDEX_BUFFER}) var<storage, read> indices: array<u32>;
@group(0) @binding(#{META_BUFFER}) var<storage, read> strand_metadata: array<StrandMeta>;
@group(0) @binding(#{GEO_BUFFER}) var<storage, read> geos: array<StrandGeo>;
@group(0) @binding(#{TILE_OFFSETS_BUFFER}) var<storage, read> tile_offsets_buffer: array<u32>;
@group(0) @binding(#{TILE_COUNTS_BUFFER}) var<storage, read> tile_counts_buffer: array<atomic<u32>>;
@group(0) @binding(#{FROXEL_TILE_BUFFER}) var<storage, read> packed_segments_buffer: array<SegmentRef>; // Read only
@group(0) @binding(#{OUTPUT_TEXTURE}) var render_target: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(#{OUTPUT_DEPTH}) var depth_target: texture_storage_2d<r32float, write>;
@group(0) @binding(#{FROXEL_CONFIG}) var<uniform> config: FroxelConfig;
@group(0) @binding(#{VIEW_UNIFORM}) var<uniform> view: View;
@group(0) @binding(#{SHADING_BUFFER}) var shading_buffer: texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(#{LIGHT_UNIFORM}) var<uniform> lights: types::Lights;

fn heatmap_precise(value: f32) -> vec3<f32> {
    let v = clamp(value, 0.0, 1.0);
    
    if (v < 0.25) {
        // Blue to cyan
        let t = v / 0.25;
        return vec3<f32>(0.0, t, 1.0);
    } else if (v < 0.5) {
        // Cyan to green
        let t = (v - 0.25) / 0.25;
        return vec3<f32>(0.0, 1.0, 1.0 - t);
    } else if (v < 0.75) {
        // Green to yellow
        let t = (v - 0.5) / 0.25;
        return vec3<f32>(t, 1.0, 0.0);
    } else {
        // Yellow to red
        let t = (v - 0.75) / 0.25;
        return vec3<f32>(1.0, 1.0 - t, 0.0);
    }
}

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
    if (l2 == 0.0) { return 0.0; } // Segment is a point
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
    if (final_alpha < 1e-6) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let final_rgb = (fg_rgb + background.rgb * background.a * (1.0 - foreground.a)) / final_alpha;
    return vec4<f32>(final_rgb, final_alpha);
}

fn get_segment_vertices(segment_ref: SegmentRef) -> mat2x4<f32> {
    // Get strand metadata
    let strand_idx = segment_ref.strand_idx;
    let strand_meta = strand_metadata[strand_idx];

    let v0_idx = segment_ref.segment_start_idx;
    let v1_idx = segment_ref.segment_start_idx + 1u;
    if ((v1_idx - strand_meta.offset) >= strand_meta.count - 1) { return mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0)); } // Safety check

    // Get segment vertex indices within the strand
    let v0_strand_idx = indices[v0_idx];
    let v1_strand_idx = indices[v1_idx];

    // Get world-space vertex positions
    return mat2x4<f32>(vertices[v0_strand_idx], vertices[v1_strand_idx]);
}

// #define DEBUG
const GAMMA: f32 = 1.0; // distribution coefficient for DOM slices.

#ifdef SHADOWS 
const DOM_SLICES: u32 = #{NUM_DOM_SLICES};
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_O}) var deep_opacity_maps: texture_storage_3d<r16float, write>; // TODO: maybe find a more compact format
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_D}) var deep_opacity_maps_depth: texture_storage_2d<r16float, write>; // TODO: maybe find a more compact format
@group(0) @binding(#{CLUSTER_INDICES}) var<storage> clusterable_object_index_lists: types::ClusterLightIndexLists;
@group(0) @binding(#{CLUSTERABLE_OBJECTS}) var<storage> clusterable_objects: types::ClusterableObjects;
@group(0) @binding(#{CLUSTER_OFFSETS_AND_COUNTS}) var<storage> cluster_offsets_and_counts: types::ClusterOffsetsAndCounts;
@group(0) @binding(#{POINT_LIGHT_DEPTH_TEXTURE}) var point_shadow_textures_linear_sampler: sampler;
@group(0) @binding(#{DIRECTIONAL_LIGHT_DEPTH_TEXTURE}) var directional_shadow_textures_linear_sampler: sampler;

@compute @workgroup_size(8,8,1)
fn rasterize_strands(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(workgroup_id)      wg : vec3u) {
    let px_i = vec2<i32>(gid.xy);
    let px_f = vec2<f32>(gid.xy) + vec2(0.5, 0.5);

    // --- constants / inputs
    let light  = lights.directional_lights[LIGHT_INDEX];
    let cas    = light.cascades[0];
    let M      = cas.clip_from_world;
    let geo    = geos[0];
    let cb     = find_clip_bounds(M, geo.aabb.min, geo.aabb.max);
    let depth_slices = config.depth_slices;

    // froxel tile coords
    let tx = wg.x; let ty = wg.y;

    // -------- pass 0: early reject based on tile counts
    var any_in_tile = false;
    for (var dz: u32 = 0u; dz < depth_slices; dz++) {
        let fc = tile_counts_buffer[calculate_froxel_index(tx, ty, dz, config)];
        any_in_tile = any_in_tile || (fc > 0u);
    }
    if (!any_in_tile) { return; }

    // -------- pass 1: find per-pixel nearest depth z0 (in your chosen convention)
    var z0 = 1e9; // if near=0..1 use +INF init, otherwise adjust
    for (var dz: u32 = 0u; dz < depth_slices; dz++) {
        let fidx = calculate_froxel_index(tx, ty, dz, config);
        if (fidx >= arrayLength(&tile_offsets_buffer)) { continue; }

        let start = tile_offsets_buffer[fidx];
        let count = tile_counts_buffer[fidx];

        for (var s: u32 = 0u; s < count; s++) {
            let idx = start + s;
            if (idx >= arrayLength(&packed_segments_buffer)) { continue; }

            let seg = packed_segments_buffer[idx];
            if (seg.strand_idx >= arrayLength(&strand_metadata)) { continue; }

            let V = get_segment_vertices(seg);
            let p0 = world_to_screen_aabbnorm(V[0], M, f32(config.screen_width), f32(config.screen_height), cb);
            let p1 = world_to_screen_aabbnorm(V[1], M, f32(config.screen_width), f32(config.screen_height), cb);

            let t  = fragment_position_line_relative(px_f, p0.xy, p1.xy);
            if (t < 0.0 || t > 1.0) { continue; }
            let p  = mix(p0, p1, t);

            // optional radius & coverage test to skip non-overlaps early
            let r  = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, 0.5*(p0.z + p1.z));
            if (clamp(1.0 - distance(px_f, p.xy)/r, 0.0, 1.0) <= 0.0) { continue; }

            // assume near=0, far=1; if reversed, invert consistently everywhere
            z0 = min(z0, p.z);
        }
    }
    if (z0 == 1e9) {
        // nothing touched this pixel: optionally clear outputs here
        return;
    }

    // -------- per-pixel slice params
    let L      = DOM_SLICES;
    // Choose a span. Common choices:
    //   (a) fixed per-light thickness in light space -> map to normalized [0,1] here
    //   (b) adaptive: cover up to z0 + span_to_far or up to nearest hair max depth observed
    // For a minimal fix, cover [z0, 1] uniformly (then gamma warp if you want):
    let span   = max(1e-6, 1.0 - z0);
    let invSpan = 1.0 / span;

    // -------- zero the slice bin accumulator
    var alpha: array<f32, DOM_SLICES>;
    for (var i: u32 = 0u; i < L; i++) { alpha[i] = 0.0; }

    // -------- pass 2: bin opacities relative to z0
    var z_nearest = z0;
    for (var dz: u32 = 0u; dz < depth_slices; dz++) {
        let fidx = calculate_froxel_index(tx, ty, dz, config);
        if (fidx >= arrayLength(&tile_offsets_buffer)) { continue; }

        let start = tile_offsets_buffer[fidx];
        let count = tile_counts_buffer[fidx];

        for (var s: u32 = 0u; s < count; s++) {
            let idx = start + s;
            if (idx >= arrayLength(&packed_segments_buffer)) { continue; }

            let seg = packed_segments_buffer[idx];
            if (seg.strand_idx >= arrayLength(&strand_metadata)) { continue; }

            let V = get_segment_vertices(seg);
            let p0 = world_to_screen_aabbnorm(V[0], M, f32(config.screen_width), f32(config.screen_height), cb);
            let p1 = world_to_screen_aabbnorm(V[1], M, f32(config.screen_width), f32(config.screen_height), cb);

            let t  = fragment_position_line_relative(px_f, p0.xy, p1.xy);
            if (t < 0.0 || t > 1.0) { continue; }
            let p  = mix(p0, p1, t);

            // coverage
            let r  = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, 0.5*(p0.z + p1.z));
            let cov = clamp(1.0 - distance(px_f, p.xy) / r, 0.0, 1.0);
            if (cov <= 0.0) { continue; }

            // map to [0,1] behind z0
            let dzp = max(0.0, p.z - z0) * invSpan;

            // optional gamma warp AFTER normalization
            let u   = pow(clamp(dzp, 0.0, 1.0), GAMMA);

            // slice index (safe clamp)
            let tL  = u * f32(L);
            let i   = min(u32(floor(tL)), L - 1u);
            // optional 2-slice linear distribution to reduce stair-steps
            let w   = fract(tL);

            // your per-fragment opacity (replace 0.5f by strand alpha/Beer)
            let a = cov * 0.5;

            // accumulate (keep α in [0,1])
            let a0 = alpha[i];
            alpha[i] = clamp(a0 + (1.0 - a0) * (1.0 - w) * a, 0.0, 1.0);
            if (i + 1u < L) {
                let a1 = alpha[i+1u];
                alpha[i+1u] = clamp(a1 + (1.0 - a1) * w * a, 0.0, 1.0);
            }

            z_nearest = min(z_nearest, p.z);
        }
    }

    // -------- write z0 (consistent convention; example: near=0..1)
    textureStore(deep_opacity_maps_depth, px_i, vec4<f32>(z_nearest, 0.0, 0.0, 0.0));

    // -------- either store per-slice α, or cumulative (pick ONE)
    // (A) store raw α per slice:
    // for (var i: u32 = 0u; i < L; i++) {
    //     textureStore(deep_opacity_maps, vec3<i32>(px_i, i32(i)), vec4<f32>(alpha[i], 0.0, 0.0, 0.0));
    // }

    // (B) if you really want cumulative opacity per slice:

    var acc = 0.0;
    for (var i: u32 = 0u; i < L; i++) {
        let a = alpha[i];
        acc = acc + (1.0 - acc) * a;   // acc = 1 - Π(1-a_k)
        textureStore(deep_opacity_maps, vec3<i32>(px_i, i32(i)), vec4<f32>(acc, 0.0, 0.0, 0.0));
    }
}


#else

// #define DEBUG

@group(0) @binding(#{DEEP_OPACITY_TEXTURE_O}) var deep_opacity_sampler: sampler; // TODO: maybe find a more compact format
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_D}) var deep_opacity_depth_sampler: sampler; // TODO: maybe find a more compact format
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_O_VIEW}) var deep_opacity_maps: texture_3d<f32>;
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_D_VIEW}) var deep_opacity_depth_maps: texture_2d<f32>;

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
    let ambient_factor = 0.03;

    let tile_coord_x = workgroup_id.x;
    let tile_coord_y = workgroup_id.y;

    #ifdef DEBUG
        var frag_count: u32 = 0;
        for (var dz: u32 = 0; dz < u32(config.depth_slices); dz = dz + 1) {
            frag_count += tile_counts_buffer[calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config)];
        }
        // Debug: Output the tile_count of this tile divided by num_elements
        let debug_color = vec4<f32>(heatmap_precise(f32(frag_count) / f32(pc.num_elements)), 0.2);
    # endif // DEBUG
    
    let geo = geos[0]; // Assuming only one strand geo for now
    let clip_bounds = find_clip_bounds(view.unjittered_clip_from_world, geo.aabb.min, geo.aabb.max);
    let aabb_znear_zfar = vec2<f32>(clip_bounds[0].z, clip_bounds[1].z);

    // light relative data
    let light: types::DirectionalLight = lights.directional_lights[LIGHT_INDEX];
    let cascade = light.cascades[0]; // TODO: select cascade based on distance
    let light_cascade_clip_from_world = cascade.clip_from_world;

    let light_clip_bounds = find_clip_bounds(light_cascade_clip_from_world, geo.aabb.min, geo.aabb.max);
    let light_aabb_znear_zfar = vec2<f32>(light_clip_bounds[0].z, light_clip_bounds[1].z);
    let texture_dims = textureDimensions(deep_opacity_maps);
    let shadow_map_dims = vec2<f32>(texture_dims.xy);
    let depth_texture_slices = texture_dims.z;
    var g_min_depth: f32 = 0.0;

    for (var dz: u32 = 0; dz < config.depth_slices; dz = dz + 1) {

        let froxel_idx = calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config);

        // bounds check
        if (froxel_idx >= arrayLength(&tile_offsets_buffer)) { continue; }

        let start_segment_offset = tile_offsets_buffer[froxel_idx];
        let segment_count_in_froxel = tile_counts_buffer[froxel_idx];

        var froxel_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        // Process all segments within this froxel
        for (var s: u32 = 0; s < segment_count_in_froxel; s = s + 1u) {
            // if s >= 300 { break; } // Limit number of segments processed per froxel
            let packed_buffer_idx = start_segment_offset + s;
             // Safety check packed buffer bounds
            if (packed_buffer_idx >= arrayLength(&packed_segments_buffer)) { continue; }

            let segment_ref = packed_segments_buffer[packed_buffer_idx];

            // Get strand metadata
            let strand_idx = segment_ref.strand_idx;
            if (strand_idx >= arrayLength(&strand_metadata)) { 
                final_color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Debug color
                continue; 
            } // Safety check
            let strand_meta = strand_metadata[strand_idx];

            let v0_idx = segment_ref.segment_start_idx;
            let v1_idx = segment_ref.segment_start_idx + 1u;
            if ((v1_idx - strand_meta.offset) >= strand_meta.count - 1) { continue; } // Safety check

            // Get segment vertex indices within the strand
            let v0_strand_idx = indices[v0_idx];
            let v1_strand_idx = indices[v1_idx];

            // Get world-space vertex positions
            let v0_world = vertices[v0_strand_idx];
            let v1_world = vertices[v1_strand_idx];

            // Project to screen space (pixels)
            // let p0_screen = world_to_screen(v0_world, view.unjittered_clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            // let p1_screen = world_to_screen(v1_world, view.unjittered_clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            let p0_screen = world_to_screen_raw(v0_world, view.unjittered_clip_from_world, vec4<f32>(0.0, 0.0, f32(config.screen_width), f32(config.screen_height)));
            let p1_screen = world_to_screen_raw(v1_world, view.unjittered_clip_from_world, vec4<f32>(0.0, 0.0, f32(config.screen_width), f32(config.screen_height)));

            // Skip if segment is fully behind camera or off-screen after projection
            if (p0_screen.x < 0.0 && p1_screen.x < 0.0) { continue; } // Basic culling

            // Calculate analytical coverage
            let t = fragment_position_line_relative(pixel_center, p0_screen.xy, p1_screen.xy);
            if (t < 0.0 || t > 1.0) { continue; } // Skip if outside segment
            let p_frag = mix(p0_screen, p1_screen, t);
            let dist = distance(pixel_center, p_frag.xy);
            // let ndc_p_frag_z = (p_frag.z * (aabb_znear_zfar.y - aabb_znear_zfar.x)) + aabb_znear_zfar.x;
            // let world_p_frag = screen_to_world(p_frag, view, aabb_znear_zfar);
            // let ndc_p_frag_z = world_to_screen_ndc(vec4<f32>(world_p_frag, 1.0), view.unjittered_clip_from_world, f32(config.screen_width), f32(config.screen_height));

            // blend hair radius from MIN to MAX based on distance to camera
            let r = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, (p0_screen.z + p1_screen.z) / 2.0);

            // Simple linear falloff based on distance
            let coverage = clamp(1.0 - dist / r, 0.0, 1.0);

            if (coverage > 0.0) {
                // Calculate color/alpha contribution of this hair segment fragment
                // sample the shading buffer (maybe a sampler here, maybe not if we'll use spline interpolation in the rasterizer directly)
                let out_row = strand_idx % MAX_TEXTURE_EXT;
                let out_col = strand_idx / MAX_TEXTURE_EXT;
                let y_coord = out_row;
                let x0_coord = out_col * (pc.workgroup_offset + 1) + (v0_idx - strand_meta.offset);
                let x1_coord = out_col * (pc.workgroup_offset + 1) + (v1_idx - strand_meta.offset);

                let shading0 = textureLoad(shading_buffer, vec2<u32>(x0_coord, y_coord));
                let shading1 = textureLoad(shading_buffer, vec2<u32>(x1_coord, y_coord));

                let hair_color = mix(shading0, shading1, clamp(t, 0.0, 1.0));

                let fragment_world_pos = vec4<f32>(screen_to_world_raw(p_frag.xyz, view, vec4<f32>(0.0, 0.0, f32(config.screen_width), f32(config.screen_height))), 1.0);
                let fragment_light = world_to_screen_aabbnorm(fragment_world_pos, light_cascade_clip_from_world, shadow_map_dims.x, shadow_map_dims.y, light_clip_bounds);

                var sample_coord = vec2<f32>(fragment_light.xy / shadow_map_dims.xy);
                
                let dom_depth = textureSampleLevel(deep_opacity_depth_maps, deep_opacity_depth_sampler, sample_coord, 0.0);
                let min_depth = dom_depth.x;
                var occlusion = 0.0;
                // if fragment_light.z > min_depth {
                let d = pow((fragment_light.z - min_depth), GAMMA);
                occlusion = textureSampleLevel(deep_opacity_maps, deep_opacity_sampler, vec3<f32>(sample_coord, d), 0.0).x;
                // }
                
                var ambient_occlusion = (1.0 - occlusion) + (lights.ambient_color.xyz / 255.0) * ambient_factor;
                let hair_fragment = vec4<f32>(hair_color.xyz * ambient_occlusion, hair_color.w * coverage);
                // let hair_fragment = vec4<f32>(occlusion, occlusion, occlusion, 0.5 * coverage);
                // transmittance accumulation
                froxel_color = blend_over(froxel_color, hair_fragment);
                g_min_depth = max(g_min_depth, p_frag.z);
                // g_min_depth = min(g_min_depth, ndc_p_frag_z);
            }
            if (froxel_color.a > 0.9995) {
                break; // Stop processing this segment
            }
        } // End loop over segments in froxel

        // Order dependent transparency (we go front to back)
        final_color = blend_over(final_color, froxel_color);

        // --- Optional Early Exit ---
        // If pixel becomes nearly opaque, we can stop processing deeper Z slices
        if (final_color.a > 0.9999) {
            break; // Stop Z loop
        }

    } // End loop over depth slices (dz)

    #ifdef DEBUG
        // this makes the debug color more visible than blending it.
        final_color = final_color + debug_color;
    # endif // DEBUG

    // Write the final accumulated color to the render target
    textureStore(render_target, pixel_coord_int, final_color);
    textureStore(depth_target, pixel_coord_int, vec4<f32>(g_min_depth, 0.0, 0.0, 0.0));
}
#endif

#ifdef SPLINE
@compute @workgroup_size(8, 8, 1) // TODO: Should match froxel_size_x, froxel_size_y
fn rasterize_strands(
    @builtin(global_invocation_id) global_id: vec3<u32>,    // Represents the pixel coordinate (x, y, 0)
    @builtin(workgroup_id) workgroup_id: vec3u,             // Represents the tile index (tx, ty, 0)
    @builtin(local_invocation_id) local_id: vec3u           // Represents pixel within tile (lx, ly, 0)
) {
    let pixel_coord_int = vec2<i32>(global_id.xy);

    let pixel_center = vec2<f32>(global_id.xy) + vec2<f32>(0.5, 0.5); // Center of the pixel
    let pixel_ndc = vec3<f32>(pixel_center.xy, 0.0);

    // Initialize final pixel color (start transparent black)
    var final_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let ambient_factor = 0.03;
    let hair_root_factor = 0.015;
    let spline_alpha = 1.0;

    let tile_coord_x = workgroup_id.x;
    let tile_coord_y = workgroup_id.y;

    #ifdef DEBUG
        var frag_count: u32 = 0;
        for (var dz: u32 = 0; dz < u32(config.depth_slices); dz = dz + 1) {
            frag_count += tile_counts_buffer[calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config)];
        }
        // Debug: Output the tile_count of this tile divided by num_elements
        let debug_color = vec4<f32>(heatmap_precise(f32(frag_count) * 10.0 / f32(pc.num_elements)), 0.2);
    # endif // DEBUG
    
    let geo = geos[0]; // Assuming only one strand geo for now
    let clip_bounds = find_clip_bounds(view.clip_from_world, geo.aabb.min, geo.aabb.max);
    let aabb_znear_zfar = vec2<f32>(clip_bounds[0].z, clip_bounds[1].z);

    // light relative data
    let light: types::DirectionalLight = lights.directional_lights[LIGHT_INDEX];
    let cascade = light.cascades[0]; // TODO: select cascade based on distance
    let light_clip_bounds = find_clip_bounds(cascade.clip_from_world, geo.aabb.min, geo.aabb.max);
    let light_aabb_znear_zfar = vec2<f32>(light_clip_bounds[0].z, light_clip_bounds[1].z);
    let texture_dims = textureDimensions(deep_opacity_maps);
    let shadow_map_dims = vec2<f32>(texture_dims.xy);
    let depth_texture_slices = texture_dims.z;

    for (var dz: u32 = 0; dz < config.depth_slices; dz = dz + 1) {

        let froxel_idx = calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config);

        // bounds check
        if (froxel_idx >= arrayLength(&tile_offsets_buffer)) { continue; }

        let start_segment_offset = tile_offsets_buffer[froxel_idx];
        let segment_count_in_froxel = tile_counts_buffer[froxel_idx];

        var froxel_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        // Process all segments within this froxel
        for (var s: u32 = 0; s < segment_count_in_froxel; s = s + 1u) {
            // if s >= 300 { break; } // Limit number of segments processed per froxel
            let packed_buffer_idx = start_segment_offset + s;
             // Safety check packed buffer bounds
            if (packed_buffer_idx >= arrayLength(&packed_segments_buffer)) { continue; }

            let segment_ref = packed_segments_buffer[packed_buffer_idx];

            // Get strand metadata
            let strand_idx = segment_ref.strand_idx;
            if (strand_idx >= arrayLength(&strand_metadata)) { 
                final_color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Debug color
                continue; 
            } // Safety check
            let strand_meta = strand_metadata[strand_idx];

            let v0_idx = segment_ref.segment_start_idx;
            let v1_idx = segment_ref.segment_start_idx + 1u;
            if ((v1_idx - strand_meta.offset) >= strand_meta.count - 1) { continue; } // Safety check
            
            // Get segment vertex indices within the strand
            let v1_strand_idx = indices[v0_idx];
            let v2_strand_idx = indices[v1_idx];
            
            var v0_world = vertices[v1_strand_idx].xyz;
            let v1_world = v0_world;
            let v2_world = vertices[v2_strand_idx].xyz;
            var v3_world = v2_world;
            let N = normalize(v2_world - v1_world);
            if start_segment_offset == 0 {
                // we use a start point offset by hair_root_factor in the direction of the segment
                v0_world = v0_world - N * hair_root_factor;
            } else {
                let vprev_idx = indices[v0_idx - 1u];
                v0_world = vertices[vprev_idx].xyz;
            }
            if (((v1_idx + 1) - strand_meta.offset) >= strand_meta.count - 1) { 
                // we use the end point offset by hair_root_factor in the direction of the segment
                v3_world = v2_world + N * hair_root_factor;
            } else {
                let vnext_idx = indices[v1_idx + 1u];
                v3_world = vertices[vnext_idx].xyz;
            }

            // Get world-space vertex positions

            // Project to screen space (pixels)
            let p0_screen = world_to_screen(vec4<f32>(v0_world, 1.0), view.unjittered_clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            let p1_screen = world_to_screen(vec4<f32>(v1_world, 1.0), view.unjittered_clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            let p2_screen = world_to_screen(vec4<f32>(v2_world, 1.0), view.unjittered_clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            let p3_screen = world_to_screen(vec4<f32>(v3_world, 1.0), view.unjittered_clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);

            // Skip if segment is fully behind camera or off-screen after projection
            if (p1_screen.x < 0.0 && p2_screen.x < 0.0) { continue; } // Basic culling

            // Calculate analytical coverage
            let t = fragment_position_line_relative(pixel_center, p1_screen.xy, p2_screen.xy);
            if (t < 0.0 || t > 1.0) { continue; } // Skip if outside segment
            let p = mix(p1_screen, p2_screen, t).xyz;
            let p_world = screen_to_world(p, view, aabb_znear_zfar);
            
            let N_ = normalize(vec3<f32>((p1_screen - p2_screen).xy, 0.0));
            let tangent = N_;
            let camera_dir = normalize(view.world_position - p_world);
            let binormal = normalize(cross(tangent, camera_dir));
            let U = world_to_screen(vec4(normalize(cross(binormal, tangent)), 1.0), view.unjittered_clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            let V = normalize(cross(N_, U));

            let plane = mat3x3<f32>(p, U, V);
            let intersection_points: mat4x3<f32> = intersect_catmull_rom_spline_3d(p0_screen.xyz, p1_screen.xyz, p2_screen.xyz, p3_screen.xyz, plane, spline_alpha);
            let mask = intersection_points[3] != vec3<f32>(0.0, 0.0, 0.0);
            if all(!mask) {
                continue; // No intersection
            }
            let closest_point: vec3<f32> = closest_point(intersection_points, view.world_position.xyz);
            let point_screen = closest_point;
            let dist = distance(pixel_center, point_screen.xy);

            // blend hair radius from MIN to MAX based on distance to camera
            let r = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, (p1_screen.z + p2_screen.z) / 2.0);

            // Simple linear falloff based on distance
            let coverage = clamp(1.0 - dist / r, 0.0, 1.0);

            if (coverage > 0.0) {
                // Calculate color/alpha contribution of this hair segment fragment
                // sample the shading buffer (maybe a sampler here, maybe not if we'll use spline interpolation in the rasterizer directly)
                let out_row = strand_idx % MAX_TEXTURE_EXT;
                let out_col = strand_idx / MAX_TEXTURE_EXT;
                let y_coord = out_row;
                let x0_coord = out_col * (pc.workgroup_offset + 1) + (v0_idx - strand_meta.offset);
                let x1_coord = out_col * (pc.workgroup_offset + 1) + (v1_idx - strand_meta.offset);

                let shading0 = textureLoad(shading_buffer, vec2<u32>(x0_coord, y_coord));
                let shading1 = textureLoad(shading_buffer, vec2<u32>(x1_coord, y_coord));

                let hair_color = mix(shading0, shading1, clamp(t, 0.0, 1.0));

                let fragment_world_pos = mix(v1_world, v2_world, clamp(t, 0.0, 1.0));
                let fragment_light = world_to_screen(vec4<f32>(fragment_world_pos, 1.0), cascade.clip_from_world, shadow_map_dims.x, shadow_map_dims.y, light_aabb_znear_zfar);
                var sample_coord = vec2<f32>(fragment_light.xy / shadow_map_dims.xy);
                
                let dom_depth = textureSampleLevel(deep_opacity_depth_maps, deep_opacity_depth_sampler, sample_coord, 0.0);
                let min_depth = dom_depth.x;
                var occlusion = 0.0;
                if fragment_light.z > min_depth {
                    let d = pow((fragment_light.z - min_depth), GAMMA);
                    occlusion = textureSampleLevel(deep_opacity_maps, deep_opacity_sampler, vec3<f32>(sample_coord, d), 0.0).x;
                }
                
                var ambient_occlusion = (1.0 - occlusion) + (lights.ambient_color.xyz / 255.0) * ambient_factor;
                let hair_fragment = vec4<f32>(hair_color.xyz * ambient_occlusion, hair_color.w * coverage);
                // transmittance accumulation
                froxel_color = blend_over(froxel_color, hair_fragment);
            }
            if (froxel_color.a > 0.9995) {
                break; // Stop processing this segment
            }
        } // End loop over segments in froxel

        // Order dependent transparency (we go front to back)
        final_color = blend_over(final_color, froxel_color);

        // --- Optional Early Exit ---
        // If pixel becomes nearly opaque, we can stop processing deeper Z slices
        if (final_color.a > 0.9999) {
            break; // Stop Z loop
        }

    } // End loop over depth slices (dz)

    #ifdef DEBUG
        // this makes the debug color more visible than blending it.
        final_color = final_color + debug_color;
    # endif // DEBUG

    // Write the final accumulated color to the render target
    textureStore(render_target, pixel_coord_int, final_color);
}
#endif // SPLINE

#endif // SHADOWS vs camera