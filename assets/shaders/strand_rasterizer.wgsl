#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::mesh_view_types as types
#import "shaders/spline.wgsl"::{ intersect_catmull_rom_spline_3d, closest_point };
#import "shaders/common.wgsl"::{ find_clip_bounds, world_to_screen, screen_to_world, calculate_froxel_index }
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
const MIN_HAIR_RADIUS_PIXELS : f32 = 1.0; // Example: Thickness in pixels
const MAX_HAIR_RADIUS_PIXELS : f32 = 1.0; // Example: Thickness in pixels

@group(0) @binding(#{VERTEX_BUFFER}) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(#{INDEX_BUFFER}) var<storage, read> indices: array<u32>;
@group(0) @binding(#{META_BUFFER}) var<storage, read> strand_metadata: array<StrandMeta>;
@group(0) @binding(#{GEO_BUFFER}) var<storage, read> geos: array<StrandGeo>;
@group(0) @binding(#{TILE_OFFSETS_BUFFER}) var<storage, read> tile_offsets_buffer: array<u32>;
@group(0) @binding(#{TILE_COUNTS_BUFFER}) var<storage, read> tile_counts_buffer: array<atomic<u32>>;
@group(0) @binding(#{FROXEL_TILE_BUFFER}) var<storage, read> packed_segments_buffer: array<SegmentRef>; // Read only
@group(0) @binding(#{OUTPUT_TEXTURE}) var render_target: texture_storage_2d<rgba8unorm, write>;
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
const GAMMA: f32 = 0.3; // distribution coefficient for DOM slices.

#ifdef SHADOWS 
const DOM_SLICES: u32 = #{NUM_DOM_SLICES};
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_O}) var deep_opacity_maps: texture_storage_3d<r16float, write>; // TODO: maybe find a more compact format
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_D}) var deep_opacity_maps_depth: texture_storage_2d<r16float, write>; // TODO: maybe find a more compact format
@group(0) @binding(#{CLUSTER_INDICES}) var<storage> clusterable_object_index_lists: types::ClusterLightIndexLists;
@group(0) @binding(#{CLUSTERABLE_OBJECTS}) var<storage> clusterable_objects: types::ClusterableObjects;
@group(0) @binding(#{CLUSTER_OFFSETS_AND_COUNTS}) var<storage> cluster_offsets_and_counts: types::ClusterOffsetsAndCounts;
@group(0) @binding(#{POINT_LIGHT_DEPTH_TEXTURE}) var point_shadow_textures_linear_sampler: sampler;
@group(0) @binding(#{DIRECTIONAL_LIGHT_DEPTH_TEXTURE}) var directional_shadow_textures_linear_sampler: sampler;

@compute @workgroup_size(8, 8, 1)
fn rasterize_strands(
    @builtin(global_invocation_id) global_id: vec3<u32>,    // Represents the pixel coordinate (x, y, 0)
    @builtin(workgroup_id) workgroup_id: vec3u,             // Represents the tile index (tx, ty, 0)
    @builtin(local_invocation_id) local_id: vec3u           // Represents pixel within tile (lx, ly, 0)
) {
    let pixel_coord_int = vec2<i32>(global_id.xy);

    let pixel_center = vec2<f32>(global_id.xy) + vec2<f32>(0.5, 0.5); // Center of the pixel

    let tile_coord_x = workgroup_id.x;
    let tile_coord_y = workgroup_id.y;

    let light: types::DirectionalLight = lights.directional_lights[LIGHT_INDEX];
    let cascade = light.cascades[0]; // TODO: select cascade based on distance
    let clip_from_world = cascade.clip_from_world;

    let geo = geos[0]; // Assuming only one strand geo for now
    let clip_bounds = find_clip_bounds(clip_from_world, geo.aabb.min, geo.aabb.max);
    let aabb_znear_zfar = vec2<f32>(clip_bounds[0].z, clip_bounds[1].z);

    // DEBUG
    var frag_counts: u32 = 0;
    var slice_start: u32 = 0;
    var slice_end: u32 = 0;
    let depth_slices = config.depth_slices;

    for (var dz: u32 = 0; dz < u32(depth_slices); dz = dz + 1) { 
        let frag_count = tile_counts_buffer[calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config)];
        frag_counts += frag_count;
        if (frag_count > 0u && slice_start == 0u) {
            slice_start = dz;
        }
        if (frag_count > 0u && dz > slice_end) {
            slice_end = dz + 1;
        }
        // let debug_color = vec4<f32>(heatmap_precise(f32(frag_count) / f32(pc.num_elements)), 0.2);
        // textureStore(deep_opacity_maps, pixel_coord_int, dz, vec4<f32>(debug_color.xy, 0.0, 0.0));
    }
    if frag_counts == 0u {
        // No segments in this tile, early exit
        return;
    }
    
    // slice buffer (constant size array)
    var slice_opacities: array<f32, DOM_SLICES>; // TODO: fit this to be compiled with number of slices
    var min_pixel_depth = f32(slice_start) / f32(depth_slices); // Decent first estimate
    var max_pixel_depth = clamp(f32(slice_end) / f32(depth_slices), 0.0, 1.0); // Decent first estimate
    var pixel_depth = min_pixel_depth;

    for (var dz: u32 = slice_start; dz < depth_slices; dz = dz + 1) {
        
        let froxel_idx = calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config);
        
        // bounds check
        if (froxel_idx >= arrayLength(&tile_offsets_buffer)) { continue; }
        
        let start_segment_offset = tile_offsets_buffer[froxel_idx];
        let segment_count_in_froxel = tile_counts_buffer[froxel_idx];
        
        // Process all segments within this froxel
        for (var s: u32 = 0; s < segment_count_in_froxel; s = s + 1u) {
            
            let packed_buffer_idx = start_segment_offset + s;
            if (packed_buffer_idx >= arrayLength(&packed_segments_buffer)) {
                continue;
            }

            let segment_ref = packed_segments_buffer[packed_buffer_idx];
            if (segment_ref.strand_idx >= arrayLength(&strand_metadata)) { 
                continue;
            } // Safety check

            let verts = get_segment_vertices(segment_ref);

            // Project to light space (pixels)
            let p0_screen = world_to_screen(verts[0], clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            let p1_screen = world_to_screen(verts[1], clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);

            // Skip if segment is fully behind camera or off-screen after projection
            // if (p0_screen.x < 0.0 && p1_screen.x < 0.0) { 
            //     continue;
            // } // Basic culling

            // Calculate analytical coverage
            let t = fragment_position_line_relative(pixel_center, p0_screen.xy, p1_screen.xy);
            let dist = point_segment_distance(pixel_center, p0_screen.xy, p1_screen.xy, t);

            // blend hair radius from MIN to MAX based on distance to camera
            let r = mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, (p0_screen.z + p1_screen.z) / 2.0); // TODO: this needs special scaling for lights.

            // Simple linear falloff based on distance
            let coverage = clamp(1.0 - dist / r, 0.0, 1.0);

            if (coverage > 0.0) {
                // Calculate color/alpha contribution of this hair segment fragment
                let depth = mix(p0_screen.z, p1_screen.z, clamp(t, 0.0, 1.0));
                if (pixel_depth == min_pixel_depth) {
                    pixel_depth = depth; // Initialize depth
                } else {
                    pixel_depth = min(pixel_depth, depth); // Update depth
                }
                let slice_idx = u32(pow(clamp(depth - pixel_depth, 0.0, 1.0), GAMMA) * f32(DOM_SLICES));
                var slice_opacity = slice_opacities[slice_idx];
                slice_opacity = slice_opacity + (1.0 - slice_opacity) * coverage * 0.1; // Accumulate opacity TODO: use hair opacity instead of 0.5
                slice_opacities[slice_idx] = slice_opacity;
            }
        } // End loop over segments in froxel
    }
    textureStore(deep_opacity_maps_depth, pixel_coord_int, vec4<f32>(pixel_depth, 0.0, 0.0, 0.0));

    var accum_opacity = 0.0;
    for (var i = 0u; i < DOM_SLICES; i = i + 1) {
        let opacity = slice_opacities[i];
        accum_opacity = accum_opacity + (1.0 - accum_opacity) * opacity;
        textureStore(deep_opacity_maps, vec3<i32>(pixel_coord_int, i32(i)), vec4<f32>(accum_opacity, 0.0, 0.0, 0.0));
    }
}

#else

#define DEBUG

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
    let ambient_factor = 0.04;

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
            let v0_strand_idx = indices[v0_idx];
            let v1_strand_idx = indices[v1_idx];

            // Get world-space vertex positions
            let v0_world = vertices[v0_strand_idx];
            let v1_world = vertices[v1_strand_idx];

            // Project to screen space (pixels)
            let p0_screen = world_to_screen(v0_world, view.clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            let p1_screen = world_to_screen(v1_world, view.clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);

            // Skip if segment is fully behind camera or off-screen after projection
            if (p0_screen.x < 0.0 && p1_screen.x < 0.0) { continue; } // Basic culling

            // Calculate analytical coverage
            let t = fragment_position_line_relative(pixel_center, p0_screen.xy, p1_screen.xy);
            let dist = point_segment_distance(pixel_center, p0_screen.xy, p1_screen.xy, t);

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

                let fragment_world_pos = mix(v0_world, v1_world, clamp(t, 0.0, 1.0));
                let fragment_light = world_to_screen(fragment_world_pos, cascade.clip_from_world, shadow_map_dims.x, shadow_map_dims.y, light_aabb_znear_zfar);
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
    let ambient_factor = 0.02;
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
        let debug_color = vec4<f32>(heatmap_precise(f32(frag_count) / f32(pc.num_elements)), 0.2);
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
            let p0_screen = world_to_screen(vec4<f32>(v1_world, 1.0), view.clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            let p1_screen = world_to_screen(vec4<f32>(v2_world, 1.0), view.clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);

            // Skip if segment is fully behind camera or off-screen after projection
            if (p0_screen.x < 0.0 && p1_screen.x < 0.0) { continue; } // Basic culling

            // Calculate analytical coverage
            let t = fragment_position_line_relative(pixel_center, p0_screen.xy, p1_screen.xy);
            let p = mix(p0_screen.xyz, p1_screen.xyz, clamp(t, 0.0, 1.0));
            let p_world = screen_to_world(p, view, aabb_znear_zfar);
            
            let N_ = normalize(screen_to_world(p0_screen - p1_screen, view, aabb_znear_zfar));
            let U = normalize(view.world_position.xyz - p_world);
            let V = normalize(cross(N_, U));

            let plane = mat3x3<f32>(p_world, U, V);
            let intersection_points: mat4x3<f32> = intersect_catmull_rom_spline_3d(v0_world.xyz, v1_world.xyz, v2_world.xyz, v3_world.xyz, plane, spline_alpha);
            let mask = intersection_points[3] != vec3<f32>(0.0, 0.0, 0.0);
            if all(!mask) {
                continue; // No intersection
            }
            let closest_point: vec3<f32> = closest_point(intersection_points, view.world_position.xyz);
            let point_screen = world_to_screen(vec4<f32>(closest_point, 1.0), view.clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            let dist = distance(pixel_center, point_screen.xy);

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