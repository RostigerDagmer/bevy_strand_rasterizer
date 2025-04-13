#import bevy_render::view::View
#import bevy_render::mesh::mesh_bindings::Instance // If needed for transforms
#import bevy_pbr::mesh_view_types as types

const LIGHT_INDEX: u32 = 0u; // Example constant for light index TODO: compute prepass -> indirect dispatch -> light index from uniforms

// --- Structures ---
struct Aabb {
    min: vec3<f32>,
    pad1: f32,
    max: vec3<f32>,
    pad2: f32,
} // align(16)

struct FroxelConfig { // Ensure this matches Rust exactly
    screen_width: u32,
    screen_height: u32,
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
}

struct SegmentRef { // Ensure this matches Rust if defined there
    strand_idx: u32,
    segment_start_idx: u32, // Index into index buffer
}

struct StrandGeo {
    strand_count: u32,
    max_segments_in_strand: u32,
    pad1: u32,
    pad2: u32,
    aabb: Aabb,
} // align(16)

struct StrandMeta { // Ensure this matches Rust exactly
    count: u32,     // Number of vertices in strand
    offset: u32,    // Start index in the original indices buffer (or vertices buffer?)
    pad1: u32,      // Padding for alignment
    pad2: u32,      // Padding for alignment
}

struct PushConstants { // Ensure this matches Rust and range covers all fields
    workgroup_offset: u32, // For dispatch_workgroup_ext compatibility
    num_elements: u32,    // Generic count (e.g., num_strands or num_tiles)
    scan_load_base: u32,
    scan_save_base: u32,
    // Add other needed constants
}
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

fn find_znear_zfar(clip_from_world: mat4x4<f32>, world_aabb_min: vec3<f32>, world_aabb_max: vec3<f32>) -> vec2<f32> {
    var transformed_z = vec2<f32>(99999.0, -99999.0); // initial min/max placeholders
    let corners = array<vec4<f32>, 8>(
        vec4(world_aabb_min, 1.0),
        vec4(world_aabb_min.x, world_aabb_min.y, world_aabb_max.z, 1.0),
        vec4(world_aabb_min.x, world_aabb_max.y, world_aabb_min.z, 1.0),
        vec4(world_aabb_min.x, world_aabb_max.y, world_aabb_max.z, 1.0),
        vec4(world_aabb_max.x, world_aabb_min.y, world_aabb_min.z, 1.0),
        vec4(world_aabb_max.x, world_aabb_min.y, world_aabb_max.z, 1.0),
        vec4(world_aabb_max.x, world_aabb_max.y, world_aabb_min.z, 1.0),
        vec4(world_aabb_max, 1.0)
    );
    for (var i = 0u; i < 8u; i = i + 1u) {
        let clip_pos = clip_from_world * corners[i];
        // For orthographic projections, w will usually be 1.0;
        // for perspective, do division if needed:
        let view_pos = clip_pos.xyz / clip_pos.w;
        // Update min and max for the z component:
        transformed_z.x = min(transformed_z.x, view_pos.z);
        transformed_z.y = max(transformed_z.y, view_pos.z);
    }
    return transformed_z;
}

fn world_to_screen(position: vec4<f32>, clip_from_world: mat4x4<f32>, screen_width: f32, screen_height: f32, aabb_znear_zfar: vec2<f32>) -> vec3<f32> {
    // Transform from world to clip space using the view-projection matrix
    let clip_pos = clip_from_world * position;

    if clip_pos.w <= 0.0 {
        // Handle point behind camera
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    
    // Perform perspective division to get NDC coordinates
    let ndc = clip_pos.xyz / clip_pos.w;
    
    // Convert NDC to screen coordinates for X and Y
    let screen_x = (ndc.x * 0.5 + 0.5) * screen_width;
    let screen_y = (ndc.y * -0.5 + 0.5) * screen_height; // Flip Y for top-left origin
    
    // Extract view-space Z of the position
    let view_pos = clip_from_world * position;
    let view_z = view_pos.z / view_pos.w;
    
    // Ensure proper ordering (min should be closer to camera)
    let z_near = aabb_znear_zfar.x;
    let z_far = aabb_znear_zfar.y;
    
    // Normalize the depth within the AABB Z range
    let normalized_depth = (view_z - z_near) / max(z_far - z_near, 0.0001);
    
    // Clamp to ensure we stay in the [0,1] range even if point is outside AABB
    let screen_z = clamp(normalized_depth, 0.0, 1.0);

    return vec3<f32>(screen_x, screen_y, 1.0 - screen_z);
}

fn calculate_froxel_index(x: u32, y: u32, z: u32, config: FroxelConfig) -> u32 {
    let froxels_x = (config.screen_width + config.froxel_size_x - 1u) / config.froxel_size_x;
    let froxels_y = (config.screen_height + config.froxel_size_y - 1u) / config.froxel_size_y;
    // Clamp coordinates to valid range before calculating index
    let clamped_x = min(x, froxels_x - 1u);
    let clamped_y = min(y, froxels_y - 1u);
    let clamped_z = min(z, config.depth_slices - 1u);
    return clamped_z * froxels_x * froxels_y + clamped_y * froxels_x + clamped_x;
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

// #define DEBUG

#ifdef SHADOWS 
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_ARRAY}) var deep_opacity_maps: texture_storage_2d_array<rg32float, write>; // TODO: maybe find a more compact format
@group(0) @binding(#{CLUSTER_INDICES}) var<storage> clusterable_object_index_lists: types::ClusterLightIndexLists;
@group(0) @binding(#{CLUSTERABLE_OBJECTS}) var<storage> clusterable_objects: types::ClusterableObjects;
@group(0) @binding(#{CLUSTER_OFFSETS_AND_COUNTS}) var<storage> cluster_offsets_and_counts: types::ClusterOffsetsAndCounts;
@group(0) @binding(#{POINT_LIGHT_DEPTH_TEXTURE}) var point_shadow_textures_linear_sampler: sampler;
@group(0) @binding(#{DIRECTIONAL_LIGHT_DEPTH_TEXTURE}) var directional_shadow_textures_linear_sampler: sampler;

@compute @workgroup_size(8, 8, 1) // TODO: Should match froxel_size_x, froxel_size_y
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
    let aabb_znear_zfar = find_znear_zfar(clip_from_world, geo.aabb.min, geo.aabb.max);

    // DEBUG
    // for (var dz: u32 = 0; dz < u32(config.depth_slices); dz = dz + 1) { 
    //     let frag_count = tile_counts_buffer[calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config)];
    //     let debug_color = vec4<f32>(heatmap_precise(f32(frag_count) / f32(pc.num_elements)), 0.2);
    //     textureStore(deep_opacity_maps, pixel_coord_int, dz, vec4<f32>(debug_color.xy, 0.0, 0.0));
    // }
    // Debug: Output the tile_count of this tile divided by num_elements

    var opacity_acc = 0.0;

    for (var dz: u32 = 0; dz < config.depth_slices; dz = dz + 1) {

        let froxel_idx = calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config);

        // bounds check
        if (froxel_idx >= arrayLength(&tile_offsets_buffer)) { continue; }

        let start_segment_offset = tile_offsets_buffer[froxel_idx];
        let segment_count_in_froxel = tile_counts_buffer[froxel_idx];

        // first component is the depth, second is the opacity
        var froxel_depth = 1e6; // Initialize to a large depth

        // Process all segments within this froxel
        for (var s: u32 = 0; s < segment_count_in_froxel; s = s + 1u) {
            if s >= 300 { break; } // Limit number of segments processed per froxel
            let packed_buffer_idx = start_segment_offset + s;
             // Safety check packed buffer bounds
            if (packed_buffer_idx >= arrayLength(&packed_segments_buffer)) { continue; }

            let segment_ref = packed_segments_buffer[packed_buffer_idx];

            // Get strand metadata
            let strand_idx = segment_ref.strand_idx;
            if (strand_idx >= arrayLength(&strand_metadata)) { 
                froxel_depth = 1.0; // Debug color
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

            // Project to light space (pixels)
            let p0_screen = world_to_screen(v0_world, clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);
            let p1_screen = world_to_screen(v1_world, clip_from_world, f32(config.screen_width), f32(config.screen_height), aabb_znear_zfar);

            // Skip if segment is fully behind camera or off-screen after projection
            // if (p0_screen.x < 0.0 && p1_screen.x < 0.0) { 
            //     // let debug_color = vec4<f32>(1.0, 1.0, 0.0, 1.0); // Debug color
            //     // textureStore(deep_opacity_maps, pixel_coord_int, dz, debug_color);
            //     continue;
            //  } // Basic culling

            // Calculate analytical coverage
            let t = fragment_position_line_relative(pixel_center, p0_screen.xy, p1_screen.xy);
            let dist = point_segment_distance(pixel_center, p0_screen.xy, p1_screen.xy, t);

            // blend hair radius from MIN to MAX based on distance to camera
            let r = 1.0; //mix(MIN_HAIR_RADIUS_PIXELS, MAX_HAIR_RADIUS_PIXELS, (p0_screen.z + p1_screen.z) / 2.0); // TODO: this needs special scaling for lights.

            // Simple linear falloff based on distance
            let coverage = clamp(1.0 - dist / r, 0.0, 1.0);

            if (coverage > 0.0) {
                // Calculate color/alpha contribution of this hair segment fragment
                opacity_acc = opacity_acc + (1.0 - opacity_acc) * coverage * 0.5; // Accumulate opacity TODO: use hair opacity instead of 0.5
                let depth = mix(p0_screen.z, p1_screen.z, clamp(t, 0.0, 1.0));
                froxel_depth = min(depth, froxel_depth); // Update depth
            }
        } // End loop over segments in froxel

        // If pixel becomes nearly opaque, we can stop processing deeper Z slices
        textureStore(deep_opacity_maps, pixel_coord_int, dz, vec4<f32>(opacity_acc, froxel_depth, 0.0, 0.0));
        // --- Optional Early Exit ---
        // if (opacity_acc.y > 0.9999) {
        //     break; // Stop Z loop
        // }
    }
}

#else

@group(0) @binding(#{DEEP_OPACITY_TEXTURE_ARRAY}) var deep_opacity_sampler: sampler; // TODO: maybe find a more compact format
@group(0) @binding(#{DEEP_OPACITY_TEXTURE_VIEW}) var deep_opacity_maps: texture_2d_array<f32>;

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

    #ifdef DEBUG
        var frag_count: u32 = 0;
        for (var dz: u32 = 0; dz < u32(config.depth_slices); dz = dz + 1) {
            frag_count += tile_counts_buffer[calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config)];
        }
        // Debug: Output the tile_count of this tile divided by num_elements
        let debug_color = vec4<f32>(heatmap_precise(f32(frag_count) / f32(pc.num_elements)), 0.2);
    # endif // DEBUG
    
    let geo = geos[0]; // Assuming only one strand geo for now
    let aabb_znear_zfar = find_znear_zfar(view.clip_from_world, geo.aabb.min, geo.aabb.max);

    // light relative data
    let light: types::DirectionalLight = lights.directional_lights[LIGHT_INDEX];
    let cascade = light.cascades[0]; // TODO: select cascade based on distance
    let light_aabb_znear_zfar = find_znear_zfar(cascade.clip_from_world, geo.aabb.min, geo.aabb.max);
    let shadow_map_dims = vec2<f32>(textureDimensions(deep_opacity_maps).xy);


    for (var dz: u32 = 0; dz < config.depth_slices; dz = dz + 1) {

        let froxel_idx = calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config);

        // bounds check
        if (froxel_idx >= arrayLength(&tile_offsets_buffer)) { continue; }

        let start_segment_offset = tile_offsets_buffer[froxel_idx];
        let segment_count_in_froxel = tile_counts_buffer[froxel_idx];

        var froxel_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        // Process all segments within this froxel
        for (var s: u32 = 0; s < segment_count_in_froxel; s = s + 1u) {
            if s >= 300 { break; } // Limit number of segments processed per froxel
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

                // transform the fragment into light space
                let p0_light = world_to_screen(v0_world, cascade.clip_from_world, shadow_map_dims.x, shadow_map_dims.y, light_aabb_znear_zfar);
                let p1_light = world_to_screen(v1_world, cascade.clip_from_world, shadow_map_dims.x, shadow_map_dims.y, light_aabb_znear_zfar);

                // Calculate the light space coordinates
                let fragment_light = mix(p0_light, p1_light, clamp(t, 0.0, 1.0));
                let dom_val = textureSampleLevel(deep_opacity_maps, deep_opacity_sampler, fragment_light.xy / shadow_map_dims.xy, u32(16.0 * (1.0 - fragment_light.z)), 0.0);

                let hair_fragment = vec4<f32>(hair_color.xyz * (dom_val.x), hair_color.w * coverage);

                // transmittance accumulation
                froxel_color += hair_fragment;
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

#endif // SHADOWS vs camera