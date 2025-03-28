// Constants
#import bevy_render::view::View
const MAX_STRANDS_PER_FROXEL: u32 = 256u;

// Structures
struct FroxelConfig {
    // Screen dimensions
    screen_width: u32,
    screen_height: u32,
    
    // Froxel dimensions
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
    
    // AABB in screen space (for optimization)
    aabb_min_x: u32,
    aabb_min_y: u32,
    aabb_min_z: f32,
    aabb_max_x: u32,
    aabb_max_y: u32,
    aabb_max_z: f32,
    // Padding
    // _padding: vec4<u32>,
}

struct FroxelTile {
    // Each froxel contains a counter followed by strand indices
    strand_count: atomic<u32>,
    strand_indices: array<u32, MAX_STRANDS_PER_FROXEL>,
}

struct PushConstants {
    // Strand count
    strand_count: u32,
}

// Bindings
@group(0) @binding(0) var<storage, read> vertices: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> strand_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> froxel_data: array<FroxelTile>;
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> config: FroxelConfig;
@group(0) @binding(5) var<uniform> view: View;

var<push_constant> push_constants: PushConstants;

// Helper function to calculate froxel index from 3D coordinates
fn calculate_froxel_index(x: u32, y: u32, z: u32) -> u32 {
    let froxels_x = (config.screen_width + config.froxel_size_x - 1u) / config.froxel_size_x;
    let froxels_y = (config.screen_height + config.froxel_size_y - 1u) / config.froxel_size_y;
    
    return z * froxels_x * froxels_y + y * froxels_x + x;
}

// Helper function to get the counter for a froxel
fn get_froxel_offset(froxel_idx: u32) -> u32 {
    return froxel_idx * (1u + MAX_STRANDS_PER_FROXEL);
}

// Helper function to get the strand indices array for a froxel
fn get_froxel_strand_offset(froxel_idx: u32) -> u32 {
    return froxel_idx * (1u + MAX_STRANDS_PER_FROXEL);
}

// Helper function to transform vertex to screen space
fn world_to_screen(position: vec3<f32>) -> vec3<f32> {
    // Apply view-projection matrix
    let clip_pos = view.view_from_world * vec4<f32>(position, 1.0);
    
    // Convert to NDC space
    let ndc = clip_pos.xyz / clip_pos.w;
    
    // Convert to screen space (pixels)
    let screen_x = (ndc.x * 0.5 + 0.5) * f32(config.screen_width);
    let screen_y = (ndc.y * 0.5 + 0.5) * f32(config.screen_height);
    
    // Keep z in NDC space [0, 1] for depth slicing
    let screen_z = ndc.z * 0.5 + 0.5;
    
    return vec3<f32>(screen_x, screen_y, screen_z);
}

// Function to add a strand to a froxel
fn add_strand_to_froxel(froxel_x: u32, froxel_y: u32, froxel_z: u32, strand_id: u32) -> bool {
    // Check if within AABB bounds
    if (froxel_x < config.aabb_min_x || froxel_x > config.aabb_max_x ||
        froxel_y < config.aabb_min_y || froxel_y > config.aabb_max_y ||
        f32(froxel_z) / f32(config.depth_slices) < config.aabb_min_z || 
        f32(froxel_z) / f32(config.depth_slices) > config.aabb_max_z) {
        return false;
    }
    
    let froxel_idx = calculate_froxel_index(froxel_x, froxel_y, froxel_z);
    // let counter_offset = get_froxel_offset(froxel_idx);
    // let counter_ptr = &froxel_data[froxel_idx].data[counter_offset];
    let counter_ptr = &froxel_data[froxel_idx].strand_count;
    
    // Atomically increment strand count and get previous value
    let prev_count = atomicAdd(counter_ptr, 1u);
    
    // Check if we have space
    if (prev_count < MAX_STRANDS_PER_FROXEL) {
        // Add strand index to froxel
        let strand_offset = get_froxel_strand_offset(counter_offset);
        // atomicStore(&froxel_data[froxel_idx].data[strand_offset + 1u + prev_count], strand_id);
        atomicStore(&froxel_data[strand_offset + 1u + prev_count], strand_id);
        return true;
    }
    
    // No space left, decrement count
    atomicSub(counter_ptr, 1u);
    return false;
}

// 3D Bresenham-like algorithm to trace a line through froxels
fn trace_segment_through_froxels(p0: vec3<f32>, p1: vec3<f32>, strand_id: u32) {
    // Convert to froxel coordinates
    let froxel_x0 = u32(p0.x / f32(config.froxel_size_x));
    let froxel_y0 = u32(p0.y / f32(config.froxel_size_y));
    let froxel_z0 = u32(p0.z * f32(config.depth_slices));
    
    let froxel_x1 = u32(p1.x / f32(config.froxel_size_x));
    let froxel_y1 = u32(p1.y / f32(config.froxel_size_y));
    let froxel_z1 = u32(p1.z * f32(config.depth_slices));
    
    // Simple case: if both endpoints are in the same froxel
    if (froxel_x0 == froxel_x1 && froxel_y0 == froxel_y1 && froxel_z0 == froxel_z1) {
        add_strand_to_froxel(froxel_x0, froxel_y0, froxel_z0, strand_id);
        return;
    }
    
    // 3D line traversal using a modified DDA algorithm
    // DDA (Digital Differential Analyzer) is simpler to implement than 3D Bresenham
    
    // Calculate delta and step direction
    let dx = f32(froxel_x1) - f32(froxel_x0);
    let dy = f32(froxel_y1) - f32(froxel_y0);
    let dz = f32(froxel_z1) - f32(froxel_z0);
    
    // Calculate step size
    let steps = max(abs(dx), max(abs(dy), abs(dz)));
    
    if (steps < 1.0) {
        // Degenerate case
        return;
    }
    
    // Calculate increment per step
    let x_inc = dx / steps;
    let y_inc = dy / steps;
    let z_inc = dz / steps;
    
    // Start position
    var x = f32(froxel_x0);
    var y = f32(froxel_y0);
    var z = f32(froxel_z0);
    
    // Add first froxel
    add_strand_to_froxel(froxel_x0, froxel_y0, froxel_z0, strand_id);
    
    // Walk along the line
    for (var i = 0u; f32(i) < steps; i = i + 1u) {
        x = x + x_inc;
        y = y + y_inc;
        z = z + z_inc;
        
        let froxel_x = u32(x);
        let froxel_y = u32(y);
        let froxel_z = u32(z);
        
        // Only add if different from the previous froxel
        if (froxel_x != froxel_x0 || froxel_y != froxel_y0 || froxel_z != froxel_z0) {
            add_strand_to_froxel(froxel_x, froxel_y, froxel_z, strand_id);
        }
    }
}

// Main compute shader function
@compute @workgroup_size(64, 1, 1)
fn bin_strands(@builtin(global_invocation_id) id: vec3<u32>) {
    var strand_id = id.x;
    
    if strand_id >= push_constants.strand_count {
        return;
    }
    
    // Process strand segments
    var index_pos = 0u;
    var in_strand = false;
    var prev_vertex_id: u32 = 0u;
    var prev_screen_pos: vec3<f32>;
    var safety: u32 = 0u;
    // Scan through index buffer to find strands
    // We're using u32::MAX as a separator between strands
    while (index_pos < arrayLength(&strand_indices) && safety < 100u) {
        safety += 1u;
        let index = strand_indices[index_pos];
        
        if (index == 0xFFFFFFFFu) {
            // End of strand marker
            in_strand = false;
            index_pos += 1u;
            continue;
        }
        
        // If this is the start of a new strand, check if it's our assigned strand
        if (!in_strand) {
            // Skip this strand if it's not our assigned ID
            if (strand_id > 0u) {
                // Skip to next strand
                while (index_pos < arrayLength(&strand_indices) && strand_indices[index_pos] != 0xFFFFFFFFu && safety < 100u) {
                    safety += 1u;
                    index_pos += 1u;
                }
                strand_id -= 1u;
                continue;
            }
            
            // This is our strand to process
            in_strand = true;
            prev_vertex_id = index;
            prev_screen_pos = world_to_screen(vertices[prev_vertex_id]);
            index_pos += 1u;
            continue;
        }
        
        // Process segment
        let current_vertex_id = index;
        let current_screen_pos = world_to_screen(vertices[current_vertex_id]);
        
        trace_segment_through_froxels(prev_screen_pos, current_screen_pos, strand_id);
        
        // Move to next vertex
        prev_vertex_id = current_vertex_id;
        prev_screen_pos = current_screen_pos;
        index_pos += 1u;
    }
}