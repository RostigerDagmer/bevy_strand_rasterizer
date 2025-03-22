// Constants
const MAX_STRANDS_PER_FROXEL: u32 = 256u;
const FROXEL_SIZE_X: u32 = 8u;
const FROXEL_SIZE_Y: u32 = 8u;
const DEPTH_SLICES: u32 = 16u;

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
};

struct Froxel {
    strand_count: atomic<u32>,
    strand_indices: array<u32, MAX_STRANDS_PER_FROXEL>,
};

// Bindings
@group(0) @binding(0) var<storage, read> vertices: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> strand_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> froxels: array<Froxel>;
@group(0) @binding(3) var<uniform> view_proj: mat4x4<f32>;
@group(0) @binding(4) var<uniform> instance_transform: mat4x4<f32>;
@group(0) @binding(5) var<uniform> froxel_config: FroxelConfig;

// Helper function to calculate froxel index from 3D coordinates
fn calculate_froxel_index(x: u32, y: u32, z: u32) -> u32 {
    let froxels_x = (froxel_config.screen_width + froxel_config.froxel_size_x - 1u) / froxel_config.froxel_size_x;
    let froxels_y = (froxel_config.screen_height + froxel_config.froxel_size_y - 1u) / froxel_config.froxel_size_y;
    
    return z * froxels_x * froxels_y + y * froxels_x + x;
}

// Helper function to transform vertex to screen space
fn world_to_screen(position: vec3<f32>) -> vec3<f32> {
    // Apply instance transform
    let world_pos = (instance_transform * vec4<f32>(position, 1.0)).xyz;
    
    // Apply view-projection matrix
    let clip_pos = view_proj * vec4<f32>(world_pos, 1.0);
    
    // Convert to NDC space
    let ndc = clip_pos.xyz / clip_pos.w;
    
    // Convert to screen space
    let screen_x = (ndc.x * 0.5 + 0.5) * f32(froxel_config.screen_width);
    let screen_y = (ndc.y * 0.5 + 0.5) * f32(froxel_config.screen_height);
    
    // Keep z in NDC space [0, 1] for depth slicing
    let screen_z = ndc.z * 0.5 + 0.5;
    
    return vec3<f32>(screen_x, screen_y, screen_z);
}

// Function to add a strand to a froxel
fn add_strand_to_froxel(froxel_x: u32, froxel_y: u32, froxel_z: u32, strand_id: u32) -> bool {
    // Check if within AABB bounds
    if (froxel_x < froxel_config.aabb_min_x || froxel_x > froxel_config.aabb_max_x ||
        froxel_y < froxel_config.aabb_min_y || froxel_y > froxel_config.aabb_max_y ||
        froxel_z < u32(froxel_config.aabb_min_z * f32(froxel_config.depth_slices)) || 
        froxel_z > u32(froxel_config.aabb_max_z * f32(froxel_config.depth_slices))) {
        return false;
    }
    
    let froxel_idx = calculate_froxel_index(froxel_x, froxel_y, froxel_z);
    
    // Atomically increment strand count
    let prev_count = atomicAdd(&froxels[froxel_idx].strand_count, 1u);
    
    // Check if we have space
    if (prev_count < MAX_STRANDS_PER_FROXEL) {
        froxels[froxel_idx].strand_indices[prev_count] = strand_id;
        return true;
    }
    
    // No space left, decrement count
    atomicSub(&froxels[froxel_idx].strand_count, 1u);
    return false;
}

// 3D Bresenham-like algorithm to trace a line through froxels
fn trace_segment_through_froxels(p0: vec3<f32>, p1: vec3<f32>, strand_id: u32) {
    // Convert to froxel coordinates
    let froxel_x0 = u32(p0.x / f32(froxel_config.froxel_size_x));
    let froxel_y0 = u32(p0.y / f32(froxel_config.froxel_size_y));
    let froxel_z0 = u32(p0.z * f32(froxel_config.depth_slices));
    
    let froxel_x1 = u32(p1.x / f32(froxel_config.froxel_size_x));
    let froxel_y1 = u32(p1.y / f32(froxel_config.froxel_size_y));
    let froxel_z1 = u32(p1.z * f32(froxel_config.depth_slices));
    
    // Simple case: if both endpoints are in the same froxel
    if (froxel_x0 == froxel_x1 && froxel_y0 == froxel_y1 && froxel_z0 == froxel_z1) {
        add_strand_to_froxel(froxel_x0, froxel_y0, froxel_z0, strand_id);
        return;
    }
    
    // 3D Bresenham-like algorithm
    // This is a simplified version - a full implementation would be more complex
    // but would ensure all intersected froxels are visited
    
    // Start at p0
    var x = froxel_x0;
    var y = froxel_y0;
    var z = froxel_z0;
    
    // Calculate deltas
    let dx = i32(froxel_x1) - i32(froxel_x0);
    let dy = i32(froxel_y1) - i32(froxel_y0);
    let dz = i32(froxel_z1) - i32(froxel_z0);
    
    // Calculate step directions
    let sx = select(-1, 1, dx > 0);
    let sy = select(-1, 1, dy > 0);
    let sz = select(-1, 1, dz > 0);
    
    // Calculate absolute deltas
    let adx = abs(dx);
    let ady = abs(dy);
    let adz = abs(dz);
    
    // Find the dominant axis
    let max_delta = max(adx, max(ady, adz));
    
    // Trace along the line
    for (var i = 0; i <= max_delta; i++) {
        add_strand_to_froxel(x, y, z, strand_id);
        
        // Move along the line
        if (i * adx / max_delta != (i - 1) * adx / max_delta) {
            x = u32(i32(x) + sx);
        }
        if (i * ady / max_delta != (i - 1) * ady / max_delta) {
            y = u32(i32(y) + sy);
        }
        if (i * adz / max_delta != (i - 1) * adz / max_delta) {
            z = u32(i32(z) + sz);
        }
    }
}

// Main compute shader function
@compute @workgroup_size(64, 1, 1)
fn bin_strands(@builtin(global_invocation_id) id: vec3<u32>) {
    let strand_id = id.x;
    
    // Find the start of this strand in the indices buffer
    // This would require a separate strand metadata buffer in a real implementation
    // For now, we'll assume strands are stored sequentially with u32::MAX separators
    
    var index_pos = 0u;
    var current_strand = 0u;
    
    // Skip to the current strand
    while (current_strand < strand_id) {
        // Find the next u32::MAX separator
        while (index_pos < arrayLength(&strand_indices) && strand_indices[index_pos] != 0xFFFFFFFFu) {
            index_pos++;
        }
        index_pos++; // Skip the separator
        current_strand++;
    }
    
    // Now we're at the start of our strand
    var strand_start = index_pos;
    var prev_vertex_id: u32 = 0xFFFFFFFFu;
    var prev_screen_pos: vec3<f32>;
    
    // Process each segment in the strand
    while (index_pos < arrayLength(&strand_indices) && strand_indices[index_pos] != 0xFFFFFFFFu) {
        let vertex_id = strand_indices[index_pos];
        let screen_pos = world_to_screen(vertices[vertex_id]);
        
        // If we have a previous vertex, we can form a segment
        if (prev_vertex_id != 0xFFFFFFFFu) {
            trace_segment_through_froxels(prev_screen_pos, screen_pos, strand_id);
        }
        
        // Update previous vertex
        prev_vertex_id = vertex_id;
        prev_screen_pos = screen_pos;
        
        index_pos++;
    }
}