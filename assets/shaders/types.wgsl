// --- Structures ---

struct FroxelConfig {
    screen_width: u32,
    screen_height: u32,
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
}

struct SegmentRef {
    strand_idx: u32,
    segment_start_idx: u32, // Index into index buffer
}

struct Aabb {
    min: vec3<f32>,
    pad_a: f32,
    max: vec3<f32>,
    pad_b: f32,
} // align(16)

struct StrandGeo {
    strand_count: u32,
    max_segments_in_strand: u32,
    pad_a: u32,
    pad_b: u32,
    aabb: Aabb,
} // align(16)

struct StrandMeta {
    count: u32,     // Number of vertices in strand
    offset: u32,    // Start index in the original indices buffer (or vertices buffer?)
    pad_a: u32,      // Padding for alignment
    pad_b: u32,      // Padding for alignment
}

struct StrandMaterial {
    absorption_color: vec4<f32>,
    specular_color: vec4<f32>,
    ambient_factor: f32,
    ao_factor: f32,
    eta: f32,
    beta: f32,
    alpha: f32,
    shift: f32,
    pad_a: u32,
    pad_b: u32,
}

struct PushConstants {
    workgroup_offset: u32, // For dispatch_workgroup_ext compatibility
    num_elements: u32,    // Generic count (e.g., num_strands or num_tiles)
    scan_load_base: u32,
    scan_save_base: u32,
    // Add other needed constants
}
