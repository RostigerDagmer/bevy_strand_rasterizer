// --- Structures ---

struct Vertices {
    vs: array<vec3<f32>>
}

struct Indices {
    is: array<u32>
}

struct Geos {
    gs: array<StrandGeo>
}

struct Meta {
    ms: array<StrandMeta>
}

struct Materials {
    mats: array<StrandMaterial>
}

struct DevicePtr {
    slab: u32,   // index into binding_array
    offset: u32, // byte offset in slab (<= 4 GB if packed in u32)
    size: u32,   // optional; useful for bounds checks
}

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
    offset: u32,
    pad_b: u32,
    aabb: Aabb,
    pad_c: vec4<u32>,
} // align(16)

struct StrandInstance {
    vertex_id: u32,
    index_id: u32,
    meta_id: u32,
    geo_id: u32,
    material_id: u32,
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
    world_from_local: mat4x4<f32>,
    local_from_world: mat4x4<f32>,
}

struct StrandMeta {
    count: u32,     // Number of vertices in strand
    offset: u32,    // Start index in the original indices buffer (or vertices buffer?)
    material_idx: u32, // material index
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
    min_radius_pixels: f32,
    max_radius_pixels: f32,
}

struct PushConstants {
    workgroup_offset: u32, // For dispatch_workgroup_ext compatibility
    num_elements: u32,    // Generic count (e.g., num_strands or num_tiles)
    frustum_count: u32,
    scan_load_base: u32,
    scan_save_base: u32,
    stochastic_cull_enabled: u32,
    target_strands_per_pixel: f32,
    min_keep_probability: f32,
    shadow_keep_probability: f32,
}
