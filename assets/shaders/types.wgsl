// --- Structures ---
struct FinePrepassTask {
    // 0–23  (6 * 4)
    geo_id:                 u32, // 0
    strand_id:              u32, // 4
    segment_base:           u32, // 8   (first segment this task covers)
    segment_count:          u32, // 12  (how many segments to process)
    cluster_xyz_packed:     u32, // 16  (x:10, y:10, z:10, 2 bits spare)
    cli_base24_count8:      u32, // 20  (base into ClusterLightIndexLists, count <= 255, you will keep <=32)

    // 24–55  (8 * 4 = 32 B)
    // Compact local light list: 32 entries max, each an 8-bit index into ClusterableObjects
    // local_light_indices[k] packs 4 indices (i0..i3) as bytes
    local_light_indices:    array<u32, 8>, // 24..55  (covers up to 32 lights)

    // 56–71  (4 * 4)
    active_mask:            u32, // 56  bit i => local entry i is used for this item/view
    shadow_mask:            u32, // 60  subset of active, bit i casts shadows (any quality)
    hq_shadow_mask:         u32, // 64  subset of shadow_mask, <= 4 bits set
    dir_mask:               u32, // 68  directional lights used (bits 0..n_dir-1)

    // 72–107 (8 * 4 = 32 B)
    // Per-light importance/LOD hints (0..255), packed 4x u8 per u32, aligned with local_light_indices
    importance4:            array<u32, 8>, // 72..103 (0=skip fast, 255=process first)
    dir_shadow_mask:        u32,           // 104 directional lights that cast shadows

    // 108–127 (5 * 4 = 20 B)
    spot_bias4:             u32, // 108 (optional: 4* u8 categorical bias modifiers for spot cutoff/PCF size buckets)
    flags:                  u32, // 112 task-level flags (see below)
    reserved0:              u32, // 116
    reserved1:              u32, // 120
    reserved2:              u32, // 124
}; // align 128 -> 1 Cacheline


struct BinningTask {
    // in l0 this is just geo_id
    // in li it becomes a tile index at the previous level
    id_info: u32,
    chunk_id: u32,
    // strand_id: u32,
    seg_idx: u32,
    // bit [0:2]: bin level bits
    // bit [2]: whether this is a render target or a shadow target.
    // bit [3:]: index into the frustrum buffer (FroxelConfig).
    packed_field: u32,
}

fn unpack_binning_field(packed_field: u32) -> vec3<u32> {
    let bin_level = packed_field & 3u;
    let is_shadow = (packed_field >> 2u) & 1u;
    let frustrum_index = (packed_field >> 3u);
    return vec3<u32>(is_shadow, bin_level, frustrum_index);
}

fn pack_binning_field(is_shadow: u32, bin_level: u32, frustrum_index: u32) -> u32 {
    var packed_field = frustrum_index << 3u;
    packed_field = packed_field & (is_shadow << 2u) & bin_level;
    return packed_field;
}

struct RasterWorkItem {
    seg_id: u32,
    frustrum_id: u32,
    tile_id: u32,
}

// Helpers: pack/unpack 4x u8 into/from a u32
fn pack4xU8(a: u32, b: u32, c: u32, d: u32) -> u32 {
    return (a & 0xFFu) |
           ((b & 0xFFu) << 8u) |
           ((c & 0xFFu) << 16u) |
           ((d & 0xFFu) << 24u);
}

fn unpackU8(v: u32, i: u32) -> u32 {
    // i in [0..3]
    return (v >> (8u * i)) & 0xFFu;
}

// Pack cluster coords (x:10, y:10, z:10) into one u32 (2 bits spare)
fn packClusterXYZ(x: u32, y: u32, z: u32) -> u32 {
    return (x & 0x3FFu) | ((y & 0x3FFu) << 10u) | ((z & 0x3FFu) << 20u);
}
fn unpackClusterX(v: u32) -> u32 { return v & 0x3FFu; }
fn unpackClusterY(v: u32) -> u32 { return (v >> 10u) & 0x3FFu; }
fn unpackClusterZ(v: u32) -> u32 { return (v >> 20u) & 0x3FFu; }

// Pack base(24 bits) + count(8 bits) to mirror your ClusterOffsetsAndCounts "compressed" format.
fn packBase24Count8(base24: u32, count8: u32) -> u32 {
    return ((base24 & 0xFFFFFFu) << 8u) | (count8 & 0xFFu);
}
fn unpackBase24(v: u32) -> u32 { return (v >> 8u) & 0xFFFFFFu; }
fn unpackCount8(v: u32) -> u32 { return v & 0xFFu; }

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
} // align(16)

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
