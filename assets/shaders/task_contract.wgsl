// Queue/task ABI contract shared across work-preparation phases.
// Keep these layouts stable so host orchestration can migrate from
// multi-dispatch to a persistent kernel without memory format churn.
//
// Binning hierarchy contract (current design target):
// - level 0: coarse insertion over view-level macro tiles
// - level i>0: refine from parent tile id in id_info
// - level == BINNING_NUM_LEVELS-1: emit raster-ready work
//
// BinningTask field semantics:
// - id_info
//   - level 0: instance/geo id
//   - level i>0: parent tile id from previous level
// - chunk_id
//   - optional chunk-pool pointer / chain head for batched local commit
// - seg_idx
//   - stable segment identifier
// - packed_field bit layout:
//   - [0..1] level
//   - [2] is_shadow
//   - [3..31] frustum index

const PHASE_BROAD: u32 = 0u;
const PHASE_FINE: u32 = 1u;
const PHASE_BINNING: u32 = 2u;
const BINNING_NUM_LEVELS: u32 = 2u;

const BINNING_LEVEL_BITS: u32 = 2u;
const BINNING_LEVEL_MASK: u32 = (1u << BINNING_LEVEL_BITS) - 1u;
const BINNING_SHADOW_BIT: u32 = 1u << BINNING_LEVEL_BITS;
const BINNING_FRUSTUM_SHIFT: u32 = BINNING_LEVEL_BITS + 1u;

struct QueueHeader {
    head: atomic<u32>,
    tail: atomic<u32>,
};

struct FinePrepassTask {
    inst_id: u32,
    strand_local: u32,
};

struct BinningTask {
    id_info: u32,
    chunk_id: u32,
    seg_idx: u32,
    packed_field: u32,
};

fn pack_binning_field(level: u32, is_shadow: u32, frustum_index: u32) -> u32 {
    return (level & BINNING_LEVEL_MASK) | ((is_shadow & 1u) << BINNING_LEVEL_BITS) | (frustum_index << BINNING_FRUSTUM_SHIFT);
}

fn unpack_binning_level(packed: u32) -> u32 {
    return packed & BINNING_LEVEL_MASK;
}

fn unpack_binning_shadow(packed: u32) -> u32 {
    return (packed & BINNING_SHADOW_BIT) >> BINNING_LEVEL_BITS;
}

fn unpack_binning_frustum(packed: u32) -> u32 {
    return packed >> BINNING_FRUSTUM_SHIFT;
}

struct RasterWorkItem {
    seg_id: u32,
    strand_id: u32,
    frustum_id: u32,
    tile_id: u32,
}



// Future work:

// struct FinePrepassTask {
//     // 0–23  (6 * 4)
//     geo_id:                 u32, // 0
//     strand_id:              u32, // 4
//     segment_base:           u32, // 8   (first segment this task covers)
//     segment_count:          u32, // 12  (how many segments to process)
//     cluster_xyz_packed:     u32, // 16  (x:10, y:10, z:10, 2 bits spare)
//     cli_base24_count8:      u32, // 20  (base into ClusterLightIndexLists, count <= 255, you will keep <=32)

//     // 24–55  (8 * 4 = 32 B)
//     // Compact local light list: 32 entries max, each an 8-bit index into ClusterableObjects
//     // local_light_indices[k] packs 4 indices (i0..i3) as bytes
//     local_light_indices:    array<u32, 8>, // 24..55  (covers up to 32 lights)

//     // 56–71  (4 * 4)
//     active_mask:            u32, // 56  bit i => local entry i is used for this item/view
//     shadow_mask:            u32, // 60  subset of active, bit i casts shadows (any quality)
//     hq_shadow_mask:         u32, // 64  subset of shadow_mask, <= 4 bits set
//     dir_mask:               u32, // 68  directional lights used (bits 0..n_dir-1)

//     // 72–107 (8 * 4 = 32 B)
//     // Per-light importance/LOD hints (0..255), packed 4x u8 per u32, aligned with local_light_indices
//     importance4:            array<u32, 8>, // 72..103 (0=skip fast, 255=process first)
//     dir_shadow_mask:        u32,           // 104 directional lights that cast shadows

//     // 108–127 (5 * 4 = 20 B)
//     spot_bias4:             u32, // 108 (optional: 4* u8 categorical bias modifiers for spot cutoff/PCF size buckets)
//     flags:                  u32, // 112 task-level flags (see below)
//     reserved0:              u32, // 116
//     reserved1:              u32, // 120
//     reserved2:              u32, // 124
// }; // align 128 -> 1 Cacheline


// Helpers: pack/unpack 4x u8 into/from a u32
// fn pack4xU8(a: u32, b: u32, c: u32, d: u32) -> u32 {
//     return (a & 0xFFu) |
//            ((b & 0xFFu) << 8u) |
//            ((c & 0xFFu) << 16u) |
//            ((d & 0xFFu) << 24u);
// }

// fn unpackU8(v: u32, i: u32) -> u32 {
//     // i in [0..3]
//     return (v >> (8u * i)) & 0xFFu;
// }

// // Pack cluster coords (x:10, y:10, z:10) into one u32 (2 bits spare)
// fn packClusterXYZ(x: u32, y: u32, z: u32) -> u32 {
//     return (x & 0x3FFu) | ((y & 0x3FFu) << 10u) | ((z & 0x3FFu) << 20u);
// }
// fn unpackClusterX(v: u32) -> u32 { return v & 0x3FFu; }
// fn unpackClusterY(v: u32) -> u32 { return (v >> 10u) & 0x3FFu; }
// fn unpackClusterZ(v: u32) -> u32 { return (v >> 20u) & 0x3FFu; }

// // Pack base(24 bits) + count(8 bits) to mirror your ClusterOffsetsAndCounts "compressed" format.
// fn packBase24Count8(base24: u32, count8: u32) -> u32 {
//     return ((base24 & 0xFFFFFFu) << 8u) | (count8 & 0xFFu);
// }
// fn unpackBase24(v: u32) -> u32 { return (v >> 8u) & 0xFFFFFFu; }
// fn unpackCount8(v: u32) -> u32 { return v & 0xFFu; }
