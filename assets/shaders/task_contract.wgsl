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
    return (level & BINNING_LEVEL_MASK) |
        ((is_shadow & 1u) << BINNING_LEVEL_BITS) |
        (frustum_index << BINNING_FRUSTUM_SHIFT);
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
