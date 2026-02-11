//! Queue/task ABI contract shared across work-preparation phases.
//!
//! Keep these layouts stable so host orchestration can evolve from
//! multi-dispatch to a persistent kernel without refactoring task memory.
//!
//! Binning hierarchy contract (current design target):
//! - `BinningTask.level == 0` means coarse insertion over view-level macro tiles.
//! - `BinningTask.level > 0` means refinement using children of `id_info` from the previous level.
//! - `BinningTask.level == BINNING_NUM_LEVELS - 1` emits raster-ready work items.
//!
//! Field semantics:
//! - `id_info`
//!   - level 0: instance/geo id for segment source
//!   - level i>0: parent tile id from level i-1
//! - `chunk_id`
//!   - optional chunk-pool pointer / chain head used for batched local commit
//! - `seg_idx`
//!   - stable global segment identifier (index-space contract chosen by producer)
//! - `packed_field`
//!   - bit [0..1]: level (supports up to 4 levels)
//!   - bit [2]: is_shadow
//!   - bit [3..31]: frustum index

/// Broad visibility/candidate generation.
pub const PHASE_BROAD: u32 = 0;
/// Fine segment/frustum expansion.
pub const PHASE_FINE: u32 = 1;
/// Binning/final work emission.
pub const PHASE_BINNING: u32 = 2;
/// Current hierarchy depth target for queue-based binning.
pub const BINNING_NUM_LEVELS: u32 = 2;
/// Sparse binning chunk size in u32 payload entries.
pub const BINNING_POOL_CHUNK_SIZE: u32 = 32;
/// Number of allocator heads used to spread atomic pressure.
pub const BINNING_POOL_NUM_HEADS: u32 = 64;
/// Minimum number of chunks reserved for sparse binning.
pub const BINNING_POOL_MIN_CHUNKS: u32 = 16_384;

pub const BINNING_LEVEL_BITS: u32 = 2;
pub const BINNING_LEVEL_MASK: u32 = (1 << BINNING_LEVEL_BITS) - 1;
pub const BINNING_SHADOW_BIT: u32 = 1 << BINNING_LEVEL_BITS;
pub const BINNING_FRUSTUM_SHIFT: u32 = BINNING_LEVEL_BITS + 1;

/// Queue header ABI used by GPU work queues.
///
/// Producer side:
/// - reserves with `tail` increments
///
/// Consumer side:
/// - advances `head`
///
/// Reset both to zero before a new frame/phase chain.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct QueueHeader {
    pub head: u32,
    pub tail: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FinePrepassTask {
    pub inst_id: u32,
    pub strand_local: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BinningTask {
    pub id_info: u32,
    pub chunk_id: u32,
    pub seg_idx: u32,
    pub packed_field: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RasterWorkItem {
    pub seg_id: u32,
    pub frustum_id: u32,
    pub tile_id: u32,
}

#[inline]
pub const fn pack_binning_field(level: u32, is_shadow: bool, frustum_index: u32) -> u32 {
    (level & BINNING_LEVEL_MASK)
        | ((is_shadow as u32) << BINNING_LEVEL_BITS)
        | (frustum_index << BINNING_FRUSTUM_SHIFT)
}

#[inline]
pub const fn unpack_binning_level(packed: u32) -> u32 {
    packed & BINNING_LEVEL_MASK
}

#[inline]
pub const fn unpack_binning_shadow(packed: u32) -> bool {
    (packed & BINNING_SHADOW_BIT) != 0
}

#[inline]
pub const fn unpack_binning_frustum(packed: u32) -> u32 {
    packed >> BINNING_FRUSTUM_SHIFT
}

pub const QUEUE_HEADER_WORDS: usize =
    std::mem::size_of::<QueueHeader>() / std::mem::size_of::<u32>();
