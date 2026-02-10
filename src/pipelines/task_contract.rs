//! Queue/task ABI contract shared across work-preparation phases.
//!
//! Keep these layouts stable so host orchestration can evolve from
//! multi-dispatch to a persistent kernel without refactoring task memory.

/// Broad visibility/candidate generation.
pub const PHASE_BROAD: u32 = 0;
/// Fine segment/frustum expansion.
pub const PHASE_FINE: u32 = 1;
/// Binning/final work emission.
pub const PHASE_BINNING: u32 = 2;

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

pub const QUEUE_HEADER_WORDS: usize =
    std::mem::size_of::<QueueHeader>() / std::mem::size_of::<u32>();
