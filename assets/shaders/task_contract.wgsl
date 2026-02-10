// Queue/task ABI contract shared across work-preparation phases.
// Keep these layouts stable so host orchestration can migrate from
// multi-dispatch to a persistent kernel without memory format churn.

const PHASE_BROAD: u32 = 0u;
const PHASE_FINE: u32 = 1u;
const PHASE_BINNING: u32 = 2u;

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
