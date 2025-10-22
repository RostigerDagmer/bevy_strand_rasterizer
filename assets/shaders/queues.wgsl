
#import "shaders/types"::{
    FinePrepassTask
}

struct FinePrepassQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    active_producers: atomic<u32>,
    active_consumers: atomic<u32>,
    tasks: array<FinePrepassTask>,
};

struct BinningQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    tasks: array<FinePrepassTask>,
}

struct ChunkHeader {
    next : u32,     // index of next chunk, or 0xFFFFFFFF
    count : u32,    // #valid items
} // align 8

struct SegmentRef {
    seg_id : u32,
    // maybe more per-hit metadata: froxel_id, macro_tile_id, etc.
}

struct Chunk {
    header : ChunkHeader,
    items  : array<SegmentRef, CHUNK_CAP>,
} // align chunk_cap + 8

struct Queue {
    head : atomic<u32>,
    tail : atomic<u32>,
} // align 8

struct Bucket {
    queues: array<Queues, N_SHARDS>
}

struct BinCoarse {
    queues: array<Bucket, N_TILES>
}

struct Results {
    high_prio: Queue,
    med_prio: Queue,
    low_prio: Queue,
}

@group(0) @binding(0) var<storage, read_write> tile_head  : array<atomic<u32>>; // UINT_MAX = empty
@group(0) @binding(0) var<storage, read_write> bin_coarse : array<Bucket>;
