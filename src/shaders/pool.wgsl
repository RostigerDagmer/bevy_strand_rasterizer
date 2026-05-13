

const CHUNK_SIZE: u32 = #{POOL_CHUNK_SIZE};
const N_HEADS: u32 = #{POOL_NUM_HEADS};

struct Chunk {
    next: u32,
    count: u32,
    items: array<u32, CHUNK_SIZE>,
}

@group(0) @binding(#{CHUNK_ALLOCATOR_POOL}) var<storage, read_write> chunk_pool : array<Chunk>;
@group(0) @binding(#{CHUNK_ALLOCATOR_HEAD}) var<storage, read_write> free_heads  : array<atomic<u32>, N_HEADS>; // lock-free stacks


fn alloc(head_id: u32, size: u32) -> vec2<u32> {
    let n_chunks = max(1u, size / CHUNK_SIZE);
    let base = atomicAdd(&free_heads[head_id], n_chunks);
    return vec2<u32>(arrayLength(&chunk_pool) / N_HEADS * head_id + base, n_chunks);
}
