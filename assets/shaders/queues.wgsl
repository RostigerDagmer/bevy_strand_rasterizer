
#import "shaders/task_contract.wgsl"::{
    FinePrepassTask,
    BinningTask,
}

struct FinePrepassQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    tasks: array<FinePrepassTask>,
};

struct BinningQueue {
    head: atomic<u32>,
    tail: atomic<u32>,
    tasks: array<BinningTask>,
};
