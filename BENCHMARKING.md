# Strand GPU benchmark

`strand_bench` measures the software-rasterizer prepass stages and raster kernel with WGPU GPU
timestamps. Always use a release build: debug-mode CPU and shader behavior is not representative.

## Running

For an automated real-asset run:

```sh
cargo run --release --example strand_bench -- \
  --case squirrel-body --headless --warmup 120 --samples 300 \
  --output results/squirrel-body.json
```

For the complete split squirrel cache, use `--case squirrel`. An arbitrary cache under this
crate's `assets/` directory can be selected with:

```sh
cargo run --release --example strand_bench -- \
  --case asset --asset path/to/file.strands --headless
```

Use the same workload in an inspectable window with `--display`:

```sh
cargo run --release --example strand_bench -- \
  --case squirrel-body --display --warmup 120 --samples 300
```

Display mode writes and prints the result, then leaves the scene open for inspection. Add
`--exit-after-bench` when a visible run should close automatically.

On Unix, closing a display benchmark uses `_exit` after the report has been written. This avoids
a crash in the current Winit/Vulkan allocator teardown path; benchmark output is flushed before
the process exits. Headless runs use normal Bevy shutdown.

## Workloads

- `squirrel`: every group in the real split squirrel cache, with monolithic-cache fallback.
- `squirrel-body`: the large Body01 group, useful for faster real-asset iteration.
- `asset`: one caller-selected `.strands` cache.
- `clustered-grid`: many entity-local clusters distributed uniformly through a 3D volume. Tune
  it with `--clusters`, `--strands-per-cluster`, and `--points-per-strand`.
- `length-skew`: a deterministic heavy-tailed strand-length distribution.
- `hotspot`: many strands packed into a small world-space area, producing concentrated screen
  occupancy.

Synthetic workloads use `--seed` and are deterministic. Keep the case, seed, resolution, and all
shape arguments fixed when comparing revisions.

## Output

The JSON report records adapter and driver identity, resolution, exact geometry/strand/vertex
counts, workload parameters, and min/mean/median/p90/p95/p99/max GPU milliseconds for every
instrumented dispatch. The measured prepass stages include depth reduction, broad and fine
classification, binning, page allocation/prefixing, segment-reference fill, and raster-work
emission. `render/strand_rasterizer/rasterize/elapsed_gpu` is the indirect raster kernel.

Use a substantial warmup to allow pipeline compilation, allocation, GPU clock ramp, and virtual
surface residency to settle. For tuning, compare medians first and p95/p99 for stability; retain
Nsight Compute for instruction-, occupancy-, cache-, and stall-level diagnosis.

Run `--help` for the complete argument list.
