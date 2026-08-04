# Squirrel raster optimization log

All measurements use the full `squirrel` case at 3840x2160 on an NVIDIA GeForce RTX 5090
(Vulkan, NVIDIA driver 610.43.03), with 120 warmup frames and 100 measured frames.

```sh
target/release/examples/strand_bench \
  --case squirrel --headless --width 3840 --height 2160 \
  --warmup 120 --samples 100 --output <result.json>
```

| Revision | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Median vs. baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8.763 | 8.394 | 10.611 | 11.087 | - |
| Hot-loop cleanups | 8.548 | 7.818 | 10.419 | 11.391 | -6.9% |
| Direct subgroup gather (rejected) | 9.364 | 8.612 | 11.847 | 12.497 | +2.6% |
| Subgroup tree reduction | 7.561 | 6.899 | 9.186 | 9.623 | -17.8% |

## Changes

- `01-baseline.json`: original shared-memory reduction.
- `02-hot-loop-cleanups.json`: avoids a redundant square root in line projection, clears only
  the batch validity flag, and removes unused segment-local work.
- `03-subgroup-reduction.json`: direct eight-shuffle gather. This was slower and is retained only
  as an experimental result.
- `04-subgroup-tree-reduction.json`: associative two-stage reduction for each four-lane pixel
  group. It removes the shared partial buffers and reduces synchronization from nine workgroup
  barriers per 16 references to two.

The final implementation is the hot-loop cleanup plus subgroup tree reduction. Percentiles are
not paired samples, so small differences between unrelated prepass stages should be treated as
run-to-run noise.
