# Squirrel GPU optimization log

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

## Fine-reference fill

| Revision | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Median vs. subgroup baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Subgroup baseline | 3.099 | 3.055 | 3.471 | 3.493 | - |
| Hoist reference construction | 3.137 | 3.075 | 3.490 | 3.504 | +0.7% |
| Remove redundant count read | 2.918 | 2.862 | 3.113 | 3.299 | -6.3% |
| Confirmation run | 2.899 | 2.833 | 3.207 | 3.243 | -7.3% |

## Changes

- `01-baseline.json`: original shared-memory reduction.
- `02-hot-loop-cleanups.json`: avoids a redundant square root in line projection, clears only
  the batch validity flag, and removes unused segment-local work.
- `03-subgroup-reduction.json`: direct eight-shuffle gather. This was slower and is retained only
  as an experimental result.
- `04-subgroup-tree-reduction.json`: associative two-stage reduction for each four-lane pixel
  group. It removes the shared partial buffers and reduces synchronization from nine workgroup
  barriers per 16 references to two.
- `05-fill-ref-hoist.json`: constructs `FineSegRef` once per task instead of once per visited
  cell. This was effectively neutral, indicating that the shader compiler already hoisted the
  invariant metadata work.
- `06-fill-ref-read-reduction.json`: also removes the redundant atomic cell-count load and cursor
  bound check. The preceding deterministic count traversal reserves exactly this range.
- `07-fill-ref-read-reduction-confirmation.json`: independent confirmation of the fill-pass win.

The final implementation includes the hot-loop cleanup, subgroup tree reduction, and fill-pass
read reduction. Percentiles are not paired samples, so small differences between unrelated stages
should be treated as run-to-run noise.
