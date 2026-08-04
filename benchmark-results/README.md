# Squirrel GPU optimization log

All measurements use the full `squirrel` case at 3840x2160 on an NVIDIA GeForce RTX 5090
(Vulkan, NVIDIA driver 610.43.03), with 120 warmup frames and 100 measured frames.

```sh
target/release/examples/strand_bench \
  --case squirrel --headless --width 3840 --height 2160 \
  --warmup 120 --samples 100 --output <result.json>
```

Prepass telemetry is enabled by default in the benchmark harness. Add `--no-telemetry` when the
goal is to measure the production path without counter atomics, aggregation, or readback.

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

## Fine-reference layout

`FineSegRef` was reduced from four words (16 bytes) to three words (12 bytes) by removing the
unused strand index and segment-local value. The remaining material index is stored as a full
`u32`, so the compact layout introduces no new range restriction.

| Revision | Fill median (ms) | Raster median (ms) |
| --- | ---: | ---: |
| 16-byte reference, run 1 | 2.862 | 7.061 |
| 16-byte reference, run 2 | 2.833 | 7.226 |
| 12-byte reference, run 1 | 2.427 | 7.018 |
| 12-byte reference, run 2 | 2.448 | 7.016 |

The two-run average fill median improved from 2.848 ms to 2.438 ms (-14.4%). Raster median
improved by 1.8%, but raster mean and tail percentiles did not improve; the raster effect should
therefore be treated as inconclusive rather than a demonstrated bandwidth win.

## Allocated-page prefix dispatch

The prefix pass now dispatches over compactly allocated count pages instead of the dense
coarse-page table. Page allocation writes the reverse tile/depth mapping into `FinePageMeta`, and
a separate 12-byte indirect-argument buffer keeps its storage write and indirect read in distinct
WGPU usage scopes.

| Revision | Prefix median (ms) | Dispatch setup median (ms) | Combined median (ms) |
| --- | ---: | ---: | ---: |
| Dense page table, run 1 | 0.185 | - | 0.185 |
| Dense page table, run 2 | 0.185 | - | 0.185 |
| Allocated pages, run 1 | 0.023 | 0.004 | 0.027 |
| Allocated pages, run 2 | 0.024 | 0.004 | 0.028 |

The two-run average combined median improved from 0.185 ms to 0.027 ms (-85.2%). The prefix
kernel itself improved by 87.3%. This saves only about 0.158 ms on squirrel because its dense
table was cheap, but it removes the sparse page-table probe that dominated the DSON trace.

## Sparse-page telemetry

Optional GPU counters now report allocated pages, generated binning tasks, segment/page
candidates, accepted fine-cell references, the number of active pages, maximum candidates per
page, and a 32-bin log2 candidate-count histogram. On the canonical squirrel frame they report:

| Metric | Median |
| --- | ---: |
| Allocated/active pages | 12,695 |
| Binning tasks | 8,627,013 |
| Segment/page candidates | 11,481,492 |
| Fine-cell references | 25,230,333 |
| Candidates per active page, mean | 904.4 |
| Candidate-count p50 upper bound | 127 |
| Candidate-count p90/p95 upper bound | 4,095 |
| Candidate-count p99 upper bound | 16,383 |
| Maximum candidates on one page | 77,165 |

This confirms a strongly skewed page distribution: most pages are comparatively light, while a
small dense tail dominates candidate processing. The log2 percentiles are bounds, not exact
quantiles.

## Cached projected segments

The fine pass now transforms, projects, clips, and quantizes each visible segment once per
frustum. It stores a 12-byte projected-segment record which the mark, count, and fill traversals
reuse. Those three passes no longer repeat index/vertex reads, world transforms, projection, or
viewport clipping.

| Revision | Fine (ms) | Mark (ms) | Count/bin (ms) | Fill (ms) | Combined (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instrumented baseline | 0.401 | 0.751 | 1.434 | 2.456 | 5.041 |
| Cached, run 1 | 0.404 | 0.527 | 1.057 | 1.919 | 3.908 |
| Cached, run 2 | 0.404 | 0.527 | 1.057 | 1.912 | 3.900 |
| Cached, telemetry off | 0.404 | 0.532 | 0.980 | 1.921 | 3.836 |

Against the telemetry-free allocated-page confirmation, the production combined median falls
from 4.900 ms to 3.836 ms (-21.7%, 1.064 ms). The cache adds about 0.008 ms to the fine pass but
saves about 1.072 ms from the three repeated traversals.

The packed viewport coordinates resolve to about 0.059 pixels at 3840 pixels wide, with depth
stored at 16-bit normalized precision. Boundary quantization changed the accepted fine-reference
count by 1,502 out of roughly 25.23 million (+0.006%); this should receive a visual regression
check before treating the representation as final.

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
- `08-fine-seg-ref-12b.json`: first measurement of the compact three-word reference.
- `09-fine-seg-ref-12b-confirmation.json`: independent compact-reference confirmation.
- `10-allocated-page-prefix.json`: prefixes only allocated count pages via indirect dispatch.
- `11-allocated-page-prefix-confirmation.json`: independent allocated-page confirmation.
- `12-prepass-telemetry-baseline.json`: canonical sparse-page counters before projected-segment
  caching.
- `13-cached-projected-segments.json`: first canonical cache measurement with telemetry enabled.
- `14-cached-projected-segments-confirmation.json`: independent instrumented confirmation.
- `15-cached-projected-segments-production.json`: cache measurement with telemetry disabled.

The final implementation includes the hot-loop cleanup, subgroup tree reduction, fill-pass read
reduction, compact fine-segment references, and allocated-page prefix dispatch. Percentiles are
not paired samples, so small differences between unrelated stages should be treated as run-to-run
noise.
