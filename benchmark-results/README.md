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

## Page-owned CSR prototype

Fine binning now has a temporary host-selected A/B backend. `segment` retains the previous
segment-owned count/prefix/fill sequence; `page-csr` constructs compact candidate ranges and lets
one workgroup own the 256 fine cells of each allocated page. Select it with:

```sh
--fine-binning segment
--fine-binning page-csr
```

The CSR candidate count is folded into the existing page-mark traversal. Allocation transfers
virtual page counts to compact physical page IDs, a one-workgroup scan prefixes the roughly
12,700 counts, and a coarse-only scatter writes 32-bit task IDs into contiguous page ranges.

| Backend/stage | Segment scatter (ms) | Page CSR (ms) |
| --- | ---: | ---: |
| Mark pages / count candidates | 0.532 | 0.657 |
| Segment fine count | 0.976 | - |
| Fine/page prefix | 0.024 | 0.028 |
| Candidate scatter | - | 0.757 |
| Segment fill | 1.909 | - |
| Page-local count/scan/fill | - | 2.096 |
| **Backend total** | **3.441** | **3.538** |

The page-owned fine build replaces 2.909 ms of segment count/prefix/fill work with 2.096 ms
(-28.0%, 0.813 ms). Candidate construction currently costs 0.910 ms, leaving this first complete
prototype 0.097 ms (+2.8%) slower overall. The default therefore remains `segment` while the CSR
path is retained for optimization.

Both instrumented backends produce 11,482,864 page candidates. Page CSR produces 25,231,840 fine
references versus 25,231,835 for segment scatter: a five-reference difference (0.00002%) at
floating-point cell boundaries. A visual regression check remains required before promoting it.

## Color raster accumulation and access validation

The color kernel now accumulates premultiplied color directly across work items, removing the
second `froxel_color` accumulator and all per-blend unpremultiply operations. The recovered live
state is spent staging the projected line vector and its inverse squared length once per segment.
Perspective correction is algebraically reduced to one division.

Fine segment references are also treated as trusted output from host-validated assets. The hot
staging path no longer branches to fallback values for invalid instance, resource, index, vertex,
material, shading-layer, or shading-atlas indices. Batch-tail and geometric visibility checks
remain because they describe valid runtime work rather than malformed input.

| Revision | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Median vs. control |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-segment batch control | 7.458 | 6.822 | 9.683 | 10.195 | - |
| Premultiplied + staged line math | 6.503 | 6.132 | 8.317 | 8.472 | -10.1% |
| Unchecked asset access | 6.062 | 5.687 | 7.819 | 8.263 | -16.6% |

The math/register change saves 0.690 ms at the median. Removing malformed-input handling saves a
further 0.446 ms, for a combined 1.136 ms median reduction. Mean time falls by 18.7% across both
changes. These results use the telemetry-free page-CSR path.

## Sparse active-tile work emission

The raster-work emitter no longer dispatches one workgroup for every fine XY tile in the padded
coarse domain. Fine-page construction marks non-empty XY tiles, a small block scan compacts those
marks in the original coarse-tile-major order, and an indirect dispatch visits only the compacted
tile indices. Queue entries are the original dense fine-stack index packed into one `u32`; the
emitter reconstructs the coarse tile and 4x4 child exactly as before.

| Revision | Emit (ms) | Scan (ms) | Compact (ms) | Setup (ms) | Combined emission (ms) | Camera raster (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense tile domain | 0.428 | - | - | - | 0.428 | 5.687 |
| Unordered atomic append | 0.168 | - | - | 0.005 | 0.173 | 7.055 |
| Ordered 12-byte descriptor | 0.180 | 0.007 | 0.006 | 0.005 | 0.197 | 5.687 |
| Ordered packed index | 0.179 | 0.007 | 0.005 | 0.005 | 0.196 | 5.696 |

The unordered prototype demonstrates why queue order is part of the performance contract: it made
emission cheap but regressed camera rasterization by 24.1% at the median. Stable compaction restores
the old coarse-major order and leaves both raster medians unchanged within run-to-run noise. The
final packed implementation cuts combined emission from 0.428 ms to 0.196 ms (-54.2%, 0.232 ms) on
squirrel. Its new storage scales with padded fine XY tiles, not fine depth cells: one word each for
the queue and mark bitset plus two words per 256-tile scan block.

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
- `16-segment-scatter-ab-control.json`: telemetry-free legacy control with the A/B switch present.
- `17-page-csr-initial.json`: telemetry-free first page-owned CSR implementation.
- `18-page-csr-telemetry.json`: page-owned workload and output-count telemetry.
- `35-color-premult-precomputed-line.json`: premultiplied direct accumulation and staged projected
  line math.
- `36-color-unchecked-asset-access.json`: removes asset-validity fallbacks from raster staging.
- `37-sparse-active-tile-emission.json`: unordered atomic active-tile append experiment. Emission
  improved, but loss of coarse-major locality substantially regressed rasterization.
- `38-ordered-sparse-active-tile-emission.json`: deterministic active-tile compaction using the
  original traversal order and a 12-byte explicit descriptor.
- `39-packed-active-tile-index.json`: final active queue representation, packing the existing dense
  fine-stack index into one word.

The final implementation includes the hot-loop cleanup, subgroup tree reduction, fill-pass read
reduction, compact fine-segment references, and allocated-page prefix dispatch. Percentiles are
not paired samples, so small differences between unrelated stages should be treated as run-to-run
noise.
