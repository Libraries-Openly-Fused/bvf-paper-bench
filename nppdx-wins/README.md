# nppdx-wins: a 5x5 box BEFORE a compile-time resize, NPPDx against FKL (smoke test)

**The question** (Oscar, through the paper coordinator, 2026-10-15): is there a use case where NPPDx beats FKL? The
candidate is a neighborhood operation before a resize with a compile-time ratio:
- NPPDx is built for this order: BoxBlur -> Resize in one kernel, the ratio a template.
- FKL cannot stage it in one kernel, because a DPP's output is a write, so it cannot feed a resize.

**Not paper numbers:** a smoke exploration with 3 warm-up and 10 timed iterations, default clocks, one process per mode.

**Setup:**
- **Code:** branch `nppdx-wins`, commit a62e87b, branched from exp3-neighborhood 96cc71d. That branch has the harness of
  Experiment 3, `BoxBVF` and FKL 3e900e6 with `executeBoxBlur`; main has neither.
- **Libraries and hardware:** NPPDx v0.1.2 for SM 890; RTX 4090, CUDA 13.3, driver 610.43.02.
- **Results:** `results/RTX4090/regions` and `results/RTX4090/frame`, the harness's files and the logs of the runs. The
  second smoke (W2 with heavier filters, commit 485266f, below) is in `results/RTX4090/frame_k9`, `frame_k5_p2` and
  `frame_gauss21`. The full-protocol W2 runs for citation (commit 57f08bc) are in `results/RTX4090-full` (section "Full
  protocol").

## Workloads

| workload | program | pipeline | batches |
|---|---|---|---|
| W1, regions | `benchmark_wins_regions` | ReadNV12 -> YUVtoRGB -> Box 5x5 over the frame -> Crop 32x16 -> Resize 1:2 to 64x32 -> Mul -> Sub -> Div -> float3 planes | N = 1, 16, 100, at Experiment 1's positions (`BENCH_REGIONS=32x16`) |
| W2, frame | `benchmark_wins_frame` | ReadNV12 -> YUVtoRGB -> Box 5x5 -> Resize 2:1 of the 3840x2160 frame to 1920x1080 -> Mul -> Sub -> Div -> one float3 plane | the frame |

## Implementations (`src/wins_common.h`)

| name | kernels | what runs |
|---|---|---|
| `nppdx` | 1 | NPPDx's NV12 ingest, FromLimitedLuma/Chroma, ColorConvert BT.601, BoxBlur<5, 5>, Resize<J, K, bilinear> on shared memory tiles, then the normalization and float3 write by hand (NPPDx has no float output), as fkl-vs-nppdx's NPPDx kernel. W1 in frame mode (exp3-exploration/box-followup.md, section 1): the halo comes from the frame pixels around the region. The fastest validated tile and block. |
| `fkl_inline` | 1 | `executeOperations`: ReadNV12 -> YUVtoRGB -> BoxBVF<5> -> [Crop ->] Resize -> Mul -> Sub -> Div -> TensorWrite. Every output value evaluates the NV12 read and conversion 4 x 25 = 100 times. |
| `fkl_staged` | 2 | MorphologyDPP's box (`executeBoxBlur`) into a float3 intermediate in memory, then one TransformDPP kernel [Crop ->] Resize -> Mul -> Sub -> Div -> TensorWrite. The fastest validated DPP tile. |

**fkl_staged's intermediate:**
- **W1:** the regions expanded by 2 pixels, 36x20 float3 = 8640 bytes per region, written then read (864 KB each way at
  N = 100). The second kernel crops the 32x16 interior, so the expanded region's replicated edge is never read.
- **W2:** the boxed frame, 3840x2160 float3 = 99.5 MB, written then read. These are nominal sizes; DRAM traffic was not
  measured.

**The harness:** the common/bench_gpu.h harness of Experiments 1 and 3.
- Three modes: latency (CPU + GPU, CUDA events), GPU only (CUPTI kernel times), and GPU only with the L2 flushed before
  every kernel. Kernel counts come from CUPTI.
- In the harness's files `fkl_staged` is "the FKL implementation", so the mean speedups are those of fkl_staged over
  `nppdx` and over `fkl_inline`. Every median is in `latency.csv`, `gpu_time.csv` and `gpu_time_l2_flushed.csv`.

### Semantics (documented, not matched)

- **FKL's Resize** samples at dst * (src / dst), with no half-pixel offset, and clamps its reads to the image it reads.
  - Both FKL forms are "box over the frame, crop, resize clamped at the crop".
  - In W2, the 2:1 takes the boxed pixel at (2x, 2y): its bilinear weights are 1 and 0, but all 4 taps are evaluated.
- **NPPDx's Resize** samples at pixel centers, (dst + 0.5) * J / K - 0.5.
  - In W1's frame mode its ingest also gives the resize the boxed frame beyond the crop.
  - In W2, the 2:1 is the mean of the 2x2 boxed pixels.
- **Edges:** in W2 both replicate the frame edge for the box.
- **Validation:** each implementation is checked against a reference of its own semantics: FKL's CPU RGB of the frame,
  then the box and resize in double, then the normalization in float. `NppdxPixelCenterFloat` is a new reference kind in
  common/bench_gpu.h, with a 1e-3 tolerance.

## Results (medians, us; validation: every implementation within 1e-3 of its own reference, max abs error <= 1.7e-6)

### W1, 32x16 regions, 1:2 to 64x32

| N | mode | nppdx | fkl_inline | fkl_staged | fastest | faster FKL form | fkl_staged mean speedup over nppdx |
|---|---|---|---|---|---|---|---|
| 1 | latency | 7.17 | 7.12 | **6.14** | fkl_staged | staged | 1.16 |
| 1 | GPU only | 5.39 | 5.22 | **3.71** | fkl_staged | staged | 1.45 |
| 1 | GPU only, L2 flushed | 5.86 | 6.08 | **4.69** | fkl_staged | staged | 1.24 |
| 16 | latency | 8.99 | 8.35 | **6.59** | fkl_staged | staged | 1.34 |
| 16 | GPU only | 6.94 | 6.11 | **4.00** | fkl_staged | staged | 1.73 |
| 16 | GPU only, L2 flushed | 7.36 | 7.10 | **5.38** | fkl_staged | staged | 1.36 |
| 100 | latency | **10.05** | 29.78 | 11.26 | **nppdx** | staged | 0.88 |
| 100 | GPU only | 8.14 | 23.49 | **7.20** | fkl_staged | staged | 1.13 |
| 100 | GPU only, L2 flushed | **8.66** | 24.53 | 8.80 | nppdx, by 1.6% | staged | 0.98 |

**Per kernel at N = 100** (latency, `kernel_breakdown.csv`):
- nppdx: 10.2 us, one kernel.
- fkl_staged: 7.2 us (DPP) + 5.1 us (crop, resize, normalize) = 12.3 us.

**Configurations chosen:**
- NPPDx: 16x8 / block 256 (N = 1), 16x16 / 128 (N = 16), 32x16 / 256 (N = 100).
- DPP: 16x8 (N = 1, 100), 32x8 (N = 16).

### W2, the 4K frame, 2:1 to 1920x1080

| mode | nppdx | fkl_inline | fkl_staged | fastest | faster FKL form | fkl_staged mean speedup over nppdx / over fkl_inline |
|---|---|---|---|---|---|---|
| latency | 232.45 | **174.85** | 297.55 | fkl_inline | inline | 0.78 / 0.59 |
| GPU only | 233.51 | **173.51** | 297.83 | fkl_inline | inline | 0.78 / 0.58 |
| GPU only, L2 flushed | 232.13 | **175.39** | 274.26 | fkl_inline | inline | 0.85 / 0.64 |

**Per kernel** (latency):
- nppdx: 228.7 us.
- fkl_inline: 172.0 us.
- fkl_staged: 157.7 us (the DPP box of the frame) + 142.3 us (resize and normalize) = 300.0 us.

**Configurations chosen:** NPPDx 64x32 / block 512; DPP 32x8.

**Explicit blocks:** NPPDx blocks smaller than a tile's cell count gave wrong output, because NPPDx's in-place color
steps are not strided (as in Experiment 3). The validation excluded them: 64x32 with 128 and 256 threads, 32x32 with
128 (`choice_sweep.csv`).

## Answer

- **NPPDx beats FKL's faster form only at W1, N = 100.**
  - In latency: 10.05 vs 11.26 us, fkl_staged's mean speedup 0.88.
  - With the L2 flushed, by 1.6% (8.66 vs 8.80 us, mean speedup 0.98), which is within what 10 iterations resolve.
  - In GPU time fkl_staged is faster at N = 100 too (7.20 vs 8.14 us). The latency difference is fkl_staged's second
    launch and its host work per execution (two IOp chains of 100 rectangles), not GPU work.
- **At W1, N = 1 and 16,** fkl_staged is the fastest in every mode, by 1.16x to 1.73x over NPPDx.
- **At W2, fkl_inline is the fastest in every mode,** 1.32x to 1.35x faster than NPPDx, although it evaluates the NV12
  read and conversion 100 times per output (207 million evaluations per frame).
  - NPPDx is 1.18x to 1.28x faster than fkl_staged there. The staged form writes and reads the 99.5 MB float3 boxed
    frame.
- **Faster FKL form per point:** staged for W1 at every N and mode; inline for W2 in every mode.

**Not done:** the 8-bit RGB write variant of W2.
- NPPDx could use its own exgest there, but FKL would need a clamping uchar3 write without SaturateCast (excluded from
  benchmark kernels) and new rounded references.
- The float3 write is 24.9 MB, at most about 25 us of NPPDx's 233 us at the 4090's bandwidth, while fkl_inline leads by
  58 us.

## Second smoke: W2 with heavier filters (commit 485266f)

The coordinator's follow-up (2026-10-15). The pipeline is W2 (the 4K frame, Resize 2:1 to 1920x1080 float3) with three
filters that make NPPDx's staging pay more, and the same protocol (3 + 10 iterations, the fastest validated
configuration, validation against each implementation's own semantics). Programs, from `src/benchmark_wins_frame.cu`:

**`benchmark_wins_frame_k9`: a 9x9 box.**
- nppdx: BoxBlur<9, 9>.
- fkl_inline: BoxBVF<9>, 4 x 81 evaluations per output.
- fkl_staged: MorphologyDPP 9x9, then the resize.

**`benchmark_wins_frame_k5_p2`: a 5x5 box in two passes.**
- nppdx: two BoxBlur<5, 5> steps with the composed halo; the second writes into the input tile.
- fkl_inline: BoxBVF<5> twice, 4 x 625 evaluations per output.
- fkl_staged: `executeBoxBlur<2>`, then the resize.
- The edges differ: FKL replicates the edge of every pass, while NPPDx filters the frame extended by the total halo.
  Each is checked against its own semantics.

**`benchmark_wins_frame_gauss21`: NPPDx's GaussianBlur<50, wide>, 21 taps, sigma 5 (separable FIR in registers).**
- fkl_staged: LinearFilterDPP 21x21 (FKL's linear filter is 2D, 441 multiply-adds per output) with the products of
  NPPDx's taps, then the resize.
- FKL has no inline Gaussian, so there is no fkl_inline.
- Added for fairness, `fkl_staged_separable`: the same LinearFilterDPP used separably, 21x1 then 1x21, then the resize.
  That is three kernels and two 99.5 MB intermediates. It is the best FKL form with existing components, not one of the
  two forms asked for.

### Medians (us), kernels per execution, winner

| filter | mode | nppdx (1) | fkl_inline (1) | fkl_staged (2) | fkl_staged_separable (3) | fastest | faster FKL form |
|---|---|---|---|---|---|---|---|
| box 9x9 | latency | 614.06 | **497.66** | 541.89 | | fkl_inline | inline |
| box 9x9 | GPU only | 614.48 | **498.08** | 543.20 | | fkl_inline | inline |
| box 9x9 | GPU only, L2 flushed | 642.64 | **510.55** | 537.92 | | fkl_inline | inline |
| box 5x5, 2 passes | latency | **315.90** | 4423.10 | 492.19 | | **nppdx** | staged |
| box 5x5, 2 passes | GPU only | **316.24** | 4423.56 | 496.77 | | **nppdx** | staged |
| box 5x5, 2 passes | GPU only, L2 flushed | **309.12** | 4313.10 | 465.15 | | **nppdx** | staged |
| Gaussian 21 taps | latency | 814.48 | | 4236.29 | **625.18** | fkl_staged_separable | separable |
| Gaussian 21 taps | GPU only | 821.17 | | 4243.78 | **625.22** | fkl_staged_separable | separable |
| Gaussian 21 taps | GPU only, L2 flushed | 862.76 | | 4414.84 | **611.54** | fkl_staged_separable | separable |

**Mean speedups of fkl_staged over nppdx** (latency / GPU only / flushed):
- box 9x9: 1.13 / 1.13 / 1.19;
- box 5x5, two passes: 0.64 / 0.63 / 0.67 (NPPDx 1.56x / 1.57x / 1.50x faster);
- Gaussian: 0.19 / 0.19 / 0.20 (NPPDx 5.2x faster than LinearFilterDPP 21x21).

**Per kernel** (latency, `kernel_breakdown.csv`):

| filter | nppdx | fkl_inline | fkl_staged | fkl_staged_separable |
|---|---|---|---|---|
| box 9x9 | 613.3 | 492.5 | 405.5 (DPP) + 140.3 (resize) | |
| box 5x5, two passes | 313.3 | 4423.7 | 355.3 (DPP, 2 passes) + 142.3 (resize) | |
| Gaussian | 799.1 | | 4105.2 (LinearFilterDPP 21x21) + 143.0 (resize) | 228.5 (rows) + 252.8 (columns) + 146.1 (resize) |

**Configurations chosen:**
- NPPDx: 64x32 / 512 (9x9 and two passes); 32x32 / recommended (Gaussian). The Gaussian's 64x32 tile needs more than
  the 4090's 101376 bytes of shared memory: the launch fails and the sweep excluded it.
- DPP tile: 32x16 (boxes), 32x8 (both Gaussian forms).

**Validation:** every implementation within 1e-3 of its own reference; the largest max abs error is 4.6e-6
(LinearFilterDPP 21x21). As in the first smoke, NPPDx blocks smaller than the tile's cell count gave wrong output and
were excluded.

### Answer

- **NPPDx wins in GPU time with the 5x5 box in two passes.** It is 1.57x faster than the faster FKL form (staged:
  MorphologyDPP with both passes, plus the resize), 1.50x with the L2 flushed and 1.56x in latency. fkl_inline is
  14x slower: 625 evaluations per tap.
- **With the 9x9 box, fkl_inline is the fastest in every mode,** 1.23x to 1.26x faster than NPPDx. fkl_staged is
  second.
- **With the 21-tap Gaussian, NPPDx is 5.2x faster than FKL's form asked for** (LinearFilterDPP 21x21, which is not
  separable).
  - FKL's LinearFilterDPP used separably (21x1, 1x21, then the resize: three kernels, two 99.5 MB intermediates) is
    1.31x faster than NPPDx in GPU time (625 vs 821 us), 1.41x with the L2 flushed and 1.30x in latency.
  - So the Gaussian is an NPPDx win only against the non-separable form.

## Full protocol: the W2 cases for citation (commit 57f08bc, FKL 3e900e6)

**What was run:** the paper coordinator's request, 2026-10-15.
- **The pipeline:** the 4K NV12 frame -> filter -> Resize 2:1 -> 1920x1080 float3, four cases (the 5x5 box in one and in
  two passes, the 9x9 box, the 21-tap Gaussian).
- **Implementations:** nppdx; fkl_inline; fkl_staged; and, for the Gaussian, fkl_staged_separable as a labelled extra.
  - fkl_inline for the Gaussian is `GaussianBVF<21>` (common/bench_neighborhood_ops.h, commit bb0c245).
- **Protocol:** the one of Experiments 1 and 3 at 57f08bc, which interleaved the implementations in every iteration; not
  the paper's current protocol. Rerun before citing next to results of the sequential protocol.
  - 100 warm-up and 1000 timed iterations in every mode (latency, GPU only, GPU only with the L2 flushed), the
    implementations interleaved in every iteration, default clocks. The harness (common/bench_gpu.h) now runs each
    implementation's complete loop before the next one's instead (the baselines first, the FKL forms last).
  - Per-iteration mean speedups in both directions: nppdx over each FKL form, and each FKL form over nppdx.
  - Medians and p10-p90, CUPTI kernel counts, validation on 21 iterations (the first, every 50th and the last timed one).
- **Run and data:**
  - Run: `~/bvf-bench/results-nppdx-wins-full/run.sh`, 2026-10-15 01:09-01:13. bvf-paper-bench 57f08bc and FKL 3e900e6
    are recorded in `commit.txt` and every `platform.txt`.
  - Data: `results/RTX4090-full/<case>`, with every iteration's time in `per_iteration_<mode>.csv`.
  - Tables: `results/RTX4090-full/tables.md` and `key_numbers.csv`, from `full_report.py`.
- **Configurations:** the fastest validated, re-swept with 1000 iterations wherever the smoke's choice was within 2% of
  its runner-up (the smoke's noise).
  - Pinned: only fkl_staged's DPP tile for the two-pass box (32x16, 6.1% ahead in the smoke), via
    `WINS_PINNED_CHOICES`.
  - Chosen: NPPDx 64x32 / block 512 (5x5, two passes), 64x32 / recommended (9x9), 32x32 / recommended (Gaussian); DPP
    32x8 (5x5, Gaussian), 32x16 (two passes, 9x9).

**Semantics (documented, not matched):**
- **Resize sampling:** FKL's samples at (2x, 2y) (bilinear weights 1 and 0). NPPDx's samples at pixel centers (the
  2x2 mean).
- **Edges with two passes:** FKL replicates the edge of every pass (BoxBVF clamps to the image before it; MorphologyDPP
  replicates the border of every pass). NPPDx's ingest replicates the frame edge into the halo, and each BoxBlur step
  filters the whole tile with its halo: NPPDx filters the frame extended by the total halo, so its second pass reads
  first-pass values beyond the frame instead of replicating the first pass's edge.
- **Where they differ:** only near the frame border; with one pass the two are the same. Each implementation is
  validated against its own semantics.

**NPPDx's Gaussian tile on the RTX 4090:** the 64x32 tile cannot launch.
- Its kernel needs 138240 bytes of shared memory: the 88x56 ingest and filtered tiles (halo 10 + 2) twice, the 32x36
  intermediate and the 32x16 output, 12 bytes per pixel.
- The RTX 4090's opt-in limit is 101376 bytes, so `cudaFuncSetAttribute` fails (invalid argument) and the sweep excluded
  it. 32x32 is the largest Gaussian tile NPPDx can run there.

### GPU only, medians (us) [p10-p90] and per-iteration mean speedups

"nppdx over it" is the mean of t_it / t_nppdx; "it over nppdx" is the mean of t_nppdx / t_it.

| case | nppdx | fkl_inline (nppdx over it / it over nppdx) | fkl_staged | fkl_staged_separable (extra) | fastest |
|---|---|---|---|---|---|
| box 5x5 | 235.23 [234.59-235.81] | **175.07** [174.88-175.30] (0.745 / 1.343) | 299.81 [298.98-300.96] (1.275 / 0.784) | | fkl_inline |
| box 5x5, two passes | **328.32** [326.44-329.51] | 4589.40 [4562.10-4604.04] (14.020 / 0.071) | 508.69 [506.28-510.34] (1.550 / 0.645) | | **nppdx** |
| box 9x9 | 615.78 [614.24-617.45] | **504.77** [504.36-505.22] (0.820 / 1.220) | 548.97 [548.26-549.70] (0.891 / 1.122) | | fkl_inline |
| Gaussian 21 taps | 827.85 [827.24-828.59] | 3446.98 [3445.92-3449.77] (4.178 / 0.239) | 4272.68 [4271.79-4273.52] (5.183 / 0.193) | **629.41** [628.49-630.21] (0.761 / 1.315) | fkl_staged_separable |

### Latency and L2 flushed, medians (us) and nppdx's mean speedup over the faster FKL form

| case | mode | nppdx | fkl_inline | fkl_staged | fkl_staged_separable | nppdx over the faster FKL form |
|---|---|---|---|---|---|---|
| box 5x5 | latency | 233.84 | **175.74** | 298.91 | | 0.750 (inline) |
| box 5x5 | L2 flushed | 233.35 | **175.94** | 275.17 | | 0.754 (inline) |
| box 5x5, two passes | latency | **326.66** | 4563.97 | 504.61 | | 1.544 (staged) |
| box 5x5, two passes | L2 flushed | **321.24** | 4483.52 | 476.17 | | 1.485 (staged) |
| box 9x9 | latency | 615.42 | **503.81** | 549.50 | | 0.819 (inline) |
| box 9x9 | L2 flushed | 612.90 | **502.40** | 522.50 | | 0.819 (inline) |
| Gaussian 21 taps | latency | 823.23 | 3443.76 | 4275.17 | **629.76** | 4.193 (inline); 0.765 over separable |
| Gaussian 21 taps | L2 flushed | 826.89 | 3440.10 | 4245.87 | **598.09** | 4.176 (inline); 0.724 over separable |

**Kernels per execution:** nppdx 1, fkl_inline 1, fkl_staged 2, fkl_staged_separable 3.

**Validation:** every implementation is within its tolerance (1e-3) in all 21 validated iterations; the largest max abs
error is 4.6e-6 (fkl_staged, the Gaussian's LinearFilterDPP 21x21).

**With the L2 flushed, fkl_staged is faster than without** (275 vs 300 us for the 5x5 box). The flush runs before every
kernel, so the DPP's dirty lines of the intermediate are written back during the flush, not during the measured
kernels.

**Answer:**
- **NPPDx wins the 5x5 box in two passes** in every mode: 1.55x over the faster FKL form (fkl_staged) in GPU time,
  1.54x in latency and 1.49x with the L2 flushed. fkl_inline is 14x slower than NPPDx there (625 frame evaluations per
  tap).
- **FKL wins the other three cases:**
  - The 5x5 box: fkl_inline, 1.34x over NPPDx in GPU time.
  - The 9x9 box: fkl_inline, 1.22x.
  - The Gaussian: of the two FKL forms asked for, fkl_inline is the faster, and NPPDx is 4.2x faster than it. FKL's
    LinearFilterDPP used separably (the extra, three kernels) is 1.31x faster than NPPDx in GPU time.

## Lines of code (`loc/`)

**What it is:** the lines of code of every FKL and NPPDx version the paper compares, in Experiments 1 and 3, counted
from the repository.
- **Request:** the paper coordinator, 2026-10-15, for Oscar's request to put the lines of code in the comparison tables.
- **Files:** the counter `loc/count_loc.py` and the line ranges `loc/manifest.json` are the coordinator's. The coordinator
  wrote the ranges and four independent reviewers checked them. After the review of c51a930, the coordinator made the
  count per program on both sides. `loc/totals.txt` and `loc/show.txt` are the counter's output.

**The rule** (the paper's Methodology): count, per program, the lines that are neither blank nor comments of the code
that builds and launches a pipeline.
- **NPPDx:**
  - The kernel and the device helpers it calls.
  - The structs with the types of its operations and tiles. Of the `Spec` structs, only the struct head, `Operator`,
    `FUNCTION` and the closing brace.
  - Its launch configuration: init with `cudaFuncSetAttribute`, grid, block, shared memory, the launch and its error
    check.
  - Its list of tile and block configurations.
- **FKL:**
  - The call that builds and launches the pipeline, with its operation chains.
  - The details and tile list of a DPP.
  - The intermediates of a staged form, with their views and coefficient images.
  - The error check of the call.
- **Not counted, on either side:**
  - The harness, references, validation, the sweeps that choose a configuration, and output buffers.
  - The parameters, constants and aliases every version shares or that derive from the workload: Normalization, crop
    origins, `RESIZE_J/K`, `HALO`, `YUVToRGB`, the `Spec` alias, the `K`, `PASSES`, halo and `GAUSSIAN` members of the
    `Spec` structs, and `GaussianSpec::taps()` (used by FKL and the references).
  - Namespace and include lines.
  - The branches the program does not instantiate, with their if/else dispatch lines:
    - the 8-bit write branch of Experiment 3's `nppdx_box.h`;
    - in part (c), the two-pass branch of `FilterResizeOps` for the one-pass programs and its one-pass branch for the
      two-pass program;
    - the box/Gaussian and one/two-pass dispatch of the shared FKL functions.
  - `BoxBVF` and `GaussianBVF`, the operations written for the benchmark (Oscar, 2026-10-15).
- **Calibration:** the rule reproduces the published 10 / 109 / 131 at 3845b60 exactly (results-v3-addendum README,
  section 4).
- **What every range is:** the manifest's `what` field gives it for every range.

**Totals** (the manifest's `expected`; `count_loc.py` checks every one):

| experiment | commit (branch) | FKL | NPPDx |
|---|---|---|---|
| 1, crop + resize | 3845b60 (main) | 10 | 109; NPPDx+NPP 131 |
| 1, NPPDx+NPP with the aligned border | b7f6e88 (nppdx-npp-border-align) | | NPPDx+NPP 140 |
| 3 (b), the box on the crops | 96cc71d (exp3-neighborhood) | MorphologyDPP pipeline 14 | 83 (94 as written, with the uninstantiated 8-bit branch) |

| Experiment 3 (c), 76a9136 (nppdx-wins; sources as 57f08bc) | NPPDx | FKL plain (BVF op inline) | FKL staged | FKL staged separable |
|---|---|---|---|---|
| box 5x5 | 107 | 10 | 21 | |
| box 5x5, two passes | 112 | 11 | 21 | |
| box 9x9 | 107 | 10 | 21 | |
| Gaussian, 21 taps | 107 | 15 | 35 (21x21) | 50 |

**Run:** from any checkout that has fetched main, nppdx-npp-border-align, exp3-neighborhood and nppdx-wins.
- The counter reads every file with `git show <commit>:<path>`, so the checked-out branch does not matter.
- `--repo` reads the commits from another checkout.
- `--show` prints every range with a `+` on each counted line.

        nppdx-wins/loc/count_loc.py              # totals, per-range subtotals, a check against "expected"
        nppdx-wins/loc/count_loc.py --show       # every range, counted lines marked

## Reproduce

    cmake -S nppdx-wins -B build-wins -DBENCH_ITERATIONS=10 -DBENCH_WARMUP_ITERATIONS=3
    cmake --build build-wins -j
    ./build-wins/benchmark_wins_regions <out>/regions
    ./build-wins/benchmark_wins_regions <out>/regions --gpu-only-l2-flushed
    ./build-wins/benchmark_wins_frame <out>/frame
    ./build-wins/benchmark_wins_frame <out>/frame --gpu-only-l2-flushed
    # the second smoke: the same two runs for benchmark_wins_frame_k9, _k5_p2 and _gauss21
    # the full protocol: build with -DBENCH_ITERATIONS=1000 -DBENCH_WARMUP_ITERATIONS=100, then
    # results-nppdx-wins-full/run.sh (copied to results/RTX4090-full/run.sh) and
    nppdx-wins/full_report.py <results directory> <tables.md>

The main run of each program validates and times every NPPDx configuration and DPP tile, and writes its choices to
`choices.csv`. The flushed run reads them.
