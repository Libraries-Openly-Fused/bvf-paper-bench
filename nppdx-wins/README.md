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
  `frame_gauss21`.

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

## Reproduce

    cmake -S nppdx-wins -B build-wins -DBENCH_ITERATIONS=10 -DBENCH_WARMUP_ITERATIONS=3
    cmake --build build-wins -j
    ./build-wins/benchmark_wins_regions <out>/regions
    ./build-wins/benchmark_wins_regions <out>/regions --gpu-only-l2-flushed
    ./build-wins/benchmark_wins_frame <out>/frame
    ./build-wins/benchmark_wins_frame <out>/frame --gpu-only-l2-flushed
    # the second smoke: the same two runs for benchmark_wins_frame_k9, _k5_p2 and _gauss21

The main run of each program validates and times every NPPDx configuration and DPP tile, and writes its choices to
`choices.csv`. The flushed run reads them.
