# FKL vs RPP on ROCm (AMD GPU and CPU)

Part of the [BVF paper benchmarks](../README.md): the Figure 1 pipeline on an AMD GPU with ROCm, FKL's HIP backend
against RPP (ROCm Performance Primitives), and FKL's CPU backends against RPP's HOST backend, on the
[workload](../README.md#workload) of every benchmark of this repository (`common/bench_geometry.h`), with the constants
and timing protocol of the CUDA benchmarks ([`common/bench_gpu.h`](../common/bench_gpu.h)). No HIP graphs anywhere.
The commands below run from the repository root.

The regions change every iteration (20x10, 32x16 and 48x24 by turns, new positions every warm-up and timed
iteration, all generated before the loops by `RegionSets`): iteration t of every loop runs every pipeline on the
regions of iteration t (warm-up 0 to W - 1, timed W to W + T - 1). Every pipeline converts the regions of every
iteration to its own arguments when it is created; inside the timed region there is only what each library needs to
receive the regions of its iteration: for FKL, `build()` of its operations with the iteration's rectangles and the
fusion, in every `executeOperations` call; for RPP, one `hipMemcpyAsync` of the iteration's ROIs from a pinned buffer
to the device ROI tensors that its kernels read (the first operation of every RPP run, like NPP's descriptors in the
CUDA benchmark). Region buffers (RPP's crop buffer, FKL VF+HF's crop tensor) have planes of the largest region,
48x24; RPP's calls take each region's size from its ROI, and FKL VF+HF's resize reads each plane through a view of
its region's size.

## GPU benchmark (`rocm/src/rocm_bench.hip`)

| Implementation | What runs |
|---|---|
| `rpp` | the RPP pipeline of the benchmark: the fastest of the five RPP variants below, which are numerically equivalent (within 1e-5 of the RPP-convention reference); a selection pass validates and times all of them (CPU + GPU, interleaved, 20 warm-up + 200 timed iterations per N) and keeps the one with the lowest geometric mean of its medians over N (`rpp_selection.csv`); only that one is measured afterwards |
| `rpp_per_op` (variant) | the upload of the iteration's ROIs (every variant starts with it), then one RPP call per operation: `rppt_yuv_to_rgb` (NV12 to 8 bit RGB, BT.601 limited range: `RpptColorStandard_BT601`, `RpptColorRange_STUDIO`; it also uploads 8 constants with `hipMemcpyToSymbol` on every call), one batched `rppt_crop_mirror_normalize` 8u to 32f (all batch images alias the frame), `hipMemsetAsync` of the output (RPP has no constant border), one batched `rppt_resize` into the rows of the output planes below the top border, `rppt_multiply_scalar`, `rppt_tensor_subtract_tensor` and `rppt_tensor_divide_tensor` on the whole tensor |
| `rpp_npp_like` (variant) | the same calls with exactly the intermediates of the paper's NPP pipeline: the resized crops in their own buffer, copied into the output rows by a batched `rppt_crop` |
| `rpp_fused` (variant) | `rppt_yuv_to_rgb`, `hipMemsetAsync` of the top border rows of the first output plane, then RPP's fused `rppt_resize_mirror_normalize` (crop + 8u to 32f + resize + Mul; it zeroes everything from its destination pointer, which starts below that border, and synchronizes the stream itself), Sub and Div on the whole tensor (the border rows also need the normalization, which `rppt_resize_mirror_normalize` cannot apply to them) |
| `rpp_normalize` (variant) | `rpp_per_op` with Mul + Sub + Div in one `rppt_normalize` |
| `rpp_fused_normalize` (variant) | `rppt_yuv_to_rgb`, `hipMemsetAsync` of the first plane's top border rows, `rppt_resize_mirror_normalize` (crop + 8u to 32f + resize) and `rppt_normalize` (Mul + Sub + Div): the fewest RPP calls |
| `fkl_vf_hf` | the four FKL kernels of `makeFklVfHf<N>` of [`fkl-ablation/src/benchmark_fkl_ablation.cu`](../fkl-ablation/src/benchmark_fkl_ablation.cu), unchanged (`makeFklVfHf<N, true>` here; `<N, false>`, with a float3 frame, is only used for the per kernel breakdown and the counters) |
| `fkl_bvf` | the single `executeOperations` call of `makeFklBvf<N>` (Listing 2 of the paper), unchanged |

Everything runs on one blocking HIP stream (so that RPP's null-stream `hipMemcpyToSymbol` calls serialize with it).
Metrics, as in the CUDA benchmark: CPU + GPU (HIP events around the complete pipeline, all implementations
interleaved in every iteration, per-execution speedups averaged), then, in a child process in which the
rocprofiler-sdk tool is registered (so that no profiler wraps the HIP API during the timed runs), kernel / memset /
memcpy counts per pipeline and per RPP call (HIP runtime API tracing), GPU time only (rocprofiler-sdk kernel dispatch
tracing, the sum of the durations of the pipeline's kernels; the HIP runtime's own memset and copy blit kernels are
not counted, like CUPTI's kernel records) and GPU time only with a flush kernel that reads 2 x (L2 + MALL) before
every GPU operation (files named `l2_flushed` as in the CUDA benchmark). Kernel dispatch tracing and hardware
counters are not available on WSL2 (ROCm runs on `/dev/dxg` there): on WSL2 the program measures CPU + GPU and the
counts, and says that GPU time only was not measured; if the profiling child fails, the CPU + GPU results are still
written. Validation, before the timing, on the timed iterations 0, 50, ..., 950 and 999 (every pipeline runs on the
regions of that iteration into a NaN-poisoned output; `validation.csv` has one row per pipeline, N and iteration), with
the rules of every GPU benchmark ([`common/bench_validation.h`](../common/bench_validation.h):
1e-3 against a float reference, one level of 8 bit RGB more against an 8 bit one, at most one value in a thousand more
than 1e-3 away; the program fails if an FKL output is outside its tolerance, and reports the RPP pipelines'): FKL VF+HF
against FKL's CPU backend with the 8 bit RGB frame, FKL BVF against FKL's CPU backend in float; the RPP pipelines against a C++ reference built from RPP's own 8 bit RGB frame with RPP's
sampling convention with each region's own scale, measured with linear ramps for every region size
(`--resize-convention`: pixel centers, clamped to [0, size - 2]), and against the paper's 8 bit reference (FKL's
conversion rounded to 8 bit, FKL's sampling); after the timed iterations of the CPU + GPU pass, every pipeline runs the
first validated iteration again and must give the same output, every value within 1e-6 (the GPU time only passes of the
profiling child run the same pipelines and check the kernel records of every execution, not the outputs). The RPP behaviour that the pipelines take into account (handle batch size, ROI
formats converted in place, 8 pixel row padding, internal copies, memsets and synchronizations) is described in
`platform.txt`.

Modes, after the results directory: (none) the benchmark, `--validate`, `--resize-convention`
(`rpp_resize_convention.csv`), `--rocprof-target <impl> <N>` (the target of `rocm/scripts/rocprof_counters.sh`),
`--check-kernel-tracing`. Without a results directory, `rocm_bench --batches` prints the batch sizes (as
the CUDA benchmarks' `--batches` does), `rocm_bench --workload` the workload and `rocm_bench --regions <t>` the 100
regions of iteration t as CSV; `rpp_cpu_bench` has `--batches` and `--workload`.

Output files: those of the CUDA benchmark with `GPU_AMD` instead of `GPU_NVIDIA` (`latency.csv`, `validation.csv`,
`kernels.csv`, `kernel_breakdown.csv`, `<baseline>_vs_fkl_bvf_[gpu_only_|gpu_only_l2_flushed_]GPU_AMD.csv`,
`gpu_time.csv`, `gpu_time_l2_flushed.csv`, `platform.txt`), plus
`kernel_counts_per_operation.csv`, `rpp_selection.csv`, `rpp_resize_convention.csv` and `regions.csv` (the sizes and
areas of the regions of the timed iterations per N, as in every results folder); the RPP pipelines' memcpy counts
include the upload of the ROIs (9 per execution: that one and the 8 of `rppt_yuv_to_rgb`); `validation.csv` has an
`iteration` column after N, and two more columns, the errors against the RPP-convention reference and against the paper's 8 bit reference, and `gpu_time*.csv` five more, the GPU time and dispatches of the HIP runtime's blit kernels (memsets, copies) of each execution, which the GPU time only sums leave out. `--validate` validates every RPP variant.

## Experiment 3: neighborhood operations (`rocm/src/rocm_neighborhood.hip`)

A box blur of fixed-size crops, without the resize and the border of Figure 1: ReadNV12 -> YUVtoRGB -> Crop 64x32 ->
Box K x K -> Mul -> Sub -> Div -> Write, into N planes of 64x32 float3. The box is the mean of the K x K window centred
on the value, per channel, replicating the edge of the crop in every pass; the crops (64x32 at random even positions,
new every iteration) and the parameters are those of `common/bench_neighborhood.h`, shared with the CUDA side. Same
frame, batch sizes, timing protocol and validation rules as `rocm_bench` (it includes `rocm_bench.hip` for its
harness). Two parts, each a set of groups (part, K, passes, N) whose implementations compute the same output:

- Part (b), the library comparison: K = `BOX_K` (5), one pass, every N. Every RPP variant starts with the upload of
  the crops of its iteration and `rppt_yuv_to_rgb` (full frame) into a buffer that starts 256 bytes into its
  allocation (`offsetInBytes`; RPP's HIP box filter needs at least 12 x (K / 2) bytes before the first pixel of its
  source, `RPP_ERROR_LOW_OFFSET`).
  - `rpp_u8box_cmn`: `rppt_box_filter` U8 on the frame with the crops as per-image XYWH ROIs (crop + box in one call:
    it replicates the ROI edge and writes each crop at (0, 0) of its own image), then `rppt_crop_mirror_normalize`
    8u->32f of the boxed crops with the whole normalization (multiplier Mul / Div, offset -Sub / Div, per channel) into
    the output tensor: 3 kernels.
  - `rpp_u8box_normalize`: the same with `rppt_crop_mirror_normalize` converting only, then `rppt_normalize`: 4 kernels.
  - `rpp_f32box_cmn`: `rppt_crop_mirror_normalize` 8u->32f of the crops with the whole normalization, then
    `rppt_box_filter` F32 into the output tensor: 3 kernels.
  - `rpp_f32box_normalize`: the same with the conversion only, then `rppt_normalize`: 4 kernels.
  - `fkl_dpp_<W>x<H>`: one FKL kernel, MorphologyDPP's `executeBoxBlur<passes>` (every pass a mean: an Add reducer
    and a Mul 1 / K^2 finalizer) on `MorphologyDPPDetails<float3, W, H, K, K>`, batched over the crops, reading the BVF
    chain up to the crop and writing through Mul, Sub, Div, for every tile of `DPP_TILES`.
- Part (a), the micro-experiment, FKL only: K = 3, 5, 7, one and two passes, N = 1 and 16: `fkl_bvf_naive` (the naive
  ReadBack operation `BoxBVF<K>` of `common/bench_neighborhood_ops.h` in the BVF chain, twice for two passes, one
  kernel) against the DPP tiles, in two variants: on the crops (operation `box`), and with the resize (operation
  `resize_box`: Figure 1's chain on Experiment 1's regions without the border, ReadNV12 -> YUVtoRGB -> Crop -> Resize
  64x32 -> Box -> Mul -> Sub -> Div -> Write, the DPP's input chain ending at the resize).

In every group every implementation is validated and timed in the same interleaved iterations; the chosen DPP tile is
the fastest (median, CPU + GPU) within its tolerance and the chosen RPP variant the fastest within its tolerance and
within `RPP_EQUIVALENCE` of its RPP-convention reference (`chosen` column); every one is recorded, and the speedups are
per iteration over the chosen DPP tile. References (`bench::neighborhood::boxReference`: the box in double, then the
normalization in float; a box does not commute with the normalization in float, so the references blur first, the
order of the pipeline and of every implementation but `rpp_f32box_cmn`, which normalizes in `rppt_crop_mirror_normalize`
before its F32 box: it is held to its tolerance against these references, and exactly to its own RPP-convention
reference, which follows its order): FKL against FKL's CPU float chain (tolerance 1e-3); RPP, whose frame is 8 bit RGB, against the same
chain with the RGB saturated and rounded to 8 bits (`fkl_cpu_8bit_rgb`), and for the U8 box with its output rounded to
8 bits (`fkl_cpu_8bit_rgb_rounded_box`), tolerance one 8 bit level; its error against the float reference, whose RGB is
not saturated, is recorded too. Each RPP variant is also checked against its RPP-convention reference, RPP 3.2.0's HIP
arithmetic on RPP's 8 bit frame operation by operation (U8 box `rint(float(sum) * scale)`, F32 box the window summed in
float row after row times the scale, `crop_mirror_normalize` `fmaf(x, multiplier, offset)`, `rppt_normalize`
`fmaf(x - mean, 1 / stdDev, 0)`). It needs FKL with MorphologyDPP's `executeBoxBlur`: the commit recorded for the
repository's `fkl` submodule (CMake builds `rocm_neighborhood` only with an FKL that has it, so FKL `63782f6` builds
`rocm_bench` alone).

Output files (in `<results dir>/neighborhood` with `run_all.sh`): `latency.csv`, `validation.csv`, `gpu_time.csv`,
`gpu_time_l2_flushed.csv`, `kernel_breakdown.csv`, `kernel_counts_per_operation.csv`, `regions.csv`,
`regions_resize.csv` and `platform.txt`,
with the group in the first columns (`part,operation,K,passes,N,implementation,chosen`) or in the implementation name
(`<part>_<operation>_k<K>_p<passes>_n<N>/<implementation>`); `regions_resize.csv` has the regions of the resize
variant. `--validate` only validates every group.

    EXPERIMENT=3 rocm/scripts/run_all.sh ../results/rocm-neighborhood

or, after the build below, `./build-rocm/rocm_neighborhood ../results/rocm-neighborhood`.

## CPU benchmark (`rocm/cpu/`)

RPP's HOST backend against FKL's CPU backends, with the same number of threads on both sides: RPP with 1 thread
against FKL's `ParArch::CPU`, and RPP with all hardware threads against FKL's `ParArch::CPU_MULTITHREAD`, with the
protocol of [`fkl-vs-opencv-cpu/`](../fkl-vs-opencv-cpu/README.md). RPP's NV12 conversion is HIP only, so the RPP HOST pipelines
get their 8 bit RGB frame from FKL's CPU conversion (one thread for the 1 thread runs, FKL's thread pool for the
others), then one of the five RPP variants on the HOST backend: for each thread count, a selection pass validates all
of them (on the regions of iterations 0, 1, 2 and 0 again, so that every crop slot has held every size: the two outputs
of iteration 0 must be identical) and times the ones numerically equivalent to `rpp_per_op` (10 warm-up + 100
iterations per N, iteration i on the regions of iteration i), and only the fastest is measured (`cpu_rpp_selection.csv`; outputs `benchmark_cpu_rpp_1thread_vs_fkl_bvf_CPU.csv` and
`benchmark_cpu_rpp_<T>threads_vs_fkl_bvf_CPU_MULTITHREAD.csv`). As on the GPU, the regions change every iteration:
each pipeline converts the regions of every iteration when it is created (RPP: its XYWH ROIs; FKL: its rectangles),
and inside the timed region RPP's calls get the ROI arrays of the iteration (host memory, read by RPP on the host) and
FKL builds its operations with the iteration's rectangles and fuses them. Validation, before the timing of each N and
on the same pipelines that are then timed, on the timed iterations 0, 50, ..., 950 and 999 (`cpu_validation.csv`, one
row per N, implementation, thread count and iteration, with the error per region size): the program fails if an FKL
output is more than 1e-3 from FKL's float reference in any of them, or if no RPP variant is numerically equivalent to
`rpp_per_op` at every N for a thread count, in the benchmark and with `--validate`. RPP has no constant border: the variants that modify the border rows of
the output in place (`rpp_per_op`, `rpp_npp_like`, `rpp_fused`) zero them with a memset in every run; the
`rppt_normalize` variants never write the border rows of their input tensor, which is zeroed once when it is allocated.
In the all-threads RPP pipeline, FKL's thread pool, which does the conversion, does not spin after it
(`setSpinTime(0)` before every timed run of that pipeline, outside of the timing; FKL's default 100 us before FKL's
runs), so that its workers do not compete with RPP's OpenMP team.
RPP HOST's sampling positions are measured with ramps for every region size (`rpp_host_sampling_positions.csv`), in
the benchmark's own crop-slot layout and with two different values outside the regions (the positions must not depend
on them: RPP must not read outside a region), and used by its reference: its `rppt_resize` and
`rppt_resize_mirror_normalize` treat the last columns and rows differently, so the two variants built on
`rppt_resize_mirror_normalize` are not equivalent to `rpp_per_op` on the CPU. The crop buffer has slots of the largest
region, 48x24, plus a spare row: RPP 3.2.0's HOST `rppt_crop_mirror_normalize` (8u to 32f, packed RGB) writes one
float (1.0) past the end of every row when the ROI width is a multiple of 16, as with the 32 and 48 pixel regions, so
the last row's would land on the next slot, which another OpenMP thread may have written already. RPP's HOST backend parallelizes over the batch images with OpenMP: the
arithmetic calls see the output as N samples, and `KMP_BLOCKTIME` is 100 us (libomp's default is 200 ms), like FKL's
thread pool, so that the idle gap before every timed run puts both pools to sleep.

## Requirements

- Linux with ROCm 10.1 (the paper's run: a Radeon RX 9060, gfx1200), CMake 3.28, Ninja, and a recent clang for the CPU
  benchmark (`run_all.sh` takes the newest `clang++-NN`, or `CXX_CPU`).
- The GPU build targets gfx1200 by default; for another AMD GPU, give its target in `-DCMAKE_HIP_ARCHITECTURES=<gfx>`
  (one target; with `run_all.sh`, in `BENCH_CMAKE_ARGS`). The default counters and the 256 B write request size of
  `summarize_counters.py` are those of gfx12.
- The ROCm 10.1 packages `amdrocm-rpp10.1` and `amdrocm-rpp-dev10.1` (RPP 3.2.0: the library and its headers),
  `amdrocm-profiler-base10.1` (rocprofiler-sdk, rocprofv3) and `amdrocm-amdsmi10.1`. Without root, unpack them with
  `apt-get download <package> && dpkg -x <package>_*.deb <prefix>` and set `ROCM_EXTRA=<prefix>/opt/rocm/core-10.1`.
- FusedKernelLibrary at `63782f6` for Experiments 1 and 2 and the CPU benchmark, and at the commit recorded for the
  repository's `fkl` submodule (with MorphologyDPP's `executeBoxBlur`) for Experiment 3. `run_all.sh` uses the
  repository's `fkl` submodule with its `thread-pool` submodule (`git submodule update --init --recursive fkl`) unless
  `FKL_GPU_DIR` / `FKL_CPU_DIR` name other checkouts, and stops if FKL is not at the commit of the experiment it runs.
  The submodule is at Experiment 3's commit, so for Experiments 1 and 2 the default is a detached worktree of the
  submodule at `63782f6` with its submodules, which `run_all.sh` makes the first time in `build-fkl-63782f6` (ignored
  by git) and reuses.

## Build and run

Everything, in one command (it writes a platform snapshot first, builds, checks that FKL is at the right commits and
unmodified and that kernel dispatch tracing and the hardware counters are available and count, then runs the GPU
benchmark, the counters and the CPU benchmark; the counters default to `FETCH_SIZE GL2C_EA_WRREQ_sum
GL2C_EA_WRREQ_64B_sum`, since rocprofiler-sdk 1.4.1 defines no `WRITE_SIZE` for gfx12):

    rocm/scripts/run_all.sh ../results/rocm

The results directory must be new or empty (the script stops otherwise), so that no file of an earlier run sits next
to this run's, and both builds are configured from scratch with `cmake --fresh`, so that no option cached by an earlier
configure applies: it deletes the build directory's `CMakeCache.txt` and `CMakeFiles/`, where the Ninja build keeps
its object files, so every run compiles both benchmarks again.

Before running it, on the bare-metal machine: nothing else running (`uptime` load below 0.5, which the script checks),
nothing else on the GPU (`amd-smi process`), and the performance level and clock lock state as they are: the script
records them (`amd-smi metric`, including `--perf-level`) and does not change them. Hardware counters need access to
the GPU's performance counters (membership of the `render` group) and, on gfx1200, a profiling performance level: at
`auto`, which leaves the GPU's clock and power gating on, the GL2C, TA, TCP and SQ instruction counters read 0, while
`STABLE_STD` (`sudo amd-smi set -l STABLE_STD`, sysfs `profile_standard`) makes them count, and pins the shader clock
(1494 MHz on the RX 9060, against about 3.3 GHz at `auto`). The script checks this before the GPU benchmark
(`rocprof_counters.sh --check`: the `FETCH_SIZE` of a cache flush kernel, which reads 2 x (L2 + MALL), must be at
least half of it) and stops if the counters read 0. To time the benchmarks at another level, run it there with
`ALLOW_NO_COUNTERS=1` and collect the counters afterwards at a profiling level with the `rocprof_counters.sh` command
it prints at the end.

What the counters measure on gfx12 (also in the `platform.txt` of the traffic directory): `FETCH_SIZE` and
`GL2C_EA_WRREQ` count the requests between L2 and the data fabric. The Infinity Cache (MALL) is behind the fabric, so
they are L2 misses and write-backs, part of which the MALL may serve, not DRAM traffic; rocprofiler-sdk lists no MALL
or DRAM counter for gfx1200. `summarize_counters.py` derives the written KiB as
((`GL2C_EA_WRREQ_sum` - `GL2C_EA_WRREQ_64B_sum`) x 256 + `GL2C_EA_WRREQ_64B_sum` x 64) / 1024 on gfx12 (a 256 MiB
streaming write takes 1,042,140 requests, none of 64 B: 257.6 B per request). Writes are undercounted:
`GL2C_EA_WRREQ` counts the write-backs while a kernel runs, and the lines still dirty in L2 when a kernel ends are
written back without being counted in any dispatch (the flush kernels after them record 0 write requests). So every
kernel can miss up to the L2 size of writes (4 MiB on the RX 9060), small intermediates can show about 0 (FKL VF+HF at
N = 100: 5 requests for its 2.4 MB resize output), and a pipeline of several kernels loses more of its writes than a
fused one. Reads are complete: with the caches flushed before every GPU operation, every kernel reads its inputs from
beyond L2. Flush leak guard: the counts of a cache flush kernel occasionally land on the dispatches right after it (in
one collection on the RX 9060, 5 of 60 profiles: `FETCH_SIZE` +3.6, +4.1, +72.7 and +96.1 MiB, and one
`GL2C_EA_WRREQ_sum` about 98,000 requests too high), so `rocprof_counters.sh` collects a profile again when its
measured execution exceeds the smallest of its three warm-up executions (the same dispatches, with the caches also
flushed) by more than 2 % and 256 KiB of traffic, up to `MAX_REPEATS` (3) rounds; the rejected profiles stay in
`raw_rejected/` and `traffic_repeats.csv`, and `traffic.csv` has their count per row and whether a leak is still
suspected. Duration on the paper's machine (bare metal, Ubuntu 26.04, Ryzen 7 5800X3D, RX 9060, performance level
`STABLE_STD`), the paper run with `SKIP_COUNTERS=1`: 436 s, of which the builds 32 s, the GPU benchmark with
its GPU time only passes 103 s and the CPU benchmark 299 s (the counters took 61 s in an earlier run with one region
size).

The workload is the one of `common/bench_geometry.h`, with the three region sizes by turns (`BENCH_REGIONS=mixed`,
the paper's); one size for every region, as the NPPDx builds use, is another CMake option in `BENCH_CMAKE_ARGS`, for
example `BENCH_CMAKE_ARGS="-DBENCH_REGIONS=32x16" rocm/scripts/run_all.sh ../results/rocm-32x16` (both builds get it;
no option cached in the build directories by earlier runs applies, and the script stops if the GPU and CPU
binaries report different workloads). `BENCH_ITERATIONS` and `BENCH_WARMUP_ITERATIONS` (default 1000 and 100, as in
every benchmark of the repository) set the timed and warm-up iterations the same way, for example
`BENCH_CMAKE_ARGS="-DBENCH_ITERATIONS=1 -DBENCH_WARMUP_ITERATIONS=0"` for a quick check; the RPP selection passes keep
their own counts (20 + 200 on the GPU, 10 + 100 on the CPU). The CPU benchmark needs region sides of at most 256 pixels:
it measures the sampling positions of RPP HOST's `rppt_resize_mirror_normalize` with 8 bit ramps.

Or step by step (the cmake commands take the same `-DBENCH_*` options):

    . rocm/scripts/env.sh   # ROCM_PATH, ROCM_EXTRA, library paths
    cmake -S rocm -B build-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ \
          -DCMAKE_HIP_COMPILER=$ROCM_PATH/bin/amdclang++ -DRPP_ROOT=$ROCM_EXTRA -DROCPROFILER_SDK_ROOT=$ROCM_EXTRA
    cmake --build build-rocm
    ./build-rocm/rocm_bench ../results/rocm/gpu
    BENCH_RPP_VARIANT=<chosen variant of rpp_selection.csv> rocm/scripts/rocprof_counters.sh ../results/rocm/traffic
    cmake -S rocm/cpu -B build-rocm-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-23 \
          -DRPP_ROOT=$ROCM_EXTRA
    cmake --build build-rocm-cpu
    ./build-rocm-cpu/rpp_cpu_bench ../results/rocm/cpu
