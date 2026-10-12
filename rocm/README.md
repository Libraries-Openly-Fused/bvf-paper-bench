# FKL vs RPP on ROCm (AMD GPU and CPU)

Part of the [BVF paper benchmarks](../README.md): the Figure 1 pipeline on an AMD GPU with ROCm, FKL's HIP backend
against RPP (ROCm Performance Primitives), and FKL's CPU backends against RPP's HOST backend, on the
[workload](../README.md#workload) of every benchmark of this repository (`common/bench_geometry.h`), with the constants
and timing protocol of the CUDA benchmarks ([`common/bench_gpu.h`](../common/bench_gpu.h)). No HIP graphs anywhere.
The commands below run from the repository root.

## GPU benchmark (`rocm/src/rocm_bench.hip`)

| Implementation | What runs |
|---|---|
| `rpp` | the RPP pipeline of the benchmark: the fastest of the five RPP variants below, which are numerically equivalent (within 1e-5 of the RPP-convention reference); a selection pass validates and times all of them (CPU + GPU, interleaved, 20 warm-up + 200 timed iterations per N) and keeps the one with the lowest geometric mean of its medians over N (`rpp_selection.csv`); only that one is measured afterwards |
| `rpp_per_op` (variant) | one RPP call per operation: `rppt_yuv_to_rgb` (NV12 to 8 bit RGB, BT.601 limited range: `RpptColorStandard_BT601`, `RpptColorRange_STUDIO`; it also uploads 8 constants with `hipMemcpyToSymbol` on every call), one batched `rppt_crop_mirror_normalize` 8u to 32f (all batch images alias the frame), `hipMemsetAsync` of the output (RPP has no constant border), one batched `rppt_resize` into the rows of the output planes below the top border, `rppt_multiply_scalar`, `rppt_tensor_subtract_tensor` and `rppt_tensor_divide_tensor` on the whole tensor |
| `rpp_npp_like` (variant) | the same calls with exactly the intermediates of the paper's NPP pipeline: the resized crops in their own buffer, copied into the output rows by a batched `rppt_crop` |
| `rpp_fused` (variant) | `rppt_yuv_to_rgb`, then RPP's fused `rppt_resize_mirror_normalize` (crop + 8u to 32f + resize + Mul; it zeroes its destination and synchronizes the stream itself), Sub and Div on the whole tensor (the border rows also need the normalization, which `rppt_resize_mirror_normalize` cannot apply to them) |
| `rpp_normalize` (variant) | `rpp_per_op` with Mul + Sub + Div in one `rppt_normalize` |
| `rpp_fused_normalize` (variant) | `rppt_yuv_to_rgb`, `rppt_resize_mirror_normalize` (crop + 8u to 32f + resize) and `rppt_normalize` (Mul + Sub + Div): the fewest RPP calls |
| `fkl_vf_hf` | the four FKL kernels of `makeFklVfHf<N, true>` of [`common/bench_gpu.h`](../common/bench_gpu.h), unchanged |
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
written. Validation, with the rules of every GPU benchmark ([`common/bench_validation.h`](../common/bench_validation.h):
1e-3 against a float reference, one level of 8 bit RGB more against an 8 bit one, at most one value in a thousand more
than 1e-3 away; the program fails if an FKL output is outside its tolerance, and reports the RPP pipelines'): FKL VF+HF
against FKL's CPU backend with the 8 bit RGB frame, FKL BVF against FKL's CPU backend in float; the RPP pipelines against a C++ reference built from RPP's own 8 bit RGB frame with RPP's
sampling convention, measured with linear ramps (`--resize-convention`: pixel centers, clamped to [0, size - 2]), and
against the paper's 8 bit reference (FKL's conversion rounded to 8 bit, FKL's sampling); every pipeline is validated
again after its timed iterations. The RPP behaviour that the pipelines take into account (handle batch size, ROI
formats converted in place, 8 pixel row padding, internal copies, memsets and synchronizations) is described in
`platform.txt`.

Modes, after the results directory: (none) the benchmark, `--validate`, `--resize-convention`
(`rpp_resize_convention.csv`), `--rocprof-target <impl> <N>` (the target of `rocm/scripts/rocprof_counters.sh`),
`--check-kernel-tracing`. Without a results directory, `rocm_bench --batches` prints the batch sizes (as
the CUDA benchmarks' `--batches` does) and `rocm_bench --workload` the workload; `rpp_cpu_bench` has the same two.

Output files: those of the CUDA benchmark with `GPU_AMD` instead of `GPU_NVIDIA` (`latency.csv`, `validation.csv`,
`kernels.csv`, `kernel_breakdown.csv`, `<baseline>_vs_fkl_bvf_[gpu_only_|gpu_only_l2_flushed_]GPU_AMD.csv`,
`gpu_time.csv`, `gpu_time_l2_flushed.csv`, `platform.txt`), plus
`kernel_counts_per_operation.csv`, `rpp_selection.csv` and `rpp_resize_convention.csv`; `validation.csv` has two more
columns, the errors against the RPP-convention reference and against the paper's 8 bit reference, and `gpu_time*.csv` five more, the GPU time and dispatches of the HIP runtime's blit kernels (memsets, copies) of each execution, which the GPU time only sums leave out. `--validate` validates every RPP variant.

## CPU benchmark (`rocm/cpu/`)

RPP's HOST backend against FKL's CPU backends, with the same number of threads on both sides: RPP with 1 thread
against FKL's `ParArch::CPU`, and RPP with all hardware threads against FKL's `ParArch::CPU_MULTITHREAD`, with the
protocol of [`fkl-vs-opencv-cpu/`](../fkl-vs-opencv-cpu/README.md). RPP's NV12 conversion is HIP only, so the RPP HOST pipelines
get their 8 bit RGB frame from FKL's CPU conversion (one thread for the 1 thread runs, FKL's thread pool for the
others), then one of the five RPP variants on the HOST backend: for each thread count, a selection pass validates all
of them and times the ones numerically equivalent to `rpp_per_op` (10 warm-up + 100 iterations per N), and only the
fastest is measured (`cpu_rpp_selection.csv`; outputs `benchmark_cpu_rpp_1thread_vs_fkl_bvf_CPU.csv` and
`benchmark_cpu_rpp_<T>threads_vs_fkl_bvf_CPU_MULTITHREAD.csv`).
RPP HOST's sampling positions are measured with ramps for this geometry (`rpp_host_sampling_positions.csv`) and used
by its reference: its `rppt_resize` and `rppt_resize_mirror_normalize` treat the last columns and rows differently, so
the two variants built on `rppt_resize_mirror_normalize` are not equivalent to `rpp_per_op` on the CPU. RPP 3.2.0's
HOST `rppt_crop_mirror_normalize` (8u to 32f, packed RGB) writes one float (1.0) past the end of every row when the
ROI width is a multiple of 16, as with the 32 pixel crops: in a tightly packed crop buffer the last row's lands on the
first float of the next crop, which another OpenMP thread may have written already, so every crop is followed by a
spare row. RPP's HOST backend parallelizes over the batch images with OpenMP: the
arithmetic calls see the output as N samples, and `KMP_BLOCKTIME` is 100 us (libomp's default is 200 ms), like FKL's
thread pool, so that the idle gap before every timed run puts both pools to sleep.

## Requirements

- Linux with ROCm 10.1 (the paper's run: a Radeon RX 9060, gfx1200), CMake 3.28, Ninja, and a recent clang for the CPU
  benchmark.
- The ROCm 10.1 packages `amdrocm-rpp-dev10.1` (RPP 3.2.0), `amdrocm-profiler-base10.1` and `amdrocm-profiler10.1`
  (rocprofiler-sdk, rocprofv3) and `amdrocm-amdsmi10.1`. Without root, unpack them with
  `apt-get download <package> && dpkg -x <package>.deb <prefix>` and set `ROCM_EXTRA=<prefix>/opt/rocm/core-10.1`.
- FusedKernelLibrary at `63782f6`, the commit of every benchmark of this repository: by default the repository's `fkl`
  submodule with its `thread-pool` submodule (`git submodule update --init --recursive fkl`), for both the GPU and the
  CPU benchmark; `run_all.sh` takes other checkouts with `FKL_GPU_DIR` / `FKL_CPU_DIR` and checks the commit.

## Build and run

Everything, in one command (it writes a platform snapshot first, builds, checks that FKL is at the right commits and
unmodified and that kernel dispatch tracing and the hardware counters are available and count, then runs the GPU
benchmark, the counters and the CPU benchmark; the counters default to `FETCH_SIZE GL2C_EA_WRREQ_sum
GL2C_EA_WRREQ_64B_sum`, since rocprofiler-sdk 1.4.1 defines no `WRITE_SIZE` for gfx12):

    rocm/scripts/run_all.sh ../results/rocm

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
suspected. Duration on the development machine (WSL2,
Ryzen 7 5800X3D, RX 9060): 431 s (builds 29 s, GPU benchmark 26 s, CPU benchmark 371 s), without the GPU time only passes and the counters, which WSL2 cannot run; on bare metal they add two more passes of 1,000 iterations per N (GPU time only, and with L2 + MALL flushed) and one rocprofv3 run per counter, pipeline and N.

The workload is the one of `common/bench_geometry.h` with its default sizes; for other sizes, give the same CMake
options as to the other projects in `BENCH_CMAKE_ARGS`, for example
`BENCH_CMAKE_ARGS="-DBENCH_CROP_W=20 -DBENCH_CROP_H=10" rocm/scripts/run_all.sh ../results/rocm-20x10` (both builds get
them; options cached in the build directories by earlier runs are reset first, and the script stops if the GPU and CPU
binaries report different workloads). The CPU benchmark needs crop sides of at most 256 pixels: it measures the
sampling positions of RPP HOST's `rppt_resize_mirror_normalize` with 8 bit ramps.

Or step by step (the cmake commands take the same `-DBENCH_*` options):

    . rocm/scripts/env.sh   # ROCM_PATH, ROCM_EXTRA, library paths
    cmake -S rocm -B build-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ \
          -DCMAKE_HIP_COMPILER=$ROCM_PATH/bin/amdclang++ -DRPP_ROOT=$ROCM_EXTRA -DROCPROFILER_SDK_ROOT=$ROCM_EXTRA
    cmake --build build-rocm
    ./build-rocm/rocm_bench ../results/rocm/gpu
    BENCH_RPP_VARIANT=<chosen variant of rpp_selection.csv> rocm/scripts/rocprof_counters.sh ../results/rocm/traffic
    cmake -S rocm/cpu -B build-rocm-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ \
          -DRPP_ROOT=$ROCM_EXTRA
    cmake --build build-rocm-cpu
    ./build-rocm-cpu/rpp_cpu_bench ../results/rocm/cpu
