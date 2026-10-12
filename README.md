# BVF paper benchmarks

Benchmarks for the Backwards Vertical Fusion (BVF) paper of the
[FusedKernelLibrary](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary) (FKL). They measure the Figure 1
pipeline of the paper,

    ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write

on one 3840x2160 NV12 frame with N in {1, 4, 16, 50, 100} crops of 32x16, each resized (bilinear) to 64x32, padded
with 16 black rows on top and bottom to 64x64, normalized as (x * mul - sub) / div and written as float3 into an N plane
tensor. The comparison of FKL with NVIDIA's NPPDx, one kernel in each library, is in
[`fkl-vs-nppdx/`](fkl-vs-nppdx/README.md) (until October 2026 the separate repository FKL-vs-NPPDx), with its RTX 4090
results.

## Workload

Every benchmark of this repository (GPU, CPU, FKL vs NPPDx and the compile-time units) takes its workload from
[`common/bench_geometry.h`](common/bench_geometry.h):

- the frame: 3840x2160 NV12, uniform random bytes (seed 1);
- the crops: `BENCH_CROP_W` x `BENCH_CROP_H` (default 32x16) at uniform random positions (seed 42; batch N uses the
  first N of one sequence of 100), with even coordinates, so that every library reads the same NV12 chroma samples, and
  at least 8 pixels from the frame edges, so that a neighborhood operation can later be inserted before the crop
  without moving them;
- the resize to `BENCH_RESIZE_W` x `BENCH_RESIZE_H` (default 64x32), with the same ratio in both dimensions (NPPDx's
  `Resize<J, K>` takes one ratio, 1:2 by default);
- the border: (`BENCH_RESIZE_W` - `BENCH_RESIZE_H`) / 2 black rows on top and bottom, which make the output square
  (16 rows and 64x64 by default);
- the batch sizes N in {1, 4, 16, 50, 100}.

The four sizes are CMake options of every project ([`common/bench_geometry.cmake`](common/bench_geometry.cmake)), and
every platform file records them. The frame bytes and the crop positions come from `std::mt19937` through a mapping
written out in the header (the one libstdc++ uses), not through `std::uniform_int_distribution`, whose algorithm
differs between standard libraries: they are the same on every platform and compiler.

## GPU benchmark (`src/bvf_bench.cu`)

| Implementation | What runs |
|---|---|
| `opencv_cuda` | NV12 to RGB of the whole frame, then per crop `cv::cuda` convertTo, resize and copyMakeBorder, then multiply, subtract and divide on the whole output tensor |
| `npp` | the same steps with NPP (one batched resize; `nppiResize` for N = 1, since `nppiResizeBatch` rejects a batch of one) |
| `fkl_vf_hf` | FKL with vertical and horizontal fusion but without BVF: four kernels, with the intermediates in memory |
| `fkl_bvf` | FKL with BVF: one kernel for the whole chain |

OpenCV, NPP and FKL VF+HF write the same intermediates (an 8 bit RGB frame, then float3 crops, resized crops and
output), because the libraries cannot write a float RGB frame; FKL BVF keeps float3 values in registers from the color
conversion to the final write. Details of every implementation are at the top of `src/bvf_bench.cu`.

In each of the 1000 timed iterations (after 100 warm-up ones) the implementations run one after the other, and FKL
BVF's speedup over each baseline is computed per iteration and then averaged, as in FKL's two-execution benchmarks.
Every run produces two benchmarks:

- CPU + GPU: CUDA events around everything the host does to enqueue a pipeline, plus its GPU execution.
- GPU only: a second pass with CUPTI kernel tracing, where the time of an execution is the sum of its kernels'
  execution times.

Other modes, selected with a second argument:

| Mode | Measures |
|---|---|
| `--gpu-only-l2-flushed` | GPU time only with the L2 cache flushed before every kernel, so no intermediate or input is assumed to stay in L2 (as when other kernels run concurrently) |
| `--profile-kernels` | GPU time of every kernel of every pipeline |
| `--host-overhead` | host time to enqueue the FKL pipelines |
| `--ncu-target <impl> <N>` | runs one execution of one pipeline inside the NVTX range `measure`, for Nsight Compute |

`bvf_bench --batches` (with no results directory) prints the batch sizes, for the Nsight Compute scripts.

No CUDA Graphs are used anywhere.

Output files: `latency.csv` (CPU + GPU: median, p10, p90 and mean in us, kernels launched, FKL BVF mean speedup),
`gpu_time.csv` (GPU only, same columns), `<baseline>_vs_fkl_bvf_gpu_only_GPU_NVIDIA.csv` and
`<baseline>_vs_fkl_bvf_GPU_NVIDIA.csv` (FKL's two-execution format, times in ms), `kernel_breakdown.csv` (time of
each kernel or stage of the multi-kernel pipelines), `validation.csv`, `kernels.csv` and `platform.txt`.

### Memory traffic with Nsight Compute

`scripts/ncu_dram.sh` (DRAM traffic, with cold and warm caches) and `scripts/ncu_l2.sh` (traffic between the SMs and
L2) profile every pipeline for every batch size with Nsight Compute through the `--ncu-target` mode of this benchmark
and of the NPPDx one. They take the batch sizes from the benchmarks (`--batches`) and NPPDx's tile and block for every
N from the configuration file of the NPPDx benchmark's run (`../results/nppdx/benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv`,
or `NPPDX_CONFIG`), so run that one first. They need root when the driver restricts the GPU performance counters to
administrators (`RmProfilingAdminOnly=1`, the default); they change nothing on the system and give the reports back to
the user who ran sudo. Summarize them afterwards, without sudo, with `scripts/ncu_summarize.py` (DRAM) or
`scripts/ncu_summarize.py l2`.

### Compile time (`compile_time/`)

How long nvcc takes to compile the pipelines. `compile_time/pipeline.cu` holds one pipeline per translation unit (TU),
selected with `-DIMPL_<NAME>` and `-DCT_BATCH=<N>` (not `BATCH`, which FKL uses as a template parameter name): FKL BVF
(`BVF`), the same with the `CONDITIONAL_WITH_DEFAULT` plane policy for 100 planes (`BVF_CWD`), FKL VF+HF (`VFHF`),
OpenCV (`OPENCV`) and NPP (`NPP`), with the calls of `src/bvf_bench.cu`, plus two baselines without a pipeline: the FKL
headers only (`HEADERS_FKL`) and every header of the other TUs (`HEADERS_ALL`). The TUs are compiled, never run.
`fkl-vs-nppdx/benchmarks/compile_time_pipeline.cu` does the same for the NPPDx kernel of `nppdx_figure1.h` and FKL's
`runFKL` (`HEADERS`, `NPPDX`, `BVF`). The NPPDx kernel of every N is compiled in the tile and block size the NPPDx
benchmark chose for that N, read from its configuration file (`NPPDX_CONFIG`, default
`../results/nppdx/benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv`). The geometry only enters the other TUs as
run-time values (sizes and rectangles), never as template arguments.

`compile_time/run.sh` compiles every TU with the nvcc command line of the benchmark target (read from
`compile_commands.json`, so configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`), 10 times after one discarded warm-up,
round-robin, one nvcc at a time on an idle machine (it waits for a load average below 0.5). It records the wall, user
and sys time and max RSS with `/usr/bin/time`, the phases with `nvcc --time` (with `--threads 0` they run in parallel,
so their sum exceeds the wall time), and from one more compile with `-Xptxas -v --keep` the kernels, registers, spills,
stack frame and PTX, cubin and object sizes. `compile_time/summarize.py` writes `compile_time_summary.csv` (one row per
TU, medians, min and max), `compile_time_kernels.csv` and `compile_time_phases.csv`, and prints a Markdown table.

    compile_time/run.sh ../results/compile_time                    # BUILD=build: src/bvf_bench.cu's command
    SET=nppdx compile_time/run.sh ../results/compile_time_nppdx    # BUILD=build-nppdx: benchmark_figure1_bvf's command
                                                                   # (after its run into ../results/nppdx)

The results of the RTX 4090 machine for FKL vs NPPDx are in `fkl-vs-nppdx/results/RTX4090/compile_time*`.

## CPU benchmark (`cpu/`)

The same pipeline on the CPU: OpenCV (with 1 thread and with all hardware threads) against FKL's CPU backends with
BVF, on one thread (`ParArch::CPU`) and on FKL's thread pool (`ParArch::CPU_MULTITHREAD`). The color conversion is
BT.601 limited range in both, because that is the only NV12 conversion of OpenCV's CPU `cvtColor`. Every timed run
follows a 1 ms idle gap, so that both libraries' thread pools are asleep when it starts. Each run writes one CSV per
comparison in FKL's two-execution format, plus `cpu_validation.csv` and `cpu_platform.txt`.

## Requirements and layout

Each benchmark has its own requirements, besides a C++20 compiler:

| Benchmark | Needs |
|---|---|
| GPU (`.`) | an NVIDIA GPU, CUDA 13.3 or newer with NPP, CUPTI and NVTX, CMake 3.28, OpenCV 4.14 built with CUDA and the opencv_contrib modules `cudaarithm` and `cudawarping` |
| CPU (`cpu`) | CMake 3.28 and OpenCV 4.14 with `core` and `imgproc`; no GPU or CUDA |
| FKL vs NPPDx (`fkl-vs-nppdx`) | an NVIDIA GPU, CUDA 13.3 or newer with CUPTI and NVTX, CMake 3.30, a Linux x86_64 or Windows x64 machine (NPPDx's platforms); no OpenCV |

The paper's results were measured on Linux; the CI also builds and runs every benchmark on Windows, and the CPU
benchmark and the GPU one on ARM64.

- FKL and NPPDx are the submodules `fkl` (FusedKernelLibrary main, with its own `thread-pool` submodule for the
  `CPU_MULTITHREAD` backend) and `nppdx` (v0.1.2): clone with
  `git clone --recursive https://github.com/Libraries-Openly-Fused/bvf-paper-bench.git`, or run
  `git submodule update --init --recursive` in an existing clone. Every benchmark takes another FKL checkout with
  `-DFKL_DIR=<path>`.

There are three CMake projects, each with its own build folder: the GPU benchmark (`.`, in `build`), the CPU
benchmark (`cpu`, in `build-cpu`) and FKL vs NPPDx (`fkl-vs-nppdx`, in `build-nppdx`). The Nsight Compute scripts
expect those folders, and a `results` directory next to this repository.

## Build and run

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<OpenCV build or install>
    cmake --build build -j
    ./build/bvf_bench ../results                          # CPU + GPU and GPU only
    ./build/bvf_bench ../results --gpu-only-l2-flushed    # adds the *_l2_flushed* files (same for the other modes)
    ./build/bvf_types                                     # C++ type of the fused read operation for N = 100

    # FKL vs NPPDx (Linux x86_64 or Windows x64), see fkl-vs-nppdx/README.md
    cmake -S fkl-vs-nppdx -B build-nppdx -DCMAKE_BUILD_TYPE=Release
    cmake --build build-nppdx -j
    ./build-nppdx/benchmark_figure1_bvf ../results/nppdx
    ./build-nppdx/benchmark_figure1_bvf ../results/nppdx --gpu-only-l2-flushed   # NPPDx as chosen by the run above

    # CPU benchmark, preferably with a recent clang; -march=native by default (CPU_ARCH_FLAGS)
    cmake -S cpu -B build-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DOpenCV_DIR=<OpenCV>
    cmake --build build-cpu -j
    ./build-cpu/cpu_bench ../results/cpu

`CUDA_ARCH` (default `native`, the GPU of the machine) selects the CUDA architectures of the GPU benchmarks.
`BENCH_ITERATIONS` and `BENCH_WARMUP_ITERATIONS` (default 1000 and 100, as for the paper's results) set the timed and
warm-up iterations of every benchmark in all three projects, and `BENCH_CROP_W`, `BENCH_CROP_H`, `BENCH_RESIZE_W` and
`BENCH_RESIZE_H` (default 32, 16, 64 and 32) the [workload](#workload); give every project the same values (also
`rocm` and `rocm/cpu`, below).

## AMD GPU and CPU: FKL vs RPP (`rocm/`)

The same pipeline, [workload](#workload) (`common/bench_geometry.h`), constants and timing protocol on an AMD GPU
with ROCm: FKL's HIP backend against RPP (ROCm Performance Primitives), and FKL's CPU backends against RPP's HOST
backend. No HIP graphs anywhere.

### GPU benchmark (`rocm/src/rocm_bench.hip`)

| Implementation | What runs |
|---|---|
| `rpp` | the RPP pipeline of the benchmark: the fastest of the five RPP variants below, which are numerically equivalent (within 1e-5 of the RPP-convention reference); a selection pass validates and times all of them (CPU + GPU, interleaved, 20 warm-up + 200 timed iterations per N) and keeps the one with the lowest geometric mean of its medians over N (`rpp_selection.csv`); only that one is measured afterwards |
| `rpp_per_op` (variant) | one RPP call per operation: `rppt_yuv_to_rgb` (NV12 to 8 bit RGB, BT.2020 full range; it also uploads 8 constants with `hipMemcpyToSymbol` on every call), one batched `rppt_crop_mirror_normalize` 8u to 32f (all batch images alias the frame), `hipMemsetAsync` of the output (RPP has no constant border), one batched `rppt_resize` into the rows of the output planes below the top border, `rppt_multiply_scalar`, `rppt_tensor_subtract_tensor` and `rppt_tensor_divide_tensor` on the whole tensor |
| `rpp_npp_like` (variant) | the same calls with exactly the intermediates of the paper's NPP pipeline: the resized crops in their own buffer, copied into the output rows by a batched `rppt_crop` |
| `rpp_fused` (variant) | `rppt_yuv_to_rgb`, then RPP's fused `rppt_resize_mirror_normalize` (crop + 8u to 32f + resize + Mul; it zeroes its destination and synchronizes the stream itself), Sub and Div on the whole tensor (the border rows also need the normalization, which `rppt_resize_mirror_normalize` cannot apply to them) |
| `rpp_normalize` (variant) | `rpp_per_op` with Mul + Sub + Div in one `rppt_normalize` |
| `rpp_fused_normalize` (variant) | `rppt_yuv_to_rgb`, `rppt_resize_mirror_normalize` (crop + 8u to 32f + resize) and `rppt_normalize` (Mul + Sub + Div): the fewest RPP calls |
| `fkl_vf_hf` | the four FKL kernels of `makeFklVfHf<N, true>` of `src/bvf_bench.cu`, unchanged |
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
written. Validation: FKL VF+HF against FKL's CPU backend with the 8 bit RGB frame, FKL BVF against FKL's CPU backend
(max abs error below 2e-6); the RPP pipelines against a C++ reference built from RPP's own 8 bit RGB frame with RPP's
sampling convention, measured with linear ramps (`--resize-convention`: pixel centers, clamped to [0, size - 2]), and
against the paper's 8 bit reference (FKL's conversion rounded to 8 bit, FKL's sampling); every pipeline is validated
again after its timed iterations. The RPP behaviour that the pipelines take into account (handle batch size, ROI
formats converted in place, 8 pixel row padding, internal copies, memsets and synchronizations) is described in
`platform.txt`.

Modes, after the results directory: (none) the benchmark, `--validate`, `--resize-convention`
(`rpp_resize_convention.csv`), `--rocprof-target <impl> <N>` (the target of `rocm/scripts/rocprof_counters.sh`),
`--check-kernel-tracing`. Without a results directory, `rocm_bench --batches` prints the batch sizes (as
`bvf_bench --batches` does) and `rocm_bench --workload` the workload; `rpp_cpu_bench` has the same two.

Output files: those of the CUDA benchmark with `GPU_AMD` instead of `GPU_NVIDIA` (`latency.csv`, `validation.csv`,
`kernels.csv`, `kernel_breakdown.csv`, `<baseline>_vs_fkl_bvf_[gpu_only_|gpu_only_l2_flushed_]GPU_AMD.csv`,
`gpu_time.csv`, `gpu_time_l2_flushed.csv`, `platform.txt`), plus
`kernel_counts_per_operation.csv`, `rpp_selection.csv` and `rpp_resize_convention.csv`; `validation.csv` has one more
column, the error against the paper's 8 bit reference, and `gpu_time*.csv` five more, the GPU time and dispatches of the HIP runtime's blit kernels (memsets, copies) of each execution, which the GPU time only sums leave out. `--validate` validates every RPP variant.

### CPU benchmark (`rocm/cpu/`)

RPP's HOST backend (with 1 thread and with all hardware threads) against FKL's `ParArch::CPU` and
`ParArch::CPU_MULTITHREAD`, with the protocol of `cpu/`. RPP's NV12 conversion is HIP only, so the RPP HOST pipelines
get their 8 bit RGB frame from FKL's CPU conversion (one thread for the 1 thread runs, FKL's thread pool for the
others), then one of the five RPP variants on the HOST backend: for each thread count, a selection pass validates all
of them and times the ones numerically equivalent to `rpp_per_op` (10 warm-up + 100 iterations per N), and only the
fastest is measured (`cpu_rpp_selection.csv`; outputs `benchmark_cpu_rpp_1thread_vs_fkl_bvf_CPU.csv`,
`benchmark_cpu_rpp_<T>threads_vs_fkl_bvf_CPU.csv` and `benchmark_cpu_rpp_<T>threads_vs_fkl_bvf_CPU_MULTITHREAD.csv`).
RPP HOST's sampling positions are measured with ramps for this geometry (`rpp_host_sampling_positions.csv`) and used
by its reference: its `rppt_resize` and `rppt_resize_mirror_normalize` treat the last columns and rows differently, so
the two variants built on `rppt_resize_mirror_normalize` are not equivalent to `rpp_per_op` on the CPU. RPP 3.2.0's
HOST `rppt_crop_mirror_normalize` (8u to 32f, packed RGB) writes one float (1.0) past the end of every row when the
ROI width is a multiple of 16, as with the 32 pixel crops: in a tightly packed crop buffer the last row's lands on the
first float of the next crop, which another OpenMP thread may have written already, so every crop is followed by a
spare row. RPP's HOST backend parallelizes over the batch images with OpenMP: the
arithmetic calls see the output as N samples, and `KMP_BLOCKTIME` is 100 us (libomp's default is 200 ms), like FKL's
thread pool, so that the idle gap before every timed run puts both pools to sleep.

### Requirements

- Linux with ROCm 10.1 (the paper's run: a Radeon RX 9060, gfx1200), CMake 3.28, Ninja, and a recent clang for the CPU
  benchmark.
- The ROCm 10.1 packages `amdrocm-rpp-dev10.1` (RPP 3.2.0), `amdrocm-profiler-base10.1` and `amdrocm-profiler10.1`
  (rocprofiler-sdk, rocprofv3) and `amdrocm-amdsmi10.1`. Without root, unpack them with
  `apt-get download <package> && dpkg -x <package>.deb <prefix>` and set `ROCM_EXTRA=<prefix>/opt/rocm/core-10.1`.
- FusedKernelLibrary at `63782f6`, the commit of every benchmark of this repository: by default the repository's `fkl`
  submodule with its `thread-pool` submodule (`git submodule update --init --recursive fkl`), for both the GPU and the
  CPU benchmark; `run_all.sh` takes other checkouts with `FKL_GPU_DIR` / `FKL_CPU_DIR` and checks the commit.

### Build and run

Everything, in one command (it writes a platform snapshot first, builds, checks that FKL is at the right commits and
unmodified and that kernel dispatch tracing and the hardware counters are available, then runs the GPU benchmark, the
counters and the CPU benchmark; the counters default to `FETCH_SIZE GL2C_EA_WRREQ_sum GL2C_EA_WRREQ_64B_sum`, since
rocprofiler-sdk 1.4.1 defines no `WRITE_SIZE` for gfx12):

    rocm/scripts/run_all.sh ../results/rocm

Before running it, on the bare-metal machine: nothing else running (`uptime` load below 0.5, which the script checks),
nothing else on the GPU (`amd-smi process`), and the performance level and clock lock state as they are: the script
records them (`amd-smi metric`, `rocm-smi --showperflevel`) and does not change them. Hardware counters need access to
the GPU's performance counters (membership of the `render` group). Duration on the development machine (WSL2,
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

## Continuous integration

[`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) runs on pull requests to `main` from branches
of this repository and by hand, on the self-hosted runners of FKL's CI with the toolchains of FKL's workflows (except
CUDA 13.0 on Windows: the benchmarks need 13.3 or newer): Linux x86_64 and ARM64 for the CPU benchmark (g++-13 and
clang++-23), Linux CUDA x86_64 and ARM64 for the GPU benchmarks (g++-13, CUDA 13.4), Windows x64 for the CPU benchmark
(cl and clang-cl) and Windows CUDA for the GPU benchmarks (cl, CUDA 13.4). The Linux and Windows ROCm runners have no
NVIDIA GPU and run the CPU benchmark only. FKL vs NPPDx runs on the x86_64 CUDA runners only, the CPUs NPPDx supports.

Every job runs [`scripts/ci/run_benchmarks.cmake`](scripts/ci/run_benchmarks.cmake), which also runs on any machine
(see its header). It builds OpenCV 4.14.0 with the modules the benchmarks use once per runner and configuration, in
`~/bvf-paper-bench-deps` (so the first run on a runner takes much longer), builds the benchmarks with one timed
iteration and no warm-up (`ITERATIONS` and `WARMUP_ITERATIONS` of the script), runs every mode except the Nsight
Compute profiles (they need root), and fails if a benchmark fails or if FKL's outputs do not match their references
(the baselines' errors are reported, not checked: OpenCV and NPP round and sample differently). The results and logs
of every job are uploaded as artifacts. Times measured on shared CI runners are not the paper's numbers.

The runners are persistent machines and this repository is public, so the workflow only runs trusted code: pull
requests from forks are skipped (push their commits to a branch of this repository after reviewing them to test them),
the checkout keeps no credentials, and the OpenCV sources that later jobs reuse are checked against their SHA-256.

## License

Apache License 2.0, like FKL.
