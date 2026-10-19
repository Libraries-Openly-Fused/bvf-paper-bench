# BVF paper benchmarks

Benchmarks for the Backwards Vertical Fusion (BVF) paper of the
[FusedKernelLibrary](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary) (FKL). They measure the Figure 1
pipeline of the paper,

    ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write

on one 3840x2160 NV12 frame with N in {1, 4, 16, 50, 100} regions of 20x10, 32x16 and 48x24 that change every
iteration, each resized (bilinear) to 64x32, padded with 16 black rows on top and bottom to 64x64, normalized as
(x * mul - sub) / div and written as float3 into an N plane tensor, with FKL and with other libraries.
[`fkl-neighborhood/`](fkl-neighborhood/README.md) measures fixed-size 64x32 crops of the same frame, blurred with a box
filter and normalized, without the resize and the black rows (the neighborhood experiment).

## Layout

One folder per library compared with FKL, each its own CMake project with its own README:

| Folder | Compares | Programs |
|---|---|---|
| [`fkl-vs-opencv-cuda/`](fkl-vs-opencv-cuda/README.md) | FKL BVF with OpenCV's CUDA modules | `benchmark_opencv_cuda` |
| [`fkl-vs-npp/`](fkl-vs-npp/README.md) | FKL BVF with NVIDIA NPP | `benchmark_npp` |
| [`fkl-vs-nppdx/`](fkl-vs-nppdx/README.md) | FKL BVF with NVIDIA NPPDx, as one kernel with hand written CUDA code and combined with NPP without hand written pixel processing code | `benchmark_nppdx`, `benchmark_nppdx_npp` |
| [`fkl-vs-opencv-cpu/`](fkl-vs-opencv-cpu/README.md) | FKL's CPU backends with OpenCV on the CPU | `benchmark_opencv_cpu` |
| [`fkl-vs-fern-cpu/`](fkl-vs-fern-cpu/README.md) | FKL's CPU backends with the code Fern (gern, PLDI 2025) generates for the pipeline from black-box subroutines | `benchmark_fern_cpu` |
| [`fkl-ablation/`](fkl-ablation/README.md) | FKL with BVF against FKL without it (vertical and horizontal fusion only), FKL's host overhead, the type FKL builds | `benchmark_fkl_ablation`, `bvf_types` |
| [`fkl-neighborhood/`](fkl-neighborhood/README.md) | The neighborhood experiment: 64x32 crops, a box blur and the normalization; FKL's BVF box operation against its MorphologyDPP, and the DPP against NPP, OpenCV CUDA and NPPDx | `benchmark_fkl_neighborhood_k<K>_p<passes>[_resize]`, `benchmark_{npp,opencv_cuda,nppdx}_neighborhood`, `test_neighborhood_ops` |
| [`rocm/`](rocm/README.md) | FKL's HIP backend with AMD RPP on an AMD GPU, and FKL's CPU backends with RPP's HOST backend, on ROCm | `rocm_bench`, `rpp_cpu_bench` |

Next to them:

- [`common/`](common): the workload of every benchmark (`bench_geometry.h`, `bench_geometry.cmake`), the harness of the
  GPU benchmarks of OpenCV CUDA, NPP and FKL's ablation (`bench_gpu.h`, `bench_gpu.cmake`), the harness of the CPU
  benchmarks (`bench_cpu.h`, `bench_cpu.cmake`, FKL's side in `bench_fkl_cpu.cpp`), the pipeline's constants and its
  plain C++ reference (`bench_pipeline.h`, `bench_reference.h`), the timing statistics and FKL's two-execution CSV
  format (`bench_timing.h`), the validation rules shared by every GPU benchmark (`bench_validation.h`), NPP's error check and NV12 conversion (`bench_npp.h`), the
  platform description (`bench_platform.h`), the common part of the compile-time units (`compile_time_unit.h`), and the
  workload and BVF box operation of the neighborhood experiment (`bench_neighborhood.h`,
  `bench_neighborhood_ops.h`, shared with `rocm/`);
- [`scripts/`](scripts): `run_all.sh` (every benchmark but `rocm/`), the memory traffic with Nsight Compute
  (`ncu_dram.sh`, `ncu_l2.sh`, `ncu_common.sh`, `ncu_summarize.py`, `ncu_traffic_model.py`), the compile time
  (`compile_time.sh`, `compile_time_summarize.py`), the runner-up NPPDx configurations (`nppdx_runner_up.py`) and the
  CI (`ci/run_benchmarks.cmake`);
- the submodules `fkl` (FusedKernelLibrary main, with its `thread-pool` submodule for the `CPU_MULTITHREAD` backend),
  `nppdx` (NVIDIA NPPDx v0.1.2) and `gern` (Fern, d05bc66: the code generator of `fkl-vs-fern-cpu`, whose generated
  code is checked in; `update = none`, so recursive clones skip it, see
  [regenerating the Fern code](fkl-vs-fern-cpu/README.md#build-and-run)).

`rocm/` has its own scripts (`rocm/scripts/run_all.sh` runs both of its benchmarks and the hardware counters).

## Workload

Every benchmark of this repository takes its workload from [`common/bench_geometry.h`](common/bench_geometry.h):

- the frame: 3840x2160 NV12, uniform random values in the legal range of limited (video) range NV12, Y in [16, 235]
  and U, V in [16, 240] (seed 1), as in real video;
- the color conversion: NV12 to RGB with BT.601 in limited range, in every implementation, because it is the only
  NV12 conversion that every library offers (OpenCV's CPU `cvtColor` has no other; NPP's through its color twist with
  the BT.601 matrix, NPPDx's through `FromLimitedLuma`, `FromLimitedChroma` and `ColorConvert<yuv_bt601, rgb>`);
- the regions, new in every iteration (warm-up and timed): 100 per iteration, batch N uses the first N. Their sizes are
  20x10, 32x16 and 48x24 by turns: region i of iteration t has the size (i + t) mod 3 of that list, so every iteration
  mixes the three and every region cycles through them. Their positions are uniform random (seed 42), with even
  coordinates, so that every library reads the same NV12 chroma samples, and drawn so that a 48x24 box at the position
  is at least 8 pixels from the frame edges (whatever the region's size), so that a neighborhood operation can later be
  inserted before the crop without moving them;
- the resize to 64x32, aspect ratio ignored (every size is 2:1, so the ratio is 5:16, 1:2 or 3:4 in both dimensions);
- the border: 16 black rows on top and bottom, which make the output square, 64x64;
- the batch sizes N in {1, 4, 16, 50, 100}.

The regions of every iteration of a run are generated before its loops from one seed, the same for every
implementation, program and machine: iteration t of every loop uses the regions of iteration t (the warm-up
iterations are 0 to 99 and the timed ones 100 to 1099), and the regions of iteration t do not depend on how many
iterations are generated. Each pipeline converts them into its library's arguments before the loops too; what stays
inside the timed region is what the library needs to receive the regions of an iteration (FKL's `build()` calls and the
fusion of the chain, NPP's descriptor and ROI upload, OpenCV's per-region views and calls). `regions.csv` in every
results folder has the statistics of the timed iterations' regions (regions of each size, mean region area, and the
smallest, mean and largest total area of an iteration) for every N.

The NPPDx kernels are compiled for one region size, so FKL vs NPPDx is built for one of the three sizes at a time (all
of its regions of that size, at the same positions as above): the CMake option `BENCH_REGIONS` of
[`common/bench_geometry.cmake`](common/bench_geometry.cmake) is `mixed` (the default, the paper's workload) or one of
`20x10`, `32x16` and `48x24`, and every platform file records the workload. The frame bytes and the region positions
come from `std::mt19937` through a mapping written out in the header (the one libstdc++ uses), not through
`std::uniform_int_distribution`, whose algorithm differs between standard libraries: they are the same on every
platform and compiler.

## The GPU benchmarks

FKL vs OpenCV CUDA, FKL vs NPP and FKL's ablation share one harness, [`common/bench_gpu.h`](common/bench_gpu.h): each
program runs its library's pipeline (the baseline) and FKL BVF, one `executeOperations` call and one kernel for the
whole chain (the paper's Listing 2). FKL vs NPPDx has its own harness, described in its README, with the same three
timings.

The implementations never run at the same time:

- **Sequential loops.** For every batch size, each implementation's complete loop (100 warm-up and 1000 timed
  iterations) runs before the next one's, on the same regions, in a fixed order: the baselines first, FKL BVF last.
  Each loop therefore starts with the L2 state its own previous iteration left.
- **Paired speedups.** FKL BVF's speedup over a baseline is computed per iteration (baseline time / FKL BVF time of the
  same iteration, identical inputs) and averaged, as in FKL's two-execution benchmarks.
- **Earlier results.** The paper's first results (`results-v3`) alternated the implementations within every
  iteration instead.
- **`rocm/`.** The programs of `rocm/` (AMD GPU and RPP on the CPU) have their own harness, which pull request #19
  moves to this protocol. Until it is merged, they still alternate the implementations within every iteration, and
  their CPU program sleeps the idle gap and sets the thread pools' spin times, so their results are not comparable with
  those of this protocol.

Every run produces two benchmarks:

- CPU + GPU: CUDA events around everything the host does to enqueue a pipeline, plus its GPU execution.
- GPU only: a second pass with CUPTI kernel tracing, where the time of an execution is the sum of its kernels'
  execution times.

Other modes, selected with a second argument after the results folder:

| Mode | Measures |
|---|---|
| `--gpu-only-l2-flushed` | GPU time only with the L2 cache flushed before every kernel, so no intermediate or input is assumed to stay in L2 (as when other kernels run concurrently) |
| `--profile-kernels` | GPU time of every kernel of every pipeline (10 executions on the regions of iterations 0 to 9, then 100 traced ones on those of iterations 10 to 109) |
| `--ncu-target <impl> <N>` | runs one pipeline on the regions of iterations 1 to 3, then once on those of iteration 0 inside the NVTX range `measure`, for Nsight Compute |
| `--host-overhead` | (`benchmark_fkl_ablation` only) host time to enqueue the FKL pipelines, median, p10 and p90 over the timed iterations |

`<program> --batches`, `<program> --workload` and `<program> --regions <t>` (with no results folder) print the batch
sizes, the whole workload the binary was built for and the regions of iteration t (CSV), for the scripts; the NPPDx
benchmarks have the first two. No CUDA Graphs are used anywhere.

Output files: `latency.csv` (CPU + GPU: median, p10, p90 and mean in us, kernels launched, FKL BVF mean speedup, and
the largest error against the implementation's own reference), `gpu_time.csv` (GPU only, same columns),
`<baseline>_vs_fkl_bvf_gpu_only_GPU_NVIDIA.csv` and `<baseline>_vs_fkl_bvf_GPU_NVIDIA.csv` (FKL's two-execution
format, times in ms), `kernel_breakdown.csv` (time of each kernel or stage of the multi-kernel pipelines),
`validation.csv`, `kernels.csv`, `regions.csv` and `platform.txt`.

Every implementation is checked before the timing, on the regions of the first, every 50th and the last timed
iteration (iterations 100, 150, ..., 1050 and 1099; one row per implementation, N and iteration in `validation.csv`;
`latency.csv` has the largest error over them), against a reference with its own conventions: FKL BVF
against the chain on FKL's CPU backend in float (tolerance 1e-3); OpenCV and FKL VF+HF, which write an 8 bit RGB
frame, against the same chain with the RGB rounded to 8 bits; NPP, which also samples its resize differently (at
`dst * scale - 0.25`, measured for upscales only), against that with NPP's sampling. Against an 8 bit reference the
tolerance is one 8 bit level (a value on a half level can round either way), and at most one value in a thousand may be
more than 1e-3 away ([`common/bench_validation.h`](common/bench_validation.h)). The errors against FKL's float
reference are in `validation.csv` too. A program fails if one of FKL's outputs is not within its tolerance in one of
the validated iterations.

## Memory traffic with Nsight Compute

`scripts/ncu_dram.sh` (DRAM traffic) and `scripts/ncu_l2.sh` (traffic between the SMs and L2) profile the OpenCV CUDA,
NPP, FKL VF+HF and FKL BVF pipelines with Nsight Compute, through the `--ncu-target` modes: one execution on the
regions of iteration 0, the same for every pipeline. `ncu_dram.sh` profiles every batch size with the caches flushed
before every kernel (cold), and N = 1 and 100 without flushing (warm: L2 holds what the executions on iterations 1 to
3 left there, as in the timed loop, where every iteration has new regions), each profile twice (`REPEATS`): the DRAM
counters are device-wide, so a display on the same GPU adds other processes' traffic to some profiles, and the summary
keeps the smallest of them. `ncu_l2.sh` profiles N = 1 and 100 (cold).
They stop unless every benchmark was built for the same workload (`--workload`), take the batch sizes from the
benchmarks (`--batches`), and write the regions of iteration 0 to `ncu_regions.csv` in the results folder
(`--regions 0`) for the traffic model. The paths and their environment variables are at the top of
`scripts/ncu_common.sh`.

They need root while the driver restricts the GPU performance counters to administrators (`RmProfilingAdminOnly=1` in
`/proc/driver/nvidia/params`, the default); with the nvidia module option `NVreg_RestrictProfilingToAdminUsers=0` they
run without sudo. They change nothing on the system and give the reports back to the user who ran sudo. Summarize
them afterwards, without sudo, with `scripts/ncu_summarize.py` (DRAM) or `scripts/ncu_summarize.py l2`, then
`scripts/ncu_traffic_model.py` puts the measured reads of every pipeline and N next to a compulsory-read model
(`traffic_model.csv`; the model is described at the top of the script), with the regions of iteration 0 and their own
sizes. `scripts/run_all.sh` with `PROFILE=1` runs all of this.

## Compile time

How long nvcc takes to compile the pipelines. Every GPU benchmark folder has a `compile_time.cu` (in `src/`, or in
`benchmarks/` for FKL vs NPPDx) with one pipeline per translation unit (TU), selected with `-DIMPL_<NAME>` and
`-DCT_BATCH=<N>` (not `BATCH`, which FKL uses as a template parameter name): the folder's headers alone (`HEADERS`),
its library's pipeline (`OPENCV`, `NPP`, `NPPDX`) and FKL BVF (`BVF`); FKL's ablation has FKL's headers alone, `BVF`,
`BVF` with the `CONDITIONAL_WITH_DEFAULT` plane policy for 100 planes (`BVF_CWD`) and FKL VF+HF (`VFHF`). The
pipelines are those of the benchmarks without their scaffolding; the TUs are compiled, never run. The NPPDx kernel is
compiled in the tile and block size the NPPDx benchmark chose for N = 100, read from its configuration file
(`NPPDX_CONFIG`), which must come from a run with the region size of the build. The regions only enter the other TUs
as run-time values (sizes and rectangles), never as template arguments.

The units of each set: `ablation` FKL's headers, `VFHF` at N = 1 and 100, `BVF` at every N and `BVF_CWD` at N = 100;
`opencv-cuda` its headers and `OPENCV` at N = 100; `npp` its headers and `NPP` at N = 100; `nppdx` its headers, `BVF`
and `NPPDX` at N = 100 (`scripts/run_all.sh` compiles the headers and `BVF` in the first size's build only, and
`NPPDX` in every size's build).

`scripts/compile_time.sh` compiles the TUs of one folder (`SET=ablation`, `opencv-cuda`, `npp` or `nppdx`) with the
nvcc command line of its benchmark (read from `compile_commands.json`, so configure with
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`), 10 times after one discarded warm-up, round-robin, one nvcc at a time on an
idle machine (it waits for a load average below 0.5). It records the wall, user and sys time and max RSS with
`/usr/bin/time`, the phases with `nvcc --time` (with `--threads 0` they run in parallel, so their sum exceeds the wall
time), and from one more compile with `-Xptxas -v --keep` the kernels, registers, spills, stack frame and PTX, cubin
and object sizes. `scripts/compile_time_summarize.py` writes `compile_time_summary.csv` (one row per TU, medians, min
and max), `compile_time_kernels.csv` and `compile_time_phases.csv`, and prints a Markdown table.

    SET=ablation BUILD=build-ablation scripts/compile_time.sh ../results/compile_time/fkl-ablation
    SET=nppdx BUILD=build-nppdx-32x16 scripts/compile_time.sh ../results/compile_time/fkl-vs-nppdx-32x16   # after benchmark_nppdx

## The CPU benchmarks

FKL vs OpenCV on the CPU shares one harness with the other CPU benchmarks (FKL vs PyTorch, Halide and Fern; not
`rocm/`'s RPP program, see [the GPU benchmarks](#the-gpu-benchmarks)), [`common/bench_cpu.h`](common/bench_cpu.h). FKL's side is [`common/bench_fkl_cpu.cpp`](common/bench_fkl_cpu.cpp):
one `executeOperations` call (BVF) on FKL's CPU backend (one thread) and on its `CPU_MULTITHREAD` backend (FKL's
thread pool, one worker per hardware thread). Each program compares its library on one thread with FKL on one thread,
and on as many threads as FKL's pool has workers with FKL's pool.

The two libraries of a comparison never run at the same time:

- **Sequential loops.** For every batch size, the library's complete loop (100 warm-up and 1000 timed iterations) runs
  first, then FKL's, on the same regions.
- **Paired speedups.** FKL's speedup is the mean over the timed iterations of (library time / FKL time) of the same
  iteration, which pairs runs on identical inputs. Every comparison CSV has a `_per_iteration.csv` sibling with the
  pairs.
- **Idle gap.** Every run follows a 1 ms idle gap. The timing thread busy-waits it, so the library's thread pool,
  whose workers spin for a while after a run and then sleep, is asleep when a run starts, while the timing thread's
  core stays out of a deep C-state. Each library keeps its own pool and idle behaviour.
- **Worker busy time.** Each program measures how long the workers of both pools stay busy after a run with little
  work: their spin and wake-up, from `/proc/self/task/*/schedstat` on Linux. It records them in its platform file
  (`cpu_platform*.txt`).
- **Validation.** Before the timing, on the regions of the first, every 50th and the last timed iteration, FKL against
  the reference with its conventions and FKL's pool against FKL on one thread, and the library on its thread count
  against itself on one thread, must agree within 1e-3: otherwise the program writes its files and exits with an
  error. The library's errors against a reference of its own conventions are written next to them.

FKL's side is the only translation unit of a program that includes FKL: FKL's headers define variable template
specializations that are not `inline`, so a program can include them in one translation unit only. It is compiled with
FKL's options only (none of the compared library's), so its object file is the same in every CPU benchmark built with
the same compiler and flags. It is linked first, and a function in it is aligned to 4096 bytes (GCC and Clang), which
aligns its code section: every FKL function then has the same address modulo 4096 in every CPU benchmark, whatever the
sections before the code (which grow with each library's imported symbols) take. The platform file records the
addresses modulo 4096 of that function (0) and of one more FKL function, the same in every CPU benchmark. The [protocol change](fkl-vs-opencv-cpu/README.md#protocol-change)
section of FKL vs OpenCV on the CPU explains the busy-waited gap and the separate translation unit.

## Requirements

Each benchmark has its own requirements, besides a C++20 compiler:

| Benchmark | Needs |
|---|---|
| `fkl-vs-opencv-cuda` | an NVIDIA GPU, CUDA 13.3 or newer with NPP, CUPTI and NVTX, CMake 3.28, OpenCV 4.14 built with CUDA and the opencv_contrib modules `cudaarithm` and `cudawarping` |
| `fkl-vs-npp`, `fkl-ablation` | an NVIDIA GPU, CUDA 13.3 or newer with NPP, CUPTI and NVTX, CMake 3.28; no OpenCV |
| `fkl-vs-nppdx` | an NVIDIA GPU, CUDA 13.3 or newer with NPP, CUPTI and NVTX, CMake 3.30, a Linux x86_64 or Windows x64 machine (NPPDx's platforms); no OpenCV |
| `fkl-neighborhood` | an NVIDIA GPU, CUDA 13.3 or newer with NPP, CUPTI and NVTX, CMake 3.28, FusedKernelLibrary with the batched `MorphologyDPP` and `executeBoxBlur` (PR #393, the `fkl` submodule), OpenCV 4.14 built with CUDA and the opencv_contrib modules `cudaarithm`, `cudafilters` and `cudaimgproc` for its OpenCV program, and for its NPPDx programs what `fkl-vs-nppdx` needs (or `-DBENCH_NPPDX=OFF`) |
| `fkl-vs-opencv-cpu` | CMake 3.28 and OpenCV 4.14 with `core` and `imgproc`; no GPU or CUDA |
| `fkl-vs-fern-cpu` | CMake 3.28; nothing else (the Fern code is checked in; regenerating it needs the `gern` submodule, CMake 3.30 and Linux or macOS) |
| `rocm` | Linux, an AMD GPU (gfx1200 by default, another one with `-DCMAKE_HIP_ARCHITECTURES`), ROCm 10.1 with RPP 3.2.0 and rocprofiler-sdk, CMake 3.28, Ninja (see [its README](rocm/README.md)); no CUDA or OpenCV |

The paper's results were measured on Linux; the CI also builds and runs every benchmark but `rocm/` on Windows, and
every benchmark but FKL vs NPPDx and `rocm/` on ARM64.

Clone with `git clone --recursive https://github.com/Libraries-Openly-Fused/bvf-paper-bench.git`, or run
`git submodule update --init --recursive` in an existing clone. Every benchmark takes another FKL checkout with
`-DFKL_DIR=<path>`.

## Build and run

Every folder is a CMake project; the scripts expect the build folder `build-<name>` in this repository (`build-opencv-cuda`,
`build-npp`, `build-ablation`, `build-opencv-cpu`, and `build-nppdx-<size>` for every region size of FKL vs NPPDx)
and a `results` folder next to it, with the results of every benchmark in a folder of its name (FKL vs NPPDx in
`fkl-vs-nppdx/<size>`):

    cmake -S fkl-vs-npp -B build-npp -DCMAKE_BUILD_TYPE=Release
    cmake --build build-npp -j
    ./build-npp/benchmark_npp ../results/fkl-vs-npp                          # CPU + GPU and GPU only
    ./build-npp/benchmark_npp ../results/fkl-vs-npp --gpu-only-l2-flushed    # adds the *_l2_flushed* files

The README of every folder has its commands. `scripts/run_all.sh` builds and runs every benchmark but `rocm/` (which has
its own `rocm/scripts/run_all.sh`): the main run and the GPU only run with the L2 flushed of every GPU benchmark, FKL's
host overhead and per-kernel times, `bvf_types`, FKL vs NPPDx for every region size (`NPPDX_SIZES`) with the runner-up
NPPDx configuration of every N re-timed against FKL (`nppdx_runner_up.py`, then `--configs`, into `runner-up/`), every
program of the neighborhood experiment, and the CPU benchmarks of OpenCV and Fern; `COMPILE_TIME=1` adds the compile-time units and `PROFILE=1` the Nsight Compute scripts, their summaries
and the traffic model, when the driver lets every user read the GPU performance counters (`RmProfilingAdminOnly: 0`);
otherwise it prints their commands, which then need sudo:

    OpenCV_DIR=<OpenCV> CPU_CXX=clang++ scripts/run_all.sh ../results

`CUDA_ARCH` (default `native`, the GPU of the machine) selects the CUDA architectures of the GPU benchmarks.
`BENCH_ITERATIONS` and `BENCH_WARMUP_ITERATIONS` (default 1000 and 100, as for the paper's results) set the timed and
warm-up iterations of every benchmark, and `BENCH_REGIONS` the region sizes of the [workload](#workload) (`mixed`, the
default, except for FKL vs NPPDx, which takes one size, `32x16` by default).

## Continuous integration

[`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) runs on pull requests to `main` from branches
of this repository and by hand, on the self-hosted runners of FKL's CI with the toolchains of FKL's workflows (except
CUDA 13.0 on Windows: the benchmarks need 13.3 or newer): Linux x86_64 and ARM64 for the CPU benchmarks (g++-13 and
clang++-23), Linux CUDA x86_64 and ARM64 for the GPU benchmarks (g++-13, CUDA 13.4), Windows x64 for the CPU benchmarks
(cl and clang-cl) and Windows CUDA for the GPU benchmarks (cl, CUDA 13.4). The Linux ROCm runner has no NVIDIA GPU
and runs every Linux x86_64 CPU job; no job runs on the Windows ROCm runner, and no job builds or runs `rocm/`. FKL vs NPPDx runs on the x86_64 CUDA runners
only, the CPUs NPPDx supports.

Every job runs [`scripts/ci/run_benchmarks.cmake`](scripts/ci/run_benchmarks.cmake), which also runs on any machine
(see its header). It builds OpenCV 4.14.0 with the modules the benchmarks use once per runner and configuration, in
`~/bvf-paper-bench-deps` (so the first run on a runner takes much longer), builds the benchmarks (FKL vs NPPDx for
every region size, `NPPDX_SIZES`, and the NPPDx neighborhood programs) with one timed iteration and no warm-up (`ITERATIONS` and `WARMUP_ITERATIONS` of the
script), runs every mode except those for Nsight Compute (`--ncu-target` and the profiling scripts), regenerates the Fern code
on Linux with CMake 3.30 or newer (gern's minimum) from gern at the submodule's commit (cloned in
`~/bvf-paper-bench-deps`) and compares it with the checked-in copy, and fails if a
benchmark fails (a CPU benchmark fails its own validation) or if FKL's outputs, or those of a library the benchmark
requires (NPPDx's, and the CPU baselines' against their own references, as each folder's README says), do not match
their references (OpenCV's and NPP's are in the validation files but not required: they come from the OpenCV build and
the CUDA toolkit of each runner). The results
and logs of every job are uploaded as artifacts. Times measured on shared CI runners are not the paper's numbers.

The runners are persistent machines and this repository is public, so the workflow only runs trusted code: pull
requests from forks are skipped (push their commits to a branch of this repository after reviewing them to test them),
the checkout keeps no credentials, and the OpenCV sources that later jobs reuse are checked against their SHA-256.

## License

Apache License 2.0, like FKL.
