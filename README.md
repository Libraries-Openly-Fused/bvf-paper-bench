# BVF paper benchmarks

Benchmarks for the Backwards Vertical Fusion (BVF) paper of the
[FusedKernelLibrary](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary) (FKL). They measure the Figure 1
pipeline of the paper,

    ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write

on one 3840x2160 NV12 frame with N in {1, 4, 16, 50, 100} crops of 32x16, each resized (bilinear) to 64x32, padded
with 16 black rows on top and bottom to 64x64, normalized as (x * mul - sub) / div and written as float3 into an N plane
tensor, with FKL and with other libraries.

## Layout

One folder per library compared with FKL, each its own CMake project with its own README:

| Folder | Compares | Programs |
|---|---|---|
| [`fkl-vs-opencv-cuda/`](fkl-vs-opencv-cuda/README.md) | FKL BVF with OpenCV's CUDA modules | `benchmark_opencv_cuda` |
| [`fkl-vs-npp/`](fkl-vs-npp/README.md) | FKL BVF with NVIDIA NPP | `benchmark_npp` |
| [`fkl-vs-nppdx/`](fkl-vs-nppdx/README.md) | FKL BVF with NVIDIA NPPDx, as one kernel with hand written CUDA code and combined with NPP without hand written pixel processing code | `benchmark_nppdx`, `benchmark_nppdx_npp` |
| [`fkl-vs-opencv-cpu/`](fkl-vs-opencv-cpu/README.md) | FKL's CPU backends with OpenCV on the CPU | `benchmark_opencv_cpu` |
| [`fkl-ablation/`](fkl-ablation/README.md) | FKL with BVF against FKL without it (vertical and horizontal fusion only), FKL's host overhead, the type FKL builds | `benchmark_fkl_ablation`, `bvf_types` |
| [`rocm/`](rocm/README.md) | FKL's HIP backend with AMD RPP on an AMD GPU, and FKL's CPU backends with RPP's HOST backend, on ROCm | `rocm_bench`, `rpp_cpu_bench` |

Next to them:

- [`common/`](common): the workload of every benchmark (`bench_geometry.h`, `bench_geometry.cmake`), the harness of the
  GPU benchmarks of OpenCV CUDA, NPP and FKL's ablation (`bench_gpu.h`, `bench_gpu.cmake`), the validation rules
  shared by every GPU benchmark (`bench_validation.h`), NPP's error check and NV12 conversion (`bench_npp.h`), the
  platform description (`bench_platform.h`) and the common part of the compile-time units (`compile_time_unit.h`);
- [`scripts/`](scripts): `run_all.sh` (every benchmark for every workload), the memory traffic with Nsight Compute
  (`ncu_dram.sh`, `ncu_l2.sh`, `ncu_common.sh`, `ncu_summarize.py`, `ncu_traffic_model.py`), the compile time
  (`compile_time.sh`, `compile_time_summarize.py`) and the CI (`ci/run_benchmarks.cmake`);
- the submodules `fkl` (FusedKernelLibrary main, with its `thread-pool` submodule for the `CPU_MULTITHREAD` backend)
  and `nppdx` (NVIDIA NPPDx v0.1.2).

`rocm/` has its own scripts (`rocm/scripts/run_all.sh` runs both of its benchmarks and the hardware counters).

## Workload

Every benchmark of this repository takes its workload from [`common/bench_geometry.h`](common/bench_geometry.h):

- the frame: 3840x2160 NV12, uniform random values in the legal range of limited (video) range NV12, Y in [16, 235]
  and U, V in [16, 240] (seed 1), as in real video;
- the color conversion: NV12 to RGB with BT.601 in limited range, in every implementation, because it is the only
  NV12 conversion that every library offers (OpenCV's CPU `cvtColor` has no other; NPP's through its color twist with
  the BT.601 matrix, NPPDx's through `FromLimitedLuma`, `FromLimitedChroma` and `ColorConvert<yuv_bt601, rgb>`);
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

## The GPU benchmarks

FKL vs OpenCV CUDA, FKL vs NPP and FKL's ablation share one harness, [`common/bench_gpu.h`](common/bench_gpu.h): each
program runs its library's pipeline (the baseline) and FKL BVF, one `executeOperations` call and one kernel for the
whole chain (the paper's Listing 2). FKL vs NPPDx has its own harness, described in its README, with the same three
timings.

In each of the 1000 timed iterations (after 100 warm-up ones) the baseline and FKL BVF run one after the other, and
FKL BVF's speedup over the baseline is computed per iteration and then averaged, as in FKL's two-execution
benchmarks. Every run produces two benchmarks:

- CPU + GPU: CUDA events around everything the host does to enqueue a pipeline, plus its GPU execution.
- GPU only: a second pass with CUPTI kernel tracing, where the time of an execution is the sum of its kernels'
  execution times.

Other modes, selected with a second argument after the results folder:

| Mode | Measures |
|---|---|
| `--gpu-only-l2-flushed` | GPU time only with the L2 cache flushed before every kernel, so no intermediate or input is assumed to stay in L2 (as when other kernels run concurrently) |
| `--profile-kernels` | GPU time of every kernel of every pipeline |
| `--ncu-target <impl> <N>` | runs one execution of one pipeline inside the NVTX range `measure`, for Nsight Compute |
| `--host-overhead` | (`benchmark_fkl_ablation` only) host time to enqueue the FKL pipelines |

`<program> --batches` and `<program> --workload` (with no results folder) print the batch sizes and the whole workload
the binary was built for, for the scripts; the NPPDx benchmarks have the same two modes. No CUDA Graphs are used
anywhere.

Output files: `latency.csv` (CPU + GPU: median, p10, p90 and mean in us, kernels launched, FKL BVF mean speedup, and
the largest error against the implementation's own reference), `gpu_time.csv` (GPU only, same columns),
`<baseline>_vs_fkl_bvf_gpu_only_GPU_NVIDIA.csv` and `<baseline>_vs_fkl_bvf_GPU_NVIDIA.csv` (FKL's two-execution
format, times in ms), `kernel_breakdown.csv` (time of each kernel or stage of the multi-kernel pipelines),
`validation.csv`, `kernels.csv` and `platform.txt`.

Every implementation is checked against a reference with its own conventions, named in `validation.csv`: FKL BVF
against the chain on FKL's CPU backend in float (tolerance 1e-3); OpenCV and FKL VF+HF, which write an 8 bit RGB
frame, against the same chain with the RGB rounded to 8 bits; NPP, which also samples its resize differently (at
`dst * scale - 0.25`, measured for upscales only), against that with NPP's sampling. Against an 8 bit reference the
tolerance is one 8 bit level (a value on a half level can round either way), and at most one value in a thousand may be
more than 1e-3 away ([`common/bench_validation.h`](common/bench_validation.h)). The errors against FKL's float
reference are in `validation.csv` too. A program fails if one of FKL's outputs is not within its tolerance.

## Memory traffic with Nsight Compute

`scripts/ncu_dram.sh` (DRAM traffic, with cold and warm caches) and `scripts/ncu_l2.sh` (traffic between the SMs and
L2) profile every pipeline of every GPU benchmark for every batch size with Nsight Compute, through their
`--ncu-target` modes. They stop unless every benchmark was built for the same workload (`--workload`) and the NPPDx
configuration files come from runs with that workload (the `Workload` line of the platform file next to each). They
take the batch sizes from the benchmarks (`--batches`) and NPPDx's tile and block for every N from the configuration
files of the NPPDx benchmarks' runs (`fkl-vs-nppdx/benchmark_nppdx_nppdx_config_GPU_NVIDIA.csv` and
`fkl-vs-nppdx/benchmark_nppdx_npp_nppdx_config_GPU_NVIDIA.csv` in the results folder of the workload), so run those
first. The paths and their environment variables are at the top of `scripts/ncu_common.sh`.

They need root while the driver restricts the GPU performance counters to administrators (`RmProfilingAdminOnly=1` in
`/proc/driver/nvidia/params`, the default); with the nvidia module option `NVreg_RestrictProfilingToAdminUsers=0` they
run without sudo. They change nothing on the system and give the reports back to the user who ran sudo. Summarize
them afterwards, without sudo, with `scripts/ncu_summarize.py` (DRAM) or `scripts/ncu_summarize.py l2`, then
`scripts/ncu_traffic_model.py` puts the measured reads of every pipeline and N next to a compulsory-read model
(`traffic_model.csv`; the model is described at the top of the script). It takes the workload from the platform file of
FKL's ablation and regenerates the crop positions, so it works for any workload. `scripts/run_all.sh` with `PROFILE=1`
runs all of this for every workload.

## Compile time

How long nvcc takes to compile the pipelines. Every GPU benchmark folder has a `compile_time.cu` (in `src/`, or in
`benchmarks/` for FKL vs NPPDx) with one pipeline per translation unit (TU), selected with `-DIMPL_<NAME>` and
`-DCT_BATCH=<N>` (not `BATCH`, which FKL uses as a template parameter name): the folder's headers alone (`HEADERS`),
its library's pipeline (`OPENCV`, `NPP`, `NPPDX`) and FKL BVF (`BVF`); FKL's ablation has FKL's headers alone, `BVF`,
`BVF` with the `CONDITIONAL_WITH_DEFAULT` plane policy for 100 planes (`BVF_CWD`) and FKL VF+HF (`VFHF`). The
pipelines are those of the benchmarks without their scaffolding; the TUs are compiled, never run. The NPPDx kernel of
every N is compiled in the tile and block size the NPPDx benchmark chose for that N, read from its configuration file
(`NPPDX_CONFIG`), which must come from a run with the workload of the build. The geometry only enters the other TUs as
run-time values (sizes and rectangles), never as template arguments.

`scripts/compile_time.sh` compiles the TUs of one folder (`SET=ablation`, `opencv-cuda`, `npp` or `nppdx`) with the
nvcc command line of its benchmark (read from `compile_commands.json`, so configure with
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`), 10 times after one discarded warm-up, round-robin, one nvcc at a time on an
idle machine (it waits for a load average below 0.5). It records the wall, user and sys time and max RSS with
`/usr/bin/time`, the phases with `nvcc --time` (with `--threads 0` they run in parallel, so their sum exceeds the wall
time), and from one more compile with `-Xptxas -v --keep` the kernels, registers, spills, stack frame and PTX, cubin
and object sizes. `scripts/compile_time_summarize.py` writes `compile_time_summary.csv` (one row per TU, medians, min
and max), `compile_time_kernels.csv` and `compile_time_phases.csv`, and prints a Markdown table.

    SET=ablation BUILD=build-ablation scripts/compile_time.sh ../results/compile_time/fkl-ablation
    SET=nppdx BUILD=build-nppdx scripts/compile_time.sh ../results/compile_time/fkl-vs-nppdx   # after benchmark_nppdx

## Requirements

Each benchmark has its own requirements, besides a C++20 compiler:

| Benchmark | Needs |
|---|---|
| `fkl-vs-opencv-cuda` | an NVIDIA GPU, CUDA 13.3 or newer with NPP, CUPTI and NVTX, CMake 3.28, OpenCV 4.14 built with CUDA and the opencv_contrib modules `cudaarithm` and `cudawarping` |
| `fkl-vs-npp`, `fkl-ablation` | an NVIDIA GPU, CUDA 13.3 or newer with NPP, CUPTI and NVTX, CMake 3.28; no OpenCV |
| `fkl-vs-nppdx` | an NVIDIA GPU, CUDA 13.3 or newer with NPP, CUPTI and NVTX, CMake 3.30, a Linux x86_64 or Windows x64 machine (NPPDx's platforms); no OpenCV |
| `fkl-vs-opencv-cpu` | CMake 3.28 and OpenCV 4.14 with `core` and `imgproc`; no GPU or CUDA |
| `rocm` | Linux, an AMD GPU, ROCm 10.1 with RPP 3.2.0 and rocprofiler-sdk, CMake 3.28, Ninja (see [its README](rocm/README.md)); no CUDA or OpenCV |

The paper's results were measured on Linux; the CI also builds and runs every benchmark on Windows, and every
benchmark but FKL vs NPPDx on ARM64.

Clone with `git clone --recursive https://github.com/Libraries-Openly-Fused/bvf-paper-bench.git`, or run
`git submodule update --init --recursive` in an existing clone. Every benchmark takes another FKL checkout with
`-DFKL_DIR=<path>`.

## Build and run

Every folder is a CMake project; the scripts expect the build folder `build-<name>` in this repository (`build-opencv-cuda`,
`build-npp`, `build-nppdx`, `build-ablation`, `build-opencv-cpu`) and a `results` folder next to it, with the results
of every benchmark in a folder of its name:

    cmake -S fkl-vs-npp -B build-npp -DCMAKE_BUILD_TYPE=Release
    cmake --build build-npp -j
    ./build-npp/benchmark_npp ../results/fkl-vs-npp                          # CPU + GPU and GPU only
    ./build-npp/benchmark_npp ../results/fkl-vs-npp --gpu-only-l2-flushed    # adds the *_l2_flushed* files

The README of every folder has its commands. `scripts/run_all.sh` builds and runs every benchmark for both workloads
of the paper, the 32x16 crops resized 1:2 and the earlier 20x10 crops resized 5:16 (`GEOMETRIES`), each with its own
build folders (`build-npp-32x16`, ...) and results folder (`../results/32x16/fkl-vs-npp`, ...); `COMPILE_TIME=1`
adds the compile-time units and `PROFILE=1` the Nsight Compute scripts, their summaries and the traffic model of each
workload, when the driver lets every user read the GPU performance counters (`RmProfilingAdminOnly: 0`); otherwise it
prints their commands, which then need sudo:

    OpenCV_DIR=<OpenCV> CPU_CXX=clang++ scripts/run_all.sh ../results

`CUDA_ARCH` (default `native`, the GPU of the machine) selects the CUDA architectures of the GPU benchmarks.
`BENCH_ITERATIONS` and `BENCH_WARMUP_ITERATIONS` (default 1000 and 100, as for the paper's results) set the timed and
warm-up iterations of every benchmark, and `BENCH_CROP_W`, `BENCH_CROP_H`, `BENCH_RESIZE_W` and `BENCH_RESIZE_H`
(default 32, 16, 64 and 32) the [workload](#workload); give every project the same values.

## Continuous integration

[`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) runs on pull requests to `main` from branches
of this repository and by hand, on the self-hosted runners of FKL's CI with the toolchains of FKL's workflows (except
CUDA 13.0 on Windows: the benchmarks need 13.3 or newer): Linux x86_64 and ARM64 for the CPU benchmark (g++-13 and
clang++-23), Linux CUDA x86_64 and ARM64 for the GPU benchmarks (g++-13, CUDA 13.4), Windows x64 for the CPU benchmark
(cl and clang-cl) and Windows CUDA for the GPU benchmarks (cl, CUDA 13.4). The Linux ROCm runner has no NVIDIA GPU
and runs every Linux x86_64 CPU job; no job runs on the Windows ROCm runner, and no job builds or runs `rocm/`. FKL vs NPPDx runs on the x86_64 CUDA runners
only, the CPUs NPPDx supports.

Every job runs [`scripts/ci/run_benchmarks.cmake`](scripts/ci/run_benchmarks.cmake), which also runs on any machine
(see its header). It builds OpenCV 4.14.0 with the modules the benchmarks use once per runner and configuration, in
`~/bvf-paper-bench-deps` (so the first run on a runner takes much longer), builds the benchmarks for both workloads
of the paper (`GEOMETRIES`) with one timed iteration and no warm-up (`ITERATIONS` and `WARMUP_ITERATIONS` of the
script), runs every mode except those for Nsight Compute (`--ncu-target` and the profiling scripts), and fails if a
benchmark fails or if FKL's or NPPDx's outputs do not match their references (OpenCV's and NPP's are in
`validation.csv` but not required: they come from the OpenCV build and the CUDA toolkit of each runner). The results
and logs of every job are uploaded as artifacts. Times measured on shared CI runners are not the paper's numbers.

The runners are persistent machines and this repository is public, so the workflow only runs trusted code: pull
requests from forks are skipped (push their commits to a branch of this repository after reviewing them to test them),
the checkout keeps no credentials, and the OpenCV sources that later jobs reuse are checked against their SHA-256.

## License

Apache License 2.0, like FKL.
