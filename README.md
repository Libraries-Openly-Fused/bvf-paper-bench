# BVF paper benchmarks

Benchmarks for the Backwards Vertical Fusion (BVF) paper of the
[FusedKernelLibrary](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary) (FKL). They measure the Figure 1
pipeline of the paper,

    ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write

on one 3840x2160 NV12 frame with N in {1, 4, 16, 50, 100} crops of 32x16, each resized (bilinear) to 64x32, padded
with 16 black rows on top and bottom to 64x64, normalized as (x * mul - sub) / div and written as float3 into an N plane
tensor. The comparisons of FKL with NVIDIA's NPPDx are in [`fkl-vs-nppdx/`](fkl-vs-nppdx/README.md) (until October
2026 the separate repository FKL-vs-NPPDx), with their RTX 4090 results: NPPDx as one kernel, with hand written CUDA
code for the steps NPPDx has no operation for, and NPPDx combined with NPP without hand written pixel processing code.

## Workload

Every benchmark of this repository (GPU, CPU, FKL vs NPPDx and the compile-time units) takes its workload from
[`common/bench_geometry.h`](common/bench_geometry.h):

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

`bvf_bench --batches` and `bvf_bench --workload` (with no results directory) print the batch sizes and the whole
workload the binary was built for, for the scripts; the two NPPDx benchmarks have the same two modes.

No CUDA Graphs are used anywhere.

Output files: `latency.csv` (CPU + GPU: median, p10, p90 and mean in us, kernels launched, FKL BVF mean speedup),
`gpu_time.csv` (GPU only, same columns), `<baseline>_vs_fkl_bvf_gpu_only_GPU_NVIDIA.csv` and
`<baseline>_vs_fkl_bvf_GPU_NVIDIA.csv` (FKL's two-execution format, times in ms), `kernel_breakdown.csv` (time of
each kernel or stage of the multi-kernel pipelines), `validation.csv`, `kernels.csv` and `platform.txt`.

### Memory traffic with Nsight Compute

`scripts/ncu_dram.sh` (DRAM traffic, with cold and warm caches) and `scripts/ncu_l2.sh` (traffic between the SMs and
L2) profile every pipeline for every batch size with Nsight Compute through the `--ncu-target` mode of this benchmark
and of the two NPPDx ones. They stop unless the three benchmarks were built for the same workload (`--workload`) and
the NPPDx configuration files come from runs with that workload (the `Workload` line of the platform file next to
each). They take the batch sizes from the benchmarks (`--batches`) and NPPDx's tile and block for every N from the
configuration files of the NPPDx benchmarks' runs (`../results/nppdx/benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv`
and `../results/nppdx_npp/benchmark_figure1_nppdx_npp_nppdx_config_GPU_NVIDIA.csv`, or `NPPDX_CONFIG` and
`NPPDX_NPP_CONFIG`), so run those first. They need root while the driver restricts the GPU performance counters
to administrators (`RmProfilingAdminOnly=1` in `/proc/driver/nvidia/params`, the default); with the nvidia module
option `NVreg_RestrictProfilingToAdminUsers=0` they run without sudo. They change nothing on the system and give the
reports back to the user who ran sudo. Summarize them afterwards, without sudo, with `scripts/ncu_summarize.py` (DRAM) or
`scripts/ncu_summarize.py l2`, then `scripts/ncu_traffic_model.py` puts the measured reads of every pipeline and N next
to a compulsory-read model (`traffic_model.csv`; the model is described at the top of the script). It takes the
workload from the NPPDx benchmark's platform file and regenerates the crop positions, so it works for any workload.

To profile another workload, for example the earlier 20x10 crops resized 5:16, build the benchmarks with the same
options into their own folders, run the NPPDx benchmarks into a results folder of that workload, and point the scripts
and the summary at them (`BVF_RESULTS` is the folder of the workload, with the NPPDx runs in `nppdx/` and
`nppdx_npp/` and the reports in `ncu/` and `ncu_l2/`):

    G="-DBENCH_CROP_W=20 -DBENCH_CROP_H=10 -DBENCH_RESIZE_W=64 -DBENCH_RESIZE_H=32"
    cmake -S . -B build-20x10 -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<OpenCV> $G && cmake --build build-20x10 -j
    cmake -S fkl-vs-nppdx -B build-nppdx-20x10 -DCMAKE_BUILD_TYPE=Release $G && cmake --build build-nppdx-20x10 -j
    ./build-nppdx-20x10/benchmark_figure1_bvf ../results/20x10/nppdx
    ./build-nppdx-20x10/benchmark_figure1_nppdx_npp ../results/20x10/nppdx_npp
    sudo PAPER_BENCH=build-20x10/bvf_bench NPPDX_BENCH=build-nppdx-20x10/benchmark_figure1_bvf \
         NPPDX_NPP_BENCH=build-nppdx-20x10/benchmark_figure1_nppdx_npp \
         BVF_RESULTS=../results/20x10 scripts/ncu_dram.sh        # and the same for scripts/ncu_l2.sh
    BVF_RESULTS=../results/20x10 scripts/ncu_summarize.py        # and scripts/ncu_summarize.py l2
    BVF_RESULTS=../results/20x10 scripts/ncu_traffic_model.py

### Compile time (`compile_time/`)

How long nvcc takes to compile the pipelines. `compile_time/pipeline.cu` holds one pipeline per translation unit (TU),
selected with `-DIMPL_<NAME>` and `-DCT_BATCH=<N>` (not `BATCH`, which FKL uses as a template parameter name): FKL BVF
(`BVF`), the same with the `CONDITIONAL_WITH_DEFAULT` plane policy for 100 planes (`BVF_CWD`), FKL VF+HF (`VFHF`),
OpenCV (`OPENCV`) and NPP (`NPP`), with the calls of `src/bvf_bench.cu`, plus two baselines without a pipeline: the FKL
headers only (`HEADERS_FKL`) and every header of the other TUs (`HEADERS_ALL`). The TUs are compiled, never run.
`fkl-vs-nppdx/benchmarks/compile_time_pipeline.cu` does the same for the NPPDx kernel of `nppdx_figure1.h` and FKL's
`runFKL` (`HEADERS`, `NPPDX`, `BVF`). The NPPDx kernel of every N is compiled in the tile and block size the NPPDx
benchmark chose for that N, read from its configuration file (`NPPDX_CONFIG`, default
`../results/nppdx/benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv`), which must come from a run with the workload
of the build (`run.sh` checks it as the Nsight Compute scripts do). The geometry only enters the other TUs as
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

The same pipeline on the CPU: OpenCV against FKL's CPU backends with BVF, with the same number of threads on both
sides: OpenCV with 1 thread against FKL on one thread (`ParArch::CPU`), and OpenCV with all hardware threads against
FKL on its thread pool (`ParArch::CPU_MULTITHREAD`). The color conversion is BT.601 limited range in both, because
that is the only NV12 conversion of OpenCV's CPU `cvtColor`. Every timed run follows a 1 ms idle gap, so that both
libraries' thread pools are asleep when it starts. Each run writes one CSV per comparison in FKL's two-execution
format, plus `cpu_validation.csv` and `cpu_platform.txt`.

## Requirements and layout

Each benchmark has its own requirements, besides a C++20 compiler:

| Benchmark | Needs |
|---|---|
| GPU (`.`) | an NVIDIA GPU, CUDA 13.3 or newer with NPP, CUPTI and NVTX, CMake 3.28, OpenCV 4.14 built with CUDA and the opencv_contrib modules `cudaarithm` and `cudawarping` |
| CPU (`cpu`) | CMake 3.28 and OpenCV 4.14 with `core` and `imgproc`; no GPU or CUDA |
| FKL vs NPPDx (`fkl-vs-nppdx`) | an NVIDIA GPU, CUDA 13.3 or newer with NPP, CUPTI and NVTX, CMake 3.30, a Linux x86_64 or Windows x64 machine (NPPDx's platforms); no OpenCV |

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
    ./build-nppdx/benchmark_figure1_nppdx_npp ../results/nppdx_npp               # NPPDx + NPP
    ./build-nppdx/benchmark_figure1_nppdx_npp ../results/nppdx_npp --gpu-only-l2-flushed

    # CPU benchmark, preferably with a recent clang; -march=native by default (CPU_ARCH_FLAGS)
    cmake -S cpu -B build-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DOpenCV_DIR=<OpenCV>
    cmake --build build-cpu -j
    ./build-cpu/cpu_bench ../results/cpu

`scripts/run_all.sh` does all of this for both workloads of the paper, the 32x16 crops resized 1:2 and the earlier
20x10 crops resized 5:16 (`GEOMETRIES`), each with its own build folders (`build-32x16`, `build-nppdx-32x16`,
`build-cpu-32x16`, ...) and results folder (`../results/32x16`, `../results/20x10`, with the FKL vs NPPDx runs in
`nppdx/` and `nppdx_npp/` and the CPU one in `cpu/`); `COMPILE_TIME=1` adds the compile-time units. When the
driver lets every user read the GPU performance counters (`RmProfilingAdminOnly: 0`), it also runs the Nsight Compute
scripts, their summaries and the traffic model of each workload; otherwise it prints their commands, which then need
sudo:

    OpenCV_DIR=<OpenCV> CPU_CXX=clang++ scripts/run_all.sh ../results

`CUDA_ARCH` (default `native`, the GPU of the machine) selects the CUDA architectures of the GPU benchmarks.
`BENCH_ITERATIONS` and `BENCH_WARMUP_ITERATIONS` (default 1000 and 100, as for the paper's results) set the timed and
warm-up iterations of every benchmark in all three projects, and `BENCH_CROP_W`, `BENCH_CROP_H`, `BENCH_RESIZE_W` and
`BENCH_RESIZE_H` (default 32, 16, 64 and 32) the [workload](#workload); give all three projects the same values.

## Continuous integration

[`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) runs on pull requests to `main` from branches
of this repository and by hand, on the self-hosted runners of FKL's CI with the toolchains of FKL's workflows (except
CUDA 13.0 on Windows: the benchmarks need 13.3 or newer): Linux x86_64 and ARM64 for the CPU benchmark (g++-13 and
clang++-23), Linux CUDA x86_64 and ARM64 for the GPU benchmarks (g++-13, CUDA 13.4), Windows x64 for the CPU benchmark
(cl and clang-cl) and Windows CUDA for the GPU benchmarks (cl, CUDA 13.4). The Linux and Windows ROCm runners have no
NVIDIA GPU and run the CPU benchmark only. FKL vs NPPDx runs on the x86_64 CUDA runners only, the CPUs NPPDx supports.

Every job runs [`scripts/ci/run_benchmarks.cmake`](scripts/ci/run_benchmarks.cmake), which also runs on any machine
(see its header). It builds OpenCV 4.14.0 with the modules the benchmarks use once per runner and configuration, in
`~/bvf-paper-bench-deps` (so the first run on a runner takes much longer), builds the benchmarks for both workloads
of the paper (`GEOMETRIES`) with one timed iteration and no warm-up (`ITERATIONS` and `WARMUP_ITERATIONS` of the
script), runs every mode except the Nsight Compute profiles (they need root), and fails if a benchmark fails or if
FKL's or NPPDx's outputs do not match their references (the errors of OpenCV and NPP are reported, not checked: they
round and sample differently). The results and logs
of every job are uploaded as artifacts. Times measured on shared CI runners are not the paper's numbers.

The runners are persistent machines and this repository is public, so the workflow only runs trusted code: pull
requests from forks are skipped (push their commits to a branch of this repository after reviewing them to test them),
the checkout keeps no credentials, and the OpenCV sources that later jobs reuse are checked against their SHA-256.

## License

Apache License 2.0, like FKL.
