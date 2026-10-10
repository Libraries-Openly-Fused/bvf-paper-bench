# BVF paper benchmarks

Benchmarks for the Backwards Vertical Fusion (BVF) paper of the
[FusedKernelLibrary](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary) (FKL). They measure the Figure 1
pipeline of the paper,

    ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write

on one 3840x2160 NV12 frame with N in {1, 4, 16, 50, 100} crops of 20x10, each resized (bilinear) to 64x32, padded
with 16 black rows on top and bottom, normalized as (x * mul - sub) / div and written as float3 into an N plane tensor.
The comparison of FKL with NVIDIA's NPPDx is in the separate repository
[FKL-vs-NPPDx](https://github.com/Libraries-Openly-Fused/FKL-vs-NPPDx).

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

No CUDA Graphs are used anywhere.

Output files: `latency.csv` (CPU + GPU: median, p10, p90 and mean in us, kernels launched, FKL BVF mean speedup),
`gpu_time.csv` (GPU only, same columns), `<baseline>_vs_fkl_bvf_gpu_only_GPU_NVIDIA.csv` and
`<baseline>_vs_fkl_bvf_GPU_NVIDIA.csv` (FKL's two-execution format, times in ms), `kernel_breakdown.csv` (time of
each kernel or stage of the multi-kernel pipelines), `validation.csv`, `kernels.csv` and `platform.txt`.

### Memory traffic with Nsight Compute

`scripts/ncu_dram.sh` (DRAM traffic, with cold and warm caches) and `scripts/ncu_l2.sh` (traffic between the SMs and
L2) profile every pipeline with Nsight Compute through the `--ncu-target` mode of this benchmark and of
FKL-vs-NPPDx's. They need root when the driver restricts the GPU performance counters to administrators
(`RmProfilingAdminOnly=1`, the default); they change nothing on the system and give the reports back to the user who
ran sudo. Summarize them afterwards, without sudo, with `scripts/ncu_summarize.py` (DRAM) or
`scripts/ncu_summarize.py l2`.

## CPU benchmark (`cpu/`)

The same pipeline on the CPU: OpenCV (with 1 thread and with all hardware threads) against FKL's CPU backends with
BVF, on one thread (`ParArch::CPU`) and on FKL's thread pool (`ParArch::CPU_MULTITHREAD`). The color conversion is
BT.601 limited range in both, because that is the only NV12 conversion of OpenCV's CPU `cvtColor`. Every timed run
follows a 1 ms idle gap, so that both libraries' thread pools are asleep when it starts. Each run writes one CSV per
comparison in FKL's two-execution format, plus `cpu_validation.csv` and `cpu_platform.txt`.

## Requirements and layout

- Linux, an NVIDIA GPU, CUDA 13 (with NPP, CUPTI and NVTX), CMake 3.28 and a C++20 compiler.
- OpenCV 4.14 built with CUDA and the opencv_contrib modules `cudaarithm` and `cudawarping` (the CPU benchmark only
  needs `core` and `imgproc`).
- A FusedKernelLibrary checkout, by default next to this repository (set another one with `-DFKL_DIR=<path>`), with
  the `AddBorder` operation: FKL `main` once
  [PR #370](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary/pull/370) is merged, or its branch
  `369-add-addborder-operation` until then. The CPU benchmark also needs the `CPU_MULTITHREAD` backend and its
  `thread-pool` submodule, which go in FKL's `cpu-multithread` branch on top of AddBorder.

The default paths assume one directory that holds this repository, FusedKernelLibrary, FKL-vs-NPPDx (only for the
Nsight Compute scripts) and a `results` directory:

    <workspace>/paper-bench
    <workspace>/FusedKernelLibrary
    <workspace>/FKL-vs-NPPDx
    <workspace>/results

## Build and run

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<OpenCV build or install>
    cmake --build build -j
    ./build/bvf_bench ../results                          # CPU + GPU and GPU only
    ./build/bvf_bench ../results --gpu-only-l2-flushed    # adds the *_l2_flushed* files (same for the other modes)
    ./build/bvf_types                                     # C++ type of the fused read operation for N = 100

    # CPU benchmark, preferably with a recent clang; -march=native by default (CPU_ARCH_FLAGS)
    cmake -S cpu -B build-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DOpenCV_DIR=<OpenCV>
    cmake --build build-cpu -j
    ./build-cpu/cpu_bench ../results/cpu

## License

Apache License 2.0, like FKL.
