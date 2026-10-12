# FKL vs OpenCV CUDA

Part of the [BVF paper benchmarks](../README.md): the Figure 1 pipeline with OpenCV's CUDA modules and with FKL, on
the [workload](../README.md#workload) of every benchmark of this repository (NV12 to RGB in BT.601 limited range).

| Implementation | What runs |
|---|---|
| `opencv_cuda` | NV12 to RGB of the whole frame with NPP (OpenCV's CUDA modules have no NV12 conversion) into an 8 bit RGB frame, then per crop `cv::cuda` `convertTo` (float3 crop), `resize` (float3 resized crop) and `copyMakeBorder` (into its output plane), then `multiply`, `subtract` and `divide` on the whole float3 output: 3N + 4 kernels |
| `fkl_bvf` | FKL with BVF: one `executeOperations` call, one kernel for the whole chain, with no intermediates in memory |

OpenCV is checked against the chain on FKL's CPU backend with the RGB rounded to 8 bits, like its 8 bit RGB frame
(OpenCV's resize samples at `dst * scale`, like FKL's). The timings, modes, output files and validation are those of
every GPU benchmark on [`common/bench_gpu.h`](../common/bench_gpu.h), described in the [main README](../README.md#the-gpu-benchmarks).

## Build and run

Needs OpenCV 4.14 built with CUDA and the opencv_contrib modules `cudaarithm` and `cudawarping`:

    cmake -S fkl-vs-opencv-cuda -B build-opencv-cuda -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<OpenCV build or install>
    cmake --build build-opencv-cuda -j
    ./build-opencv-cuda/benchmark_opencv_cuda ../results/fkl-vs-opencv-cuda                         # CPU + GPU, GPU only
    ./build-opencv-cuda/benchmark_opencv_cuda ../results/fkl-vs-opencv-cuda --gpu-only-l2-flushed
    ./build-opencv-cuda/benchmark_opencv_cuda ../results/fkl-vs-opencv-cuda --profile-kernels

The compile-time units are in [`src/compile_time.cu`](src/compile_time.cu) (`HEADERS`, `OPENCV`, `BVF`; with
`SET=opencv-cuda scripts/compile_time.sh`, see the [main README](../README.md#compile-time)).
