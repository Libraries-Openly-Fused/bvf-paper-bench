# FKL vs OpenCV CUDA

Part of the [BVF paper benchmarks](../README.md): the Figure 1 pipeline with OpenCV's CUDA modules and with FKL, on
the [workload](../README.md#workload) of every benchmark of this repository (NV12 to RGB in BT.601 limited range).

| Implementation | What runs |
|---|---|
| `opencv_cuda` | OpenCV's own NV12 to RGB of the whole frame into an 8 bit RGB frame: cudacodec's `createNVSurfaceToColorConverter(ColorSpaceStandard::BT601, false)` and `convert(SF_NV12, ColorFormat::RGB, BitDepth::EIGHT, interleaved)`, one kernel, on the raw NV12 frame without a decoder session ([`common/bench_opencv_cuda.h`](../common/bench_opencv_cuda.h)). It converts whole surfaces only (its kernel reads the chroma of luma row y at nHeight - y / 2 rows below it, nHeight the surface's luma height, so a crop of the frame cannot be given on its own). Then per region `cv::cuda` `convertTo` (float3 crop, into a buffer of the largest region size through a view of the region's size), `resize` (float3 resized crop) and `copyMakeBorder` (into its output plane), then `multiply`, `subtract` and `divide` on the whole float3 output: 3N + 4 kernels. The `GpuMat` views of every region (of the RGB frame and of its crop buffer) are made in the iteration, inside the timed region |
| `fkl_bvf` | FKL with BVF: one `executeOperations` call, one kernel for the whole chain, with no intermediates in memory |

OpenCV's converter computes in float with the standard's coefficients and truncates to 8 bits, so OpenCV is checked
against the chain on FKL's CPU backend with the RGB truncated to 8 bits (`fkl_cpu_8bit_rgb_truncated`; OpenCV's resize
samples at `dst * scale`, like FKL's). `platform.txt` compares its conversion of the whole frame with FKL's, rounded and
truncated. The timings, modes, output files and validation are those of
every GPU benchmark on [`common/bench_gpu.h`](../common/bench_gpu.h), described in the [main README](../README.md#the-gpu-benchmarks).

## Build and run

Needs OpenCV 4.14 built with CUDA and the opencv_contrib modules `cudaarithm`, `cudacodec` and `cudawarping` (the
converter needs OpenCV's CUDA build only, not the Video Codec SDK):

    cmake -S fkl-vs-opencv-cuda -B build-opencv-cuda -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<OpenCV build or install>
    cmake --build build-opencv-cuda -j
    ./build-opencv-cuda/benchmark_opencv_cuda ../results/fkl-vs-opencv-cuda                         # CPU + GPU, GPU only
    ./build-opencv-cuda/benchmark_opencv_cuda ../results/fkl-vs-opencv-cuda --gpu-only-l2-flushed
    ./build-opencv-cuda/benchmark_opencv_cuda ../results/fkl-vs-opencv-cuda --profile-kernels

The compile-time units are in [`src/compile_time.cu`](src/compile_time.cu) (`HEADERS`, `OPENCV`, `BVF`; with
`SET=opencv-cuda scripts/compile_time.sh`, see the [main README](../README.md#compile-time)).
