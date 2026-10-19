# FKL vs OpenCV CUDA

Part of the [BVF paper benchmarks](../README.md): the Figure 1 pipeline with OpenCV's CUDA modules and with FKL, on
the [workload](../README.md#workload) of every benchmark of this repository (NV12 to RGB in BT.601 limited range).

| Implementation | What runs |
|---|---|
| `opencv_cuda` | OpenCV's own NV12 to RGB of the whole frame into an 8 bit RGB frame: cudacodec's `createNVSurfaceToColorConverter(ColorSpaceStandard::BT601, false)` and `convert(SF_NV12, ColorFormat::RGB, BitDepth::EIGHT, interleaved)`, one kernel, on the raw NV12 frame without a decoder session ([`common/bench_opencv_cuda.h`](../common/bench_opencv_cuda.h)). Then per region `cv::cuda` `convertTo` (float3 crop, into a buffer of the largest region size through a view of the region's size), `resize` (float3 resized crop) and `copyMakeBorder` (into its output plane), then `multiply`, `subtract` and `divide` on the whole float3 output: 3N + 4 kernels. The `GpuMat` views of every region (of the RGB frame and of its crop buffer) are made in the iteration, inside the timed region |
| `opencv_cuda_crop_nv12` | The same pipeline with the conversion of each region's columns only: per region, one `convert` of the full-height strip of the frame's columns [x, x + w) (a `GpuMat` view of 2160 x w at column x, with the frame's pitch) into the same columns of the 8 bit RGB frame, which the rest of the pipeline reads as `opencv_cuda`: 4N + 3 kernels. The converter cannot restrict the rows: its kernel reads the chroma of luma row y at nHeight - y / 2 rows below it, nHeight the surface's luma height, so a crop's rows cannot be given on their own. It can restrict the columns: the strip's chroma is where the kernel reads it, x being even. The frame is written only at the columns of the iteration's regions, and only pixels of the regions are read; before every validated run, the 8 bit frame is filled with 0 |
| `fkl_bvf` | FKL with BVF: one `executeOperations` call, one kernel for the whole chain, with no intermediates in memory |

The two OpenCV forms each have their own loop and speedup in every mode; per batch size, the one with the lower latency
median is OpenCV's, in every mode (the paper's rule for a library's variants).

OpenCV's converter computes in float with the standard's coefficients and truncates to 8 bits, so both OpenCV forms are
checked against the chain on FKL's CPU backend with the RGB truncated to 8 bits (`fkl_cpu_8bit_rgb_truncated`; OpenCV's
resize samples at `dst * scale`, like FKL's). `platform.txt` compares its conversion of the whole frame with FKL's,
rounded and truncated. The timings, modes, output files and validation are those of every GPU benchmark on
[`common/bench_gpu.h`](../common/bench_gpu.h), described in the [main README](../README.md#the-gpu-benchmarks).

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
