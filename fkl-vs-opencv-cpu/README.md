# FKL vs OpenCV on the CPU

Part of the [BVF paper benchmarks](../README.md): the Figure 1 pipeline on the CPU with OpenCV and with FKL's CPU
backends (BVF), on the [workload](../README.md#workload) of every benchmark of this repository (NV12 to RGB in
BT.601 limited range, the only NV12 conversion of OpenCV's CPU `cvtColor`).

Two comparisons, each with the same number of threads on both sides:

- OpenCV with 1 thread against FKL on one thread (`ParArch::CPU`);
- OpenCV with as many threads as FKL's thread pool has workers (one per hardware thread) against FKL on its thread
  pool (`ParArch::CPU_MULTITHREAD`).

OpenCV runs `cvtColor` on the whole frame (8 bit RGB), then per region `convertTo` (into a buffer of the largest region
size, through a view of the region's size), `resize` and `copyMakeBorder`, then multiply, subtract and divide on the
whole tensor; FKL runs one `executeOperations` call. Both run on the regions of the iteration, which change every
iteration ([workload](../README.md#workload)). Every timed run follows a 1 ms idle gap, so that both libraries' thread
pools are asleep when it starts. Each run writes one CSV per comparison in FKL's two-execution format, plus
`cpu_validation.csv` (before the timing, on the regions of the first, every 50th and the last timed iteration, one row
per iteration), `regions.csv` and `cpu_platform.txt`.

## Build and run

Preferably with a recent clang; `-march=native` by default (`CPU_ARCH_FLAGS`):

    cmake -S fkl-vs-opencv-cpu -B build-opencv-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ \
          -DOpenCV_DIR=<OpenCV build or install>
    cmake --build build-opencv-cpu -j
    ./build-opencv-cpu/benchmark_opencv_cpu ../results/fkl-vs-opencv-cpu
