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
iteration ([workload](../README.md#workload)). The harness is the one of every CPU benchmark
([`common/bench_cpu.h`](../common/bench_cpu.h), [the CPU benchmarks](../README.md#the-cpu-benchmarks)): every timed
run follows a 1 ms idle gap, busy-waited by the timing thread, so that both libraries' thread pools are asleep when it
starts, and FKL's pipelines are compiled in their own translation unit
([`common/bench_fkl_cpu.cpp`](../common/bench_fkl_cpu.cpp)). Each run writes one CSV per comparison in FKL's
two-execution format, plus `fkl_control.csv` (FKL alone with the same protocol), `cpu_validation.csv` (before the
timing, on the regions of the first, every 50th and the last timed iteration, one row per iteration), `regions.csv`
and `cpu_platform.txt` (with how long each pool's workers stay busy after a run).

### Protocol change

The paper's first results of this benchmark (`results-v3`) were measured with the timing thread sleeping through the
1 ms gap and with FKL's pipelines compiled in the program's own translation unit. Both made FKL's single-thread time
depend on the binary rather than on FKL. Measured on the paper's Ryzen 9 5950X with FKL d1fba2f, built with clang
23.1.3 `-O3 -march=native`, with the same FKL code everywhere (medians, us, N = 1 / 4 / 16 / 50 / 100):

| FKL on one thread, measured in | Gap | Median |
|---|---|---|
| this benchmark's binary | sleep | 34.7 / 139.8 / 562 / 1753 / 3556 |
| FKL vs PyTorch's CPU binary, FKL alone (PyTorch never called) | sleep | 91 / 371 / 785 / 1897 / 6888 |
| the same binary | busy-wait | 35.3 / 262 / 558 / 1757 / 6642 |
| a program with FKL only | sleep | 49 / 196 / 781 / 1917 / 3882 |
| the same program | busy-wait, or none | 35 / 139 / 558 / 1755 / 3508 |

Two effects:

- **Waking from a sleep.** A run that follows a sleep pays the core's wake-up from its deepest C-state. The cost
  depends on the binary: nothing here, +14 us in the FKL-only program, +56 us in the PyTorch one at N = 1.
- **Code placement.** In some binaries N = 4 and N = 100 are about twice as slow even without sleeping. FKL's CPU loop
  calls the out-of-line `InterpolateComplete::exec` for every pixel, and its speed depends on where that code lands
  (an observation; this machine gives no access to the performance counters that would confirm it).

The busy-waited gap and the shared, separately compiled FKL code remove both effects: every CPU benchmark now runs the
same FKL object file, and FKL's control matches between them. OpenCV also gains, since its runs no longer start from a
sleeping core.

## Build and run

Preferably with a recent clang; `-march=native` by default (`CPU_ARCH_FLAGS`):

    cmake -S fkl-vs-opencv-cpu -B build-opencv-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ \
          -DOpenCV_DIR=<OpenCV build or install>
    cmake --build build-opencv-cpu -j
    ./build-opencv-cpu/benchmark_opencv_cpu ../results/fkl-vs-opencv-cpu
