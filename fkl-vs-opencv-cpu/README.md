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
whole tensor. FKL runs one `executeOperations` call. Both run on the regions of the iteration, which change every
iteration ([workload](../README.md#workload)).

The harness is the one of every CPU benchmark ([`common/bench_cpu.h`](../common/bench_cpu.h),
[the CPU benchmarks](../README.md#the-cpu-benchmarks)):

- **Sequential loops.** For every batch size, OpenCV's complete loop (100 warm-up and 1000 timed iterations) runs
  first, then FKL's on the same regions; the two libraries never run at the same time.
- **Idle gap.** Every run follows a 1 ms idle gap, busy-waited by the timing thread, so that the library's thread pool
  is asleep when the run starts.
- **Paired speedups.** The speedup is the mean of the per-iteration ratios, pairing the runs of the same iteration.
- **FKL compiled apart.** FKL's pipelines are compiled in their own translation unit
  ([`common/bench_fkl_cpu.cpp`](../common/bench_fkl_cpu.cpp)).

Each run writes one CSV per comparison in FKL's two-execution format and its `_per_iteration.csv` (every iteration's
pair of times, in ms, and their ratio). It also writes `cpu_validation.csv`, `regions.csv`, and `cpu_platform.txt`
(with how long each pool's workers stay busy after a run). The validation runs before the timing, on the regions of
the first, every 50th and the last timed iteration, one row per iteration:

- FKL against the reference with its conventions, FKL's pool against FKL on one thread, and OpenCV on as many threads
  as FKL's pool has workers against OpenCV on one must agree within 1e-3, or the program exits with an error after
  writing its files;
- OpenCV against the reference with OpenCV's conventions (8-bit intermediates), and OpenCV against FKL, are reported.

### Protocol change

The paper's first results of this benchmark (`results-v3`) were measured with a different protocol:

- OpenCV and FKL alternated within every iteration;
- the timing thread slept through the 1 ms gap;
- FKL's pipelines were compiled in the program's own translation unit.

Now each library runs its complete loop on its own, and the two loops follow each other (a rule of the paper's author
for every CPU comparison). The timing thread busy-waits the gap, and FKL's code is shared and compiled apart.

The last two changes are there because, in development runs on the paper's Ryzen 9 5950X (FKL d1fba2f, clang 23.1.3
`-O3 -march=native`; their files are not kept, so this section gives no numbers), FKL's single-thread time depended on
the binary rather than on FKL. Two effects:

- **Waking from a sleep.** A run that follows a sleep pays the core's wake-up from its deepest C-state, and how much
  depended on the binary: from nothing in this benchmark's binary to several tens of microseconds at N = 1 in others.
- **Code placement.** In some binaries, N = 4 and N = 100 were about twice as slow even without sleeping. FKL's CPU
  loop calls the out-of-line `InterpolateComplete::exec` for every pixel, and its speed depends on where that code lands
  (an observation; this machine gives no access to the performance counters that would confirm it).

The busy-waited gap removes the first. For the second, FKL's code is compiled apart, linked first and page-aligned
(see [the CPU benchmarks](../README.md#the-cpu-benchmarks)): every CPU benchmark runs the same FKL object file, with
every FKL function at the same address modulo 4096, which each platform file records. Whether FKL's times then agree
between the CPU benchmarks is checked on the paper's runs, where every program measures FKL on the same regions. OpenCV
also gains from the busy-waited gap, since its runs no longer start from a sleeping core.

## Build and run

Preferably with a recent clang; `-march=native` by default (`CPU_ARCH_FLAGS`):

    cmake -S fkl-vs-opencv-cpu -B build-opencv-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ \
          -DOpenCV_DIR=<OpenCV build or install>
    cmake --build build-opencv-cpu -j
    ./build-opencv-cpu/benchmark_opencv_cpu ../results/fkl-vs-opencv-cpu
