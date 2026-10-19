# FKL vs Halide on the CPU

Part of the [BVF paper benchmarks](../README.md). This benchmark runs the Figure 1 pipeline on the CPU with
[Halide](https://halide-lang.org) and with FKL's CPU backends (BVF). It uses the [workload](../README.md#workload) of
every benchmark of this repository and the harness of every CPU benchmark
([`common/bench_cpu.h`](../common/bench_cpu.h), [the CPU benchmarks](../README.md#the-cpu-benchmarks)).

Halide is the binary release of **Halide 21.0.0**, the latest release (2025-09-16, built from commit
[b629c80](https://github.com/halide/Halide/commit/b629c80de18f1534ec71fddd8b567aa7027a0876)):

| Platform | Release archive | SHA-256 |
|---|---|---|
| Linux x86_64 | `Halide-21.0.0-x86-64-linux-b629c80de18f1534ec71fddd8b567aa7027a0876.tar.gz` | `b56139ddc5d863486b9b339e1c9b7cc3f6aadd4dd8a2eff2202e79ca68706091` |
| Linux ARM64 | `Halide-21.0.0-arm-64-linux-b629c80de18f1534ec71fddd8b567aa7027a0876.tar.gz` | `6fa8be9a556ddf3e899a0db59af84d69d0bebbf91ead90c03f90f731ab314d95` |
| Windows x64 | `Halide-21.0.0-x86-64-windows-b629c80de18f1534ec71fddd8b567aa7027a0876.zip` | `4efec94b7c8958b1ae0125a73245a148e4c98dbb54f2678bc34c6abe36ee899a` |

The archives come from the [v21.0.0 release](https://github.com/halide/Halide/releases/tag/v21.0.0). The SHA-256 are
the ones GitHub publishes for them, and the CI checks them. The program records in its platform files the SHA-256 of the
`libHalide` it links (computed at configure time): in the Linux x86_64 archive, `libHalide.so.21.0.0` has SHA-256
`4bd7f53061c70d3217a2878d8f37cb052c57208a5c59e447eee81a018e112f4e`.

## The Halide pipeline

The pipeline is one `Func`, `out(c, x, y, n)`, over the N x 64 x 64 x 3 output. `c` is innermost, which is FKL's
interleaved float3 layout. Every producer is inlined into it. For each output value it:

1. reads the region of plane `n` from the **region table**, an input buffer of 4 x N int32 (x, y, width, height);
2. samples the frame with **FKL's bilinear convention** (fused_kernel's `resize.h` and `interpolation.h`):
   `src = dst * (size / output size)` (64 for x, 32 for y), `x1 = floor(src)`, `x2 = x1 + 1` read at `min(x2, size - 1)`, and FKL's weights
   `(x2 - sx)(y2 - sy)`, ...;
3. converts the four NV12 samples with **FKL's BT.601 limited range matrix** (chroma replicated, float, no
   clamping), taken from FKL itself (`common/bench_reference.h`);
4. adds the black rows on top and bottom;
5. computes `(x * MUL - SUB) / DIV`.

Choices in how the pipeline is written:

- **Regions are run-time data.** They change every iteration without recompiling, as FKL's `Crop` parameters do. The
  table (4 x N int32: x, y, width, height) is filled from the regions of the iteration in every run, which is timed, as
  FKL's `build()` calls take the iteration's rectangles in every run.
- **Nothing is computed over the frame.** The frame coordinates are wrapped in `unsafe_promise_clamped(..., 0, size - 1)`,
  which is valid because every region lies inside the frame ([workload](../README.md#workload)). Bounds inference then
  bounds the frame reads by the frame itself. No producer is scheduled outside the output's loops, so Halide converts
  only the four samples of each output value, as FKL does.
- **The border rows compute nothing.** The border is `select(border, 0, likely(resized))`, Halide's idiom for boundary
  conditions. Loop partitioning splits the `y` loop into the top border, the 32 resized rows and the bottom border (or
  branches per row in the parallel schedule). Only the resized rows compute the resize, as FKL's `AddBorder` reads only
  there. The row is also clamped to the 32 resized rows, so that where Halide would still evaluate the sample, it reads
  inside the region.
- **Compiled once.** The pipeline is JIT-compiled once per schedule before any loop into a `Halide::Callable`, the
  low-overhead JIT entry point, for the host target. On the paper's Ryzen 9 5950X that target is
  `x86-64-linux-tune_znver3-avx-avx2-f16c-fma-sse41`.

The program writes Halide's lowered statement of each schedule (`halide_default.stmt`, `halide_parallel.stmt`): its
loops after bounds inference and loop partitioning.

Two schedules:

| Name | Schedule |
|---|---|
| `halide_default` | none: the output's loops in their default order (n, y, x, c), serial, every producer inlined; the direct analogue of BVF |
| `halide_parallel` | `out.bound(c, 0, 3).reorder(c, x, n, y).unroll(c).vectorize(x, 8).parallel(y)`: one task per output row for all the planes, 8 output pixels per AVX2 vector, the 3 channels unrolled |

## Comparisons and thread counts

FKL runs one `executeOperations` call with the whole chain on its CPU backend (one thread, `ParArch::CPU`) and on its
`CPU_MULTITHREAD` backend (FKL's thread pool, one worker per hardware thread): both schedules against FKL CPU on one
thread, and `halide_parallel` against FKL CPU_MULTITHREAD on the pool size, at the same thread count on both sides:

| Halide threads | Schedules | Compared with | Files |
|---|---|---|---|
| 1 | both | FKL CPU | `benchmark_cpu_halide_default_1thread_vs_fkl_bvf_CPU.csv`, `benchmark_cpu_halide_parallel_1thread_vs_fkl_bvf_CPU.csv`, and their `_per_iteration.csv` |
| FKL's pool size (32 on the paper's machine) | `halide_parallel` | FKL CPU_MULTITHREAD | `benchmark_cpu_halide_parallel_32threads_vs_fkl_bvf_CPU_MULTITHREAD.csv` and its `_per_iteration.csv` |

`halide_default` has no parallel loop: on 32 threads it would still run on one, against FKL on 32, so the
multithreaded process does not time it.

**One process per thread count.** Halide's threads are set with `Halide::Internal::JITSharedRuntime::set_num_threads`,
the JIT runtime's `halide_set_num_threads`. That call cannot take threads away from a pool whose workers have started.
After a run on 32 threads, `set_num_threads(1)` reports 1 thread, but the started workers keep taking the tasks of
`halide_parallel` (seen in development, where the other threads then used CPU time in every run). So the program sets
the thread count once, before any run, and runs once per thread count:

    benchmark_halide_cpu <results folder> 1          # both schedules against FKL on one thread
    benchmark_halide_cpu <results folder> pool       # halide_parallel against FKL's thread pool, on as many threads

The 1-thread run checks this. It measures the CPU time of the process's other threads per `halide_parallel` run on
100 regions on one Halide thread, records it in `cpu_platform_1thread.txt`, and fails above 5 us (a Halide worker
taking tasks uses hundreds) or if the threads' CPU times cannot be read. This check reads `/proc`: on Linux only, it
is not made on Windows.

The protocol is the one of every CPU benchmark:

- **Sequential loops.** For every batch size and schedule, Halide's complete loop (100 warm-up and 1000 timed
  iterations) runs first, then FKL's on the same regions. The two libraries never run at the same time.
- **Idle gap.** Every run follows a 1 ms idle gap, busy-waited by the timing thread.
- **Paired speedups.** The speedup is the mean of the per-iteration ratios, pairing the runs of the same iteration.
  Every comparison CSV has its `_per_iteration.csv` with the pairs.

Each process records in `cpu_platform_<threads>.txt` how long Halide's and FKL's workers stay busy after a parallel run
(the idle gap must outlast both), and writes the validation (`cpu_validation_<threads>.csv`): before the timing, on the
regions of the first, every 50th and the last timed iteration, the schedules it times and FKL against the plain C++
reference with FKL's conventions ([`common/bench_reference.h`](../common/bench_reference.h)), FKL's pool against FKL,
and Halide against FKL's output. The schedules and FKL must be within 1e-3 of the reference, and FKL's pool of FKL, or
the process exits with an error after writing its files.

## What each side does that the other does not

Halide, compared with FKL:

- **Vectorizes.** `halide_parallel` computes 8 output pixels per AVX2 instruction (the four samples' loads are
  gathered). FKL's CPU backend runs its fused loop one output pixel at a time, without SIMD across pixels.
- **Simplifies the arithmetic (`halide_parallel`).** With the channel loop bounded to 3 and unrolled, Halide's
  simplifier folds each channel's constants, and `(x * MUL - SUB) / DIV` becomes one multiply-add
  (`halide_parallel.stmt`). `halide_default` keeps the channel a loop variable: its `.stmt` selects the constants per
  channel (`mux`), divides, and samples the frame inside the channel loop. FKL computes each operation of the chain as
  written. All must stay within 1e-3 of the reference; the validation files have their errors.
- **Compiles for the exact CPU.** The JIT compiles for the host CPU (`tune_znver3`), at run time and once. FKL's code
  is compiled ahead of time with `-march=native`, the same CPU.
- **Has the regions in a table.** Halide reads the regions from a table in memory, which every run fills from the
  regions of its iteration; FKL's are parameters of its operations, which every run builds (`build()` calls). Both are
  timed.
- **Calls through the JIT.** Every run is a call through the JIT `Callable`, which validates the buffers' shapes; the
  bounds queries are part of the compiled pipeline.

Both convert 4 source pixels for each output value: neither materializes the crop.

## Build and run

Preferably with a recent clang, `-march=native` by default (`CPU_ARCH_FLAGS`), with `Halide_DIR` the CMake folder of
the release:

    cmake -S fkl-vs-halide-cpu -B build-halide-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ \
          -DHalide_DIR=<Halide-21.0.0-x86-64-linux>/lib/cmake/Halide
    cmake --build build-halide-cpu -j
    ./build-halide-cpu/benchmark_halide_cpu ../results/fkl-vs-halide-cpu 1
    ./build-halide-cpu/benchmark_halide_cpu ../results/fkl-vs-halide-cpu pool
