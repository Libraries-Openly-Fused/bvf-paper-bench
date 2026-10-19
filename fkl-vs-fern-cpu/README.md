# FKL vs Fern on the CPU

Part of the [BVF paper benchmarks](../README.md). This benchmark runs the Figure 1 pipeline on the CPU with **Fern**
("Lightweight and Locality-Aware Composition of Black-Box Subroutines", PLDI 2025) and with FKL's CPU backends (BVF).
It uses the [workload](../README.md#workload) of every benchmark of this repository and the harness of every CPU
benchmark ([`common/bench_cpu.h`](../common/bench_cpu.h), [the CPU benchmarks](../README.md#the-cpu-benchmarks)).

Fern's code is the repository [manya-bansal/gern](https://github.com/manya-bansal/gern). It has no release and is
marked "under construction". It is the submodule `gern` of this repository, pinned at
[d05bc66](https://github.com/manya-bansal/gern/commit/d05bc6648ccd40af71f616242ecafbb3ffe762b5), the head of its main
branch on 2026-10-19.

## How Fern expresses the pipeline

Fern composes **black-box subroutines**. The user writes the data structures and the subroutines in plain C++, and
annotates them for Fern: which subset of each input a subroutine consumes to produce a subset of its output, and how
its output can be tiled. Fern composes and tiles them, allocates the intermediates and generates C++. Here
([`fern/`](fern)):

| File | Contents |
|---|---|
| [`fern_impl.h`](fern/fern_impl.h) | What the user writes in C++: the data structures `NV12View` (a view of the frame), `Rect` (a region) and `ImageF3` (a float3 image or a view of one); the subroutines `nv12_to_rgb`, `resize_bilinear`, `letterbox` and `normalize`, with FKL's conventions and constants. The constants are checked against FKL's at run time. |
| [`fern_pipeline_gen.cpp`](fern/fern_pipeline_gen.cpp) | What the user writes for Fern: the annotations of the data structures and subroutines, the composition, the schedule, and the code generation (gern's `CodeGenerator`) |
| [`generated/fern_band16.cpp`](fern/generated/fern_band16.cpp) | The code gern generates for the benchmark, checked in |
| [`generated/pitfall_*.cpp`](fern/generated) | Three compositions that generate wrong code without an error, as evidence for the notes below |
| [`regenerate.cmake`](fern/regenerate.cmake) | Rebuilds gern from the submodule and the generator, regenerates `generated/`, and checks it against the checked-in copy, or replaces the copy with `-DWRITE=ON` |

**One region per call.** Fern has no batch construct (one output per program), so the generated function processes
one region. The benchmark calls it once per region, through Fern's own calling convention: `hook_fern_band16(void**)`,
what gern's runner calls after `dlopen`.

**The region is run-time data.** The region's position and size are the fields of a `Rect` data structure, used in
the subset expressions (`rect.x`, ...). One generated function serves every region of every iteration, with no
recompilation. Run-time scalars cannot be used for this: they lower to 0 in the subset expressions
(`src/codegen/concretize.cpp:346-347`), and are dropped from the generated function's arguments (evidence:
`generated/pitfall_scalar_region.cpp`).

**The crop is a subset, not a subroutine.** `nv12_to_rgb` consumes the subset of the frame at
`(rect.y, rect.x) + (row, col)`, so the crop is the offset of the frame view Fern queries.

**Schedule.** `Tile(out["row"], 16)` around the whole chain. The generated function does this, per region:

1. queries the frame at the region;
2. allocates the crop (h x w float3) and the resized image (32 x 64 float3);
3. runs `nv12_to_rgb` and `resize_bilinear` once (Fern's finalizer hoists them out of the band loop, since they do not
   depend on the band);
4. for each band of 16 output rows, allocates a 16 x 64 padded band, letterboxes into it and normalizes it into the
   output plane.

The band height is a compile-time template parameter that must divide the 64 rows, because Fern never clamps the last
tile (`src/codegen/codegen.cpp:336-344`). The research prototype measured 8, 16 and 64 rows on one thread: 8 and 16
were the same, and 64 about 10 % slower.

**The top-level `Tile` is required.** Without it, or with only the tail tiled, Fern treats the program's inputs as
already restricted to the subsets they consume. It passes the whole frame to `nv12_to_rgb` without the region's
offset, and the output is wrong (11.86 away from the reference) with no error (`generated/pitfall_untiled.cpp`,
`generated/pitfall_tail_tiled.cpp`).

**Compiled ahead of time.** Fern's runner compiles generated code with `g++` and no optimization flag (so `-O0`),
then `dlopen`s it (`src/codegen/runner.cpp:26-42`). The benchmark does not use the runner: the generated source is
compiled with the benchmark, with the flags of FKL's code (clang, `-O3 -march=native` for the paper), and called
directly. There is no compilation or `dlopen` in the timed loop.

**Intermediates.** The intermediates come from a per-thread stack arena in `fern_impl.h`'s `ImageF3::allocate`, so
there is no `malloc` in the timed loop. Fern lets the data structure decide how to allocate. With `malloc`, the
prototype was about 1 us per region slower.

## Comparisons and thread counts

Fern generates **serial** CPU code. The annotations take a `parallel` flag
(`include/gern/annotations/stmt.h:107-111, 195-199`), but the code generator builds its loops with
`For::make(start, cond, step, code)` and never passes it (`src/codegen/codegen.cpp:343`). The printer's `cilk_for`
(`src/codegen/codegen_printer.cpp:236-239`) is unreachable, and would need OpenCilk anyway. Fern's only parallel
mapping is to CUDA blocks and threads. So:

| Comparison | Fern | FKL | File |
|---|---|---|---|
| 1 thread | one call per region, on the calling thread | CPU backend (`ParArch::CPU`) | `benchmark_cpu_fern_1thread_vs_fkl_bvf_CPU.csv` |
| FKL's pool size (32 on the paper's machine) | **"Fern code, FKL thread pool"**: Fern's generated serial code, parallelized by the benchmark. The per-region calls run as the chunks of one execution on FKL's own thread pool (`runOnFklPool`), one chunk per region, so at most N workers are busy | `CPU_MULTITHREAD` backend on the same pool, one chunk per output row of each region | `benchmark_cpu_fern_code_fkl_thread_pool_32threads_vs_fkl_bvf_CPU_MULTITHREAD.csv` |

Fern has no parallel code of its own, so the multithreaded row is not Fern parallelizing anything. It is what a user of
Fern's serial code can do, on the same pool, workers, spin and wake-up as FKL. The protocol is the one of every CPU
benchmark:

- **Sequential loops.** For every batch size, Fern's complete loop (100 warm-up and 1000 timed iterations) runs first,
  then FKL's on the same regions. The two libraries never run at the same time.
- **Idle gap.** Every run follows a 1 ms idle gap, busy-waited by the timing thread.
- **Paired speedups.** The speedup is the mean of the per-iteration ratios, pairing the runs of the same iteration.
  Every comparison CSV has its `_per_iteration.csv` with the pairs.

The program also writes the validation (`cpu_validation.csv`): before the timing, on the regions of the first, every
50th and the last timed iteration, Fern and FKL against the plain C++ reference with FKL's conventions
([`common/bench_reference.h`](../common/bench_reference.h)), Fern against FKL's output, and Fern on the pool against
Fern.

## What each side does that the other does not

Fern, compared with FKL:

- **Writes intermediates to memory.** Per region: the crop (up to 48 x 24 float3, 13.5 KB), the resized image (32 x 64
  float3, 24 KB) and a padded band (16 x 64 float3, 12 KB) per band. They are small enough to stay in the L1 and L2
  caches. FKL writes nothing but the output.
- **Makes N opaque calls.** It calls the generated function once per region per iteration, where FKL makes one
  `executeOperations` call for the batch. Building each call's `Rect` and plane view is timed, as FKL's `build()`
  calls are.
- **Converts each source pixel once.** Fern converts w x h pixels per region: 200 to 1152. FKL's backwards fusion
  converts the 4 source pixels of every interior output pixel: 8192 per region. This is the main reason Fern is faster
  on one thread.
- **Has fixed shapes.** The intermediates' shapes and the pipeline's constants are known when the black boxes are
  compiled. The `letterbox` and `normalize` loops are auto-vectorized by clang; the conversion and resize loops stay
  scalar, as FKL's loop does.
- **Has a schedule chosen by hand.** The tile and the black boxes were written for this pipeline. FKL's pipeline is
  its generic operations.

## Build and run

The benchmark needs neither gern nor its dependencies: the generated code is checked in. Preferably with a recent
clang, `-march=native` by default (`CPU_ARCH_FLAGS`):

    cmake -S fkl-vs-fern-cpu -B build-fern-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
    cmake --build build-fern-cpu -j
    ./build-fern-cpu/benchmark_fern_cpu ../results/fkl-vs-fern-cpu

To regenerate the Fern code from the `gern` submodule (`git submodule update --init gern`), on Linux or macOS with
CMake 3.30 or newer:

    cmake -DCXX=clang++ -P fkl-vs-fern-cpu/fern/regenerate.cmake            # checks generated/ against it
    cmake -DCXX=clang++ -DWRITE=ON -P fkl-vs-fern-cpu/fern/regenerate.cmake # replaces generated/

gern at d05bc66 calls `std::find` in `src/codegen/codegen.cpp` without including `<algorithm>`, which libstdc++ 15 no
longer includes indirectly. `regenerate.cmake` forces the header on every gern source (`-include algorithm`) instead of
patching gern. The generated text is the same with g++ 15 and clang 23, except for the numbers in the names of Fern's
temporaries, which the comparison renames.
