/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// The harness of the CPU benchmarks that compare a library with FKL's CPU backends (BVF) on the Figure 1 pipeline
// (fkl-vs-opencv-cpu; fkl-vs-pytorch, fkl-vs-halide-cpu and fkl-vs-fern-cpu are added by pull requests #18, #16, #17):
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// One 3840x2160 NV12 frame in host memory and N regions, new every iteration, with the sizes and positions of
// common/bench_geometry.h, bilinear resize to 64x32, black rows on top and bottom up to a square, then
// (x * mul - sub) / div written as float3 into an N-plane tensor (N x 64 x 64 x 3 floats).
//
// FKL's side is common/bench_fkl_cpu.cpp, the only translation unit of a CPU benchmark that includes FKL, linked first
// in every one of them (add_cpu_bench_target, common/bench_cpu.cmake) and page-aligned: the same FKL code, at the same
// addresses modulo 4096, in every CPU benchmark, whatever the library adds. This header is plain C++.
//   makeFklCpu(N, false)  one executeOperations call with the whole chain (BVF) on FKL's CPU backend, one thread
//   makeFklCpu(N, true)   the same call on the CPU_MULTITHREAD backend: FKL's thread pool (one worker per hardware
//                         thread), in chunks of one output row of one crop; it returns when the output is written
//
// The two libraries of a comparison never run at the same time: the library's complete loop (warm-up and timed
// iterations) runs first, then FKL's, on the same regions. Iteration t of each loop uses the regions of iteration t,
// generated before the loops; FKL converts them into its rectangles before the loops, and its build() calls and fusion
// are timed; each program says what its library does with them inside its timed run (CpuPlatformText). FKL's speedup is
// the mean over the timed iterations of (library time / FKL time) of the same iteration, which pairs runs on identical
// inputs. Every timed run is preceded by a 1 ms idle gap, as when a pipeline runs once per video frame: a pool whose
// workers spin for less than that after a run (FKL's, OpenCV's) is asleep when the run starts, one that spins longer
// (libgomp's by default, PyTorch's in #18) is still spinning; each library keeps its own threads, if it has any (a
// library without threads runs on FKL's pool through runOnFklPool, Fern in #17), and its own idle behaviour. The timing thread busy-waits the gap instead of sleeping: a sleeping
// core enters its deepest C-state, and the run right after it then pays a wake-up whose cost depends on each binary's
// code (fkl-vs-opencv-cpu/README.md).

#ifndef BVF_BENCH_CPU_H
#define BVF_BENCH_CPU_H

#include "bench_geometry.h"
#include "bench_pipeline.h"
#include "bench_platform.h"
#include "bench_timing.h"
#include "bench_validation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace bench::cpu {

// Frame, region, resize, border and output sizes, batch sizes, seeds and region positions, shared by every benchmark
using namespace geometry;
using namespace pipeline;
using validation::validatedIterations;
// BENCH_WARMUP_ITERATIONS and BENCH_ITERATIONS come from CMake: 100 and 1000 for the paper's results, 0 and 1 in the CI
constexpr int WARMUP_ITERS = BENCH_WARMUP_ITERATIONS;
constexpr int ITERS = BENCH_ITERATIONS;
static_assert(WARMUP_ITERS >= 0 && ITERS >= 1, "at least one timed iteration");
constexpr auto IDLE_GAP = std::chrono::milliseconds(1);

// ------------------------------------------------------------------------------------------------
// FKL's side, common/bench_fkl_cpu.cpp
// ------------------------------------------------------------------------------------------------

// The NV12 frame of every benchmark (common/bench_geometry.h's generateFrame) in host memory, owned by FKL's side:
// FRAME_H * 3 / 2 rows of FRAME_W bytes, *pitch bytes apart
const uint8_t* fklFrame(size_t* pitch);

struct Workload {
    const uint8_t* frame{nullptr};
    size_t pitch{0};
    // The regions of every iteration (warm-up and timed), generated before the loops
    std::unique_ptr<RegionSets> regions;
};

inline void initWorkload(Workload& w) {
    w.frame = fklFrame(&w.pitch);
    w.regions = std::make_unique<RegionSets>(static_cast<size_t>(WARMUP_ITERS + ITERS));
}

// FKL with BVF on n regions per iteration
class FklCpu {
public:
    virtual ~FklCpu() = default;
    // On the regions of iteration t; returns when the output is written
    virtual void run(size_t t) = 0;
    // The n x OUT_H x OUT_W x 3 output of the last run
    virtual std::vector<float> output() const = 0;
};

// Where FKL's code is, modulo 4096 (bench_fkl_cpu.cpp), for the platform file
std::string fklCodeAlignment();

// n is one of BATCHES; on FKL's thread pool if multithread
std::unique_ptr<FklCpu> makeFklCpu(size_t n, bool multithread, const Workload& w);

// The plain C++ reference of common/bench_reference.h with the given conventions, on the first n regions of
// iteration t
std::vector<float> referenceOutput(const Workload& w, size_t n, size_t t, Conventions conventions);

// FKL's NV12 -> RGB conversion as a matrix and offsets, for the libraries that convert with their own operations
Conversion fklConversion();

// The workers of FKL's thread pool (one per hardware thread): the thread count of the multithreaded comparisons
int poolThreads();

// Runs function(0) ... function(chunks - 1) on FKL's thread pool, the way FKL's CPU_MULTITHREAD executions run their
// chunks (an execution of a CPU_MULTITHREAD stream that enqueues them in the pool), and returns when all of them have
// finished: for a library whose generated code is serial, parallelized by the benchmark (fkl-vs-fern-cpu)
void runOnFklPool(size_t chunks, const std::function<void(size_t)>& function);

// ------------------------------------------------------------------------------------------------
// Timing in FKL's two-execution format: the library's loop, then FKL's, on the same regions, each run timed with
// steady_clock after the idle gap
// ------------------------------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;

// The idle gap before a timed run, busy-waited by the timing thread
inline void idleGap() {
    const auto end = Clock::now() + IDLE_GAP;
    while (Clock::now() < end) {
    }
}

// One comparison for one batch size: the statistics of both loops, every timed iteration's times (ms, in iteration
// order) and the mean of the per-iteration speedups
struct Row {
    size_t batch;
    TimingStats baseline, fkl;
    double meanSpeedup;
    std::vector<float> baselineMs, fklMs;
};

// The warm-up iterations (0 to WARMUP_ITERS - 1) and the timed ones (from WARMUP_ITERS) of one library, each run after
// the idle gap; returns the times of the timed ones (ms)
inline std::vector<float> timeLoop(const std::function<void(size_t)>& run) {
    std::vector<float> ms(ITERS);
    for (int i = 0; i < WARMUP_ITERS + ITERS; ++i) {
        idleGap();
        const auto t0 = Clock::now();
        run(static_cast<size_t>(i));
        const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (i >= WARMUP_ITERS) {
            ms[i - WARMUP_ITERS] = static_cast<float>(elapsed);
        }
    }
    return ms;
}

// The library's complete loop, then FKL's
inline Row timeTwo(size_t batch, const std::function<void(size_t)>& baseline, const std::function<void(size_t)>& fkl) {
    Row row{batch, {}, {}, 0., timeLoop(baseline), {}};
    row.fklMs = timeLoop(fkl);
    double speedup = 0.;
    for (int i = 0; i < ITERS; ++i) {
        speedup += static_cast<double>(row.baselineMs[i]) / row.fklMs[i];
    }
    row.baseline = computeStats(row.baselineMs);
    row.fkl = computeStats(row.fklMs);
    row.meanSpeedup = speedup / ITERS;
    return row;
}

// One comparison in FKL's two-execution format (times in ms), one row per batch size, at path, and every timed
// iteration's pair of times with its speedup next to it (<path>_per_iteration.csv): the iteration (its regions), the
// library's time, FKL's time and their ratio, which the mean speedup averages
inline void writeCSV(const std::filesystem::path& path, const std::string& baselineLabel, const std::vector<Row>& rows,
                     const std::string& fklLabel) {
    std::ofstream f(path);
    writeTwoExecutionHeader(f, "Batch size", baselineLabel, fklLabel);
    for (const auto& r : rows) {
        writeTwoExecutionRow(f, static_cast<long long>(r.batch), r.baseline, r.fkl, r.meanSpeedup);
    }
    std::filesystem::path perIteration = path;
    perIteration.replace_filename(path.stem().string() + "_per_iteration" + path.extension().string());
    std::ofstream p(perIteration);
    p << "Batch size, Iteration, " << baselineLabel << " Time (ms), " << fklLabel << " Time (ms), Speedup" << std::endl;
    for (const auto& r : rows) {
        for (size_t i = 0; i < r.baselineMs.size(); ++i) {
            p << r.batch << ", " << WARMUP_ITERS + i << ", " << r.baselineMs[i] << ", " << r.fklMs[i] << ", "
              << static_cast<double>(r.baselineMs[i]) / r.fklMs[i] << std::endl;
        }
    }
}

// How long the workers of a thread pool stay on their cores after a parallel run before they sleep, in us: `rounds`
// times, a parallel run, then 20 ms of sleep of the calling thread; the CPU time of the process's other threads over
// the rounds (Linux: /proc/self/task/<tid>/schedstat) divided by rounds * workers is the time each worker stays busy per
// run, its share of the work plus its spin. workers: the pool's threads other than the caller (FKL's pool: all of them,
// the caller waits; OpenMP, OpenCV and Halide: their thread count minus one, the caller takes part). With little work
// it is the pool's spin and wake-up, which the idle gap must exceed for the pool to be asleep when a run starts. NAN
// where the threads' counters cannot be read.
inline double workerBusyUs(const std::function<void()>& parallelRun, int workers, int rounds = 50) {
#ifdef _WIN32
    (void)parallelRun, (void)workers, (void)rounds;
    return NAN;
#else
    const std::filesystem::path tasks = "/proc/self/task";
    std::error_code error;
    if (!std::filesystem::exists(tasks, error) || workers < 1) {
        return NAN;
    }
    const std::string self = std::to_string(getpid());
    const auto threadCpuNs = [&]() {
        std::map<std::string, long long> ns;
        for (const auto& task : std::filesystem::directory_iterator(tasks, error)) {
            const std::string tid = task.path().filename().string();
            std::ifstream f(task.path() / "schedstat");
            long long runNs = 0;
            if (tid != self && f >> runNs) {
                ns[tid] = runNs;
            }
        }
        return ns;
    };
    parallelRun(); // starts the workers
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto before = threadCpuNs();
    for (int r = 0; r < rounds; ++r) {
        parallelRun();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const auto after = threadCpuNs();
    if (before.empty() || after.empty()) {
        return NAN;
    }
    long long busy = 0;
    for (const auto& [tid, ns] : after) {
        const auto b = before.find(tid);
        busy += ns - (b == before.end() ? 0 : b->second);
    }
    return busy / 1e3 / rounds / workers;
#endif
}

// Calls f.template operator()<N>() for every batch size, in order
template <typename F>
void forEachBatch(F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (f.template operator()<BATCHES[Is]>(), ...);
    }(std::make_index_sequence<BATCHES.size()>{});
}

// Whether a validation error (a maximum absolute difference) is finite and within the tolerance of FKL's float outputs.
// Every CPU benchmark requires it of FKL against its reference and of FKL's pool against FKL on one thread, and of
// what it checks of its library (its CpuPlatformText says what), and exits with an error after writing its files
// otherwise
inline bool withinTolerance(float error) {
    return std::isfinite(error) && error <= validation::TOLERANCE;
}

// The larger of two validation errors, NaN if either is (std::max would keep a NaN only as its first argument, and lose
// it at the next iteration): the worst error over the validated iterations
inline float worstError(float a, float b) {
    return std::isnan(a) || std::isnan(b) ? NAN : std::max(a, b);
}

// What a program says about its library in the lines every CPU benchmark shares
struct CpuPlatformText {
    // The program's validation CSV
    std::string validationFile = "cpu_validation.csv";
    // What the program requires of its library, besides FKL's checks (within the tolerance)
    std::string libraryValidation;
    // What the library does with the regions of an iteration, and whether it is timed
    std::string libraryRegions;
};

// The lines of the platform file that every CPU benchmark shares: the CPU, FKL, the OS, the workload, the regions, the
// timing and the validation; the program writes its library's lines before them
inline void writeCpuPlatform(std::ostream& p, int threads, const CpuPlatformText& text) {
    p << "CPU: " << cpuName() << ", " << std::thread::hardware_concurrency() << " hardware threads; a multithreaded "
      << "comparison gives each side " << threads << " threads, FKL's pool size (the program's lines say how the "
      << "library uses them)\n"
#ifdef _WIN32
      << "SIMD of the CPU: see the CPU model\n"
#else
      << "SIMD of the CPU: "
      << shell("grep -o -w -E 'avx|avx2|fma|avx512f|avx512bw|avx512vl|asimd|sve|sve2' /proc/cpuinfo | sort -u | paste -sd' '")
      << "\n"
#endif
      << "SIMD the code of this repository is compiled for: " << compiledSimd() << "\n"
      << "FKL: commit " << BENCH_FKL_COMMIT << ", CPU backend (single thread) and CPU_MULTITHREAD backend (thread pool of "
      << poolThreads() << " workers, thread-pool " << BENCH_THREAD_POOL_COMMIT
      << ", workers spin 100 us before sleeping, one chunk per output row of each crop), compiled with "
      << BENCH_COMPILER << ", flags " << BENCH_CXX_FLAGS
      << ", in its own translation unit (common/bench_fkl_cpu.cpp), the first of the program; " << fklCodeAlignment()
      << "\n"
      << "Benchmark repository: commit " << BENCH_REPO_COMMIT << "\n"
      << "OS: " << osName() << "\n"
      << "Workload: " << describe() << "\n"
      << "Pipeline: as the GPU benchmarks, with BT.601 limited range for NV12 -> RGB\n"
      << "Regions: iteration t of each loop (the " << WARMUP_ITERS << " warm-up iterations are t in [0, " << WARMUP_ITERS
      << "), the " << ITERS << " timed ones t in [" << WARMUP_ITERS << ", " << WARMUP_ITERS + ITERS
      << ")) runs on the regions of iteration t, generated before the loops (statistics of the timed ones in "
      << "regions.csv). FKL converts them into its rectangles before the loops, and its build() calls and fusion are "
      << "timed; " << text.libraryRegions << ".\n"
      << "Timing: steady_clock around each pipeline; for every batch size, the library's loop (" << WARMUP_ITERS
      << " warm-up and " << ITERS << " timed iterations) runs to completion first, then FKL's on the same regions: the "
      << "two libraries never run at the same time. Every run is preceded by a "
      << std::chrono::milliseconds(IDLE_GAP).count()
      << " ms idle gap, busy-waited by the timing thread (so that its core does not enter a deep C-state), in which "
         "the worker threads of the previous run fall asleep if they spin for less than that (the lines above say how "
         "long each pool's workers stay busy after a run); each library keeps its own threads, if it has any, and its "
         "own idle behaviour. "
         "Speedup: mean over "
         "the timed iterations of (library time / FKL time) of the same iteration (the *_per_iteration.csv files, times "
         "in ms, as in the comparison CSVs).\n"
      << "Validation: before the timing, on the regions of the first, every " << validation::VALIDATION_STRIDE
      << "th and the last timed iteration (" << text.validationFile << ", one row per iteration): FKL against the "
      << "reference with its conventions, FKL's pool against FKL on one thread, and " << text.libraryValidation
      << " must agree within " << validation::TOLERANCE << " (maximum absolute difference); the program exits with an "
      << "error otherwise.\n";
}

} // namespace bench::cpu

#endif // BVF_BENCH_CPU_H
