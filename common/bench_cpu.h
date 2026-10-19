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
// (fkl-vs-opencv-cpu, fkl-vs-pytorch, fkl-vs-halide-cpu, fkl-vs-fern-cpu):
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// One 3840x2160 NV12 frame in host memory and N regions, new every iteration, with the sizes and positions of
// common/bench_geometry.h, bilinear resize to 64x32, black rows on top and bottom up to a square, then
// (x * mul - sub) / div written as float3 into an N-plane tensor (N x 64 x 64 x 3 floats).
//
// FKL's side is common/bench_fkl_cpu.cpp, the only translation unit of a CPU benchmark that includes FKL, compiled
// first in every one of them (add_cpu_bench_target, common/bench_cpu.cmake): the same FKL code, placed the same way,
// in every CPU benchmark, whatever the library adds. This header is plain C++.
//   makeFklCpu(N, false)  one executeOperations call with the whole chain (BVF) on FKL's CPU backend, one thread
//   makeFklCpu(N, true)   the same call on the CPU_MULTITHREAD backend: FKL's thread pool (one worker per hardware
//                         thread), in chunks of one output row of one crop; it returns when the output is written
//
// Iteration t of every loop runs both pipelines of a comparison on the regions of iteration t, generated before the
// loops; each converts them into its own arguments before the loops too, and what it does with them in the iteration
// (FKL's build() calls and fusion, the library's per-region work) is timed. Every timed run is preceded by a 1 ms idle
// gap, so that the thread pools, whose workers spin for a while after a run and then sleep, are asleep when the run
// starts: neither library steals cores from the other, and both start from sleeping workers, as when a pipeline runs
// once per video frame. The timing thread busy-waits the gap instead of sleeping: a sleeping core enters its deepest
// C-state, and the run right after it then pays a wake-up whose cost depends on each binary's code (README.md of each
// benchmark). Each benchmark also times FKL alone with the same protocol (a no-op in the library's place), the FKL-only
// control, to compare FKL's times between benchmarks.

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

// n is one of BATCHES; on FKL's thread pool if multithread
std::unique_ptr<FklCpu> makeFklCpu(size_t n, bool multithread, const Workload& w);

// The plain C++ reference of common/bench_reference.h with the given conventions, on the first n regions of
// iteration t
std::vector<float> referenceOutput(const Workload& w, size_t n, size_t t, Conventions conventions);

// FKL's NV12 -> RGB conversion as a matrix and offsets, for the libraries that convert with their own operations
Conversion fklConversion();

// The workers of FKL's thread pool (one per hardware thread): the thread count of the multithreaded comparisons
int poolThreads();

// ------------------------------------------------------------------------------------------------
// Two-execution timing in FKL's format: in every iteration the library and then FKL on the regions of that iteration,
// each timed with steady_clock after the idle gap
// ------------------------------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;

// The idle gap before a timed run, busy-waited by the timing thread
inline void idleGap(Clock::duration gap = IDLE_GAP) {
    const auto end = Clock::now() + gap;
    while (Clock::now() < end) {
    }
}

struct Row {
    size_t batch;
    TimingStats baseline, fkl;
    double meanSpeedup;
};

inline Row timeTwo(size_t batch, const std::function<void(size_t)>& baseline, const std::function<void(size_t)>& fkl,
                   Clock::duration gap = IDLE_GAP) {
    const auto time = [gap](const std::function<void(size_t)>& f, size_t t) {
        idleGap(gap);
        const auto t0 = Clock::now();
        f(t);
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        time(baseline, static_cast<size_t>(i));
        time(fkl, static_cast<size_t>(i));
    }
    std::vector<float> a(ITERS), b(ITERS);
    double speedup = 0.;
    for (int i = 0; i < ITERS; ++i) {
        const double ta = time(baseline, static_cast<size_t>(WARMUP_ITERS + i));
        const double tb = time(fkl, static_cast<size_t>(WARMUP_ITERS + i));
        a[i] = static_cast<float>(ta);
        b[i] = static_cast<float>(tb);
        speedup += ta / tb;
    }
    return {batch, computeStats(a), computeStats(b), speedup / ITERS};
}

// Every timed iteration's time of both libraries (ms), in iteration order
struct PerIteration {
    size_t batch;
    std::vector<float> baseline, fkl;
};

// The same comparison with each library in a loop of its own, for a library whose thread pool cannot be interleaved
// with FKL's (fkl-vs-pytorch's multithreaded comparison, README.md): the library's warm-up and timed iterations, then
// FKL's, on the same regions, each run after the idle gap. The speedup is still the mean over the timed iterations of
// (library time / FKL time) in the same iteration, which pairs runs on identical inputs; perIteration gets the times.
inline Row timeSeparately(size_t batch, const std::function<void(size_t)>& baseline,
                          const std::function<void(size_t)>& fkl, PerIteration& perIteration,
                          Clock::duration gap = IDLE_GAP) {
    const auto loop = [gap](const std::function<void(size_t)>& f) {
        std::vector<float> ms(ITERS);
        for (int i = 0; i < WARMUP_ITERS + ITERS; ++i) {
            idleGap(gap);
            const auto t0 = Clock::now();
            f(static_cast<size_t>(i));
            const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            if (i >= WARMUP_ITERS) {
                ms[i - WARMUP_ITERS] = static_cast<float>(elapsed);
            }
        }
        return ms;
    };
    perIteration = {batch, loop(baseline), loop(fkl)};
    double speedup = 0.;
    for (int i = 0; i < ITERS; ++i) {
        speedup += static_cast<double>(perIteration.baseline[i]) / perIteration.fkl[i];
    }
    return {batch, computeStats(perIteration.baseline), computeStats(perIteration.fkl), speedup / ITERS};
}

// The per-iteration times of timeSeparately (ms): one row per batch size and timed iteration, with the iteration's
// regions (iteration index t) and the ratio that the mean speedup averages
inline void writePerIterationCSV(const std::filesystem::path& path, const std::string& baselineLabel,
                                 const std::vector<PerIteration>& rows, const std::string& fklLabel) {
    std::ofstream f(path);
    f << "Batch size, Iteration, " << baselineLabel << " Time, " << fklLabel << " Time, Speedup" << std::endl;
    for (const auto& r : rows) {
        for (size_t i = 0; i < r.baseline.size(); ++i) {
            f << r.batch << ", " << WARMUP_ITERS + i << ", " << r.baseline[i] << ", " << r.fkl[i] << ", "
              << static_cast<double>(r.baseline[i]) / r.fkl[i] << std::endl;
        }
    }
}

// One comparison in FKL's two-execution format (times in ms), one row per batch size
inline void writeCSV(const std::filesystem::path& path, const std::string& baselineLabel, const std::vector<Row>& rows,
                     const std::string& fklLabel) {
    std::ofstream f(path);
    writeTwoExecutionHeader(f, "Batch size", baselineLabel, fklLabel);
    for (const auto& r : rows) {
        writeTwoExecutionRow(f, static_cast<long long>(r.batch), r.baseline, r.fkl, r.meanSpeedup);
    }
}

// The FKL-only control: FKL on one thread and on its thread pool, timed with the same protocol (warm-up, idle gaps,
// iterations and regions) with a no-op in the library's place
struct Control {
    size_t batch;
    TimingStats single, multi;
};

inline Control timeFklControl(size_t batch, FklCpu& single, FklCpu& multi) {
    const auto none = [](size_t) {};
    return {batch, timeTwo(batch, none, [&](size_t t) { single.run(t); }).fkl,
            timeTwo(batch, none, [&](size_t t) { multi.run(t); }).fkl};
}

// fkl_control.csv: the control's medians, p10 and p90 (ms)
inline void writeControlCSV(const std::filesystem::path& path, const std::vector<Control>& controls) {
    std::ofstream f(path);
    f << "Batch size, FKL CPU MedianTime, FKL CPU P10Time, FKL CPU P90Time, FKL CPU_MULTITHREAD MedianTime, "
         "FKL CPU_MULTITHREAD P10Time, FKL CPU_MULTITHREAD P90Time" << std::endl;
    for (const auto& c : controls) {
        f << c.batch;
        for (const TimingStats* s : {&c.single, &c.multi}) {
            f << ", " << s->median / 1000. << ", " << s->p10 / 1000. << ", " << s->p90 / 1000.;
        }
        f << std::endl;
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

// The lines of cpu_platform.txt that every CPU benchmark shares: the CPU, FKL, the OS, the workload, the regions, the
// timing and the validation; the program writes its library's lines before them
inline void writeCpuPlatform(std::ostream& p, int threads) {
    p << "CPU: " << cpuName() << ", " << std::thread::hardware_concurrency() << " hardware threads; the multithreaded "
      << "comparisons use " << threads << " threads on both sides\n"
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
      << ", in its own translation unit (common/bench_fkl_cpu.cpp), the first of the program\n"
      << "Benchmark repository: commit " << BENCH_REPO_COMMIT << "\n"
      << "OS: " << osName() << "\n"
      << "Workload: " << describe() << "\n"
      << "Pipeline: as the GPU benchmarks, with BT.601 limited range for NV12 -> RGB\n"
      << "Regions: iteration t (the " << WARMUP_ITERS << " warm-up iterations are t in [0, " << WARMUP_ITERS << "), the "
      << ITERS << " timed ones t in [" << WARMUP_ITERS << ", " << WARMUP_ITERS + ITERS
      << ")) runs both pipelines of a comparison on the regions of iteration t, generated before the loops and "
      << "converted into each library's arguments before them too (statistics of the timed ones in regions.csv); "
      << "FKL's build() calls and fusion and the library's per-region work are timed.\n"
      << "Timing: steady_clock around each pipeline, " << WARMUP_ITERS << " warm-up and " << ITERS
      << " timed iterations, the library and FKL one after the other in each, each run preceded by a "
      << std::chrono::milliseconds(IDLE_GAP).count()
      << " ms idle gap (busy-waited by the timing thread, so that its core does not enter a deep C-state, while the "
         "thread pools fall asleep); speedup computed per iteration and averaged. FKL-only control (fkl_control.csv): "
         "FKL alone with the same protocol, a no-op in the library's place.\n"
      << "Validation: before the timing, on the regions of the first, every " << validation::VALIDATION_STRIDE
      << "th and the last timed iteration (cpu_validation.csv, one row per iteration).\n";
}

} // namespace bench::cpu

#endif // BVF_BENCH_CPU_H
