// FKL vs Fern on the CPU, Figure 1 pipeline: the code Fern (gern) generates for the pipeline, composed from black-box
// subroutines, vs FKL's CPU backends (BVF).
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// The harness, the workload and FKL's pipelines are common/bench_cpu.h and common/bench_fkl_cpu.cpp.
//
// Fern's side (fern/): the data structures and black-box subroutines a Fern user writes (fern_impl.h: NV12 -> RGB,
// bilinear resize, letterbox, normalize, with FKL's conventions), their annotations and the composition
// (fern_pipeline_gen.cpp), and the code gern generates from them, checked in (generated/fern_band16.cpp; regenerate.cmake
// rebuilds it from the gern submodule and compares). Fern composes one region: the crop is the offset of the subset of
// the frame that nv12_to_rgb consumes, the region's position and size are the fields of a Rect data structure (run-time
// data, no recompilation), and the generated function fuses the chain under one loop over bands of 16 output rows (its
// finalizer runs the conversion and the resize once per region, before the band loop). It is compiled ahead of time
// with the flags of FKL's code and called through Fern's calling convention (hook_<name>(void**)), with no compilation
// and no dlopen in the program.
//
//   fern          one call of the generated function per region, on the calling thread (Fern generates serial code)
//   fern on pool  the same calls, one per region, as the chunks of one execution on FKL's thread pool
//                 (runOnFklPool): Fern has no parallel CPU code generation, so the benchmark parallelizes its calls,
//                 over regions (at most N workers run Fern's code; every run wakes all of the pool's sleeping workers,
//                 which spin as after FKL's runs)
//   fkl           one executeOperations<TransformDPP<ParArch::CPU>> call with the whole chain (BVF), one thread
//   fkl mt        the same call with ParArch::CPU_MULTITHREAD, on FKL's thread pool, in chunks of one output row
//
// Two comparisons, each with the same number of threads on both sides: fern against fkl, and fern on pool ("Fern code,
// FKL thread pool": Fern has no parallel code of its own) against fkl mt. In each, Fern's complete loop runs first and
// FKL's after it, on the same regions (common/bench_cpu.h).

#include "bench_cpu.h"
#include "fern_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// The generated function, in Fern's calling convention: args are the frame, the region and the output plane
extern "C" void hook_fern_band16(void** args);

namespace bench::cpu {

// The conversion and normalization constants of the black boxes must be FKL's
void checkFernConstants() {
    const Conversion k = fklConversion();
    bool same = k.lumaOffset == fern_bvf::Y_SUB && k.chromaOffset == fern_bvf::C_SUB;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            same = same && k.matrix[r][c] == fern_bvf::M[r][c];
        }
        same = same && MUL[r] == fern_bvf::MUL && SUB[r] == fern_bvf::SUB[r] && DIV[r] == fern_bvf::DIV[r];
    }
    if (!same) {
        throw std::runtime_error("fern/fern_impl.h's constants are not FKL's conversion and normalization");
    }
}

// Fern's generated function on the first n regions of an iteration, one call per region
class FernPipeline {
public:
    FernPipeline(const Workload& w, size_t n)
        : m_n(n),
          m_frame{w.frame, w.frame + static_cast<size_t>(FRAME_H) * w.pitch, static_cast<int64_t>(w.pitch), 0, 0,
                  FRAME_H, FRAME_W},
          m_out(n * OUT_FLOATS) {
        // The regions of every iteration, converted before the loops into Fern's Rect, as FKL's into fk::Rect
        for (size_t t = 0; t < w.regions->iterations(); ++t) {
            for (size_t i = 0; i < n; ++i) {
                const Region& r = w.regions->at(t, i);
                m_rects.push_back(fern_bvf::Rect{r.x, r.y, r.width, r.height});
            }
        }
    }

    // Region i of iteration t into output plane i
    void runRegion(size_t t, size_t i) {
        fern_bvf::NV12View frame = m_frame;
        fern_bvf::Rect rect = m_rects[t * m_n + i];
        fern_bvf::ImageF3 plane{m_out.data() + i * OUT_FLOATS, OUT_H, OUT_W, OUT_W, 0, 0};
        void* args[3] = {&frame, &rect, &plane};
        hook_fern_band16(args);
    }

    // On the regions of iteration t, on the calling thread
    void run(size_t t) {
        for (size_t i = 0; i < m_n; ++i) {
            runRegion(t, i);
        }
    }

    // On the regions of iteration t, one chunk per region on FKL's thread pool
    void runOnPool(size_t t) {
        runOnFklPool(m_n, [this, t](size_t i) { runRegion(t, i); });
    }

    const std::vector<float>& output() const { return m_out; }
    void fillNaN() { std::fill(m_out.begin(), m_out.end(), std::numeric_limits<float>::quiet_NaN()); }

private:
    size_t m_n;
    fern_bvf::NV12View m_frame;
    std::vector<fern_bvf::Rect> m_rects;
    std::vector<float> m_out;
};

// The errors on the regions of one validated iteration
struct Check {
    size_t batch, iteration;
    float fklVsReference, fernVsReference, fernVsFkl, fernPoolVsFern, fklMultithreadVsFkl;
    // What the program requires (withinTolerance): FKL and Fern against the reference with FKL's conventions (Fern's
    // too), FKL's pool against FKL, and Fern's code on the pool against Fern on one thread
    bool passed() const {
        return withinTolerance(fklVsReference) && withinTolerance(fernVsReference) &&
               withinTolerance(fernPoolVsFern) && withinTolerance(fklMultithreadVsFkl);
    }
};

template <size_t N>
void runBatch(const Workload& w, std::vector<Row>& singleThread, std::vector<Row>& multiThread,
              std::vector<Check>& checks, int threads) {
    FernPipeline fern(w, N);
    FernPipeline fernPool(w, N);
    const auto fkl = makeFklCpu(N, false, w);
    const auto fklMT = makeFklCpu(N, true, w);
    // Before the timing, on the regions of the first, every 50th and the last timed iteration
    Check worst{N, 0, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (const size_t k : validatedIterations(ITERS)) {
        const size_t t = WARMUP_ITERS + k;
        fern.fillNaN();
        fern.run(t);
        fernPool.fillNaN();
        fernPool.runOnPool(t);
        fkl->run(t);
        fklMT->run(t);
        const auto fklOut = fkl->output();
        const auto reference = referenceOutput(w, N, t, Conventions::Fkl);
        const Check c{N, t, maxAbsError(fklOut, reference), maxAbsError(fern.output(), reference),
                      maxAbsError(fern.output(), fklOut), maxAbsError(fernPool.output(), fern.output()),
                      maxAbsError(fklMT->output(), fklOut)};
        checks.push_back(c);
        // std::max keeps a NaN only as its first argument: the error goes first
        worst = Check{N, t, std::max(c.fklVsReference, worst.fklVsReference),
                      std::max(c.fernVsReference, worst.fernVsReference), std::max(c.fernVsFkl, worst.fernVsFkl),
                      std::max(c.fernPoolVsFern, worst.fernPoolVsFern),
                      std::max(c.fklMultithreadVsFkl, worst.fklMultithreadVsFkl)};
    }
    const Check& c = worst;

    singleThread.push_back(timeTwo(N, [&](size_t t) { fern.run(t); }, [&](size_t t) { fkl->run(t); }));
    multiThread.push_back(timeTwo(N, [&](size_t t) { fernPool.runOnPool(t); }, [&](size_t t) { fklMT->run(t); }));
    const Row& s = singleThread.back();
    const Row& m = multiThread.back();
    std::printf("N %3zu | fern %8.1f us, FKL CPU %8.1f us, mean speedup %6.2fx | Fern code, FKL thread pool (%d threads) "
                "%8.1f us, FKL CPU_MULTITHREAD %8.1f us, mean speedup %6.2fx | max err: FKL vs "
                "reference %.2e, fern vs reference %.2e, fern vs FKL %.2e, fern on pool vs fern %.2e, FKL "
                "CPU_MULTITHREAD vs FKL CPU %.2e\n",
                N, s.baseline.median, s.fkl.median, s.meanSpeedup, threads, m.baseline.median, m.fkl.median,
                m.meanSpeedup, c.fklVsReference,
                c.fernVsReference, c.fernVsFkl, c.fernPoolVsFern, c.fklMultithreadVsFkl);
}

} // namespace bench::cpu

int main(int argc, char** argv) {
    using namespace bench::cpu;
    try {
        const std::filesystem::path outDir = argc > 1 ? argv[1] : "results";
        std::filesystem::create_directories(outDir);
        checkFernConstants();
        // Both multithreaded sides run on FKL's thread pool, one worker per hardware thread
        const int threads = poolThreads();
        Workload w;
        initWorkload(w);
        // How long the pool's workers stay busy after a run with little work (their spin and wake-up), which the idle
        // gap must outlast: the pool of both multithreaded sides (FKL's caller waits, so every thread is a worker)
        const auto spinProbe = makeFklCpu(BATCHES.front(), true, w);
        const double fklBusyUs = workerBusyUs([&]() { spinProbe->run(WARMUP_ITERS); }, threads);

        std::vector<Row> singleThread, multiThread;
        std::vector<Check> checks;
        forEachBatch([&]<size_t N>() { runBatch<N>(w, singleThread, multiThread, checks, threads); });

        writeCSV(outDir / "benchmark_cpu_fern_1thread_vs_fkl_bvf_CPU.csv", "Fern 1 thread", singleThread, "FKL CPU");
        writeCSV(outDir / ("benchmark_cpu_fern_code_fkl_thread_pool_" + std::to_string(threads) +
                           "threads_vs_fkl_bvf_CPU_MULTITHREAD.csv"),
                 "Fern code FKL thread pool " + std::to_string(threads) + " threads", multiThread, "FKL CPU_MULTITHREAD");
        std::ofstream v(outDir / "cpu_validation.csv");
        v << "Batch size, Iteration, FKL MaxAbsError vs FKL-convention reference, Fern MaxAbsError vs FKL-convention "
             "reference, Fern vs FKL MaxAbsDifference, Fern code FKL thread pool vs Fern MaxAbsDifference, FKL "
             "CPU_MULTITHREAD vs FKL CPU MaxAbsDifference, Within tolerance" << std::endl;
        bool passed = true;
        for (const auto& c : checks) {
            v << c.batch << ", " << c.iteration << ", " << c.fklVsReference << ", " << c.fernVsReference << ", "
              << c.fernVsFkl << ", " << c.fernPoolVsFern << ", " << c.fklMultithreadVsFkl << ", "
              << (c.passed() ? "yes" : "no") << std::endl;
            passed &= c.passed();
        }
        writeRegionStats((outDir / "regions.csv").string(), *w.regions, WARMUP_ITERS, ITERS);
        std::ofstream p(outDir / "cpu_platform.txt");
        p << "Fern: gern (https://github.com/manya-bansal/gern) commit " << BENCH_GERN_COMMIT << " (the gern submodule); "
          << "fern/generated/fern_band16.cpp generated by fern/fern_pipeline_gen.cpp with it (checked by "
             "fern/regenerate.cmake) and compiled with the flags of FKL's code below\n"
          << "Fern pipeline: black boxes nv12_to_rgb (the crop is the offset of the subset it consumes), "
             "resize_bilinear, letterbox and normalize (fern/fern_impl.h, FKL's conventions and constants, checked at "
             "run time); the region a Rect data structure whose fields are used in the subsets; composed under one "
             "Tile over bands of 16 output rows: per region, the crop (h x w) and the resized image (32 x 64) once, "
             "then per band a 16 x 64 padded band, letterboxed and normalized into the output plane; intermediates from "
             "a per-thread stack arena; one call of the generated function per region (hook_fern_band16)\n"
          << "Multithreaded comparison, \"Fern code, FKL thread pool\": Fern generates serial CPU code and has no "
             "parallel code of its own, so the benchmark runs Fern's per-region calls as the chunks of one execution on "
             "FKL's thread pool (one chunk per region: at most N workers run Fern's code, and every run wakes all of the "
             "pool's sleeping workers, which spin as after FKL's runs), against FKL's CPU_MULTITHREAD backend on the "
             "same pool (one chunk per output row of each region). Fern has no threads of its own: both sides of the "
             "multithreaded comparison use FKL's pool\n"
          << "Thread pool, measured before the benchmark: each of FKL's workers stays busy " << fklBusyUs << " us per "
          << "FKL CPU_MULTITHREAD run on 1 region (spin and wake-up), which the "
          << std::chrono::milliseconds(IDLE_GAP).count() << " ms idle gap must outlast\n";
        writeCpuPlatform(p, threads);
        std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
        if (!passed) {
            throw std::runtime_error("validation failed (FKL or Fern against the reference, FKL's pool against FKL, or "
                                     "Fern's code on the pool against Fern, beyond the tolerance): see "
                                     "cpu_validation.csv");
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
