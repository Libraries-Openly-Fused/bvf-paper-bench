// FKL vs Halide on the CPU, Figure 1 pipeline: a Halide pipeline (JIT, compiled once before the timed loops) vs FKL's
// CPU backends (BVF).
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// The harness, the workload and FKL's pipelines are common/bench_cpu.h and common/bench_fkl_cpu.cpp.
//
// Halide's pipeline is one Func, out(c, x, y, n), over the N x 64 x 64 x 3 output (c innermost, FKL's interleaved
// float3 layout), whose every producer is inlined: for each output value it reads the region of plane n from a region
// table (an input buffer of 4 x N int32: x, y, width, height), samples the frame with FKL's bilinear convention (source
// coordinate dst * (in / out), the second sample of each axis clamped to the region), converts the four NV12 samples
// with FKL's BT.601 limited range matrix (chroma replicated, float, no clamping), adds the black rows (a select whose
// resized value is marked likely, so that Halide partitions the y loop and the border rows compute nothing) and
// normalizes. The region
// table changes every iteration without recompiling: the regions are run-time data, as FKL's Crop parameters. The frame
// coordinates are wrapped in unsafe_promise_clamped (every region lies inside the frame, common/bench_geometry.h), so
// that bounds inference bounds the frame reads by the frame itself; nothing is computed over the frame, because no
// producer is scheduled outside the output's loops.
//
//   halide_default   no scheduling directive: the output's loops in their default order (n, y, x, c), serial
//   halide_parallel  out.bound(c, 0, 3).reorder(c, x, n, y).unroll(c).vectorize(x, 8).parallel(y): one task per output
//                    row (y) for all the planes, 8 output pixels per vector, the 3 channels unrolled
//   fkl              one executeOperations<TransformDPP<ParArch::CPU>> call with the whole chain (BVF), one thread
//   fkl mt           the same call with ParArch::CPU_MULTITHREAD, on FKL's thread pool
//
// The comparisons run each library's complete loop on its own, Halide's first and FKL's after it, on the same regions
// (common/bench_cpu.h). Three comparisons, each with the same number of threads on both sides: halide_default and
// halide_parallel with Halide's thread pool set to 1 thread against fkl, and halide_parallel with as many threads as
// FKL's thread pool has workers against fkl mt. halide_default has no parallel loop, so on more threads it would still
// run on one: the multithreaded process does not time it. Halide's
// threads are set once per process, before any run, with Halide::Internal::JITSharedRuntime::set_num_threads (the JIT
// runtime's halide_set_num_threads), which does not take workers away once they have started: one process runs the
// single-thread comparisons and another the multithreaded one.
//
//     benchmark_halide_cpu <results folder> 1      both schedules against FKL on one thread
//     benchmark_halide_cpu <results folder> pool   halide_parallel against FKL's thread pool, on as many Halide threads
//                                                  (or the pool size as a number)

#include "bench_cpu.h"

#include <Halide.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace bench::cpu {

enum class Schedule { Default, Parallel };

const char* scheduleName(Schedule s) {
    return s == Schedule::Default ? "halide_default" : "halide_parallel";
}

// The pipeline of one schedule, JIT compiled for the host; its lowered statement (Halide's loops, after bounds inference
// and loop partitioning) is written to stmtFile
// compileMs: plus the JIT compilation's time (compile_to_callable only, not the lowering to the .stmt file)
Halide::Callable compilePipeline(Schedule schedule, const Halide::Target& target, const std::filesystem::path& stmtFile,
                                 double& compileMs) {
    using namespace Halide;
    const Conversion k = fklConversion();
    ImageParam frame(UInt(8), 2, "frame");     // FRAME_W x FRAME_H * 3 / 2 bytes: the luma plane, then the chroma one
    ImageParam regions(Int(32), 2, "regions"); // 4 x N: x, y, width, height of every region of the iteration
    Var c("c"), x("x"), y("y"), n("n");

    // The region of plane n and the source coordinates of FKL's bilinear resize (fused_kernel's resize.h and
    // interpolation.h): src = dst * (size / out size), x1 = floor(src), x2 = x1 + 1, read at min(x2, size - 1)
    const Expr rx = regions(0, n), ry = regions(1, n), rw = regions(2, n), rh = regions(3, n);
    const Expr srcX = cast<float>(x) * (cast<float>(rw) / RES_W);
    // The row of the resized image (outside the border); clamped so that the samples of a border row, which the select
    // below discards, still read inside the region where Halide evaluates them (the border rows of a loop that Halide
    // does not partition)
    const Expr resizedY = y - BORDER_TOP;
    const Expr srcY = cast<float>(clamp(resizedY, 0, RES_H - 1)) * (cast<float>(rh) / RES_H);
    const Expr x1 = cast<int>(floor(srcX)), y1 = cast<int>(floor(srcY));
    const Expr x2 = x1 + 1, y2 = y1 + 1;
    const Expr x2Read = min(x2, rw - 1), y2Read = min(y2, rh - 1);

    // Channel c of FKL's conversion of the NV12 pixel at (px, py) of the frame; the coordinates are promised inside the
    // frame, so that bounds inference bounds the frame reads by the frame
    const auto rgb = [&](const Expr& regionX, const Expr& regionY) {
        const Expr px = unsafe_promise_clamped(rx + regionX, 0, FRAME_W - 1);
        const Expr py = unsafe_promise_clamped(ry + regionY, 0, FRAME_H - 1);
        const Expr chromaX = (px / 2) * 2, chromaY = FRAME_H + py / 2;
        const Expr luma = cast<float>(frame(px, py)) - k.lumaOffset;
        const Expr u = cast<float>(frame(chromaX, chromaY)) - k.chromaOffset;
        const Expr v = cast<float>(frame(chromaX + 1, chromaY)) - k.chromaOffset;
        std::vector<Expr> channels;
        for (int ch = 0; ch < 3; ++ch) {
            channels.push_back(luma * k.matrix[ch][0] + u * k.matrix[ch][1] + v * k.matrix[ch][2]);
        }
        return mux(c, channels);
    };
    const Expr fx2 = cast<float>(x2), fy2 = cast<float>(y2), fx1 = cast<float>(x1), fy1 = cast<float>(y1);
    const Expr resized = rgb(x1, y1) * ((fx2 - srcX) * (fy2 - srcY)) + rgb(x2Read, y1) * ((srcX - fx1) * (fy2 - srcY)) +
                         rgb(x1, y2Read) * ((fx2 - srcX) * (srcY - fy1)) +
                         rgb(x2Read, y2Read) * ((srcX - fx1) * (srcY - fy1));
    // The black rows on top and bottom, then (x * MUL - SUB) / DIV. The resized value is marked likely, Halide's idiom
    // for boundary conditions: loop partitioning then splits the y loop into the top border, the resized rows and the
    // bottom border, and the border rows compute nothing but the constant
    const Expr bordered = select(resizedY < 0, 0.f, resizedY >= RES_H, 0.f, likely(resized));
    Func out("out");
    out(c, x, y, n) = (bordered * mux(c, {MUL[0], MUL[1], MUL[2]}) - mux(c, {SUB[0], SUB[1], SUB[2]})) /
                      mux(c, {DIV[0], DIV[1], DIV[2]});

    // FKL's output layout: N x 64 x 64 x float3, contiguous; the region table: 4 x N int32, contiguous
    out.output_buffer().dim(0).set_bounds(0, 3).set_stride(1);
    out.output_buffer().dim(1).set_bounds(0, OUT_W).set_stride(3);
    out.output_buffer().dim(2).set_bounds(0, OUT_H).set_stride(3 * OUT_W);
    out.output_buffer().dim(3).set_min(0).set_stride(3 * OUT_W * OUT_H);
    regions.dim(0).set_bounds(0, 4).set_stride(1);
    regions.dim(1).set_min(0).set_stride(4);
    frame.dim(0).set_bounds(0, FRAME_W).set_stride(1);
    frame.dim(1).set_bounds(0, FRAME_H * 3 / 2);
    if (schedule == Schedule::Parallel) {
        out.bound(c, 0, 3).reorder(c, x, n, y).unroll(c).vectorize(x, 8).parallel(y);
    }
    out.compile_to_lowered_stmt(stmtFile.string(), {frame, regions}, Halide::Text, target);
    const auto start = Clock::now();
    Halide::Callable callable = out.compile_to_callable({frame, regions}, target);
    compileMs += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return callable;
}

// One schedule on n regions per iteration: the region table (4 x n int32: x, y, width, height), filled in every run
// from the regions of the iteration, as FKL's build() calls take the iteration's rectangles, and the output, as FKL's
// pipelines
class HalidePipeline {
public:
    HalidePipeline(Halide::Callable& callable, const Workload& w, size_t n)
        : m_callable(callable), m_n(n), m_regions(*w.regions),
          m_frame(const_cast<uint8_t*>(w.frame),
                  {halide_dimension_t{0, FRAME_W, 1}, halide_dimension_t{0, FRAME_H * 3 / 2, static_cast<int32_t>(w.pitch)}}),
          m_table(4, static_cast<int>(n)),
          // Dense c x x x y x n: FKL's N x 64 x 64 x float3 layout
          m_out(3, OUT_W, OUT_H, static_cast<int>(n)) {}

    // On the regions of iteration t
    void run(size_t t) {
        for (size_t i = 0; i < m_n; ++i) {
            const Region& r = m_regions.at(t, i);
            const int ii = static_cast<int>(i);
            m_table(0, ii) = r.x;
            m_table(1, ii) = r.y;
            m_table(2, ii) = r.width;
            m_table(3, ii) = r.height;
        }
        const int error = m_callable(m_frame, m_table, m_out);
        if (error != 0) {
            throw std::runtime_error("Halide pipeline failed with error " + std::to_string(error));
        }
    }

    std::vector<float> output() const {
        return std::vector<float>(m_out.data(), m_out.data() + m_n * OUT_FLOATS);
    }

    void fillNaN() { m_out.fill(std::numeric_limits<float>::quiet_NaN()); }

private:
    Halide::Callable& m_callable;
    size_t m_n;
    const RegionSets& m_regions;
    Halide::Runtime::Buffer<uint8_t, 2> m_frame;
    Halide::Runtime::Buffer<int32_t, 2> m_table;
    Halide::Runtime::Buffer<float, 4> m_out;
};

// Halide's thread pool, set once per process before any parallel run: the JIT runtime's halide_set_num_threads
// (Halide::Internal::JITSharedRuntime::set_num_threads) does not take workers away once they have started, so a process
// that changed it between runs would still run on every worker started earlier
void setHalideThreads(int threads) {
    Halide::Internal::JITSharedRuntime::set_num_threads(threads);
    if (Halide::Internal::JITSharedRuntime::get_num_threads() != threads) {
        throw std::runtime_error("Halide's thread pool did not take " + std::to_string(threads) + " threads");
    }
}

// The errors on the regions of one validated iteration (defaultVsReference: NAN in the multithreaded process, which
// does not run halide_default)
struct Check {
    size_t batch, iteration;
    float fklVsReference, defaultVsReference, parallelVsReference, parallelVsFkl, fklMultithreadVsFkl;
    // What the program requires (withinTolerance): FKL and the timed schedules against the reference with FKL's
    // conventions (Halide's too), and FKL's pool against FKL
    bool passed() const {
        return withinTolerance(fklVsReference) && withinTolerance(parallelVsReference) &&
               withinTolerance(fklMultithreadVsFkl) && (std::isnan(defaultVsReference) || withinTolerance(defaultVsReference));
    }
};

struct Results {
    // On one thread, both schedules against FKL CPU; on FKL's pool size, halide_parallel against FKL CPU_MULTITHREAD
    std::vector<Row> defaultRows, parallelRows;
    std::vector<Check> checks;
};

template <size_t N>
void runBatch(const Workload& w, Halide::Callable& defaultCallable, Halide::Callable& parallelCallable, Results& res,
              bool multithreaded) {
    HalidePipeline halideDefault(defaultCallable, w, N);
    HalidePipeline halideParallel(parallelCallable, w, N);
    const auto fkl = makeFklCpu(N, false, w);
    const auto fklMT = makeFklCpu(N, true, w);
    // Before the timing, on the regions of the first, every 50th and the last timed iteration
    Check worst{N, 0, 0.f, multithreaded ? NAN : 0.f, 0.f, 0.f, 0.f};
    for (const size_t k : validatedIterations(ITERS)) {
        const size_t t = WARMUP_ITERS + k;
        halideParallel.fillNaN();
        halideParallel.run(t);
        fkl->run(t);
        fklMT->run(t);
        const auto fklOut = fkl->output();
        const auto reference = referenceOutput(w, N, t, Conventions::Fkl);
        float defaultVsReference = NAN;
        if (!multithreaded) {
            halideDefault.fillNaN();
            halideDefault.run(t);
            defaultVsReference = maxAbsError(halideDefault.output(), reference);
        }
        const Check c{N, t, maxAbsError(fklOut, reference), defaultVsReference,
                      maxAbsError(halideParallel.output(), reference),
                      maxAbsError(halideParallel.output(), fklOut), maxAbsError(fklMT->output(), fklOut)};
        res.checks.push_back(c);
        worst = Check{N, t, worstError(c.fklVsReference, worst.fklVsReference),
                      worstError(c.defaultVsReference, worst.defaultVsReference),
                      worstError(c.parallelVsReference, worst.parallelVsReference),
                      worstError(c.parallelVsFkl, worst.parallelVsFkl),
                      worstError(c.fklMultithreadVsFkl, worst.fklMultithreadVsFkl)};
    }
    const Check& c = worst;

    FklCpu& fklSide = multithreaded ? *fklMT : *fkl;
    const char* fklName = multithreaded ? "CPU_MULTITHREAD" : "CPU";
    if (!multithreaded) {
        res.defaultRows.push_back(
            timeTwo(N, [&](size_t t) { halideDefault.run(t); }, [&](size_t t) { fklSide.run(t); }));
        const Row& d = res.defaultRows.back();
        std::printf("N %3zu | halide_default %8.1f us, FKL %s %8.1f us, mean speedup %6.2fx |", N, d.baseline.median,
                    fklName, d.fkl.median, d.meanSpeedup);
    } else {
        std::printf("N %3zu |", N);
    }
    res.parallelRows.push_back(timeTwo(N, [&](size_t t) { halideParallel.run(t); }, [&](size_t t) { fklSide.run(t); }));
    const Row& p = res.parallelRows.back();
    std::printf(" halide_parallel %8.1f us, FKL %s %8.1f us, mean speedup %6.2fx | max "
                "err vs reference: FKL %.2e, halide_default %.2e, halide_parallel %.2e; halide_parallel vs FKL %.2e, FKL "
                "CPU_MULTITHREAD vs FKL CPU %.2e\n",
                p.baseline.median, fklName, p.fkl.median, p.meanSpeedup, c.fklVsReference,
                c.defaultVsReference, c.parallelVsReference, c.parallelVsFkl, c.fklMultithreadVsFkl);
}

} // namespace bench::cpu

int main(int argc, char** argv) {
    using namespace bench::cpu;
    try {
        if (argc < 3) {
            throw std::runtime_error("Usage: benchmark_halide_cpu <results folder> <threads: 1, or pool (FKL's thread "
                                     "pool size, " + std::to_string(poolThreads()) + ")>");
        }
        const std::filesystem::path outDir = argv[1];
        std::filesystem::create_directories(outDir);
        // One thread: halide_default and halide_parallel against FKL on one thread. As many threads as FKL's thread
        // pool has workers (std::thread::hardware_concurrency()): halide_parallel against FKL's thread pool
        // (halide_default has no parallel loop: it is not timed there).
        const int poolSize = poolThreads();
        const int threads = std::string(argv[2]) == "pool" ? poolSize : std::stoi(argv[2]);
        if (threads != 1 && threads != poolSize) {
            throw std::runtime_error("The comparisons run on 1 thread or on " + std::to_string(poolSize) +
                                     " threads (FKL's thread pool), not " + std::to_string(threads));
        }
        // pool is the multithreaded comparison whatever the pool's size (on a one-worker pool too)
        const bool multithreaded = std::string(argv[2]) == "pool" || threads > 1;
        const std::string threadLabel = multithreaded ? std::to_string(threads) + "threads" : "1thread";
        Workload w;
        initWorkload(w);

        // Both schedules JIT compiled once, before any loop, then Halide's thread count, once
        const Halide::Target target = Halide::get_host_target();
        double compileMs = 0.;
        Halide::Callable defaultCallable = compilePipeline(
            Schedule::Default, target, outDir / (std::string(scheduleName(Schedule::Default)) + ".stmt"), compileMs);
        Halide::Callable parallelCallable = compilePipeline(
            Schedule::Parallel, target, outDir / (std::string(scheduleName(Schedule::Parallel)) + ".stmt"), compileMs);
        setHalideThreads(threads);

        // How long each pool's workers stay busy after a parallel run with little work (their spin and wake-up), which
        // the idle gap must outlast; Halide has workers only on more than one thread
        double halideBusyUs = NAN;
        if (multithreaded) {
            HalidePipeline probe(parallelCallable, w, BATCHES.front());
            // Halide's thread count includes the caller; FKL's pool does not (the caller waits)
            halideBusyUs = workerBusyUs([&]() { probe.run(WARMUP_ITERS); }, threads - 1);
        }
        const auto fklProbe = makeFklCpu(BATCHES.front(), true, w);
        const double fklBusyUs = workerBusyUs([&]() { fklProbe->run(WARMUP_ITERS); }, poolSize);
        // On one thread, a check that Halide runs on the calling thread only: the CPU time of the process's other
        // threads (Halide has none, FKL's are asleep) per halide_parallel run on 100 regions
        double singleThreadCheckUs = NAN;
        if (!multithreaded) {
            HalidePipeline check(parallelCallable, w, BATCHES.back());
            singleThreadCheckUs = workerBusyUs([&]() { check.run(WARMUP_ITERS); }, 1, 20);
        }
        std::printf("Halide %d.%d.%d, target %s, both schedules compiled in %.0f ms, %d Halide thread(s); workers busy "
                    "after a parallel run: Halide %.1f us, FKL %.1f us\n",
                    HALIDE_VERSION_MAJOR, HALIDE_VERSION_MINOR, HALIDE_VERSION_PATCH, target.to_string().c_str(),
                    compileMs, threads, halideBusyUs, fklBusyUs);

        Results res;
        forEachBatch([&]<size_t N>() { runBatch<N>(w, defaultCallable, parallelCallable, res, multithreaded); });

        const std::string fklSuffix = multithreaded ? "CPU_MULTITHREAD" : "CPU";
        const std::string label = multithreaded ? std::to_string(threads) + " threads" : "1 thread";
        if (!multithreaded) {
            writeCSV(outDir / ("benchmark_cpu_halide_default_" + threadLabel + "_vs_fkl_bvf_" + fklSuffix + ".csv"),
                     "Halide default " + label, res.defaultRows, "FKL " + fklSuffix);
        }
        writeCSV(outDir / ("benchmark_cpu_halide_parallel_" + threadLabel + "_vs_fkl_bvf_" + fklSuffix + ".csv"),
                 "Halide parallel " + label, res.parallelRows, "FKL " + fklSuffix);
        std::ofstream v(outDir / ("cpu_validation_" + threadLabel + ".csv"));
        v << "Batch size, Iteration, FKL MaxAbsError vs FKL-convention reference, Halide default MaxAbsError vs "
             "FKL-convention reference, Halide parallel MaxAbsError vs FKL-convention reference, Halide parallel vs FKL "
             "MaxAbsDifference, FKL CPU_MULTITHREAD vs FKL CPU MaxAbsDifference, Within tolerance" << std::endl;
        bool passed = true;
        for (const auto& c : res.checks) {
            v << c.batch << ", " << c.iteration << ", " << c.fklVsReference << ", ";
            if (!std::isnan(c.defaultVsReference)) {
                v << c.defaultVsReference;
            }
            v << ", " << c.parallelVsReference << ", " << c.parallelVsFkl << ", " << c.fklMultithreadVsFkl << ", "
              << (c.passed() ? "yes" : "no") << std::endl;
            passed &= c.passed();
        }
        writeRegionStats((outDir / "regions.csv").string(), *w.regions, WARMUP_ITERS, ITERS);
        std::ofstream p(outDir / ("cpu_platform_" + threadLabel + ".txt"));
        p << "Halide: " << HALIDE_VERSION_MAJOR << "." << HALIDE_VERSION_MINOR << "." << HALIDE_VERSION_PATCH
          << " (release binary " << BENCH_HALIDE_RELEASE << ", libHalide " << BENCH_HALIDE_LIBRARY << " SHA-256 "
          << BENCH_HALIDE_LIBRARY_SHA256 << "), JIT target " << target.to_string()
          << ", both schedules JIT-compiled once before the loops in " << compileMs << " ms (compile_to_callable only; "
          << "lowered statements written first to halide_default.stmt and halide_parallel.stmt, not timed)\n"
          << "Halide pipeline: one Func out(c, x, y, n), every producer inlined; region table (4 x N int32) as an input "
             "buffer, filled from the regions of the iteration in every run (timed); FKL's bilinear sampling and BT.601 "
             "limited range "
             "matrix; frame coordinates in unsafe_promise_clamped; the border rows a select with the resized value "
             "marked likely (loop partitioning). Schedules: halide_default (none) and halide_parallel (out.bound(c, 0, "
             "3).reorder(c, x, n, y).unroll(c).vectorize(x, 8).parallel(y))\n"
          << "Halide's threads: " << threads << ", set once before any run with "
          << "Halide::Internal::JITSharedRuntime::set_num_threads (this process runs the "
          << (multithreaded ? "multithreaded comparison: halide_parallel only, halide_default has no parallel loop"
                            : "single-thread comparisons")
          << ")\n"
          << "Thread pools, measured before the benchmark: each worker stays busy " << halideBusyUs << " us per "
          << "halide_parallel run on 1 region ("
#ifdef _WIN32
          << "NAN: not measured on Windows"
#else
          << (multithreaded ? "Halide's workers" : "NAN: one thread, no workers")
#endif
          << "), and " << fklBusyUs << " us per FKL "
          << "CPU_MULTITHREAD run on 1 region (spin and wake-up), which the "
          << std::chrono::milliseconds(IDLE_GAP).count() << " ms idle gap must outlast\n";
        // A Halide worker taking tasks would use hundreds of us per run on 100 regions
        constexpr double SINGLE_THREAD_LIMIT_US = 5.;
        if (!multithreaded) {
            p << "Validation of the thread count: CPU time of the process's other threads per halide_parallel run on "
              << BATCHES.back() << " regions, on one Halide thread: " << singleThreadCheckUs << " us (0: Halide ran on "
              << "the calling thread only; the program fails above " << SINGLE_THREAD_LIMIT_US << " us, or if the "
              << "threads' CPU times cannot be read on Linux; not checked on Windows)\n";
            std::printf("Other threads' CPU time per halide_parallel run on one Halide thread: %.1f us\n",
                        singleThreadCheckUs);
        }
        writeCpuPlatform(p, poolSize,
                         {"cpu_validation_" + threadLabel + ".csv",
                          "each timed Halide schedule, on this process's thread count, against the reference with "
                          "FKL's conventions",
                          "Halide's region table (4 x N int32) is filled from the regions of the iteration in every "
                          "run, which is timed"});
        std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
        if (!passed) {
            throw std::runtime_error("validation failed (FKL or Halide against the reference, or FKL's pool against "
                                     "FKL, beyond the tolerance): see cpu_validation_" + threadLabel + ".csv");
        }
#ifndef _WIN32
        // On Linux the threads' CPU times are read from /proc: a value that cannot be read fails the check too
        if (!multithreaded && !(std::isfinite(singleThreadCheckUs) && singleThreadCheckUs <= SINGLE_THREAD_LIMIT_US)) {
            throw std::runtime_error("Halide used other threads on one thread, or their CPU time could not be read: " +
                                     std::to_string(singleThreadCheckUs) + " us per run");
        }
#endif
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
