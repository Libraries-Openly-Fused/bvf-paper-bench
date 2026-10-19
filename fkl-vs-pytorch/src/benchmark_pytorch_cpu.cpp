// FKL vs PyTorch on the CPU, Figure 1 pipeline: eager PyTorch (LibTorch's C++ API) vs FKL's CPU backends (BVF).
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// The harness, the workload and FKL's pipelines are common/bench_cpu.h and common/bench_fkl_cpu.cpp; PyTorch's
// pipeline is src/pytorch_pipeline.h, the same code as on the GPU.
//
//   pytorch: per region, views of the region's planes, NV12 -> float RGB with addmm and a broadcast add, interpolate
//            (bilinear) and a copy into its output plane; the border rows set to 0 and multiply, subtract and divide on
//            the whole tensor once per iteration. PyTorch's own CPU kernels (its runtime SIMD dispatch) and its
//            intra-op thread pool (OpenMP, libgomp, in the official wheels), under c10::InferenceMode.
//   fkl:     one executeOperations<TransformDPP<ParArch::CPU>> call with the whole chain (BVF), on one thread.
//   fkl mt:  the same call with ParArch::CPU_MULTITHREAD, on FKL's thread pool (one worker per hardware thread).
//
// Two comparisons, each with the same number of threads on both sides: PyTorch with at::set_num_threads(1) against
// fkl, and PyTorch with as many threads as FKL's thread pool has workers against fkl mt. In each, PyTorch's complete
// loop runs first and FKL's after it, on the same regions (common/bench_cpu.h). PyTorch keeps libgomp's default spin
// (its workers spin for GOMP_SPINCOUNT iterations after a parallel region before they sleep: about 5.4 ms with the
// default 300000 on the paper's machine); the program measures how long the workers of both pools stay busy after a
// run (--omp-spin alone prints it) and records it in cpu_platform.txt.
//
//     benchmark_pytorch_cpu <results folder>
//     benchmark_pytorch_cpu --omp-spin      how long libgomp's and FKL's workers stay busy after a parallel run, us

#include "bench_cpu.h"
#include "pytorch_pipeline.h"

#include <ATen/Parallel.h>
#include <ATen/Version.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace bench::cpu {

// The errors on the regions of one validated iteration
struct Check {
    size_t batch, iteration;
    float fklVsReference, pytorchVsReference, pytorchVsFkl, fklMultithreadVsFkl, pytorchThreadsVsOneThread;
    // What the program requires (withinTolerance): FKL and PyTorch against the references of their conventions, FKL's
    // pool against FKL, and PyTorch on `threads` threads against PyTorch on one
    bool passed() const {
        return withinTolerance(fklVsReference) && withinTolerance(pytorchVsReference) &&
               withinTolerance(fklMultithreadVsFkl) && withinTolerance(pytorchThreadsVsOneThread);
    }
};

template <size_t N>
void runBatch(Workload& w, std::vector<Row>& singleThread, std::vector<Row>& multiThread, std::vector<Check>& checks,
              int threads) {
    pytorch::Pipeline torchPipeline(w.frame, w.pitch, torch::kCPU, N, fklConversion());
    const auto fkl = makeFklCpu(N, false, w);
    const auto fklMT = makeFklCpu(N, true, w);
    const auto runPytorch = [&](size_t t) { torchPipeline.run(&w.regions->at(t, 0)); };
    // Before the timing, on the regions of the first, every 50th and the last timed iteration: PyTorch on 1 thread
    // against the pixel-center reference and FKL, and on `threads` threads against itself on 1
    Check worst{N, 0, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (const size_t k : validatedIterations(ITERS)) {
        const size_t t = WARMUP_ITERS + k;
        at::set_num_threads(1);
        torchPipeline.fillNaN();
        runPytorch(t);
        const auto torchOut = torchPipeline.download();
        at::set_num_threads(threads);
        torchPipeline.fillNaN();
        runPytorch(t);
        const auto torchThreadsOut = torchPipeline.download();
        fkl->run(t);
        fklMT->run(t);
        const auto fklOut = fkl->output();
        const Check c{N, t, maxAbsError(fklOut, referenceOutput(w, N, t, Conventions::Fkl)),
                      maxAbsError(torchOut, referenceOutput(w, N, t, Conventions::PixelCenters)),
                      maxAbsError(torchOut, fklOut), maxAbsError(fklMT->output(), fklOut),
                      maxAbsError(torchThreadsOut, torchOut)};
        checks.push_back(c);
        // std::max keeps a NaN only as its first argument: the error goes first
        worst = Check{N, t, std::max(c.fklVsReference, worst.fklVsReference),
                      std::max(c.pytorchVsReference, worst.pytorchVsReference),
                      std::max(c.pytorchVsFkl, worst.pytorchVsFkl),
                      std::max(c.fklMultithreadVsFkl, worst.fklMultithreadVsFkl),
                      std::max(c.pytorchThreadsVsOneThread, worst.pytorchThreadsVsOneThread)};
    }
    const Check& c = worst;

    at::set_num_threads(1);
    singleThread.push_back(timeTwo(N, runPytorch, [&](size_t t) { fkl->run(t); }));
    at::set_num_threads(threads);
    multiThread.push_back(timeTwo(N, runPytorch, [&](size_t t) { fklMT->run(t); }));
    at::set_num_threads(1);
    const Row& s = singleThread.back();
    const Row& m = multiThread.back();
    std::printf("N %3zu | PyTorch 1 thread %8.1f us, FKL CPU %8.1f us, mean speedup %6.2fx | PyTorch %d threads %8.1f us, "
                "FKL CPU_MULTITHREAD %8.1f us, mean speedup %6.2fx | max err: FKL vs "
                "reference %.2e, PyTorch vs pixel-center reference %.2e, PyTorch vs FKL %.2e, FKL CPU_MULTITHREAD vs "
                "FKL CPU %.2e, PyTorch %d threads vs 1 thread %.2e\n",
                N, s.baseline.median, s.fkl.median, s.meanSpeedup, threads, m.baseline.median, m.fkl.median,
                m.meanSpeedup, c.fklVsReference,
                c.pytorchVsReference, c.pytorchVsFkl, c.fklMultithreadVsFkl, threads, c.pytorchThreadsVsOneThread);
}

// libgomp's workers (through PyTorch, on an add of 32768 floats per thread, PyTorch's grain size, so that every thread
// takes part) and FKL's workers (the CPU_MULTITHREAD pipeline on 1 region: 64 chunks of one output row, so that every
// worker takes part): little work, so that the time is mostly each pool's spin and wake-up
struct WorkerSpin {
    double ompBusyUs, fklBusyUs;
};

WorkerSpin measureWorkerSpin(const Workload& w, int threads) {
    constexpr int ROUNDS = 50;
    at::set_num_threads(threads);
    torch::Tensor x = torch::zeros({static_cast<int64_t>(threads) * 32768});
    // libgomp's team includes the caller; FKL's pool does not (the caller waits)
    const double omp = workerBusyUs([&]() { x.add_(1.f); }, threads - 1, ROUNDS);
    at::set_num_threads(1);
    const auto fklMT = makeFklCpu(BATCHES.front(), true, w);
    const double fklBusy = workerBusyUs([&]() { fklMT->run(WARMUP_ITERS); }, threads, ROUNDS);
    return {omp, fklBusy};
}

std::string envOr(const char* name, const char* unset) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string(unset);
}

} // namespace bench::cpu

int main(int argc, char** argv) {
    using namespace bench::cpu;
    try {
        // PyTorch runs on as many threads as FKL's thread pool has workers (std::thread::hardware_concurrency())
        const int threads = poolThreads();
        Workload w;
        initWorkload(w);
        if (argc > 1 && std::string(argv[1]) == "--omp-spin") {
            const WorkerSpin spin = measureWorkerSpin(w, threads);
            std::printf("GOMP_SPINCOUNT=%s OMP_WAIT_POLICY=%s: libgomp worker busy %.1f us per parallel run, FKL "
                        "CPU_MULTITHREAD worker busy %.1f us per run\n",
                        envOr("GOMP_SPINCOUNT", "unset").c_str(), envOr("OMP_WAIT_POLICY", "unset").c_str(),
                        spin.ompBusyUs, spin.fklBusyUs);
            return 0;
        }
        const std::filesystem::path outDir = argc > 1 ? argv[1] : "results";
        std::filesystem::create_directories(outDir);
        const WorkerSpin spin = measureWorkerSpin(w, threads);
        std::printf("Workers busy after a parallel run: libgomp %.1f us, FKL %.1f us\n", spin.ompBusyUs,
                    spin.fklBusyUs);

        std::vector<Row> singleThread, multiThread;
        std::vector<Check> checks;
        forEachBatch([&]<size_t N>() { runBatch<N>(w, singleThread, multiThread, checks, threads); });

        writeCSV(outDir / "benchmark_cpu_pytorch_1thread_vs_fkl_bvf_CPU.csv", "PyTorch 1 thread", singleThread,
                 "FKL CPU");
        writeCSV(outDir / ("benchmark_cpu_pytorch_" + std::to_string(threads) + "threads_vs_fkl_bvf_CPU_MULTITHREAD.csv"),
                 "PyTorch " + std::to_string(threads) + " threads", multiThread, "FKL CPU_MULTITHREAD");
        std::ofstream v(outDir / "cpu_validation.csv");
        v << "Batch size, Iteration, FKL MaxAbsError vs FKL-convention reference, PyTorch MaxAbsError vs "
             "pixel-center reference, PyTorch vs FKL MaxAbsDifference, FKL CPU_MULTITHREAD vs FKL CPU "
             "MaxAbsDifference, PyTorch "
          << threads << " threads vs PyTorch 1 thread MaxAbsDifference, Within tolerance" << std::endl;
        bool passed = true;
        for (const auto& c : checks) {
            v << c.batch << ", " << c.iteration << ", " << c.fklVsReference << ", " << c.pytorchVsReference << ", "
              << c.pytorchVsFkl << ", " << c.fklMultithreadVsFkl << ", " << c.pytorchThreadsVsOneThread << ", "
              << (c.passed() ? "yes" : "no") << std::endl;
            passed &= c.passed();
        }
        writeRegionStats((outDir / "regions.csv").string(), *w.regions, WARMUP_ITERS, ITERS);
        std::ofstream(outDir / "pytorch_build_information.txt") << at::show_config() << "\n" << at::get_parallel_info();
        std::ofstream p(outDir / "cpu_platform.txt");
        p << "PyTorch: " << BENCH_TORCH_VERSION << " (LibTorch C++ API of the wheel, eager mode, c10::InferenceMode), "
          << "CPU capability " << at::get_cpu_capability()
          << ", intra-op threads set with at::set_num_threads (1, then " << threads
          << "); full configuration in pytorch_build_information.txt\n"
          << "PyTorch's OpenMP (libgomp) workers: GOMP_SPINCOUNT=" << envOr("GOMP_SPINCOUNT", "unset (libgomp's default)")
          << ", OMP_WAIT_POLICY=" << envOr("OMP_WAIT_POLICY", "unset") << "; measured before the benchmark, each worker "
          << "stays busy " << spin.ompBusyUs << " us per parallel run of PyTorch (an add with 32768 floats per thread) "
          << "and each of FKL's CPU_MULTITHREAD workers " << spin.fklBusyUs << " us per run (1 region): work plus spin "
          << "before sleeping (libgomp's default spin is kept)\n"
          << "PyTorch pipeline: per region, views of the region's luma and chroma planes, chroma to float and addmm "
             "with FKL's chroma coefficients and every offset folded into the bias, a broadcast add of the 8-bit luma "
             "with its coefficient (chroma replicated to 2x2 pixels; the add converts the luma), interpolate "
             "(bilinear, align_corners=false: "
             "pixel-center sampling) and a copy into its output plane; per iteration the border rows set to 0 and "
             "mul_, sub_, div_ on the whole N x 64 x 64 x 3 tensor\n";
        writeCpuPlatform(p, threads);
        std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
        if (!passed) {
            throw std::runtime_error("validation failed (FKL or PyTorch against its reference, FKL's pool against FKL, "
                                     "or PyTorch on " + std::to_string(threads) + " threads against PyTorch on one, "
                                     "beyond the tolerance): see cpu_validation.csv");
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
