// FKL vs OpenCV on the CPU, Figure 1 pipeline: OpenCV's CPU functions vs FKL's CPU backends (BVF).
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// One 3840x2160 NV12 frame in host memory and N regions, new every iteration, with the sizes and positions of
// common/bench_geometry.h (20x10, 32x16 and 48x24 by turns), bilinear resize to 64x32, black rows added on top and
// bottom up to a square, then (x * mul - sub) / div written as float3 into an N-plane tensor. The harness, the workload
// and FKL's pipelines are common/bench_cpu.h and common/bench_fkl_cpu.cpp, shared by every CPU benchmark.
//
// The color conversion is BT.601 limited range in both, because OpenCV's CPU cvtColor only implements that one for
// NV12; every benchmark of this repository uses it (common/bench_geometry.h).
//
//   opencv: cvtColor(COLOR_YUV2RGB_NV12) on the full frame (8 bit RGB), then per region convertTo(CV_32F) (into a
//           buffer of the largest region size, through a view of the region's size), resize and copyMakeBorder, then
//           multiply, subtract and divide on the whole tensor. OpenCV uses its own SIMD dispatch
//           (up to AVX2 here) and its thread pool.
//   opencv crop NV12: the same pipeline with the NV12 -> RGB of each region only: cvtColorTwoPlane(COLOR_YUV2RGB_NV12)
//           of the region's Y view and its UV view (every region has even coordinates and sizes) into the region's
//           view of the 8 bit RGB frame, which the rest of the pipeline reads as before.
//   fkl:    one executeOperations<TransformDPP<ParArch::CPU>> call with the whole chain (BVF). FKL's CPU backend runs
//           on one thread, compiled with -march=native.
//   fkl mt: the same call with TransformDPP<ParArch::CPU_MULTITHREAD>: the stream enqueues the execution in FKL's shared
//           thread pool (one worker per hardware thread), in chunks of one output row of one crop, and sync() waits.
//
// Two comparisons per form of OpenCV's conversion, each with the same number of threads on both sides: OpenCV with 1
// thread against fkl, and OpenCV with as many threads as FKL's thread pool has workers (one per hardware thread)
// against fkl mt. In each, OpenCV's complete loop runs first and FKL's after it, on the same regions
// (common/bench_cpu.h). Per batch size and thread count, the form with the lower median is OpenCV's.

#include "bench_cpu.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace cpubench {

using namespace bench::cpu;

// ------------------------------------------------------------------------------------------------
// OpenCV on the CPU
// ------------------------------------------------------------------------------------------------
struct OpenCVCpu {
    size_t n;
    // The NV12 -> RGB of each region only (cvtColorTwoPlane), instead of the whole frame (cvtColor)
    bool cropNv12;
    cv::Mat nv12; // the frame of the workload, not copied
    // Its luma and its interleaved chroma, as the two planes cvtColorTwoPlane takes (views, not copied)
    cv::Mat luma, chroma;
    cv::Mat rgb8;
    // The regions of every iteration, n per iteration (rects[t * n + i])
    std::vector<cv::Rect> rects;
    // One float3 crop of the largest region size per region; a region of another size uses its top-left corner
    std::vector<cv::Mat> cropped, resized;
    cv::Mat tensor;

    OpenCVCpu(const Workload& w, size_t batch, bool cropNv12_)
        : n(batch),
          cropNv12(cropNv12_),
          nv12(FRAME_H * 3 / 2, FRAME_W, CV_8UC1, const_cast<uint8_t*>(w.frame), w.pitch),
          luma(nv12.rowRange(0, FRAME_H)),
          chroma(FRAME_H / 2, FRAME_W / 2, CV_8UC2,
                 const_cast<uint8_t*>(w.frame) + static_cast<size_t>(FRAME_H) * w.pitch, w.pitch) {
        rgb8.create(FRAME_H, FRAME_W, CV_8UC3);
        for (size_t t = 0; t < w.regions->iterations(); ++t) {
            for (size_t i = 0; i < n; ++i) {
                const Region& r = w.regions->at(t, i);
                rects.emplace_back(r.x, r.y, r.width, r.height);
            }
        }
        for (size_t i = 0; i < n; ++i) {
            cropped.emplace_back(MAX_REGION_H, MAX_REGION_W, CV_32FC3);
            resized.emplace_back(RES_H, RES_W, CV_32FC3);
        }
        tensor.create(static_cast<int>(n) * OUT_H, OUT_W, CV_32FC3);
    }

    // On the regions of iteration t
    void run(size_t t) {
        if (cropNv12) {
            // Each region's Y view and UV view into the region's view of the 8 bit RGB frame: the frame is written
            // only where the regions of the iteration land, and the rest of the pipeline reads only those pixels
            for (size_t i = 0; i < n; ++i) {
                const cv::Rect& r = rects[t * n + i];
                cv::Mat dst = rgb8(r);
                cv::cvtColorTwoPlane(luma(r), chroma(cv::Rect(r.x / 2, r.y / 2, r.width / 2, r.height / 2)), dst,
                                     cv::COLOR_YUV2RGB_NV12);
            }
        } else {
            cv::cvtColor(nv12, rgb8, cv::COLOR_YUV2RGB_NV12);
        }
        for (size_t i = 0; i < n; ++i) {
            const cv::Rect& r = rects[t * n + i];
            cv::Mat crop = cropped[i](cv::Rect(0, 0, r.width, r.height));
            rgb8(r).convertTo(crop, CV_32F);
            cv::resize(crop, resized[i], cv::Size(RES_W, RES_H), 0., 0., cv::INTER_LINEAR);
            cv::Mat plane = tensor.rowRange(static_cast<int>(i) * OUT_H, static_cast<int>(i + 1) * OUT_H);
            cv::copyMakeBorder(resized[i], plane, BORDER_TOP, BORDER_BOTTOM, 0, 0, cv::BORDER_CONSTANT,
                               cv::Scalar(0., 0., 0.));
        }
        cv::multiply(tensor, cv::Scalar(MUL[0], MUL[1], MUL[2]), tensor);
        cv::subtract(tensor, cv::Scalar(SUB[0], SUB[1], SUB[2]), tensor);
        cv::divide(tensor, cv::Scalar(DIV[0], DIV[1], DIV[2]), tensor);
    }

    std::vector<float> output() const {
        const float* data = reinterpret_cast<const float*>(tensor.data);
        return std::vector<float>(data, data + n * OUT_W * OUT_H * 3);
    }

    // Before a validated run: the 8 bit frame of the crop form keeps the regions of earlier runs, so it is filled with
    // 0 (a pixel the regions of the validated iteration do not convert, if the pipeline read it, would be 0)
    void poison() {
        if (cropNv12) {
            rgb8.setTo(cv::Scalar::all(0));
        }
    }
};

// The errors on the regions of one validated iteration
struct Check {
    size_t batch, iteration;
    float fklVsReference, opencvVsReference, opencvVsFkl, fklMultithreadVsFkl, opencvThreadsVsOneThread;
    // The crop form: against the OpenCV-convention reference, against the full-frame form, and on `threads` threads
    // against one
    float cropVsReference, cropVsFrame, cropThreadsVsOneThread;
    // What the program requires (withinTolerance): FKL against its reference, FKL's pool against FKL, OpenCV on
    // `threads` threads against OpenCV on one (both forms), and the crop form against the full-frame form
    bool passed() const {
        return withinTolerance(fklVsReference) && withinTolerance(fklMultithreadVsFkl) &&
               withinTolerance(opencvThreadsVsOneThread) && withinTolerance(cropVsFrame) &&
               withinTolerance(cropThreadsVsOneThread);
    }
};

template <size_t N>
void runBatch(const Workload& w, std::vector<Row>& singleThread, std::vector<Row>& multiThread,
              std::vector<Row>& cropSingleThread, std::vector<Row>& cropMultiThread, std::vector<Check>& checks,
              int threads) {
    OpenCVCpu opencv(w, N, false);
    OpenCVCpu opencvCrop(w, N, true);
    const auto fkl = makeFklCpu(N, false, w);
    const auto fklMT = makeFklCpu(N, true, w);
    // Before the timing, on the regions of the first, every 50th and the last timed iteration: OpenCV on one thread
    // and on `threads` threads, FKL on one thread and on its pool
    Check worst{N, 0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (const size_t k : validatedIterations(ITERS)) {
        const size_t t = WARMUP_ITERS + k;
        cv::setNumThreads(threads);
        opencv.run(t);
        opencvCrop.poison();
        opencvCrop.run(t);
        const auto ocvThreadsOut = opencv.output();
        const auto cropThreadsOut = opencvCrop.output();
        cv::setNumThreads(1);
        opencv.run(t);
        opencvCrop.poison();
        opencvCrop.run(t);
        fkl->run(t);
        fklMT->run(t);
        const auto ocvOut = opencv.output();
        const auto cropOut = opencvCrop.output();
        const auto fklOut = fkl->output();
        const auto ocvReference = referenceOutput(w, N, t, Conventions::OpenCV);
        const Check c{N, t, maxAbsError(fklOut, referenceOutput(w, N, t, Conventions::Fkl)),
                      maxAbsError(ocvOut, ocvReference), maxAbsError(ocvOut, fklOut),
                      maxAbsError(fklMT->output(), fklOut), maxAbsError(ocvThreadsOut, ocvOut),
                      maxAbsError(cropOut, ocvReference), maxAbsError(cropOut, ocvOut),
                      maxAbsError(cropThreadsOut, cropOut)};
        checks.push_back(c);
        worst = Check{N, t, worstError(c.fklVsReference, worst.fklVsReference),
                      worstError(c.opencvVsReference, worst.opencvVsReference), worstError(c.opencvVsFkl, worst.opencvVsFkl),
                      worstError(c.fklMultithreadVsFkl, worst.fklMultithreadVsFkl),
                      worstError(c.opencvThreadsVsOneThread, worst.opencvThreadsVsOneThread),
                      worstError(c.cropVsReference, worst.cropVsReference),
                      worstError(c.cropVsFrame, worst.cropVsFrame),
                      worstError(c.cropThreadsVsOneThread, worst.cropThreadsVsOneThread)};
    }
    const Check& c = worst;

    // Per thread count: the full-frame form's comparison, then the crop form's, each with its own FKL loop
    singleThread.push_back(timeTwo(N, [&](size_t t) { opencv.run(t); }, [&](size_t t) { fkl->run(t); }));
    cropSingleThread.push_back(timeTwo(N, [&](size_t t) { opencvCrop.run(t); }, [&](size_t t) { fkl->run(t); }));
    cv::setNumThreads(threads);
    multiThread.push_back(timeTwo(N, [&](size_t t) { opencv.run(t); }, [&](size_t t) { fklMT->run(t); }));
    cropMultiThread.push_back(timeTwo(N, [&](size_t t) { opencvCrop.run(t); }, [&](size_t t) { fklMT->run(t); }));
    cv::setNumThreads(1);
    const Row& s = singleThread.back();
    const Row& m = multiThread.back();
    const Row& cs = cropSingleThread.back();
    const Row& cm = cropMultiThread.back();
    std::printf("N %3zu | 1 thread: OpenCV %8.1f us, crop NV12 %8.1f us, FKL CPU %8.1f / %8.1f us, mean speedup "
                "%6.2fx / %6.2fx | %d threads: OpenCV %8.1f us, crop NV12 %8.1f us, FKL CPU_MULTITHREAD %8.1f / "
                "%8.1f us, mean speedup %6.2fx / %6.2fx | max err: FKL vs reference %.2e, OpenCV vs OpenCV-convention "
                "reference %.2e (crop NV12 %.2e, vs OpenCV %.2e), OpenCV vs FKL %.2e, FKL CPU_MULTITHREAD vs FKL CPU "
                "%.2e, "
                "OpenCV %d threads vs 1 thread %.2e (crop NV12 %.2e)\n",
                N, s.baseline.median, cs.baseline.median, s.fkl.median, cs.fkl.median, s.meanSpeedup, cs.meanSpeedup,
                threads, m.baseline.median, cm.baseline.median, m.fkl.median, cm.fkl.median, m.meanSpeedup,
                cm.meanSpeedup, c.fklVsReference, c.opencvVsReference, c.cropVsReference, c.cropVsFrame, c.opencvVsFkl,
                c.fklMultithreadVsFkl, threads, c.opencvThreadsVsOneThread, c.cropThreadsVsOneThread);
}

} // namespace cpubench

int main(int argc, char** argv) {
    using namespace cpubench;
    const std::filesystem::path outDir = argc > 1 ? argv[1] : "results";
    std::filesystem::create_directories(outDir);
    // OpenCV runs on as many threads as FKL's thread pool has workers (std::thread::hardware_concurrency()), which can
    // differ from cv::getNumberOfCPUs() under an affinity mask or a CPU quota
    const int threads = poolThreads();

    Workload w;
    initWorkload(w);
    // How long each pool's workers stay busy after a parallel run with little work (their spin and wake-up), which the
    // idle gap must outlast
    // workerBusyUs divides by the pool's workers other than the caller: OpenCV's calling thread takes part in its
    // parallel_for_ (threads - 1 workers), FKL's waits while every one of its pool's threads works
    const int opencvWorkers = threads - 1;
    const int fklWorkers = poolThreads();
    cv::setNumThreads(threads);
    const double opencvBusyUs = workerBusyUs([&]() {
        cv::parallel_for_(cv::Range(0, threads), [](const cv::Range&) {}, threads);
    }, opencvWorkers);
    cv::setNumThreads(1);
    const auto spinProbe = makeFklCpu(BATCHES.front(), true, w);
    const double fklBusyUs = workerBusyUs([&]() { spinProbe->run(WARMUP_ITERS); }, fklWorkers);
    std::printf("Workers busy after a parallel run: OpenCV %.1f us, FKL %.1f us\n", opencvBusyUs, fklBusyUs);

    std::vector<Row> singleThread, multiThread, cropSingleThread, cropMultiThread;
    std::vector<Check> checks;
    forEachBatch([&]<size_t N>() {
        runBatch<N>(w, singleThread, multiThread, cropSingleThread, cropMultiThread, checks, threads);
    });

    writeCSV(outDir / "benchmark_cpu_opencv_1thread_vs_fkl_bvf_CPU.csv", "OpenCV 1 thread", singleThread, "FKL CPU");
    writeCSV(outDir / ("benchmark_cpu_opencv_" + std::to_string(threads) +
                       "threads_vs_fkl_bvf_CPU_MULTITHREAD.csv"),
             "OpenCV " + std::to_string(threads) + " threads", multiThread, "FKL CPU_MULTITHREAD");
    writeCSV(outDir / "benchmark_cpu_opencv_crop_nv12_1thread_vs_fkl_bvf_CPU.csv", "OpenCV crop NV12 1 thread",
             cropSingleThread, "FKL CPU");
    writeCSV(outDir / ("benchmark_cpu_opencv_crop_nv12_" + std::to_string(threads) +
                       "threads_vs_fkl_bvf_CPU_MULTITHREAD.csv"),
             "OpenCV crop NV12 " + std::to_string(threads) + " threads", cropMultiThread, "FKL CPU_MULTITHREAD");
    std::ofstream v(outDir / "cpu_validation.csv");
    v << "Batch size, Iteration, FKL MaxAbsError vs FKL-convention reference, OpenCV MaxAbsError vs OpenCV-convention "
         "reference, OpenCV vs FKL MaxAbsDifference, FKL CPU_MULTITHREAD vs FKL CPU MaxAbsDifference, OpenCV "
      << threads << " threads vs OpenCV 1 thread MaxAbsDifference, OpenCV crop NV12 MaxAbsError vs OpenCV-convention "
         "reference, OpenCV crop NV12 vs OpenCV MaxAbsDifference, OpenCV crop NV12 "
      << threads << " threads vs OpenCV crop NV12 1 thread MaxAbsDifference, Within tolerance" << std::endl;
    bool passed = true;
    for (const auto& c : checks) {
        v << c.batch << ", " << c.iteration << ", " << c.fklVsReference << ", " << c.opencvVsReference << ", "
          << c.opencvVsFkl << ", " << c.fklMultithreadVsFkl << ", " << c.opencvThreadsVsOneThread << ", "
          << c.cropVsReference << ", " << c.cropVsFrame << ", " << c.cropThreadsVsOneThread << ", "
          << (c.passed() ? "yes" : "no") << std::endl;
        passed &= c.passed();
    }
    writeRegionStats((outDir / "regions.csv").string(), *w.regions, WARMUP_ITERS, ITERS);
    const std::string opencvBuild = cv::getBuildInformation();
    std::ofstream(outDir / "cpu_opencv_build_information.txt") << opencvBuild;
    std::ofstream p(outDir / "cpu_platform.txt");
    p << "OpenCV: " << CV_VERSION << ", built with " << bench::buildInfoValue(opencvBuild, "C++ Compiler:")
      << " (baseline " << bench::buildInfoValue(opencvBuild, "Baseline:") << ", runtime dispatch "
      << bench::buildInfoValue(opencvBuild, "Dispatched code generation:") << "; features of this CPU: "
      << cv::getCPUFeaturesLine() << "), parallel framework " << bench::buildInfoValue(opencvBuild, "Parallel framework:")
      << ", " << cv::getNumberOfCPUs() << " CPUs available to OpenCV; full configuration in "
      << "cpu_opencv_build_information.txt\n"
      << "Thread pools, measured before the benchmark: each worker stays busy " << opencvBusyUs << " us per OpenCV "
      << "parallel_for_ with no work, and " << fklBusyUs << " us per FKL CPU_MULTITHREAD run on 1 region (spin and "
      << "wake-up), which the " << std::chrono::milliseconds(IDLE_GAP).count() << " ms idle gap must outlast\n";
    p << "OpenCV's two forms of the NV12 -> RGB: cvtColor(COLOR_YUV2RGB_NV12) of the whole frame (opencv), or "
         "cvtColorTwoPlane(COLOR_YUV2RGB_NV12) of each region's Y view and UV view (CV_8UC2) into the region's view of "
         "the 8 bit RGB frame (opencv crop NV12; the frame is written only where the regions of the iteration land, "
         "and the rest of the pipeline reads only those pixels; before every validated run the frame is filled with "
         "0). Each form has its own comparison with FKL per thread count (benchmark_cpu_opencv_*.csv and "
         "benchmark_cpu_opencv_crop_nv12_*.csv); per batch size and thread count, the one with the lower median is "
         "OpenCV's.\n";
    writeCpuPlatform(p, threads,
                     {"cpu_validation.csv",
                      "OpenCV on its thread count against OpenCV on one thread, for both forms, and the crop form "
                      "against the full-frame form",
                      "OpenCV's cv::Rects are built before the loops, and its per-region calls are timed"});
    std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
    if (!passed) {
        std::cerr << "ERROR: validation failed (FKL against its reference, FKL's pool against FKL, OpenCV on "
                  << threads << " threads against OpenCV on one, or OpenCV's crop form against its full-frame form "
                  << "beyond " << bench::validation::TOLERANCE
                  << "): see cpu_validation.csv" << std::endl;
        return 1;
    }
    return 0;
}
