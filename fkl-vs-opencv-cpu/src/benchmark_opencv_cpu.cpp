// FKL vs OpenCV on the CPU, Figure 1 pipeline: OpenCV's CPU functions vs FKL's CPU backends (BVF).
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// One 3840x2160 NV12 frame in host memory and N regions, new every iteration, with the sizes and positions of
// common/bench_geometry.h (20x10, 32x16 and 48x24 by turns), bilinear resize to 64x32, black rows added on top and
// bottom up to a square, then (x * mul - sub) / div written as float3 into an N-plane tensor. Iteration t of every loop
// runs both libraries on the regions of iteration t, generated before the loops; each converts them into its own
// arguments before the loops too, and what it does with them in the iteration (FKL's build() calls and fusion,
// OpenCV's per-region views and calls) is timed.
//
// The color conversion is BT.601 limited range in both, because OpenCV's CPU cvtColor only implements that one for
// NV12; every benchmark of this repository uses it (common/bench_geometry.h).
//
//   opencv: cvtColor(COLOR_YUV2RGB_NV12) on the full frame (8 bit RGB), then per region convertTo(CV_32F) (into a
//           buffer of the largest region size, through a view of the region's size), resize and copyMakeBorder, then
//           multiply, subtract and divide on the whole tensor. OpenCV uses its own SIMD dispatch
//           (up to AVX2 here) and its thread pool.
//   fkl:    one executeOperations<TransformDPP<ParArch::CPU>> call with the whole chain (BVF). FKL's CPU backend runs
//           on one thread, compiled with -march=native.
//   fkl mt: the same call with TransformDPP<ParArch::CPU_MULTITHREAD>: the stream enqueues the execution in FKL's shared
//           thread pool (one worker per hardware thread), in chunks of one output row of one crop, and sync() waits.
//
// Two comparisons, each with the same number of threads on both sides: OpenCV with 1 thread against fkl, and OpenCV
// with as many threads as FKL's thread pool has workers (one per hardware thread) against fkl mt.
//
// Both libraries' thread pools spin for a while after finishing before their workers go to sleep, so every timed run
// is preceded by an idle gap that lets them fall asleep: neither steals cores from the other, and both start from
// sleeping workers, as when a pipeline runs once per video frame.

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/execution_model/cpu_multithread.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "bench_geometry.h"
#include "bench_platform.h"
#include "bench_validation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace cpubench {

// Frame, region, resize, border and output sizes, batch sizes, seeds and region positions, shared by every benchmark
using namespace bench::geometry;
using bench::validation::validatedIterations;
// BENCH_WARMUP_ITERATIONS and BENCH_ITERATIONS come from CMake: 100 and 1000 for the paper's results, 0 and 1 in the CI
constexpr int WARMUP_ITERS = BENCH_WARMUP_ITERATIONS;
constexpr int ITERS = BENCH_ITERATIONS;
static_assert(WARMUP_ITERS >= 0 && ITERS >= 1, "at least one timed iteration");
constexpr auto IDLE_GAP = std::chrono::milliseconds(1);

constexpr float MUL[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
constexpr float SUB[3] = {0.485f, 0.456f, 0.406f};
constexpr float DIV[3] = {0.229f, 0.224f, 0.225f};

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Limited, fk::ColorPrimitives::bt601>;

struct Workload {
    std::unique_ptr<NV12Image> frame;
    cv::Mat nv12; // the same memory as frame
    // The regions of every iteration (warm-up and timed), generated before the loops
    std::unique_ptr<RegionSets> regions;
};

void initWorkload(Workload& w) {
    w.frame = std::make_unique<NV12Image>(FRAME_W, FRAME_H, fk::MemType::Host);
    auto data = w.frame->getData();
    generateFrame([&](int x, int y, uint8_t value) { data.at(x, y) = value; });
    const auto raw = data.ptr();
    w.nv12 = cv::Mat(FRAME_H * 3 / 2, FRAME_W, CV_8UC1, raw.data, raw.dims.pitch);
    w.regions = std::make_unique<RegionSets>(static_cast<size_t>(WARMUP_ITERS + ITERS));
}

// ------------------------------------------------------------------------------------------------
// OpenCV on the CPU
// ------------------------------------------------------------------------------------------------
struct OpenCVCpu {
    const Workload& w;
    size_t n;
    cv::Mat rgb8;
    // The regions of every iteration, n per iteration (rects[t * n + i])
    std::vector<cv::Rect> rects;
    // One float3 crop of the largest region size per region; a region of another size uses its top-left corner
    std::vector<cv::Mat> cropped, resized;
    cv::Mat tensor;

    OpenCVCpu(const Workload& workload, size_t batch) : w(workload), n(batch) {
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
        cv::cvtColor(w.nv12, rgb8, cv::COLOR_YUV2RGB_NV12);
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
};

// ------------------------------------------------------------------------------------------------
// FKL CPU backends, BVF: one executeOperations call, on one thread (ParArch::CPU) or on FKL's thread pool
// (ParArch::CPU_MULTITHREAD)
// ------------------------------------------------------------------------------------------------
template <size_t N, fk::ParArch PA = fk::ParArch::CPU>
struct FklCpu {
    const Workload& w;
    // The rectangles of every iteration
    std::vector<std::array<fk::Rect, N>> crops;
    fk::Tensor<fk::float3> out{OUT_W, OUT_H, N, 1, fk::MemType::Host};
    fk::Stream_<PA> stream;

    explicit FklCpu(const Workload& workload) : w(workload), crops(workload.regions->iterations()) {
        for (size_t t = 0; t < crops.size(); ++t) {
            for (size_t i = 0; i < N; ++i) {
                const Region& r = w.regions->at(t, i);
                crops[t][i] = fk::Rect(r.x, r.y, r.width, r.height);
            }
        }
    }

    // On the regions of iteration t
    void run(size_t t) {
        fk::executeOperations<fk::TransformDPP<PA>>(stream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*w.frame),
            YUVToRGB::build(),
            fk::Crop<>::build(crops[t]),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, fk::float3{0.f, 0.f, 0.f}),
            fk::Mul<fk::float3>::build(fk::float3{MUL[0], MUL[1], MUL[2]}),
            fk::Sub<fk::float3>::build(fk::float3{SUB[0], SUB[1], SUB[2]}),
            fk::Div<fk::float3>::build(fk::float3{DIV[0], DIV[1], DIV[2]}),
            fk::TensorWrite<fk::float3>::build(out.ptr()));
        stream.sync();
    }

    std::vector<float> output() const {
        const float* data = reinterpret_cast<const float*>(out.ptr().data);
        return std::vector<float>(data, data + N * OUT_W * OUT_H * 3);
    }
};

// ------------------------------------------------------------------------------------------------
// Independent host references: the same chain in plain C++, with FKL's conversion matrix and either FKL's conventions
// (src = dst * scale, float RGB) or OpenCV's (Y clamped to at least 16 before the conversion, 8 bit RGB, pixel centers
// src = (dst + 0.5) * scale - 0.5)
// ------------------------------------------------------------------------------------------------
std::vector<float> hostReference(const Workload& w, size_t n, size_t t, bool openCVConventions) {
    std::vector<float> out(n * OUT_W * OUT_H * 3);
    for (size_t i = 0; i < n; ++i) {
        const Region& r = w.regions->at(t, i);
        const int cw = r.width, ch = r.height;
        const float sx = static_cast<float>(cw) / RES_W;
        const float sy = static_cast<float>(ch) / RES_H;
        std::vector<float> cropPixels(static_cast<size_t>(cw) * ch * 3);
        const auto crop = [&](int y, int x, int c) -> float& {
            return cropPixels[(static_cast<size_t>(y) * cw + x) * 3 + c];
        };
        for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
                fk::uchar3 yuv = w.frame->readAt(fk::Point{static_cast<int>(r.x) + x, static_cast<int>(r.y) + y, 0});
                if (openCVConventions) {
                    // OpenCV's NV12 -> RGB clamps Y below 16 (limited range black) to 16 before converting
                    yuv.x = std::max<uchar>(yuv.x, 16);
                }
                const fk::float3 rgb = YUVToRGB::exec(yuv);
                const float c3[3] = {rgb.x, rgb.y, rgb.z};
                for (int c = 0; c < 3; ++c) {
                    crop(y, x, c) = openCVConventions ? std::floor(std::clamp(c3[c], 0.f, 255.f) + 0.5f) : c3[c];
                }
            }
        }
        for (int y = 0; y < OUT_H; ++y) {
            for (int x = 0; x < OUT_W; ++x) {
                const int ry = y - BORDER_TOP;
                float v[3] = {0.f, 0.f, 0.f};
                if (ry >= 0 && ry < RES_H) {
                    float fx = openCVConventions ? (x + 0.5f) * sx - 0.5f : x * sx;
                    float fy = openCVConventions ? (ry + 0.5f) * sy - 0.5f : ry * sy;
                    fx = std::clamp(fx, 0.f, static_cast<float>(cw - 1));
                    fy = std::clamp(fy, 0.f, static_cast<float>(ch - 1));
                    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
                    const int x1 = std::min(x0 + 1, cw - 1), y1 = std::min(y0 + 1, ch - 1);
                    const float ax = fx - x0, ay = fy - y0;
                    for (int c = 0; c < 3; ++c) {
                        v[c] = (crop(y0, x0, c) * (1.f - ax) + crop(y0, x1, c) * ax) * (1.f - ay) +
                               (crop(y1, x0, c) * (1.f - ax) + crop(y1, x1, c) * ax) * ay;
                    }
                }
                float* dst = &out[((i * OUT_H + y) * OUT_W + x) * 3];
                for (int c = 0; c < 3; ++c) {
                    dst[c] = ((v[c] * MUL[c]) - SUB[c]) / DIV[c];
                }
            }
        }
    }
    return out;
}

float maxAbsError(const std::vector<float>& a, const std::vector<float>& b) {
    float e = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::abs(a[i] - b[i]);
        e = std::isfinite(d) ? std::max(e, d) : INFINITY;
    }
    return e;
}

// ------------------------------------------------------------------------------------------------
// Two-execution timing in FKL's format: in every iteration OpenCV and then FKL on the regions of that iteration, each
// timed with steady_clock after an idle gap
// ------------------------------------------------------------------------------------------------
struct Stats {
    double mean{0.}, variance{0.}, max{0.}, min{0.}, median{0.}, p10{0.}, p90{0.};
};

double percentile(const std::vector<double>& sorted, double p) {
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

Stats stats(std::vector<double> ms) {
    Stats s;
    double sum = 0.;
    for (const double v : ms) sum += v;
    s.mean = sum / ms.size();
    double sq = 0.;
    for (const double v : ms) sq += (v - s.mean) * (v - s.mean);
    s.variance = ms.size() > 1 ? sq / (ms.size() - 1) : 0.;
    std::sort(ms.begin(), ms.end());
    s.min = ms.front();
    s.max = ms.back();
    s.median = percentile(ms, 0.5);
    s.p10 = percentile(ms, 0.1);
    s.p90 = percentile(ms, 0.9);
    return s;
}

struct Row {
    size_t batch;
    Stats opencv, fkl;
    double meanSpeedup;
};

Row timeTwo(size_t batch, const std::function<void(size_t)>& opencv, const std::function<void(size_t)>& fkl) {
    using Clock = std::chrono::steady_clock;
    const auto time = [](const std::function<void(size_t)>& f, size_t t) {
        std::this_thread::sleep_for(IDLE_GAP);
        const auto t0 = Clock::now();
        f(t);
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        time(opencv, static_cast<size_t>(i));
        time(fkl, static_cast<size_t>(i));
    }
    std::vector<double> a(ITERS), b(ITERS);
    double speedup = 0.;
    for (int i = 0; i < ITERS; ++i) {
        a[i] = time(opencv, static_cast<size_t>(WARMUP_ITERS + i));
        b[i] = time(fkl, static_cast<size_t>(WARMUP_ITERS + i));
        speedup += a[i] / b[i];
    }
    return {batch, stats(a), stats(b), speedup / ITERS};
}

void writeCSV(const std::filesystem::path& path, const std::string& opencvLabel, const std::vector<Row>& rows,
              const std::string& fklLabel = "FKL CPU") {
    std::ofstream f(path);
    f << "Batch size";
    for (const std::string label : {opencvLabel, fklLabel}) {
        f << ", " << label << " MeanTime, " << label << " TimeVariance, " << label << " MaxTime, " << label << " MinTime";
    }
    f << ", Mean Speedup";
    for (const std::string label : {opencvLabel, fklLabel}) {
        f << ", " << label << " MedianTime, " << label << " P10Time, " << label << " P90Time";
    }
    f << std::endl;
    for (const auto& r : rows) {
        f << r.batch;
        for (const Stats* s : {&r.opencv, &r.fkl}) {
            f << ", " << s->mean << ", " << s->variance << ", " << s->max << ", " << s->min;
        }
        f << ", " << r.meanSpeedup;
        for (const Stats* s : {&r.opencv, &r.fkl}) {
            f << ", " << s->median << ", " << s->p10 << ", " << s->p90;
        }
        f << std::endl;
    }
}

// The errors on the regions of one validated iteration
struct Check {
    size_t batch, iteration;
    float fklVsReference, opencvVsReference, opencvVsFkl, fklMultithreadVsFkl;
};

template <size_t N>
void runBatch(Workload& w, std::vector<Row>& singleThread, std::vector<Row>& multiThread, std::vector<Check>& checks,
              int threads) {
    OpenCVCpu opencv(w, N);
    FklCpu<N> fkl(w);
    FklCpu<N, fk::ParArch::CPU_MULTITHREAD> fklMT(w);
    cv::setNumThreads(1);
    // Before the timing, on the regions of the first, every 50th and the last timed iteration
    Check worst{N, 0, 0.f, 0.f, 0.f, 0.f};
    for (const size_t k : validatedIterations(ITERS)) {
        const size_t t = WARMUP_ITERS + k;
        opencv.run(t);
        fkl.run(t);
        fklMT.run(t);
        const auto ocvOut = opencv.output();
        const auto fklOut = fkl.output();
        const Check c{N, t, maxAbsError(fklOut, hostReference(w, N, t, false)),
                      maxAbsError(ocvOut, hostReference(w, N, t, true)), maxAbsError(ocvOut, fklOut),
                      maxAbsError(fklMT.output(), fklOut)};
        checks.push_back(c);
        worst = Check{N, t, std::max(worst.fklVsReference, c.fklVsReference),
                      std::max(worst.opencvVsReference, c.opencvVsReference), std::max(worst.opencvVsFkl, c.opencvVsFkl),
                      std::max(worst.fklMultithreadVsFkl, c.fklMultithreadVsFkl)};
    }
    const Check& c = worst;

    singleThread.push_back(timeTwo(N, [&](size_t t) { opencv.run(t); }, [&](size_t t) { fkl.run(t); }));
    cv::setNumThreads(threads);
    multiThread.push_back(timeTwo(N, [&](size_t t) { opencv.run(t); }, [&](size_t t) { fklMT.run(t); }));
    cv::setNumThreads(1);
    const Row& s = singleThread.back();
    const Row& m = multiThread.back();
    std::printf("N %3zu | OpenCV 1 thread %8.1f us, FKL CPU %8.1f us, mean speedup %6.2fx | OpenCV %d threads %8.1f us, "
                "FKL CPU_MULTITHREAD %8.1f us, mean speedup %6.2fx | max err: FKL vs reference %.2e, OpenCV vs "
                "OpenCV-convention reference %.2e, OpenCV vs FKL %.2e, FKL CPU_MULTITHREAD vs FKL CPU %.2e\n",
                N, s.opencv.median * 1000., s.fkl.median * 1000., s.meanSpeedup, threads,
                m.opencv.median * 1000., m.fkl.median * 1000., m.meanSpeedup, c.fklVsReference, c.opencvVsReference,
                c.opencvVsFkl, c.fklMultithreadVsFkl);
}

} // namespace cpubench

int main(int argc, char** argv) {
    using namespace cpubench;
    const std::filesystem::path outDir = argc > 1 ? argv[1] : "results";
    std::filesystem::create_directories(outDir);
    // OpenCV runs on as many threads as FKL's thread pool has workers (std::thread::hardware_concurrency()), which can
    // differ from cv::getNumberOfCPUs() under an affinity mask or a CPU quota
    const int threads = static_cast<int>(fk::cpuThreadPool().size());

    Workload w;
    initWorkload(w);
    std::vector<Row> singleThread, multiThread;
    std::vector<Check> checks;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (runBatch<BATCHES[Is]>(w, singleThread, multiThread, checks, threads), ...);
    }(std::make_index_sequence<BATCHES.size()>{});

    writeCSV(outDir / "benchmark_cpu_opencv_1thread_vs_fkl_bvf_CPU.csv", "OpenCV 1 thread", singleThread);
    writeCSV(outDir / ("benchmark_cpu_opencv_" + std::to_string(threads) +
                       "threads_vs_fkl_bvf_CPU_MULTITHREAD.csv"),
             "OpenCV " + std::to_string(threads) + " threads", multiThread, "FKL CPU_MULTITHREAD");
    std::ofstream v(outDir / "cpu_validation.csv");
    v << "Batch size, Iteration, FKL MaxAbsError vs FKL-convention reference, OpenCV MaxAbsError vs OpenCV-convention "
         "reference, OpenCV vs FKL MaxAbsDifference, FKL CPU_MULTITHREAD vs FKL CPU MaxAbsDifference" << std::endl;
    for (const auto& c : checks) {
        v << c.batch << ", " << c.iteration << ", " << c.fklVsReference << ", " << c.opencvVsReference << ", "
          << c.opencvVsFkl << ", " << c.fklMultithreadVsFkl << std::endl;
    }
    writeRegionStats((outDir / "regions.csv").string(), *w.regions, WARMUP_ITERS, ITERS);
    const std::string opencvBuild = cv::getBuildInformation();
    std::ofstream(outDir / "cpu_opencv_build_information.txt") << opencvBuild;
    std::ofstream p(outDir / "cpu_platform.txt");
    p << "CPU: " << bench::cpuName() << ", " << std::thread::hardware_concurrency() << " hardware threads, "
      << cv::getNumberOfCPUs() << " available to OpenCV; the multithreaded comparison uses " << threads
      << " threads on both sides\n"
#ifdef _WIN32
      << "SIMD of the CPU: see the CPU model\n"
#else
      << "SIMD of the CPU: "
      << bench::shell("grep -o -w -E 'avx|avx2|fma|avx512f|avx512bw|avx512vl|asimd|sve|sve2' /proc/cpuinfo | sort -u | paste -sd' '")
      << "\n"
#endif
      << "SIMD the FKL code is compiled for: " << bench::compiledSimd() << "\n"
      << "FKL: commit " << BENCH_FKL_COMMIT << ", CPU backend (single thread) and CPU_MULTITHREAD backend (thread pool of "
      << fk::cpuThreadPool().size() << " workers, thread-pool " << BENCH_THREAD_POOL_COMMIT
      << ", workers spin 100 us before sleeping, one chunk per output row of each crop), compiled with "
      << BENCH_COMPILER << ", flags " << BENCH_CXX_FLAGS << "\n"
      << "OpenCV: " << CV_VERSION << ", built with " << bench::buildInfoValue(opencvBuild, "C++ Compiler:")
      << " (baseline " << bench::buildInfoValue(opencvBuild, "Baseline:") << ", runtime dispatch "
      << bench::buildInfoValue(opencvBuild, "Dispatched code generation:") << "; features of this CPU: "
      << cv::getCPUFeaturesLine() << "), parallel framework " << bench::buildInfoValue(opencvBuild, "Parallel framework:")
      << "; full configuration in cpu_opencv_build_information.txt\n"
      << "OS: " << bench::osName() << "\n"
      << "Workload: " << describe() << "\n"
      << "Pipeline: as the GPU benchmarks, with BT.601 limited range for NV12 -> RGB (the only NV12 conversion of "
         "OpenCV's CPU cvtColor)\n"
      << "Regions: iteration t (the " << WARMUP_ITERS << " warm-up iterations are t in [0, " << WARMUP_ITERS << "), the "
      << ITERS << " timed ones t in [" << WARMUP_ITERS << ", " << WARMUP_ITERS + ITERS
      << ")) runs both libraries on the regions of iteration t, generated before "
      << "the loops and converted into each library's arguments before them too (statistics of the timed ones in "
      << "regions.csv); FKL's build() calls and fusion and OpenCV's per-region views and calls are timed.\n"
      << "Timing: steady_clock around each pipeline, " << WARMUP_ITERS << " warm-up and " << ITERS
      << " timed iterations, OpenCV and FKL one after the other in each, each run preceded by a "
      << std::chrono::milliseconds(IDLE_GAP).count()
      << " ms idle gap so that both thread pools are asleep; speedup computed per iteration and averaged.\n"
      << "Validation: before the timing, on the regions of the first, every " << bench::validation::VALIDATION_STRIDE
      << "th and the last timed iteration (cpu_validation.csv, one row per iteration).\n";
    std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
    return 0;
}
