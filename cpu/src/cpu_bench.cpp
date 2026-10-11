// Exploratory CPU benchmark of the Figure 1 pipeline: OpenCV on the CPU vs FKL's CPU backend (BVF).
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// One 3840x2160 NV12 frame in host memory, N crops of 20x10, bilinear resize to 64x32, 16 black rows on top and
// bottom, then (x * mul - sub) / div written as float3 into an N-plane tensor.
//
// The color conversion is BT.601 limited range in both, because OpenCV's CPU cvtColor only implements that one for
// NV12 (the GPU benchmarks use BT.2020 full range).
//
//   opencv: cvtColor(COLOR_YUV2RGB_NV12) on the full frame (8 bit RGB), then per crop convertTo(CV_32F), resize and
//           copyMakeBorder, then multiply, subtract and divide on the whole tensor. OpenCV uses its own SIMD dispatch
//           (up to AVX2 here) and its thread pool; it is timed with 1 thread and with all the hardware threads.
//   fkl:    one executeOperations<TransformDPP<ParArch::CPU>> call with the whole chain (BVF). FKL's CPU backend runs
//           on one thread, compiled with -march=native.
//   fkl mt: the same call with TransformDPP<ParArch::CPU_MULTITHREAD>: the stream enqueues the execution in FKL's shared
//           thread pool (one worker per hardware thread), in chunks of one output row of one crop, and sync() waits.
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

#include "bench_platform.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpubench {

constexpr int FRAME_W = 3840;
constexpr int FRAME_H = 2160;
constexpr int CROP_W = 20;
constexpr int CROP_H = 10;
constexpr int RES_W = 64;
constexpr int RES_H = 32;
constexpr int BORDER_TOP = 16;
constexpr int BORDER_BOTTOM = 16;
constexpr int OUT_W = RES_W;
constexpr int OUT_H = RES_H + BORDER_TOP + BORDER_BOTTOM;
constexpr int MAX_BATCH = 100;
constexpr std::array<size_t, 5> BATCHES{1, 4, 16, 50, 100};
// BENCH_WARMUP_ITERATIONS and BENCH_ITERATIONS come from CMake: 100 and 1000 for the paper's results, 0 and 1 in the CI
constexpr int WARMUP_ITERS = BENCH_WARMUP_ITERATIONS;
constexpr int ITERS = BENCH_ITERATIONS;
static_assert(WARMUP_ITERS >= 0 && ITERS >= 1, "at least one timed iteration");
constexpr auto IDLE_GAP = std::chrono::milliseconds(1);
constexpr unsigned FRAME_SEED = 1;
constexpr unsigned CROP_SEED = 42;

constexpr float MUL[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
constexpr float SUB[3] = {0.485f, 0.456f, 0.406f};
constexpr float DIV[3] = {0.229f, 0.224f, 0.225f};

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Limited, fk::ColorPrimitives::bt601>;

struct Workload {
    std::unique_ptr<NV12Image> frame;
    cv::Mat nv12; // the same memory as frame
    std::array<fk::Rect, MAX_BATCH> crops{};
};

void initWorkload(Workload& w) {
    w.frame = std::make_unique<NV12Image>(FRAME_W, FRAME_H, fk::MemType::Host);
    auto data = w.frame->getData();
    std::mt19937 gen(FRAME_SEED);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int y = 0; y < FRAME_H * 3 / 2; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            data.at(x, y) = static_cast<uchar>(byte(gen));
        }
    }
    const auto raw = data.ptr();
    w.nv12 = cv::Mat(FRAME_H * 3 / 2, FRAME_W, CV_8UC1, raw.data, raw.dims.pitch);
    std::mt19937 cropGen(CROP_SEED);
    std::uniform_int_distribution<int> distX(0, FRAME_W - CROP_W);
    std::uniform_int_distribution<int> distY(0, FRAME_H - CROP_H);
    for (auto& crop : w.crops) {
        const int x = distX(cropGen);
        const int y = distY(cropGen);
        crop = fk::Rect(x, y, CROP_W, CROP_H);
    }
}

// ------------------------------------------------------------------------------------------------
// OpenCV on the CPU
// ------------------------------------------------------------------------------------------------
struct OpenCVCpu {
    const Workload& w;
    size_t n;
    cv::Mat rgb8;
    std::vector<cv::Rect> rects;
    std::vector<cv::Mat> cropped, resized;
    cv::Mat tensor;

    OpenCVCpu(const Workload& workload, size_t batch) : w(workload), n(batch) {
        rgb8.create(FRAME_H, FRAME_W, CV_8UC3);
        for (size_t i = 0; i < n; ++i) {
            const fk::Rect& r = w.crops[i];
            rects.emplace_back(r.x, r.y, r.width, r.height);
            cropped.emplace_back(CROP_H, CROP_W, CV_32FC3);
            resized.emplace_back(RES_H, RES_W, CV_32FC3);
        }
        tensor.create(static_cast<int>(n) * OUT_H, OUT_W, CV_32FC3);
    }

    void run() {
        cv::cvtColor(w.nv12, rgb8, cv::COLOR_YUV2RGB_NV12);
        for (size_t i = 0; i < n; ++i) {
            rgb8(rects[i]).convertTo(cropped[i], CV_32F);
            cv::resize(cropped[i], resized[i], cv::Size(RES_W, RES_H), 0., 0., cv::INTER_LINEAR);
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
    std::array<fk::Rect, N> crops{};
    fk::Tensor<fk::float3> out{OUT_W, OUT_H, N, 1, fk::MemType::Host};
    fk::Stream_<PA> stream;

    explicit FklCpu(const Workload& workload) : w(workload) {
        std::copy_n(w.crops.begin(), N, crops.begin());
    }

    void run() {
        fk::executeOperations<fk::TransformDPP<PA>>(stream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*w.frame),
            YUVToRGB::build(),
            fk::Crop<>::build(crops),
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
std::vector<float> hostReference(const Workload& w, size_t n, bool openCVConventions) {
    std::vector<float> out(n * OUT_W * OUT_H * 3);
    const float sx = static_cast<float>(CROP_W) / RES_W;
    const float sy = static_cast<float>(CROP_H) / RES_H;
    for (size_t i = 0; i < n; ++i) {
        const fk::Rect& r = w.crops[i];
        float crop[CROP_H][CROP_W][3];
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                fk::uchar3 yuv = w.frame->readAt(fk::Point{static_cast<int>(r.x) + x, static_cast<int>(r.y) + y, 0});
                if (openCVConventions) {
                    // OpenCV's NV12 -> RGB clamps Y below 16 (limited range black) to 16 before converting
                    yuv.x = std::max<uchar>(yuv.x, 16);
                }
                const fk::float3 rgb = YUVToRGB::exec(yuv);
                const float c3[3] = {rgb.x, rgb.y, rgb.z};
                for (int c = 0; c < 3; ++c) {
                    crop[y][x][c] = openCVConventions ? std::floor(std::clamp(c3[c], 0.f, 255.f) + 0.5f) : c3[c];
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
                    fx = std::clamp(fx, 0.f, static_cast<float>(CROP_W - 1));
                    fy = std::clamp(fy, 0.f, static_cast<float>(CROP_H - 1));
                    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
                    const int x1 = std::min(x0 + 1, CROP_W - 1), y1 = std::min(y0 + 1, CROP_H - 1);
                    const float ax = fx - x0, ay = fy - y0;
                    for (int c = 0; c < 3; ++c) {
                        v[c] = (crop[y0][x0][c] * (1.f - ax) + crop[y0][x1][c] * ax) * (1.f - ay) +
                               (crop[y1][x0][c] * (1.f - ax) + crop[y1][x1][c] * ax) * ay;
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
// Two-execution timing in FKL's format: in every iteration OpenCV and then FKL, each timed with steady_clock after an
// idle gap
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

Row timeTwo(size_t batch, const std::function<void()>& opencv, const std::function<void()>& fkl) {
    using Clock = std::chrono::steady_clock;
    const auto time = [](const std::function<void()>& f) {
        std::this_thread::sleep_for(IDLE_GAP);
        const auto t0 = Clock::now();
        f();
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        time(opencv);
        time(fkl);
    }
    std::vector<double> a(ITERS), b(ITERS);
    double speedup = 0.;
    for (int i = 0; i < ITERS; ++i) {
        a[i] = time(opencv);
        b[i] = time(fkl);
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

struct Check {
    size_t batch;
    float fklVsReference, opencvVsReference, opencvVsFkl, fklMultithreadVsFkl;
};

template <size_t N>
void runBatch(Workload& w, std::vector<Row>& singleThread, std::vector<Row>& multiThread,
              std::vector<Row>& multiThreadFkl, std::vector<Check>& checks, int hardwareThreads) {
    OpenCVCpu opencv(w, N);
    FklCpu<N> fkl(w);
    FklCpu<N, fk::ParArch::CPU_MULTITHREAD> fklMT(w);
    cv::setNumThreads(1);
    opencv.run();
    fkl.run();
    fklMT.run();
    const auto ocvOut = opencv.output();
    const auto fklOut = fkl.output();
    const Check c{N, maxAbsError(fklOut, hostReference(w, N, false)), maxAbsError(ocvOut, hostReference(w, N, true)),
                  maxAbsError(ocvOut, fklOut), maxAbsError(fklMT.output(), fklOut)};
    checks.push_back(c);

    singleThread.push_back(timeTwo(N, [&]() { opencv.run(); }, [&]() { fkl.run(); }));
    cv::setNumThreads(hardwareThreads);
    multiThread.push_back(timeTwo(N, [&]() { opencv.run(); }, [&]() { fkl.run(); }));
    multiThreadFkl.push_back(timeTwo(N, [&]() { opencv.run(); }, [&]() { fklMT.run(); }));
    cv::setNumThreads(1);
    const Row& s = singleThread.back();
    const Row& m = multiThread.back();
    const Row& mt = multiThreadFkl.back();
    std::printf("N %3zu | OpenCV 1 thread %8.1f us, OpenCV %d threads %8.1f us, FKL CPU %8.1f us, FKL CPU_MULTITHREAD "
                "%8.1f us | mean speedup: FKL CPU vs OpenCV 1 thread %6.2fx, FKL CPU vs OpenCV %d threads %6.2fx, FKL "
                "CPU_MULTITHREAD vs OpenCV %d threads %6.2fx | max err: FKL vs reference %.2e, OpenCV vs "
                "OpenCV-convention reference %.2e, OpenCV vs FKL %.2e, FKL CPU_MULTITHREAD vs FKL CPU %.2e\n",
                N, s.opencv.median * 1000., hardwareThreads, m.opencv.median * 1000., s.fkl.median * 1000.,
                mt.fkl.median * 1000., s.meanSpeedup, hardwareThreads, m.meanSpeedup, hardwareThreads, mt.meanSpeedup,
                c.fklVsReference, c.opencvVsReference, c.opencvVsFkl, c.fklMultithreadVsFkl);
}

} // namespace cpubench

int main(int argc, char** argv) {
    using namespace cpubench;
    const std::filesystem::path outDir = argc > 1 ? argv[1] : "results/cpu";
    std::filesystem::create_directories(outDir);
    const int hardwareThreads = cv::getNumberOfCPUs();

    Workload w;
    initWorkload(w);
    std::vector<Row> singleThread, multiThread, multiThreadFkl;
    std::vector<Check> checks;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (runBatch<BATCHES[Is]>(w, singleThread, multiThread, multiThreadFkl, checks, hardwareThreads), ...);
    }(std::make_index_sequence<BATCHES.size()>{});

    writeCSV(outDir / "benchmark_cpu_opencv_1thread_vs_fkl_bvf_CPU.csv", "OpenCV 1 thread", singleThread);
    writeCSV(outDir / ("benchmark_cpu_opencv_" + std::to_string(hardwareThreads) + "threads_vs_fkl_bvf_CPU.csv"),
             "OpenCV " + std::to_string(hardwareThreads) + " threads", multiThread);
    writeCSV(outDir / ("benchmark_cpu_opencv_" + std::to_string(hardwareThreads) +
                       "threads_vs_fkl_bvf_CPU_MULTITHREAD.csv"),
             "OpenCV " + std::to_string(hardwareThreads) + " threads", multiThreadFkl, "FKL CPU_MULTITHREAD");
    std::ofstream v(outDir / "cpu_validation.csv");
    v << "Batch size, FKL MaxAbsError vs FKL-convention reference, OpenCV MaxAbsError vs OpenCV-convention reference, "
         "OpenCV vs FKL MaxAbsDifference, FKL CPU_MULTITHREAD vs FKL CPU MaxAbsDifference" << std::endl;
    for (const auto& c : checks) {
        v << c.batch << ", " << c.fklVsReference << ", " << c.opencvVsReference << ", " << c.opencvVsFkl << ", "
          << c.fklMultithreadVsFkl << std::endl;
    }
    const std::string opencvBuild = cv::getBuildInformation();
    std::ofstream(outDir / "cpu_opencv_build_information.txt") << opencvBuild;
    std::ofstream p(outDir / "cpu_platform.txt");
    p << "CPU: " << bench::cpuName() << ", " << hardwareThreads << " hardware threads\n"
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
      << "Pipeline: as the GPU benchmarks, with BT.601 limited range for NV12 -> RGB (the only NV12 conversion of "
         "OpenCV's CPU cvtColor)\n"
      << "Timing: steady_clock around each pipeline, " << WARMUP_ITERS << " warm-up and " << ITERS
      << " timed iterations, OpenCV and FKL one after the other in each, each run preceded by a "
      << std::chrono::milliseconds(IDLE_GAP).count()
      << " ms idle gap so that both thread pools are asleep; speedup computed per iteration and averaged.\n";
    std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
    return 0;
}
