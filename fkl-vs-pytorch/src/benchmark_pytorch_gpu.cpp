// FKL vs PyTorch on the GPU, Figure 1 pipeline: eager PyTorch (LibTorch's C++ API) vs FKL BVF.
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// with the frame, region sizes, batch sizes and the regions of every iteration of common/bench_geometry.h. The same
// source builds against LibTorch for CUDA and for ROCm (USE_ROCM, defined by LibTorch's CMake for ROCm), where PyTorch's
// device is still torch::kCUDA and FKL's side (src/fkl_gpu.cu) is compiled as HIP.
//
//   pytorch  src/pytorch_pipeline.h on the GPU: per region, views of the region's planes, two conversions to float, an
//            addmm, a broadcast add, interpolate and a device to device copy into its output plane; per iteration two
//            zero fills for the border rows and mul_, sub_, div_ on the whole tensor. PyTorch's caching allocator
//            serves the temporaries of every region.
//   fkl_bvf  one FKL kernel for the whole chain (BVF), src/fkl_gpu.cu.
// Both run on one PyTorch stream (a stream of PyTorch's pool, made current), FKL through the stream's handle.
//
// The protocol of the GPU benchmarks of common/bench_gpu.h, with PyTorch's device-generic APIs instead of CUDA's:
// - validation before the timing, on the regions of the first, every 50th and the last timed iteration: PyTorch
//   against a reference with its pixel-center sampling, FKL against FKL's chain on its CPU backend, both 1e-3;
// - latency: in every iteration PyTorch and then FKL, each between two device events on the stream (c10::Event, all
//   host enqueue work included) and followed by a synchronization; FKL's speedup computed per iteration and averaged;
// - GPU time only: a second pass with the same interleaved iterations, traced with PyTorch's profiler (Kineto: CUPTI on
//   CUDA, the ROCm tracer on ROCm), in chunks of PROFILE_CHUNK iterations; the GPU time of one execution is the sum of
//   the execution times of its device activities (kernels, and PyTorch's device to device copies and fills);
// - kernel, memset and memcpy counts of one execution of each, from the same trace.

#include <torch/torch.h>
#include <ATen/Context.h>
#include <ATen/Version.h>
#include <c10/core/Event.h>
#include <kineto/ActivityType.h>
#include <torch/csrc/autograd/profiler_kineto.h>
// LibTorch for ROCm keeps the c10::cuda names of streams and guards, in hipified headers
#ifdef USE_ROCM
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "bench_geometry.h"
#include "bench_platform.h"
#include "bench_timing.h"
#include "bench_validation.h"
#include "fkl_gpu.h"
#include "pytorch_pipeline.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace bench::pytorch {

using namespace geometry;
// BENCH_WARMUP_ITERATIONS and BENCH_ITERATIONS come from CMake: 100 and 1000 for the paper's results, 0 and 1 in the CI
constexpr int WARMUP_ITERS = BENCH_WARMUP_ITERATIONS;
constexpr int TIMED_ITERS = BENCH_ITERATIONS;
static_assert(WARMUP_ITERS >= 0 && TIMED_ITERS >= 1, "at least one timed iteration");
// Iterations per trace of the GPU time pass
constexpr int PROFILE_CHUNK = 100;
using validation::TOLERANCE;
using validation::validatedIterations;
using ErrorStats = validation::Errors;
using validation::compare;

const char* const PYTORCH = "pytorch";
const char* const FKL_BVF = "fkl_bvf";
const char* const PYTORCH_REFERENCE = "pytorch_pixel_center_float";
const char* const FKL_REFERENCE = "fkl_cpu_float";

struct Implementation {
    std::string name;
    std::string reference;
    std::function<void(size_t)> run;
    std::function<std::vector<float>()> download;
    std::function<void()> fillNaN;
};

// Device activities of one execution, from the trace
struct Activities {
    size_t kernels{0}, memsets{0}, memcpys{0};
    uint64_t busyNs{0}, firstStartNs{0}, lastEndNs{0};
};

struct Result {
    std::string name;
    size_t batch{0};
    std::string reference;
    TimingStats timing;
    // Mean over the timed iterations of (this implementation's time / FKL BVF's time) in the same iteration
    double meanSpeedupOfBvf{NAN};
    TimingStats gpuTiming, gpuSpan;
    double meanGpuSpeedupOfBvf{NAN};
    Activities counts;
    // Over the validated iterations together, and per validated iteration (iteration, against its own reference,
    // against FKL's float reference)
    ErrorStats errorVsOwnReference, errorVsFklReference;
    std::vector<std::tuple<size_t, ErrorStats, ErrorStats>> errorsPerIteration;
    bool withinTolerance() const {
        return std::all_of(errorsPerIteration.begin(), errorsPerIteration.end(),
                           [](const auto& e) { return validation::withinTolerance(std::get<1>(e), TOLERANCE); });
    }
};

struct Context {
    c10::Stream stream;
    // The stream's cudaStream_t (hipStream_t on ROCm), for FKL
    void* handle;
    std::unique_ptr<RegionSets> regions;
    pipeline::Conversion conversion;
};

template <size_t N>
std::vector<Implementation> implementations(Context& ctx) {
    size_t pitch = 0;
    const uint8_t* frame = hostFrame(&pitch);
    auto torchPipeline =
        std::make_shared<Pipeline>(frame, pitch, torch::Device(torch::kCUDA, ctx.stream.device_index()), N,
                                   ctx.conversion);
    std::shared_ptr<FklGpu> fkl = makeFklGpu(N, ctx.handle, *ctx.regions);
    const RegionSets* regions = ctx.regions.get();
    std::vector<Implementation> impls;
    impls.push_back({PYTORCH, PYTORCH_REFERENCE, [torchPipeline, regions](size_t t) {
                         torchPipeline->run(&regions->at(t, 0));
                     },
                     [torchPipeline]() { return torchPipeline->download(); },
                     [torchPipeline]() { torchPipeline->fillNaN(); }});
    impls.push_back({FKL_BVF, FKL_REFERENCE, [fkl](size_t t) { fkl->run(t); }, [fkl]() { return fkl->download(); },
                     [fkl]() { fkl->fillNaN(); }});
    return impls;
}

// Latency of each complete pipeline, as common/bench_gpu.h measures it: in every iteration the implementations one
// after the other on the regions of that iteration, each between two events on the stream and followed by a
// synchronization. Returns the TIMED_ITERS execution times of each implementation, in ms, in iteration order.
std::vector<std::vector<float>> measureInterleaved(const Context& ctx, const std::vector<Implementation>& impls) {
    c10::Event start(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
    c10::Event stop(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
    std::vector<std::vector<float>> ms(impls.size(), std::vector<float>(TIMED_ITERS));
    for (int i = 0; i < WARMUP_ITERS + TIMED_ITERS; ++i) {
        for (size_t k = 0; k < impls.size(); ++k) {
            start.record(ctx.stream);
            impls[k].run(static_cast<size_t>(i));
            stop.record(ctx.stream);
            stop.synchronize();
            if (i >= WARMUP_ITERS) {
                ms[k][i - WARMUP_ITERS] = static_cast<float>(start.elapsedTime(stop));
            }
        }
    }
    return ms;
}

// The device activities of iterations [first, first + count): every iteration runs PyTorch and then FKL, each followed
// by a synchronization, under PyTorch's profiler. In start time order, each FKL kernel (the only kernel whose name has
// "fk::") ends one iteration, and the activities before it since the previous one are PyTorch's execution of that
// iteration.
std::vector<std::pair<Activities, Activities>> traceIterations(const Context& ctx,
                                                               const std::vector<Implementation>& impls,
                                                               int first, int count) {
    namespace prof = torch::profiler::impl;
    const prof::ProfilerConfig config(prof::ProfilerState::KINETO);
    const std::set<prof::ActivityType> activities{prof::ActivityType::CUDA};
    torch::autograd::profiler::prepareProfiler(config, activities);
    torch::autograd::profiler::enableProfiler(config, activities);
    for (int i = first; i < first + count; ++i) {
        for (const auto& impl : impls) {
            impl.run(static_cast<size_t>(i));
            ctx.stream.synchronize();
        }
    }
    const auto result = torch::autograd::profiler::disableProfiler();
    struct Record {
        uint64_t start, end;
        uint8_t type;
        bool fkl;
    };
    std::vector<Record> records;
    for (const auto& e : result->events()) {
        const auto type = static_cast<libkineto::ActivityType>(e.activityType());
        if (e.deviceType() != c10::DeviceType::CUDA && e.deviceType() != c10::DeviceType::HIP) {
            continue;
        }
        if (type != libkineto::ActivityType::CONCURRENT_KERNEL && type != libkineto::ActivityType::GPU_MEMCPY &&
            type != libkineto::ActivityType::GPU_MEMSET) {
            continue;
        }
        records.push_back({e.startNs(), e.startNs() + e.durationNs(), e.activityType(),
                           type == libkineto::ActivityType::CONCURRENT_KERNEL &&
                               e.name().find("fk::") != std::string::npos});
    }
    std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) { return a.start < b.start; });
    std::vector<std::pair<Activities, Activities>> iterations;
    Activities torchExecution;
    const auto add = [](Activities& a, const Record& r) {
        const auto type = static_cast<libkineto::ActivityType>(r.type);
        a.kernels += type == libkineto::ActivityType::CONCURRENT_KERNEL;
        a.memcpys += type == libkineto::ActivityType::GPU_MEMCPY;
        a.memsets += type == libkineto::ActivityType::GPU_MEMSET;
        a.busyNs += r.end - r.start;
        a.firstStartNs = a.kernels + a.memcpys + a.memsets == 1 ? r.start : std::min(a.firstStartNs, r.start);
        a.lastEndNs = std::max(a.lastEndNs, r.end);
    };
    for (const Record& r : records) {
        if (r.fkl) {
            Activities fklExecution;
            add(fklExecution, r);
            if (torchExecution.kernels == 0) {
                throw std::runtime_error("GPU time: an FKL kernel without PyTorch kernels before it");
            }
            iterations.emplace_back(torchExecution, fklExecution);
            torchExecution = Activities{};
        } else {
            add(torchExecution, r);
        }
    }
    if (iterations.size() != static_cast<size_t>(count) || torchExecution.kernels + torchExecution.memcpys +
                                                                   torchExecution.memsets != 0) {
        throw std::runtime_error("GPU time: the trace of " + std::to_string(count) + " iterations has " +
                                 std::to_string(iterations.size()) + " FKL kernels and " +
                                 std::to_string(records.size()) + " device activities");
    }
    return iterations;
}

// Validation, latency and GPU time of batch N
template <size_t N>
void runBatch(Context& ctx, std::vector<Result>& results) {
    std::cout << "=== N = " << N << std::endl;
    const auto impls = implementations<N>(ctx);
    const size_t bvf = impls.size() - 1;
    std::vector<Result> batchResults(impls.size());
    for (size_t k = 0; k < impls.size(); ++k) {
        batchResults[k].name = impls[k].name;
        batchResults[k].batch = N;
        batchResults[k].reference = impls[k].reference;
    }
    // Before the timing, on the regions of the validated timed iterations: every pipeline against the reference of
    // its own conventions, and against FKL's float reference
    for (const size_t k : validatedIterations(TIMED_ITERS)) {
        const size_t t = WARMUP_ITERS + k;
        const auto fklReference = fklCpuReference(N, *ctx.regions, t);
        const auto pytorchReference = pixelCenterReference(N, *ctx.regions, t);
        for (size_t i = 0; i < impls.size(); ++i) {
            impls[i].fillNaN();
            impls[i].run(t);
            const auto output = impls[i].download();
            Result& r = batchResults[i];
            const ErrorStats own = compare(output, impls[i].name == PYTORCH ? pytorchReference : fklReference);
            const ErrorStats vsFkl = compare(output, fklReference);
            r.errorsPerIteration.emplace_back(t, own, vsFkl);
            r.errorVsOwnReference = r.errorsPerIteration.size() == 1 ? own : validation::combine(r.errorVsOwnReference, own);
            r.errorVsFklReference =
                r.errorsPerIteration.size() == 1 ? vsFkl : validation::combine(r.errorVsFklReference, vsFkl);
        }
    }
    const auto ms = measureInterleaved(ctx, impls);
    for (size_t k = 0; k < impls.size(); ++k) {
        Result& r = batchResults[k];
        r.timing = computeStats(ms[k]);
        if (k != bvf) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += ms[k][i] / ms[bvf][i];
            }
            r.meanSpeedupOfBvf = sum / TIMED_ITERS;
        }
    }
    // GPU time only: warm-up as in the latency pass, then the timed iterations traced in chunks
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        for (const auto& impl : impls) {
            impl.run(static_cast<size_t>(i));
            ctx.stream.synchronize();
        }
    }
    std::vector<std::vector<float>> gpuMs(impls.size()), spanMs(impls.size());
    for (int first = 0; first < TIMED_ITERS; first += PROFILE_CHUNK) {
        const int count = std::min(PROFILE_CHUNK, TIMED_ITERS - first);
        const auto iterations = traceIterations(ctx, impls, WARMUP_ITERS + first, count);
        for (const auto& [torchExecution, fklExecution] : iterations) {
            size_t k = 0;
            for (const Activities* a : {&torchExecution, &fklExecution}) {
                gpuMs[k].push_back(static_cast<float>(a->busyNs * 1e-6));
                spanMs[k].push_back(static_cast<float>((a->lastEndNs - a->firstStartNs) * 1e-6));
                ++k;
            }
        }
        if (first == 0) {
            batchResults[0].counts = iterations.front().first;
            batchResults[1].counts = iterations.front().second;
        }
    }
    for (size_t k = 0; k < impls.size(); ++k) {
        Result& r = batchResults[k];
        r.gpuTiming = computeStats(gpuMs[k]);
        r.gpuSpan = computeStats(spanMs[k]);
        if (k != bvf) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += gpuMs[k][i] / gpuMs[bvf][i];
            }
            r.meanGpuSpeedupOfBvf = sum / TIMED_ITERS;
        }
        std::cout << std::left << std::setw(8) << r.name << std::right << std::fixed << std::setprecision(1)
                  << " median " << std::setw(8) << r.timing.median << " us  p10 " << std::setw(8) << r.timing.p10
                  << "  p90 " << std::setw(8) << r.timing.p90 << "  GPU time median " << std::setw(8)
                  << r.gpuTiming.median << " us  kernels " << r.counts.kernels << ", memcpys " << r.counts.memcpys
                  << ", memsets " << r.counts.memsets;
        if (k != bvf) {
            std::cout << "  FKL BVF mean speedup " << std::setprecision(2) << r.meanSpeedupOfBvf << "x (GPU time "
                      << r.meanGpuSpeedupOfBvf << "x)";
        }
        std::cout << "  max err " << std::scientific << std::setprecision(2) << r.errorVsOwnReference.max() << " ("
                  << r.reference << "), vs FKL's reference " << r.errorVsFklReference.max() << std::defaultfloat
                  << std::endl;
        results.push_back(r);
    }
}

// Calls f.template operator()<N>() for every batch size, in order
template <typename F>
void forEachBatch(F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (f.template operator()<BATCHES[Is]>(), ...);
    }(std::make_index_sequence<BATCHES.size()>{});
}

void writePlatform(const std::filesystem::path& outDir, const Context& ctx) {
    std::ofstream os(outDir / "platform.txt");
    os << "Benchmark: FKL vs PyTorch on the GPU (fkl-vs-pytorch)\n" << describeGpuAndFkl();
#ifndef USE_ROCM
    os << "NVIDIA driver: " << shell("nvidia-smi --query-gpu=driver_version --format=csv,noheader") << "\n";
#endif
    os << "PyTorch: " << BENCH_TORCH_VERSION << " (LibTorch C++ API of the wheel, eager mode, c10::InferenceMode, "
       << "device " << ctx.stream.device() << ", one stream of PyTorch's pool; cuBLAS TF32 "
       << (at::globalContext().allowTF32CuBLAS() ? "allowed" : "not allowed")
       << "); full configuration in pytorch_build_information.txt\n"
       << "PyTorch's code: compiled with " << hostCompiler() << "\n"
       << "Benchmark repository: commit " << BENCH_REPO_COMMIT << "\n"
       << "CPU: " << cpuName() << "\n"
       << "OS: " << osName() << ", kernel " << kernelVersion() << "\n\n"
       << "Workload: " << describe() << ". AddBorder (black), then Mul (1/255), Sub (0.485, 0.456, 0.406), Div (0.229, "
       << "0.224, 0.225), float3 tensor of N planes " << OUT_W << "x" << OUT_H << "\n"
       << "PyTorch pipeline: per region, views of the region's luma and chroma planes, chroma to float and addmm with "
          "FKL's chroma coefficients and every offset folded into the bias, luma to float and a broadcast add with "
          "its coefficient (chroma replicated to 2x2 pixels), interpolate (bilinear, align_corners=false: "
          "pixel-center sampling) and a device to device copy into its output plane; per iteration the border rows "
          "set to 0 and mul_, sub_, div_ on the whole N x 64 x 64 x 3 tensor. Temporaries from PyTorch's caching "
          "allocator.\n"
       << "FKL BVF: one executeOperations call, one kernel: no intermediates in memory, float3 from the color "
          "conversion to the write.\n"
       << "Regions: iteration t of every loop (the " << WARMUP_ITERS << " warm-up iterations are t in [0, "
       << WARMUP_ITERS << "), the " << TIMED_ITERS << " timed ones t in [" << WARMUP_ITERS << ", "
       << WARMUP_ITERS + TIMED_ITERS << ")) runs both pipelines on the regions of iteration t, generated before the "
       << "loops (statistics of the timed ones in regions.csv); FKL's build() calls and fusion and PyTorch's "
          "per-region calls are timed.\n"
       << "Timing: " << WARMUP_ITERS << " warm-up then " << TIMED_ITERS << " timed iterations. In every iteration "
       << "pytorch then FKL BVF, each timed with device events (c10::Event) on one stream around the complete pipeline "
       << "(all host enqueue work included) and followed by a synchronization. FKL BVF's speedup is computed per "
       << "iteration (pytorch time / BVF time) and averaged. Median, p10, p90 with linear interpolation.\n"
       << "GPU time only: a second pass with the same interleaved iterations and regions, traced with PyTorch's "
       << "profiler (Kineto) in chunks of " << PROFILE_CHUNK << " iterations; the GPU time of one execution is the "
       << "sum of the execution times of its device activities (kernels, memcpys and memsets: PyTorch's device to "
       << "device copies are memcpys), span = first start to last end; speedups per execution, averaged.\n"
       << "Validation: before the timing, both pipelines on the regions of the first, every "
       << validation::VALIDATION_STRIDE << "th and the last timed iteration (validation.csv, one row per iteration): "
       << "PyTorch against the plain C++ reference with its pixel-center sampling (common/bench_reference.h), FKL BVF "
       << "against FKL's chain on its CPU backend, both 1e-3; max_abs_err_vs_float_reference is against FKL's "
          "reference for both.\n"
       << "Counts: kernels, memcpys and memsets of one execution (the first timed iteration), from the trace.\n";
}

int run(int argc, char** argv) {
    const std::filesystem::path outDir = argc > 1 ? argv[1] : "results";
    std::filesystem::create_directories(outDir);
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("PyTorch sees no GPU");
    }
    // A stream of PyTorch's pool, current for PyTorch's operations and given to FKL
    const c10::cuda::CUDAStream torchStream = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(torchStream);
    Context ctx{torchStream, static_cast<void*>(torchStream.stream()), std::make_unique<RegionSets>(static_cast<size_t>(WARMUP_ITERS + TIMED_ITERS)),
                fklConversion()};
    std::cout << "=== FKL vs PyTorch on the GPU (fkl-vs-pytorch)" << std::endl;
    std::vector<Result> results;
    forEachBatch([&]<size_t N>() { runBatch<N>(ctx, results); });

    std::ofstream latency(outDir / "latency.csv");
    latency << "implementation,N,median_us,p10_us,p90_us,iterations,max_abs_err,kernels_launched,mean_us,"
            << FKL_BVF << "_mean_speedup\n";
    std::ofstream validationCsv(outDir / "validation.csv");
    validationCsv << "implementation,N,iteration,reference,max_abs_err_r,max_abs_err_g,max_abs_err_b,mean_abs_err,"
                     "non_finite_values,max_abs_err_vs_float_reference,values_beyond_1e-3,values,tolerance,"
                     "within_tolerance\n";
    std::ofstream kernels(outDir / "kernels.csv");
    kernels << "implementation,N,kernels,memsets,memcpys\n";
    std::ofstream gpuTime(outDir / "gpu_time.csv");
    gpuTime << "implementation,N,gpu_median_us,gpu_p10_us,gpu_p90_us,gpu_mean_us,iterations,kernels_launched,"
            << FKL_BVF << "_mean_gpu_speedup,span_median_us,memcpys,memsets\n";
    bool failed = false;
    for (const auto& r : results) {
        latency << r.name << "," << r.batch << "," << std::fixed << std::setprecision(2) << r.timing.median << ","
                << r.timing.p10 << "," << r.timing.p90 << "," << TIMED_ITERS << "," << std::scientific
                << std::setprecision(3) << r.errorVsOwnReference.max() << "," << r.counts.kernels << ","
                << std::fixed << std::setprecision(2) << r.timing.mean << ",";
        if (!std::isnan(r.meanSpeedupOfBvf)) {
            latency << std::setprecision(3) << r.meanSpeedupOfBvf;
        }
        latency << "\n";
        for (const auto& [iteration, e, vsFkl] : r.errorsPerIteration) {
            validationCsv << r.name << "," << r.batch << "," << iteration << "," << r.reference << ","
                          << std::scientific << std::setprecision(3) << e.maxPerChannel[0] << "," << e.maxPerChannel[1]
                          << "," << e.maxPerChannel[2] << "," << e.mean << "," << e.nonFinite << "," << vsFkl.max()
                          << "," << e.beyond << "," << e.values << "," << TOLERANCE << ","
                          << (validation::withinTolerance(e, TOLERANCE) ? "yes" : "no") << "\n";
        }
        kernels << r.name << "," << r.batch << "," << r.counts.kernels << "," << r.counts.memsets << ","
                << r.counts.memcpys << "\n";
        gpuTime << r.name << "," << r.batch << "," << std::fixed << std::setprecision(2) << r.gpuTiming.median << ","
                << r.gpuTiming.p10 << "," << r.gpuTiming.p90 << "," << r.gpuTiming.mean << "," << TIMED_ITERS << ","
                << r.counts.kernels << ",";
        if (!std::isnan(r.meanGpuSpeedupOfBvf)) {
            gpuTime << std::setprecision(3) << r.meanGpuSpeedupOfBvf;
        }
        gpuTime << "," << std::setprecision(2) << r.gpuSpan.median << "," << r.counts.memcpys << ","
                << r.counts.memsets << "\n";
        // Both pipelines must match the reference of their conventions
        failed = failed || !r.withinTolerance();
    }
    // FKL's two-execution format (times in ms): pytorch_vs_fkl_bvf_<ParArch>.csv (latency) and
    // pytorch_vs_fkl_bvf_gpu_only_<ParArch>.csv
    for (const bool gpuOnly : {false, true}) {
        std::ofstream two(outDir / (std::string(PYTORCH) + "_vs_" + FKL_BVF + "_" + (gpuOnly ? "gpu_only_" : "") +
                                    parArchName() + ".csv"));
        writeTwoExecutionHeader(two, "Batch size", PYTORCH, FKL_BVF);
        for (const auto& b : results) {
            if (b.name != PYTORCH) {
                continue;
            }
            const auto f = std::find_if(results.begin(), results.end(),
                                        [&](const Result& r) { return r.name == FKL_BVF && r.batch == b.batch; });
            writeTwoExecutionRow(two, static_cast<long long>(b.batch), gpuOnly ? b.gpuTiming : b.timing,
                                 gpuOnly ? f->gpuTiming : f->timing,
                                 gpuOnly ? b.meanGpuSpeedupOfBvf : b.meanSpeedupOfBvf);
        }
    }
    std::ofstream(outDir / "pytorch_build_information.txt") << at::show_config();
    writePlatform(outDir, ctx);
    writeRegionStats((outDir / "regions.csv").string(), *ctx.regions, WARMUP_ITERS, TIMED_ITERS);
    std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
    if (failed) {
        std::cerr << "ERROR: an output is not within its tolerance, see validation.csv" << std::endl;
        return 1;
    }
    return 0;
}

} // namespace bench::pytorch

int main(int argc, char** argv) {
    try {
        return bench::pytorch::run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
