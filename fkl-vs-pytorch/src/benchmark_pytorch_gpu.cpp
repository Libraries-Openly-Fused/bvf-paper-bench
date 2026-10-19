// FKL vs PyTorch on the GPU, Figure 1 pipeline: eager PyTorch (LibTorch's C++ API) vs FKL BVF.
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// with the frame, region sizes, batch sizes and the regions of every iteration of common/bench_geometry.h. The same
// source builds against LibTorch for CUDA and for ROCm (USE_ROCM, defined by LibTorch's CMake for ROCm), where PyTorch's
// device is still torch::kCUDA and FKL's side (src/fkl_gpu.cu) is compiled as HIP.
//
//   pytorch  src/pytorch_pipeline.h on the GPU: the chroma terms, a broadcast add with the 8-bit luma, the resize and
//            a copy into the output planes, in the fastest of six variants (pytorch_pipeline.h's VARIANTS, chosen per
//            batch size): per region (views of the region's planes, interpolate), or batched (the regions' parameters
//            uploaded in one copy, every region's 48 x 24 window gathered, one grid_sample); per iteration two zero
//            fills for the border rows and mul_, sub_, div_ on the whole tensor. PyTorch's caching allocator serves
//            the temporaries.
//   fkl_bvf  one FKL kernel for the whole chain (BVF), src/fkl_gpu.cu.
// Both run on one PyTorch stream (a stream of PyTorch's pool, made current), FKL through the stream's handle.
//
// The protocol of the GPU benchmarks of common/bench_gpu.h, with PyTorch's device-generic APIs instead of CUDA's:
// - validation before the timing, on the regions of the first, every 50th and the last timed iteration: PyTorch
//   against a reference with its pixel-center sampling, FKL against FKL's chain on its CPU backend, both 1e-3;
// - the two libraries never run at the same time: PyTorch's complete loop (warm-up and timed iterations) runs first,
//   then FKL's, on the same regions; FKL's speedup is computed per iteration (the executions of the same iteration,
//   identical inputs) and averaged, and every pair is written next to the summary (*_per_iteration.csv);
// - latency: every run between two device events on the stream (c10::Event, all host enqueue work included) and
//   followed by a synchronization;
// - GPU time only: a second pass with the same loops, PyTorch's then FKL's, the timed iterations traced with PyTorch's
//   profiler (Kineto: CUPTI on CUDA, the ROCm tracer on ROCm) in chunks of PROFILE_CHUNK iterations, an empty marker
//   kernel after every execution splitting the trace; the GPU time of one execution is the sum of the execution times
//   of its device activities (kernels, and memcpys and memsets when there are any);
// - kernel, memset and memcpy counts of one execution of each, from the same trace.

#include <torch/torch.h>
#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
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
    // PyTorch's: the variant of the next runs (empty for FKL)
    std::function<void(Variant)> setVariant;
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
    // Every timed iteration's latency and GPU time (ms), in iteration order
    std::vector<float> latencyMs, gpuMs;
    // Device memory segments PyTorch's caching allocator allocated (cudaMalloc/hipMalloc calls) during this
    // implementation's timed latency iterations
    int64_t timedSegmentAllocations{0};
    Activities counts;
    // Over the validated iterations together, and per validated iteration (iteration, against its own reference,
    // against FKL's float reference)
    ErrorStats errorVsOwnReference, errorVsFklReference;
    // (and the variant: PyTorch's, validated with each; "-" for FKL)
    std::vector<std::tuple<size_t, std::string, ErrorStats, ErrorStats>> errorsPerIteration;
    // PyTorch's variant in the timed loops: the fastest of the sweep (latencyBatch)
    std::string variant{"-"};
    bool withinTolerance() const {
        return std::all_of(errorsPerIteration.begin(), errorsPerIteration.end(),
                           [](const auto& e) { return validation::withinTolerance(std::get<2>(e), TOLERANCE); });
    }
};

struct Context {
    c10::Stream stream;
    // The stream's cudaStream_t (hipStream_t on ROCm), for FKL
    void* handle;
    std::unique_ptr<RegionSets> regions;
    pipeline::Conversion conversion;
    // PyTorch's variant chosen for each batch size by latencyBatch's sweep, used by gpuOnlyBatch too
    std::map<size_t, Variant> variants;
};

// PyTorch's latency loop with one variant of its pipeline, before the comparison of a batch size
struct SweepRow {
    size_t batch;
    Variant variant;
    TimingStats timing;
    bool chosen;
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
                     [torchPipeline]() { torchPipeline->fillNaN(); },
                     [torchPipeline](Variant v) { torchPipeline->setVariant(v); }});
    impls.push_back({FKL_BVF, FKL_REFERENCE, [fkl](size_t t) { fkl->run(t); }, [fkl]() { return fkl->download(); },
                     [fkl]() { fkl->fillNaN(); }, {}});
    return impls;
}

// Latency of each complete pipeline, as common/bench_gpu.h measures it, with the two libraries never running at the
// same time: each implementation's complete loop (warm-up and timed iterations, on the regions of each iteration)
// runs before the next one's, PyTorch first and FKL BVF last, each run between two events on the stream and followed
// by a synchronization. Returns the TIMED_ITERS execution times of each implementation, in ms, in iteration order.
// segments[k]: the device memory segments the caching allocator allocated during implementation k's timed iterations.
std::vector<std::vector<float>> measureSequential(const Context& ctx, const std::vector<Implementation>& impls,
                                                  std::vector<int64_t>& segments) {
    c10::Event start(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
    c10::Event stop(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
    std::vector<std::vector<float>> ms(impls.size(), std::vector<float>(TIMED_ITERS));
    segments.assign(impls.size(), 0);
    const auto segmentsAllocated = [&]() {
        return at::accelerator::getDeviceStats(ctx.stream.device_index())
            .segment[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)]
            .allocated;
    };
    for (size_t k = 0; k < impls.size(); ++k) {
        int64_t segmentsBefore = 0;
        for (int i = 0; i < WARMUP_ITERS + TIMED_ITERS; ++i) {
            if (i == WARMUP_ITERS) {
                segmentsBefore = segmentsAllocated();
            }
            start.record(ctx.stream);
            impls[k].run(static_cast<size_t>(i));
            stop.record(ctx.stream);
            stop.synchronize();
            if (i >= WARMUP_ITERS) {
                ms[k][i - WARMUP_ITERS] = static_cast<float>(start.elapsedTime(stop));
            }
        }
        segments[k] = segmentsAllocated() - segmentsBefore;
    }
    return ms;
}

// The device activities of every execution of one implementation on iterations [first, first + count), each followed
// by a synchronization and the marker kernel (launchExecutionMarker), under PyTorch's profiler: in start time order,
// every marker ends one execution, and the activities before it since the previous one are that execution's
std::vector<Activities> traceExecutions(const Context& ctx, const Implementation& impl, int first, int count) {
    namespace prof = torch::profiler::impl;
    const prof::ProfilerConfig config(prof::ProfilerState::KINETO);
    const std::set<prof::ActivityType> activities{prof::ActivityType::CUDA};
    torch::autograd::profiler::prepareProfiler(config, activities);
    torch::autograd::profiler::enableProfiler(config, activities);
    for (int i = first; i < first + count; ++i) {
        impl.run(static_cast<size_t>(i));
        ctx.stream.synchronize();
        launchExecutionMarker(ctx.handle);
        ctx.stream.synchronize();
    }
    const auto result = torch::autograd::profiler::disableProfiler();
    struct Record {
        uint64_t start, end;
        libkineto::ActivityType type;
        bool marker;
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
        records.push_back({e.startNs(), e.startNs() + e.durationNs(), type,
                           type == libkineto::ActivityType::CONCURRENT_KERNEL &&
                               e.name().find(EXECUTION_MARKER) != std::string::npos});
    }
    std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) { return a.start < b.start; });
    std::vector<Activities> executions;
    Activities execution;
    size_t activitiesSeen = 0;
    for (const Record& r : records) {
        if (r.marker) {
            if (activitiesSeen == 0) {
                throw std::runtime_error("GPU time: an execution of " + impl.name + " with no device activity");
            }
            executions.push_back(execution);
            execution = Activities{};
            activitiesSeen = 0;
            continue;
        }
        execution.kernels += r.type == libkineto::ActivityType::CONCURRENT_KERNEL;
        execution.memcpys += r.type == libkineto::ActivityType::GPU_MEMCPY;
        execution.memsets += r.type == libkineto::ActivityType::GPU_MEMSET;
        execution.busyNs += r.end - r.start;
        execution.firstStartNs = activitiesSeen == 0 ? r.start : std::min(execution.firstStartNs, r.start);
        execution.lastEndNs = std::max(execution.lastEndNs, r.end);
        ++activitiesSeen;
    }
    if (executions.size() != static_cast<size_t>(count) || activitiesSeen != 0) {
        throw std::runtime_error("GPU time: the trace of " + std::to_string(count) + " executions of " + impl.name +
                                 " has " + std::to_string(executions.size()) + " markers and " +
                                 std::to_string(records.size()) + " device activities");
    }
    // Every execution of a pipeline launches the same work: a different count means a lost or extra record
    for (const Activities& e : executions) {
        if (e.kernels != executions.front().kernels || e.memcpys != executions.front().memcpys ||
            e.memsets != executions.front().memsets) {
            throw std::runtime_error("GPU time: the executions of " + impl.name + " in the trace have different " +
                                     "kernel, memcpy or memset counts (" + std::to_string(e.kernels) + " and " +
                                     std::to_string(executions.front().kernels) + " kernels)");
        }
    }
    return executions;
}

// Validation and latency of batch N. Every batch's latency is measured before any profiler session of the process
// (gpuOnlyBatch), as in common/bench_gpu.h.
template <size_t N>
void latencyBatch(Context& ctx, std::vector<Result>& results, std::vector<SweepRow>& sweep) {
    std::cout << "=== N = " << N << ": validation and latency" << std::endl;
    const auto impls = implementations<N>(ctx);
    const size_t bvf = impls.size() - 1;
    std::vector<Result> batchResults(impls.size());
    for (size_t k = 0; k < impls.size(); ++k) {
        batchResults[k].name = impls[k].name;
        batchResults[k].batch = N;
        batchResults[k].reference = impls[k].reference;
    }
    // Before the timing, on the regions of the validated timed iterations: every pipeline against the reference of
    // its own conventions, and against FKL's float reference; PyTorch with each variant
    for (const size_t k : validatedIterations(TIMED_ITERS)) {
        const size_t t = WARMUP_ITERS + k;
        const auto fklReference = fklCpuReference(N, *ctx.regions, t);
        const auto pytorchReference = pixelCenterReference(N, *ctx.regions, t);
        for (size_t i = 0; i < impls.size(); ++i) {
            std::vector<std::string> variants{"-"};
            if (impls[i].setVariant) {
                variants.clear();
                for (const Variant f : VARIANTS) {
                    variants.push_back(variantName(f));
                }
            }
            for (size_t v = 0; v < variants.size(); ++v) {
                if (impls[i].setVariant) {
                    impls[i].setVariant(VARIANTS[v]);
                }
                impls[i].fillNaN();
                impls[i].run(t);
                const auto output = impls[i].download();
                Result& r = batchResults[i];
                const ErrorStats own = compare(output, impls[i].name == PYTORCH ? pytorchReference : fklReference);
                const ErrorStats vsFkl = compare(output, fklReference);
                r.errorsPerIteration.emplace_back(t, variants[v], own, vsFkl);
                r.errorVsOwnReference =
                    r.errorsPerIteration.size() == 1 ? own : validation::combine(r.errorVsOwnReference, own);
                r.errorVsFklReference =
                    r.errorsPerIteration.size() == 1 ? vsFkl : validation::combine(r.errorVsFklReference, vsFkl);
            }
        }
    }
    // The sweep: PyTorch's latency loop with each variant (events as in the comparison); the comparison uses the one
    // with the lower median
    Variant best = VARIANTS[0];
    double bestMedian = INFINITY;
    const size_t first = sweep.size();
    for (const Variant f : VARIANTS) {
        impls[0].setVariant(f);
        std::vector<int64_t> unused;
        const TimingStats timing = computeStats(measureSequential(ctx, {impls[0]}, unused)[0]);
        sweep.push_back({N, f, timing, false});
        if (timing.median < bestMedian) {
            bestMedian = timing.median;
            best = f;
        }
    }
    for (size_t i = first; i < sweep.size(); ++i) {
        sweep[i].chosen = sweep[i].variant.chromaTerms == best.chromaTerms &&
                          sweep[i].variant.resizeInput == best.resizeInput;
    }
    impls[0].setVariant(best);
    ctx.variants[N] = best;
    batchResults[0].variant = variantName(best);
    std::vector<int64_t> segments;
    const auto ms = measureSequential(ctx, impls, segments);
    for (size_t k = 0; k < impls.size(); ++k) {
        Result& r = batchResults[k];
        r.timing = computeStats(ms[k]);
        r.latencyMs = ms[k];
        r.timedSegmentAllocations = segments[k];
        if (k != bvf) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += ms[k][i] / ms[bvf][i];
            }
            r.meanSpeedupOfBvf = sum / TIMED_ITERS;
        }
        results.push_back(r);
    }
}

// GPU time of batch N, after every batch's latency: each implementation's warm-up, then its timed iterations traced in
// chunks, PyTorch first; fills in the results latencyBatch<N> wrote
template <size_t N>
void gpuOnlyBatch(Context& ctx, std::vector<Result>& results) {
    std::cout << "=== N = " << N << ": GPU time only" << std::endl;
    const auto impls = implementations<N>(ctx);
    // The variant of the latency comparison
    impls[0].setVariant(ctx.variants.at(N));
    const size_t bvf = impls.size() - 1;
    std::vector<Result*> batchResults;
    for (const auto& impl : impls) {
        const auto r = std::find_if(results.begin(), results.end(),
                                    [&](const Result& x) { return x.name == impl.name && x.batch == N; });
        if (r == results.end()) {
            throw std::runtime_error("GPU time: no latency results for " + impl.name);
        }
        batchResults.push_back(&*r);
    }
    std::vector<std::vector<float>> gpuMs(impls.size()), spanMs(impls.size());
    for (size_t k = 0; k < impls.size(); ++k) {
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            impls[k].run(static_cast<size_t>(i));
            ctx.stream.synchronize();
        }
        for (int first = 0; first < TIMED_ITERS; first += PROFILE_CHUNK) {
            const int count = std::min(PROFILE_CHUNK, TIMED_ITERS - first);
            const auto executions = traceExecutions(ctx, impls[k], WARMUP_ITERS + first, count);
            for (const Activities& a : executions) {
                gpuMs[k].push_back(static_cast<float>(a.busyNs * 1e-6));
                spanMs[k].push_back(static_cast<float>((a.lastEndNs - a.firstStartNs) * 1e-6));
            }
            // Every chunk is its own profiler session: its executions must have the first chunk's counts too
            const Activities& e = executions.front();
            if (first == 0) {
                batchResults[k]->counts = e;
            } else if (e.kernels != batchResults[k]->counts.kernels || e.memcpys != batchResults[k]->counts.memcpys ||
                       e.memsets != batchResults[k]->counts.memsets) {
                throw std::runtime_error("GPU time: the executions of " + impls[k].name + " from iteration " +
                                         std::to_string(WARMUP_ITERS + first) + " have other kernel, memcpy or " +
                                         "memset counts than the first chunk's (" + std::to_string(e.kernels) +
                                         " and " + std::to_string(batchResults[k]->counts.kernels) + " kernels)");
            }
        }
    }
    for (size_t k = 0; k < impls.size(); ++k) {
        Result& r = *batchResults[k];
        r.gpuMs = gpuMs[k];
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
                  << ", memsets " << r.counts.memsets << ", cudaMallocs in the timed iterations "
                  << r.timedSegmentAllocations;
        if (k != bvf) {
            std::cout << "  FKL BVF mean speedup " << std::setprecision(2) << r.meanSpeedupOfBvf << "x (GPU time "
                      << r.meanGpuSpeedupOfBvf << "x)";
        }
        std::cout << "  max err " << std::scientific << std::setprecision(2) << r.errorVsOwnReference.max() << " ("
                  << r.reference << "), vs FKL's reference " << r.errorVsFklReference.max() << std::defaultfloat
                  << std::endl;
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
    // nvidia-smi on this device only (its PCI bus ID): nvidia-smi's indices and CUDA's need not agree, and
    // CUDA_VISIBLE_DEVICES does not apply to nvidia-smi
    const std::string busId = gpuPciBusId();
    os << "NVIDIA driver: " << shell("nvidia-smi --id=" + busId + " --query-gpu=driver_version --format=csv,noheader")
       << "; this GPU in nvidia-smi: "
       << shell("nvidia-smi --id=" + busId + " --query-gpu=index,pci.bus_id,uuid --format=csv,noheader") << "\n"
       << "GPU clocks: the benchmark does not lock them (locking needs root); maximum SM and memory clocks "
          "(nvidia-smi clocks.max.sm, clocks.max.mem): "
       << shell("nvidia-smi --id=" + busId + " --query-gpu=clocks.max.sm,clocks.max.mem --format=csv,noheader")
       << "\n";
#endif
    os << "PyTorch: " << BENCH_TORCH_VERSION << " (LibTorch C++ API of the wheel, eager mode, c10::InferenceMode, "
       << "device " << ctx.stream.device() << ", one stream of PyTorch's pool; cuBLAS TF32 "
       << (at::globalContext().allowTF32CuBLAS() ? "allowed" : "not allowed")
       << "); full configuration in pytorch_build_information.txt\n"
       << "The benchmark's code (the calls into LibTorch and the harness): compiled with " << hostCompiler()
       << "; LibTorch itself is the wheel's build (its compilers and CUDA architectures in "
       << "pytorch_build_information.txt)\n"
       << "Benchmark repository: commit " << BENCH_REPO_COMMIT << "\n"
       << "CPU: " << cpuName() << "\n"
       << "OS: " << osName() << ", kernel " << kernelVersion() << "\n\n"
       << "Workload: " << describe() << ". AddBorder (black), then Mul (1/255), Sub (0.485, 0.456, 0.406), Div (0.229, "
       << "0.224, 0.225), float3 tensor of N planes " << OUT_W << "x" << OUT_H << "\n"
       << "PyTorch pipeline: the chroma terms (FKL's chroma coefficients and every offset folded into the bias: "
          "addmm on the chroma converted to float, or two addcmul on the 8-bit U and V views) and a broadcast add of "
          "the 8-bit luma with its coefficient (chroma replicated to 2x2 pixels; the add converts the luma), then the "
          "resize and a copy into the output planes; either per region, on views of the region's luma and chroma "
          "planes, with interpolate (bilinear, align_corners=false: pixel-center sampling) on a view of the crop that "
          "PyTorch treats as NCHW or as channels-last and a copy into its output plane (a transposing one from NCHW), "
          "or batched: the regions' window starts and grid coordinates computed on the host and uploaded in one copy "
          "from pinned memory, the 48 x 24 window at every region's origin gathered with one take from the frame, the "
          "conversion on the whole batch, one grid_sample (bilinear, align_corners=false, each region's pixel-center "
          "coordinates clamped to the region) and one copy into the output planes; per iteration the border rows set "
          "to 0 and mul_, sub_, div_ on the whole N x 64 x 64 x 3 tensor (the constants as 64 x 64 x 3 planes). "
          "Temporaries from PyTorch's caching "
          "allocator. Variants: the two chroma terms with the NCHW, channels-last or batched resize, six; for every "
          "batch size PyTorch's latency loop runs with each before the comparison, which uses the one with the lower "
          "median, also in the GPU time pass (pytorch_variant_sweep.csv; kernels.csv's variant)\n"
       << "FKL BVF: one executeOperations call, one kernel: no intermediates in memory, float3 from the color "
          "conversion to the write.\n"
       << "Regions: iteration t of each pipeline's loop (the " << WARMUP_ITERS << " warm-up iterations are t in [0, "
       << WARMUP_ITERS << "), the " << TIMED_ITERS << " timed ones t in [" << WARMUP_ITERS << ", "
       << WARMUP_ITERS + TIMED_ITERS << ")) runs that pipeline on the regions of iteration t, generated before the "
       << "loops (statistics of the timed ones in regions.csv); FKL's build() calls and fusion and PyTorch's "
          "per-region calls, or the batched variants' parameters and their upload, are timed.\n"
       << "Timing: the two libraries never run at the same time: for every batch size, pytorch's complete loop ("
       << WARMUP_ITERS << " warm-up then " << TIMED_ITERS << " timed iterations) runs first, then FKL BVF's on the same "
       << "regions, every run timed with device events (c10::Event) on one stream around the complete pipeline "
       << "(all host enqueue work included) and followed by a synchronization. FKL BVF's speedup is computed per "
       << "iteration (pytorch time / BVF time of the same iteration) and averaged (*_per_iteration.csv have every "
       << "pair). Median, p10, p90 with linear interpolation. Every batch's latency is measured before the process "
       << "starts any profiler session. kernels.csv has the device memory segments PyTorch's caching allocator "
       << "allocated during each implementation's timed iterations (0: no cudaMalloc there).\n"
       << "GPU time only: after every batch's latency, a second pass with the same loops, pytorch's then FKL BVF's, "
       << "the timed iterations traced with "
       << "PyTorch's profiler (Kineto) in chunks of " << PROFILE_CHUNK << " iterations, an empty marker kernel after "
       << "every execution splitting the trace; the GPU time of one execution is the "
       << "sum of the execution times of its device activities: kernels, and memcpys and memsets when there are any "
       << "(PyTorch's copy into the output plane is a device to device memcpy in the channels-last variants and a "
       << "kernel otherwise, the batched variants upload their parameters with a host to device memcpy, and cuBLASLt "
       << "may add a memset; its fills are kernels; kernels.csv has the counts, the same for every traced execution, "
       << "checked), span = first start to last end "
       << "of its activities in the trace (measured under the profiler, so it includes the profiler's cost per launch "
       << "and can exceed the latency); "
       << "speedups per execution, averaged.\n"
       << "Validation: before the timing, both pipelines on the regions of the first, every "
       << validation::VALIDATION_STRIDE << "th and the last timed iteration (validation.csv, one row per iteration and "
       << "variant): "
       << "PyTorch against the plain C++ reference with its pixel-center sampling (common/bench_reference.h), FKL BVF "
       << "against FKL's chain on its CPU backend, both 1e-3; max_abs_err_vs_float_reference is against FKL's "
          "reference for both. latency.csv's max_abs_err is, for pytorch, the largest over every validated variant "
          "(validation.csv has each variant's).\n"
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
    // Every batch's validation and latency first, then every batch's GPU time: no profiler session before a latency
    std::vector<SweepRow> sweep;
    forEachBatch([&]<size_t N>() { latencyBatch<N>(ctx, results, sweep); });
    forEachBatch([&]<size_t N>() { gpuOnlyBatch<N>(ctx, results); });

    std::ofstream latency(outDir / "latency.csv");
    latency << "implementation,N,median_us,p10_us,p90_us,iterations,max_abs_err,kernels_launched,mean_us,"
            << FKL_BVF << "_mean_speedup\n";
    std::ofstream validationCsv(outDir / "validation.csv");
    validationCsv << "implementation,variant,N,iteration,reference,max_abs_err_r,max_abs_err_g,max_abs_err_b,mean_abs_err,"
                     "non_finite_values,max_abs_err_vs_float_reference,values_beyond_1e-3,values,tolerance,"
                     "within_tolerance\n";
    std::ofstream sweepCsv(outDir / "pytorch_variant_sweep.csv");
    sweepCsv << "Batch size, Chroma terms, Resize input, MedianTime, P10Time, P90Time, MeanTime, Chosen\n";
    for (const auto& r : sweep) {
        sweepCsv << r.batch << ", " << chromaTermsName(r.variant.chromaTerms) << ", "
                 << resizeInputName(r.variant.resizeInput) << ", " << r.timing.median / 1000. << ", "
                 << r.timing.p10 / 1000. << ", " << r.timing.p90 / 1000. << ", " << r.timing.mean / 1000. << ", "
                 << (r.chosen ? "yes" : "no") << "\n";
    }
    std::ofstream kernels(outDir / "kernels.csv");
    kernels << "implementation,variant,N,kernels,memsets,memcpys,device_mallocs_in_timed_iterations\n";
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
        for (const auto& [iteration, variant, e, vsFkl] : r.errorsPerIteration) {
            validationCsv << r.name << "," << variant << "," << r.batch << "," << iteration << "," << r.reference << ","
                          << std::scientific << std::setprecision(3) << e.maxPerChannel[0] << "," << e.maxPerChannel[1]
                          << "," << e.maxPerChannel[2] << "," << e.mean << "," << e.nonFinite << "," << vsFkl.max()
                          << "," << e.beyond << "," << e.values << "," << TOLERANCE << ","
                          << (validation::withinTolerance(e, TOLERANCE) ? "yes" : "no") << "\n";
        }
        kernels << r.name << "," << r.variant << "," << r.batch << "," << r.counts.kernels << "," << r.counts.memsets << ","
                << r.counts.memcpys << "," << r.timedSegmentAllocations << "\n";
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
        // Every timed iteration's pair of times (ms) and their ratio, which the mean speedup averages
        std::ofstream pairs(outDir / (std::string(PYTORCH) + "_vs_" + FKL_BVF + "_" + (gpuOnly ? "gpu_only_" : "") +
                                      parArchName() + "_per_iteration.csv"));
        pairs << "Batch size, Iteration, " << PYTORCH << " Time (ms), " << FKL_BVF << " Time (ms), Speedup" << std::endl;
        for (const auto& b : results) {
            if (b.name != PYTORCH) {
                continue;
            }
            const auto f = std::find_if(results.begin(), results.end(),
                                        [&](const Result& r) { return r.name == FKL_BVF && r.batch == b.batch; });
            const std::vector<float>& pt = gpuOnly ? b.gpuMs : b.latencyMs;
            const std::vector<float>& fk = gpuOnly ? f->gpuMs : f->latencyMs;
            for (size_t i = 0; i < pt.size(); ++i) {
                pairs << b.batch << ", " << WARMUP_ITERS + i << ", " << pt[i] << ", " << fk[i] << ", "
                      << static_cast<double>(pt[i]) / fk[i] << std::endl;
            }
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
