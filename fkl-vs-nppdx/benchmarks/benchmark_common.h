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

#ifndef FKL_VS_NPPDX_BENCHMARK_COMMON_H
#define FKL_VS_NPPDX_BENCHMARK_COMMON_H

// Two-execution benchmarks in the same format as FKL's benchmarks/twoExecutionsBenchmark.h and
// cvGPUSpeedup's tests/testsCommon.cuh: in every iteration the first (baseline) and the second
// implementation are timed one after the other with CUDA events, and one CSV per benchmark is written
// with the mean, variance, max and min of each (in ms) and the mean of the per-iteration speedups
// (first / second). Median, p10 and p90 columns are appended for the paper.

#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/utils/utils.h>

#include <cupti.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace benchmark {

constexpr int WARMUP_ITERS = 100;
constexpr int ITERS = 1000;

struct TimeStats {
    float mean{0.f}, variance{0.f}, max{0.f}, min{0.f}, median{0.f}, p10{0.f}, p90{0.f};
};

struct TwoExecutionsResult {
    int variableDimensionValue{0};
    TimeStats first, second;
    float meanSpeedup{0.f};
};

inline float percentile(const std::vector<float>& sorted, const double p) {
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return static_cast<float>(sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]));
}

inline TimeStats computeStats(const std::vector<float>& times) {
    TimeStats stats;
    double sum = 0.;
    for (const float t : times) {
        sum += t;
    }
    stats.mean = static_cast<float>(sum / times.size());
    double sumOfDiff = 0.;
    for (const float t : times) {
        sumOfDiff += (t - stats.mean) * (t - stats.mean);
    }
    stats.variance = static_cast<float>(sumOfDiff / (times.size() - 1));
    std::vector<float> sorted(times);
    std::sort(sorted.begin(), sorted.end());
    stats.min = sorted.front();
    stats.max = sorted.back();
    stats.median = percentile(sorted, 0.5);
    stats.p10 = percentile(sorted, 0.1);
    stats.p90 = percentile(sorted, 0.9);
    return stats;
}

// Times first() and second() back to back in every iteration, on one stream, synchronizing after each
inline TwoExecutionsResult runTwoExecutions(cudaStream_t stream, const int variableDimensionValue,
                                            const std::function<void()>& first,
                                            const std::function<void()>& second) {
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    const auto timeOne = [&](const std::function<void()>& run) {
        gpuErrchk(cudaEventRecord(start, stream));
        run();
        gpuErrchk(cudaEventRecord(stop, stream));
        gpuErrchk(cudaEventSynchronize(stop));
        float ms = 0.f;
        gpuErrchk(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        timeOne(first);
        timeOne(second);
    }
    std::vector<float> firstTimes(ITERS), secondTimes(ITERS);
    double speedupSum = 0.;
    for (int i = 0; i < ITERS; ++i) {
        firstTimes[i] = timeOne(first);
        secondTimes[i] = timeOne(second);
        speedupSum += firstTimes[i] / secondTimes[i];
    }
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return {variableDimensionValue, computeStats(firstTimes), computeStats(secondTimes),
            static_cast<float>(speedupSum / ITERS)};
}

// Times one implementation alone, same protocol as runTwoExecutions
inline TimeStats runOneExecution(cudaStream_t stream, const std::function<void()>& run) {
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    std::vector<float> times(ITERS);
    for (int i = 0; i < WARMUP_ITERS + ITERS; ++i) {
        gpuErrchk(cudaEventRecord(start, stream));
        run();
        gpuErrchk(cudaEventRecord(stop, stream));
        gpuErrchk(cudaEventSynchronize(stop));
        if (i >= WARMUP_ITERS) {
            gpuErrchk(cudaEventElapsedTime(&times[i - WARMUP_ITERS], start, stop));
        }
    }
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return computeStats(times);
}

// GPU time only: CUPTI kernel activity tracing gives the start and end of every kernel on the GPU. The GPU time of
// one execution is the sum of the execution times of its kernels: no host work and no launch gaps.
namespace cupti {
struct KernelSpan {
    uint64_t start, end;
};
inline std::vector<KernelSpan>& kernels() {
    static std::vector<KernelSpan> records;
    return records;
}

inline void check(CUptiResult result, const char* what) {
    if (result != CUPTI_SUCCESS) {
        const char* message = nullptr;
        cuptiGetResultString(result, &message);
        throw std::runtime_error(std::string("CUPTI error ") + (message ? message : "") + " in " + what);
    }
}

inline void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *size = 8 << 20;
    *buffer = static_cast<uint8_t*>(std::aligned_alloc(8, *size));
    *maxNumRecords = 0;
}

inline void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            const auto* k = reinterpret_cast<const CUpti_ActivityKernel12*>(record);
            // The L2 flush kernels of the flushed benchmark are not part of any pipeline
            if (!(k->name && std::strstr(k->name, "flushL2Kernel"))) {
                kernels().push_back({k->start, k->end});
            }
        }
    }
    std::free(buffer);
}

inline void init() {
    static bool initialized = false;
    if (!initialized) {
        check(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted), "cuptiActivityRegisterCallbacks");
        initialized = true;
    }
}

// Runs run() `times` times with kernel tracing on and returns the kernel records in start order
inline std::vector<KernelSpan> trace(cudaStream_t stream, const std::function<void()>& run, int times) {
    init();
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "cuptiActivityFlushAll");
    kernels().clear();
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "cuptiActivityEnable");
    for (int i = 0; i < times; ++i) {
        run();
        gpuErrchk(cudaStreamSynchronize(stream));
        if (i % 100 == 99) {
            check(cuptiActivityFlushAll(0), "cuptiActivityFlushAll");
        }
    }
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "cuptiActivityDisable");
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "cuptiActivityFlushAll");
    std::vector<KernelSpan> records;
    records.swap(kernels());
    std::sort(records.begin(), records.end(), [](const KernelSpan& a, const KernelSpan& b) { return a.start < b.start; });
    return records;
}
} // namespace cupti

// L2 flush for the GPU only benchmark with a flushed L2: reads a buffer of twice the L2 size, which evicts every other
// line (writing back the dirty ones during the flush) and leaves L2 clean, so that the next kernel finds neither its
// input nor any previous result in L2, as on a GPU that runs other kernels in between
__global__ void flushL2Kernel(const int4* buffer, size_t count, int* sink) {
    int acc = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int4 v = __ldcg(buffer + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x7fffffff) {
        *sink = acc; // never true for the zeroed buffer, keeps the loads
    }
}

class L2Flusher {
public:
    explicit L2Flusher(cudaStream_t stream) : m_stream(stream) {
        int device = 0;
        gpuErrchk(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        gpuErrchk(cudaGetDeviceProperties(&prop, device));
        m_bytes = 2 * static_cast<size_t>(prop.l2CacheSize);
        m_blocks = prop.multiProcessorCount * 8;
        gpuErrchk(cudaMalloc(&m_buffer, m_bytes));
        gpuErrchk(cudaMemset(m_buffer, 0, m_bytes));
        gpuErrchk(cudaMalloc(&m_sink, sizeof(int)));
    }
    ~L2Flusher() {
        cudaFree(m_buffer);
        cudaFree(m_sink);
    }
    void flush() const {
        flushL2Kernel<<<m_blocks, 256, 0, m_stream>>>(m_buffer, m_bytes / sizeof(int4), m_sink);
    }
    size_t bytes() const { return m_bytes; }
private:
    cudaStream_t m_stream;
    int4* m_buffer{nullptr};
    int* m_sink{nullptr};
    size_t m_bytes{0};
    int m_blocks{0};
};

// Same iterations as runTwoExecutions, but each execution is timed by the sum of its kernels' GPU execution times.
// With a flusher, the L2 is flushed before every execution (each implementation here is a single kernel).
inline TwoExecutionsResult runTwoExecutionsGpuOnly(cudaStream_t stream, const int variableDimensionValue,
                                                   const std::function<void()>& firstRun,
                                                   const std::function<void()>& secondRun,
                                                   const L2Flusher* flusher = nullptr) {
    const std::function<void()> first = [&]() {
        if (flusher) flusher->flush();
        firstRun();
    };
    const std::function<void()> second = [&]() {
        if (flusher) flusher->flush();
        secondRun();
    };
    const size_t firstKernels = cupti::trace(stream, first, 1).size();
    const size_t secondKernels = cupti::trace(stream, second, 1).size();
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        first();
        gpuErrchk(cudaStreamSynchronize(stream));
        second();
        gpuErrchk(cudaStreamSynchronize(stream));
    }
    const auto records = cupti::trace(stream, [&]() {
        first();
        gpuErrchk(cudaStreamSynchronize(stream));
        second();
    }, ITERS);
    if (records.size() != (firstKernels + secondKernels) * ITERS) {
        throw std::runtime_error("GPU only benchmark: unexpected number of kernel records");
    }
    std::vector<float> firstTimes(ITERS), secondTimes(ITERS);
    double speedupSum = 0.;
    size_t idx = 0;
    for (int i = 0; i < ITERS; ++i) {
        uint64_t ns = 0;
        for (size_t j = 0; j < firstKernels; ++j, ++idx) {
            ns += records[idx].end - records[idx].start;
        }
        firstTimes[i] = static_cast<float>(ns * 1e-6);
        ns = 0;
        for (size_t j = 0; j < secondKernels; ++j, ++idx) {
            ns += records[idx].end - records[idx].start;
        }
        secondTimes[i] = static_cast<float>(ns * 1e-6);
        speedupSum += firstTimes[i] / secondTimes[i];
    }
    return {variableDimensionValue, computeStats(firstTimes), computeStats(secondTimes),
            static_cast<float>(speedupSum / ITERS)};
}

// <benchmarkName>_<ParArch>.csv, as FKL names its benchmark files
inline void writeTwoExecutionsCSV(const std::filesystem::path& outDir, const std::string& benchmarkName,
                                  const std::string& variableDimension, const std::string& firstLabel,
                                  const std::string& secondLabel, const std::vector<TwoExecutionsResult>& results) {
    std::filesystem::create_directories(outDir);
    const std::string fileName = benchmarkName + "_" + std::string(fk::toStrView(fk::defaultParArch)) + ".csv";
    std::ofstream file(outDir / fileName);
    file << variableDimension;
    for (const auto& label : {firstLabel, secondLabel}) {
        file << ", " << label << " MeanTime, " << label << " TimeVariance, " << label << " MaxTime, " << label
             << " MinTime";
    }
    file << ", Mean Speedup";
    for (const auto& label : {firstLabel, secondLabel}) {
        file << ", " << label << " MedianTime, " << label << " P10Time, " << label << " P90Time";
    }
    file << std::endl;
    for (const auto& r : results) {
        file << r.variableDimensionValue;
        for (const auto* s : {&r.first, &r.second}) {
            file << ", " << s->mean << ", " << s->variance << ", " << s->max << ", " << s->min;
        }
        file << ", " << r.meanSpeedup;
        for (const auto* s : {&r.first, &r.second}) {
            file << ", " << s->median << ", " << s->p10 << ", " << s->p90;
        }
        file << std::endl;
    }
}

} // namespace benchmark

#endif // FKL_VS_NPPDX_BENCHMARK_COMMON_H
