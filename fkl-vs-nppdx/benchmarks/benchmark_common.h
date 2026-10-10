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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
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
