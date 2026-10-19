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

// Timing statistics and FKL's two-execution CSV format, shared by the benchmarks of this repository: the GPU harness
// (common/bench_gpu.h), the CPU harness (common/bench_cpu.h) and fkl-vs-pytorch's GPU program (added by pull request
// #18), which builds against LibTorch for CUDA or ROCm and so cannot include the CUDA-only common/bench_gpu.h. Plain C++,
// no GPU API.

#ifndef BVF_BENCH_TIMING_H
#define BVF_BENCH_TIMING_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace bench {

// Statistics of a set of timed executions, in microseconds
struct TimingStats {
    double mean{0.}, variance{0.}, max{0.}, min{0.}, median{0.}, p10{0.}, p90{0.};
};

inline double percentile(const std::vector<float>& sorted, double p) {
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

inline TimingStats computeStats(std::vector<float> ms) {
    TimingStats t;
    double sum = 0.;
    for (const float v : ms) {
        sum += v;
    }
    t.mean = sum / ms.size() * 1000.;
    double sumOfDiff = 0.;
    for (const float v : ms) {
        sumOfDiff += (v * 1000. - t.mean) * (v * 1000. - t.mean);
    }
    t.variance = ms.size() > 1 ? sumOfDiff / (ms.size() - 1) : 0.;
    std::sort(ms.begin(), ms.end());
    t.min = ms.front() * 1000.;
    t.max = ms.back() * 1000.;
    t.median = percentile(ms, 0.5) * 1000.;
    t.p10 = percentile(ms, 0.1) * 1000.;
    t.p90 = percentile(ms, 0.9) * 1000.;
    return t;
}

// FKL's two-execution CSV format (times in ms): one row per value of the first column
inline void writeTwoExecutionHeader(std::ostream& two, const std::string& firstColumn, const std::string& baseline,
                                    const std::string& fkl) {
    two << firstColumn;
    for (const std::string label : {baseline, fkl}) {
        two << ", " << label << " MeanTime, " << label << " TimeVariance, " << label << " MaxTime, " << label
            << " MinTime";
    }
    two << ", Mean Speedup";
    for (const std::string label : {baseline, fkl}) {
        two << ", " << label << " MedianTime, " << label << " P10Time, " << label << " P90Time";
    }
    two << std::endl;
}

inline void writeTwoExecutionRow(std::ostream& two, long long key, const TimingStats& tb, const TimingStats& tf,
                                 double speedup) {
    two << key;
    for (const TimingStats* t : std::array<const TimingStats*, 2>{&tb, &tf}) {
        two << ", " << t->mean / 1000. << ", " << t->variance / 1e6 << ", " << t->max / 1000. << ", " << t->min / 1000.;
    }
    two << ", " << speedup;
    for (const TimingStats* t : std::array<const TimingStats*, 2>{&tb, &tf}) {
        two << ", " << t->median / 1000. << ", " << t->p10 / 1000. << ", " << t->p90 / 1000.;
    }
    two << std::endl;
}

} // namespace bench

#endif // BVF_BENCH_TIMING_H
