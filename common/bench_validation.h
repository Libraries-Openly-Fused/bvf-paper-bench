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

// How the GPU benchmarks (common/bench_gpu.h and fkl-vs-nppdx) check an output against its reference: the errors in
// normalized units, the tolerances, and the rule that decides whether an output is within its tolerance.

#ifndef BVF_BENCH_VALIDATION_H
#define BVF_BENCH_VALIDATION_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bench::validation {

// Tolerance against a reference computed in float
constexpr float TOLERANCE = 1e-3f;
// Tolerance against a reference whose RGB is rounded to 8 bits: a pipeline that rounds RGB to 8 bits can round a value
// that lies on a half level differently from the CPU reference (different floating point contraction), which is one
// level of 8 bit RGB after the normalization (1 / 255 / 0.224, the largest of the three channels)
constexpr float TOLERANCE_8BIT = 1.f / 255.f / 0.224f + TOLERANCE;

// Errors of an output of interleaved RGB values against a reference
struct Errors {
    std::array<float, 3> maxPerChannel{0.f, 0.f, 0.f}; // INFINITY in a channel with a value that is not finite
    double mean{0.};                                   // over the finite errors
    size_t nonFinite{0};
    size_t beyond{0}; // finite errors larger than TOLERANCE
    size_t values{0};
    float max() const { return std::max({maxPerChannel[0], maxPerChannel[1], maxPerChannel[2]}); }
};

inline Errors compare(const std::vector<float>& output, const std::vector<float>& reference) {
    Errors e;
    e.values = output.size();
    double sum = 0.;
    for (size_t i = 0; i < output.size(); ++i) {
        const float err = std::abs(output[i] - reference[i]);
        if (!std::isfinite(err)) {
            e.nonFinite++;
            e.maxPerChannel[i % 3] = INFINITY;
            continue;
        }
        e.maxPerChannel[i % 3] = std::max(e.maxPerChannel[i % 3], err);
        e.beyond += err > TOLERANCE;
        sum += err;
    }
    const size_t finite = e.values - e.nonFinite;
    e.mean = finite ? sum / static_cast<double>(finite) : 0.;
    return e;
}

// Within the tolerance: every value finite, the largest error at most the tolerance, and at most one value in a
// thousand more than TOLERANCE away (ties on a half level of an 8 bit pipeline are rare, while a systematic one-level
// error, truncating instead of rounding for example, is not)
inline bool withinTolerance(const Errors& e, float tolerance) {
    return e.nonFinite == 0 && e.max() <= tolerance && e.beyond * 1000 <= e.values;
}

} // namespace bench::validation

#endif // BVF_BENCH_VALIDATION_H
