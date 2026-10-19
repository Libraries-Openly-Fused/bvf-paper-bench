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

// The constants of the Figure 1 pipeline and the conventions of its references, in plain C++ (no FKL), for the code
// that compares a library with FKL without including FKL: the CPU benchmarks on common/bench_cpu.h (FKL's headers are
// included by one translation unit only, common/bench_fkl_cpu.cpp) and the PyTorch side of fkl-vs-pytorch's GPU
// program.

#ifndef BVF_BENCH_PIPELINE_H
#define BVF_BENCH_PIPELINE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bench::pipeline {

// The ImageNet normalization of 8 bit RGB at the end of the pipeline: (x * MUL - SUB) / DIV
constexpr float MUL[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
constexpr float SUB[3] = {0.485f, 0.456f, 0.406f};
constexpr float DIV[3] = {0.229f, 0.224f, 0.225f};

// FKL's NV12 -> RGB conversion (BT.601 limited range) as a matrix and offsets, for the libraries that convert with
// their own operations: (R, G, B) = matrix * (Y - lumaOffset, U - chromaOffset, V - chromaOffset), row c of the
// matrix giving channel c
struct Conversion {
    float matrix[3][3];
    float lumaOffset, chromaOffset;
};

// The conventions of a reference of the pipeline (common/bench_reference.h):
//   Fkl           FKL's bilinear sampling, src = dst * (in / out) (fused_kernel's resize.h), float RGB
//   PixelCenters  the pixel-center sampling of PyTorch's interpolate with align_corners=false and of OpenCV's resize,
//                 src = (dst + 0.5) * (in / out) - 0.5, float RGB
//   OpenCV        OpenCV's CPU pipeline: pixel-center sampling, Y clamped to at least 16 before the conversion and the
//                 RGB rounded to 8 bits (OpenCV's cvtColor writes 8 bit RGB)
// In all of them, the source coordinate is clamped to [0, size - 1] and the second sample of each axis is the next
// pixel, or the last one at the edge.
enum class Conventions { Fkl, PixelCenters, OpenCV };

// The largest absolute difference between two outputs, INFINITY if one of them is not finite or their sizes differ
inline float maxAbsError(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    float e = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::abs(a[i] - b[i]);
        e = std::isfinite(d) ? std::max(e, d) : INFINITY;
    }
    return e;
}

} // namespace bench::pipeline

#endif // BVF_BENCH_PIPELINE_H
