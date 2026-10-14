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

// The workload of the neighborhood experiment (Experiment 3), shared by the CUDA (fkl-neighborhood/) and the ROCm
// (rocm/) programs: the Figure 1 pipeline with a morphology stage after the resize and without the letterbox,
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> Morph (K x K) -> Mul -> Sub -> Div -> Write
// on the frame, regions and batch sizes of bench_geometry.h. Every region is resized to PLANE_W x PLANE_H, and the
// output is N planes of PLANE_W x PLANE_H fk::float3, one after the other.
//
// The morphology is an erosion (one pass, minimum) or an opening (two passes: minimum, then maximum) with a K x K
// rectangular mask, anchored at its center. Every pass replicates the border of the PLANE_W x PLANE_H image it reads,
// including the second pass of the opening, which reads the result of the first one. The minimum and the maximum are
// taken per channel.

#ifndef BVF_BENCH_NEIGHBORHOOD_H
#define BVF_BENCH_NEIGHBORHOOD_H

#include <fused_kernel/core/data/vector_types.h>

#include "bench_geometry.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace bench::neighborhood {

// The resized regions, without the letterbox of Figure 1 (geometry::OUT_W x OUT_H)
constexpr int PLANE_W = geometry::RES_W;
constexpr int PLANE_H = geometry::RES_H;
// Floats of one output plane (fk::float3)
constexpr size_t PLANE_FLOATS = static_cast<size_t>(PLANE_W) * PLANE_H * 3;

enum class MorphOp { Erode, Open };

// Passes of the operation: the erosion is one minimum, the opening a minimum and then a maximum
constexpr int passes(const MorphOp op) { return op == MorphOp::Erode ? 1 : 2; }

constexpr const char* name(const MorphOp op) { return op == MorphOp::Erode ? "erode" : "open"; }

// Part (b), the comparison with the libraries: both operations with a MORPH_K x MORPH_K mask, on every batch size of
// geometry::BATCHES
constexpr int MORPH_K = 5;
constexpr std::array<MorphOp, 2> MORPH_OPS{MorphOp::Erode, MorphOp::Open};

// Part (a), FKL's operations against MorphologyDPP: every mask size, number of passes (1: erode, 2: open) and batch
// size of these lists
constexpr std::array<int, 3> PART_A_KS{3, 5, 7};
constexpr std::array<int, 2> PART_A_PASSES{1, 2};
constexpr std::array<size_t, 2> PART_A_BATCHES{1, 16};

// The tiles FKL's MorphologyDPP is compiled for (template parameters): every one is validated and timed, and the
// fastest correct one is used and recorded, for each batch size
struct Tile {
    int width;
    int height;
};
constexpr std::array<Tile, 3> DPP_TILES{{{16, 8}, {32, 8}, {32, 16}}};

// Host reference of the morphology on one plane: in and out are PLANE_W x PLANE_H, row-major, interleaved fk::float3
// (out must not alias in); the passes of op with a k x k mask anchored at its center, each one replicating the border
// of the image it reads, per channel. Applied to the reference of the pipeline without the morphology, after the
// normalization (Mul, Sub, Div), it is exact: x -> (x * mul - sub) / div with mul and div > 0 is non-decreasing in
// IEEE round to nearest, so it commutes with the minimum and the maximum.
inline void hostMorphPlane(const fk::float3* in, fk::float3* out, const int k, const MorphOp op) {
    std::vector<fk::float3> current(in, in + static_cast<size_t>(PLANE_W) * PLANE_H);
    std::vector<fk::float3> next(current.size());
    const int anchor = k / 2;
    for (int pass = 0; pass < passes(op); ++pass) {
        const bool maximum = pass == 1;
        for (int y = 0; y < PLANE_H; ++y) {
            for (int x = 0; x < PLANE_W; ++x) {
                fk::float3 value{};
                bool first = true;
                for (int my = 0; my < k; ++my) {
                    const int sy = std::clamp(y + my - anchor, 0, PLANE_H - 1);
                    for (int mx = 0; mx < k; ++mx) {
                        const int sx = std::clamp(x + mx - anchor, 0, PLANE_W - 1);
                        const fk::float3 sample = current[static_cast<size_t>(sy) * PLANE_W + sx];
                        if (first) {
                            value = sample;
                            first = false;
                        } else if (maximum) {
                            value = {std::max(value.x, sample.x), std::max(value.y, sample.y),
                                     std::max(value.z, sample.z)};
                        } else {
                            value = {std::min(value.x, sample.x), std::min(value.y, sample.y),
                                     std::min(value.z, sample.z)};
                        }
                    }
                }
                next[static_cast<size_t>(y) * PLANE_W + x] = value;
            }
        }
        current.swap(next);
    }
    std::copy(current.begin(), current.end(), out);
}

// hostMorphPlane on every one of the planes of a batch, one after the other in in and out
inline void hostMorph(const fk::float3* in, fk::float3* out, const size_t planes, const int k, const MorphOp op) {
    const size_t pixels = static_cast<size_t>(PLANE_W) * PLANE_H;
    for (size_t plane = 0; plane < planes; ++plane) {
        hostMorphPlane(in + plane * pixels, out + plane * pixels, k, op);
    }
}

} // namespace bench::neighborhood

#endif // BVF_BENCH_NEIGHBORHOOD_H
