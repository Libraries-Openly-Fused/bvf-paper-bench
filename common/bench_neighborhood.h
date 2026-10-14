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
// (rocm/) programs: fixed-size crops of the frame of bench_geometry.h, blurred and normalized,
//   ReadNV12 -> YUVtoRGB -> Crop (CROP_W x CROP_H) -> Box (K x K) -> Mul -> Sub -> Div -> Write
// without the resize and the letterbox of Figure 1: a pipeline that both FKL (MorphologyDPP with its BVF read and VF
// write) and NPPDx (its area step between the color conversion and the write) fuse into one kernel. The output is N
// planes of CROP_W x CROP_H fk::float3, one after the other.
//
// The regions are CROP_W x CROP_H crops at uniform random even positions at least geometry::REGION_MARGIN pixels from
// the frame edges, new every iteration, from geometry::REGION_SEED (cropRegionSets: RegionSets with a fixed size), the
// same for every implementation, program and machine. The box blur is the mean of the K x K window centered on the
// output value, per channel, replicating the edge of the crop; with two passes, the second blurs the result of the
// first, again replicating the edge of the crop.

#ifndef BVF_BENCH_NEIGHBORHOOD_H
#define BVF_BENCH_NEIGHBORHOOD_H

#include "bench_geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bench::neighborhood {

// The size of every crop, and of every output plane
constexpr int CROP_W = 64;
constexpr int CROP_H = 32;
constexpr geometry::RegionSize CROP_SIZE{CROP_W, CROP_H};
constexpr int PLANE_W = CROP_W;
constexpr int PLANE_H = CROP_H;
constexpr size_t PLANE_PIXELS = static_cast<size_t>(PLANE_W) * PLANE_H;
// Floats of one output plane (fk::float3)
constexpr size_t PLANE_FLOATS = PLANE_PIXELS * 3;

// The crops of iterations 0 to iterations - 1 (batch N uses the first N of every iteration)
inline geometry::RegionSets cropRegionSets(const size_t iterations) {
    return geometry::RegionSets(iterations, CROP_SIZE);
}

// Part (b), the comparison with the libraries: one pass of a BOX_K x BOX_K box on every batch size of geometry::BATCHES
constexpr int BOX_K = 5;

// Part (a), FKL's BoxBVF operation against MorphologyDPP: every box size, number of passes and batch size of these lists
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

// ------------------------------------------------------------------------------------------------
// Host references. Every reference blurs the RGB values and then normalizes them: a box is linear, and the
// normalization is affine and rounds, so the two do not commute exactly (unlike the minimum and the maximum).
// ------------------------------------------------------------------------------------------------

// How a library's box output is rounded to 8 bits, if it is: not at all (a float box), to nearest with ties up (NPPDx's
// 8 bit write, floor(v + 0.5)), or down (NPP's and OpenCV's 8 bit box filters, floor(sum / K^2) on 8 bit input)
enum class Rounding { None, HalfUp, Floor };

inline double round8bit(const double v, const Rounding r) {
    switch (r) {
        case Rounding::HalfUp: return std::clamp(std::floor(v + 0.5), 0.0, 255.0);
        case Rounding::Floor: return std::clamp(std::floor(v), 0.0, 255.0);
        default: return v;
    }
}

// The box blur of one plane: in and out are PLANE_W x PLANE_H, row-major, interleaved RGB (out must not alias in);
// passes passes of the mean of the k x k window centered on each value, per channel, each one replicating the edge of
// the plane it reads; sums in double
inline void hostBoxPlane(const double* in, double* out, const int k, const int passes) {
    std::vector<double> current(in, in + PLANE_FLOATS);
    std::vector<double> next(PLANE_FLOATS);
    const int anchor = k / 2;
    for (int pass = 0; pass < passes; ++pass) {
        for (int y = 0; y < PLANE_H; ++y) {
            for (int x = 0; x < PLANE_W; ++x) {
                for (int c = 0; c < 3; ++c) {
                    double sum = 0.0;
                    for (int my = 0; my < k; ++my) {
                        const int sy = std::clamp(y + my - anchor, 0, PLANE_H - 1);
                        for (int mx = 0; mx < k; ++mx) {
                            const int sx = std::clamp(x + mx - anchor, 0, PLANE_W - 1);
                            sum += current[(static_cast<size_t>(sy) * PLANE_W + sx) * 3 + c];
                        }
                    }
                    next[(static_cast<size_t>(y) * PLANE_W + x) * 3 + c] = sum / (static_cast<double>(k) * k);
                }
            }
        }
        current.swap(next);
    }
    std::copy(current.begin(), current.end(), out);
}

// The reference output of a batch: rgb holds planes planes of RGB values (PLANE_FLOATS each, one after the other: the
// crops, in float or rounded to 8 bits, with the conventions of the implementation); each plane is blurred
// (hostBoxPlane), rounded with r, and normalized in float as the pipelines do, ((v * mul) - sub) / div per channel
inline std::vector<float> boxReference(const std::vector<double>& rgb, const size_t planes, const int k,
                                       const int passes, const Rounding r, const float* mul, const float* sub,
                                       const float* div) {
    std::vector<float> out(planes * PLANE_FLOATS);
    std::vector<double> blurred(PLANE_FLOATS);
    for (size_t p = 0; p < planes; ++p) {
        hostBoxPlane(rgb.data() + p * PLANE_FLOATS, blurred.data(), k, passes);
        for (size_t i = 0; i < PLANE_FLOATS; ++i) {
            const int c = static_cast<int>(i % 3);
            const float v = static_cast<float>(round8bit(blurred[i], r));
            out[p * PLANE_FLOATS + i] = ((v * mul[c]) - sub[c]) / div[c];
        }
    }
    return out;
}

} // namespace bench::neighborhood

#endif // BVF_BENCH_NEIGHBORHOOD_H
