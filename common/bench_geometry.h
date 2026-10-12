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

// The workload of every benchmark of this repository, in one place: the frame, the crop, resize, border and output
// sizes, the batch sizes, and the bytes of the frame and the crop positions, which are generated the same way on every
// platform and standard library. The crop and resize sizes come from the CMake options BENCH_CROP_W, BENCH_CROP_H,
// BENCH_RESIZE_W and BENCH_RESIZE_H of common/bench_geometry.cmake (default 32x16 crops resized to 64x32).

#ifndef BVF_BENCH_GEOMETRY_H
#define BVF_BENCH_GEOMETRY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#if !defined(BENCH_CROP_W) || !defined(BENCH_CROP_H) || !defined(BENCH_RESIZE_W) || !defined(BENCH_RESIZE_H)
#error "Define BENCH_CROP_W, BENCH_CROP_H, BENCH_RESIZE_W and BENCH_RESIZE_H (common/bench_geometry.cmake does)"
#endif

namespace bench::geometry {

// One 3840x2160 NV12 frame
constexpr int FRAME_W = 3840;
constexpr int FRAME_H = 2160;
// N crops of CROP_W x CROP_H, each resized (bilinear) to RES_W x RES_H and letterboxed to a square OUT_W x OUT_H with
// black rows on top and bottom
constexpr int CROP_W = BENCH_CROP_W;
constexpr int CROP_H = BENCH_CROP_H;
constexpr int RES_W = BENCH_RESIZE_W;
constexpr int RES_H = BENCH_RESIZE_H;
constexpr int BORDER_TOP = (RES_W - RES_H) / 2;
constexpr int BORDER_BOTTOM = RES_W - RES_H - BORDER_TOP;
constexpr int OUT_W = RES_W;
constexpr int OUT_H = RES_H + BORDER_TOP + BORDER_BOTTOM;
// Floats of one output plane (float3)
constexpr size_t OUT_FLOATS = static_cast<size_t>(OUT_W) * OUT_H * 3;
// Batch sizes of every benchmark; batch N uses the first N crops of makeCrops()
constexpr std::array<size_t, 5> BATCHES{1, 4, 16, 50, 100};
constexpr size_t MAX_BATCH = BATCHES.back();
constexpr unsigned FRAME_SEED = 1;
constexpr unsigned CROP_SEED = 42;
// Crops stay at least this far from the frame edges, so that a neighborhood operation (a convolution or mask of up
// to 17x17) can be inserted between the color conversion and the crop without moving them or clamping at the edges
constexpr int CROP_MARGIN = 8;
// The resize ratio as NPPDx expresses it: J source pixels become K output pixels. NPPDx takes one ratio for both
// dimensions, so the crop and the resized crop have the same aspect ratio.
constexpr unsigned RESIZE_J = static_cast<unsigned>(CROP_W / std::gcd(CROP_W, RES_W));
constexpr unsigned RESIZE_K = static_cast<unsigned>(RES_W / std::gcd(CROP_W, RES_W));

static_assert(CROP_W > 0 && CROP_H > 0 && RES_W > 0 && RES_H > 0, "Sizes must be positive");
static_assert(CROP_W % 2 == 0 && CROP_H % 2 == 0, "NV12 crops have even sizes (whole chroma samples)");
static_assert(CROP_W * RESIZE_K == RES_W * RESIZE_J && CROP_H * RESIZE_K == RES_H * RESIZE_J,
              "The resize must have the same ratio in both dimensions (NPPDx's Resize<J, K>)");
static_assert(RES_W >= RES_H, "The output is letterboxed to a square with rows on top and bottom");
static_assert(CROP_W + 2 * CROP_MARGIN <= FRAME_W && CROP_H + 2 * CROP_MARGIN <= FRAME_H,
              "The crops must fit in the frame with their margins");
static_assert(CROP_MARGIN % 2 == 0, "Crop origins are even");

// Uniform integer in [lo, hi] from a 32 bit mt19937, with Lemire's multiply and shift and rejection. The standard fixes
// the output of std::mt19937 but not the algorithm of std::uniform_int_distribution, which differs between libstdc++,
// libc++ and MSVC's STL; this one is written out so that every platform gets the same numbers. It is the algorithm
// libstdc++ uses for these ranges, so on Linux with GCC the values are those of std::uniform_int_distribution.
inline int uniformInt(std::mt19937& gen, int lo, int hi) {
    const uint32_t range = static_cast<uint32_t>(hi - lo) + 1u;
    uint64_t product = static_cast<uint64_t>(static_cast<uint32_t>(gen())) * range;
    uint32_t low = static_cast<uint32_t>(product);
    if (low < range) {
        const uint32_t threshold = (0u - range) % range;
        while (low < threshold) {
            product = static_cast<uint64_t>(static_cast<uint32_t>(gen())) * range;
            low = static_cast<uint32_t>(product);
        }
    }
    return lo + static_cast<int>(product >> 32);
}

// Every benchmark converts NV12 to RGB with BT.601 in limited (video) range, the only NV12 conversion that every
// library offers (OpenCV's CPU cvtColor has no other); its legal values are Y in [16, 235] and U, V in [16, 240]
constexpr int LUMA_MIN = 16, LUMA_MAX = 235, CHROMA_MIN = 16, CHROMA_MAX = 240;

// The synthetic NV12 frame: uniform random legal limited-range values (seed FRAME_SEED) in row-major order, FRAME_H
// luma rows in [LUMA_MIN, LUMA_MAX] and then FRAME_H / 2 rows of interleaved chroma in [CHROMA_MIN, CHROMA_MAX], as in
// real video, so that no library has to handle values outside the standard (OpenCV clamps Y below 16, the others do
// not). Calls set(x, y, value) for every byte.
template <typename Set>
void generateFrame(Set&& set) {
    std::mt19937 gen(FRAME_SEED);
    for (int y = 0; y < FRAME_H * 3 / 2; ++y) {
        const bool luma = y < FRAME_H;
        for (int x = 0; x < FRAME_W; ++x) {
            const int value = luma ? uniformInt(gen, LUMA_MIN, LUMA_MAX) : uniformInt(gen, CHROMA_MIN, CHROMA_MAX);
            set(x, y, static_cast<uint8_t>(value));
        }
    }
}

struct CropRect {
    int x, y, width, height;
};

// MAX_BATCH crops of CROP_W x CROP_H at uniform random positions (seed CROP_SEED); batch N uses the first N. The
// origins are even, so that every library reads the same NV12 chroma samples, and at least CROP_MARGIN pixels from the
// frame edges.
inline std::array<CropRect, MAX_BATCH> makeCrops() {
    std::mt19937 gen(CROP_SEED);
    std::array<CropRect, MAX_BATCH> crops{};
    for (auto& crop : crops) {
        const int x = 2 * uniformInt(gen, CROP_MARGIN / 2, (FRAME_W - CROP_W - CROP_MARGIN) / 2);
        const int y = 2 * uniformInt(gen, CROP_MARGIN / 2, (FRAME_H - CROP_H - CROP_MARGIN) / 2);
        crop = CropRect{x, y, CROP_W, CROP_H};
    }
    return crops;
}

// The workload, for the platform files of every benchmark
inline std::string describe() {
    std::ostringstream os;
    os << FRAME_W << "x" << FRAME_H << " NV12 frame (uniform random limited-range values, Y in [" << LUMA_MIN << ", "
       << LUMA_MAX << "], U and V in [" << CHROMA_MIN << ", " << CHROMA_MAX << "], seed " << FRAME_SEED
       << "), NV12->RGB BT.601 limited range, N in {";
    for (size_t i = 0; i < BATCHES.size(); ++i) {
        os << (i ? ", " : "") << BATCHES[i];
    }
    os << "} crops of " << CROP_W << "x" << CROP_H << " at uniform random even positions at least " << CROP_MARGIN
       << " pixels from the frame edges (seed " << CROP_SEED << ", the first N of one sequence), bilinear resize to "
       << RES_W << "x" << RES_H << " (ratio " << RESIZE_J << ":" << RESIZE_K << "), " << BORDER_TOP
       << " black rows added on top and " << BORDER_BOTTOM << " at the bottom, output " << OUT_W << "x" << OUT_H
       << " per crop (CMake options BENCH_CROP_W=" << CROP_W << " BENCH_CROP_H=" << CROP_H
       << " BENCH_RESIZE_W=" << RES_W << " BENCH_RESIZE_H=" << RES_H << ")";
    return os.str();
}

} // namespace bench::geometry

#endif // BVF_BENCH_GEOMETRY_H
