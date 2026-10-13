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

// The workload of every benchmark of this repository, in one place: the frame, the region sizes, the resize, border
// and output sizes, the batch sizes, and the bytes of the frame and the regions of every iteration, which are
// generated the same way on every platform and standard library.
//
// The regions change every iteration. Their sizes are 20x10, 32x16 and 48x24 by turns: region i of iteration t has
// size REGION_SIZES[(i + t) % 3], so every iteration mixes the three and every region cycles through them. A build
// with the CMake option BENCH_REGIONS set to one of the sizes (common/bench_geometry.cmake; the NPPDx benchmarks, whose
// kernels are compiled for one size) gives every region that size instead. Positions do not depend on the sizes: they
// are drawn for the largest size, so every build of every benchmark gets the same positions.
//
// Iteration t of a run uses the regions of iteration t: the warm-up iterations are 0 to W - 1 and the timed ones W to
// W + T - 1. RegionSets generates them all before the loop, from one seed; the regions of iteration t do not depend on
// how many iterations are generated.

#ifndef BVF_BENCH_GEOMETRY_H
#define BVF_BENCH_GEOMETRY_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef BENCH_REGIONS
#error "Define BENCH_REGIONS (common/bench_geometry.cmake does): -1 for the three sizes by turns, or 0, 1 or 2 for one"
#endif

namespace bench::geometry {

// One 3840x2160 NV12 frame
constexpr int FRAME_W = 3840;
constexpr int FRAME_H = 2160;

struct RegionSize {
    int width, height;
};

// The region sizes of the paper's workload, from the smallest
constexpr std::array<RegionSize, 3> PAPER_REGION_SIZES{{{20, 10}, {32, 16}, {48, 24}}};
// The sizes of this build: the three by turns (BENCH_REGIONS = -1), or one of them (its index)
constexpr int REGION_SIZE_INDEX = BENCH_REGIONS;
static_assert(REGION_SIZE_INDEX >= -1 && REGION_SIZE_INDEX < static_cast<int>(PAPER_REGION_SIZES.size()),
              "BENCH_REGIONS is -1 or the index of one of PAPER_REGION_SIZES");
constexpr size_t REGION_SIZE_COUNT = REGION_SIZE_INDEX < 0 ? PAPER_REGION_SIZES.size() : 1;
constexpr std::array<RegionSize, REGION_SIZE_COUNT> REGION_SIZES = []() {
    std::array<RegionSize, REGION_SIZE_COUNT> sizes{};
    for (size_t s = 0; s < REGION_SIZE_COUNT; ++s) {
        sizes[s] = PAPER_REGION_SIZES[REGION_SIZE_INDEX < 0 ? s : static_cast<size_t>(REGION_SIZE_INDEX)];
    }
    return sizes;
}();
constexpr bool MIXED_SIZES = REGION_SIZE_COUNT > 1;

// The size of region i of iteration t
constexpr RegionSize regionSize(size_t t, size_t i) { return REGION_SIZES[(i + t) % REGION_SIZE_COUNT]; }

// The largest region of the paper's workload: every region fits in it, positions are drawn for it, and the buffers of
// the pipelines that store regions have planes of this size
constexpr int MAX_REGION_W = 48;
constexpr int MAX_REGION_H = 24;

// Every region is resized (bilinear, aspect ratio ignored) to RES_W x RES_H and letterboxed to a square OUT_W x OUT_H
// with black rows on top and bottom
constexpr int RES_W = 64;
constexpr int RES_H = 32;
constexpr int BORDER_TOP = (RES_W - RES_H) / 2;
constexpr int BORDER_BOTTOM = RES_W - RES_H - BORDER_TOP;
constexpr int OUT_W = RES_W;
constexpr int OUT_H = RES_H + BORDER_TOP + BORDER_BOTTOM;
// Floats of one output plane (float3)
constexpr size_t OUT_FLOATS = static_cast<size_t>(OUT_W) * OUT_H * 3;
// Batch sizes of every benchmark; batch N uses the first N regions of every iteration
constexpr std::array<size_t, 5> BATCHES{1, 4, 16, 50, 100};
constexpr size_t MAX_BATCH = BATCHES.back();
constexpr unsigned FRAME_SEED = 1;
constexpr unsigned REGION_SEED = 42;
// Regions stay at least this far from the frame edges, so that a neighborhood operation (a convolution or mask of up
// to 17x17) can be inserted between the color conversion and the crop without moving them or clamping at the edges
constexpr int REGION_MARGIN = 8;

static_assert([]() {
    for (const RegionSize s : PAPER_REGION_SIZES) {
        if (s.width <= 0 || s.height <= 0 || s.width % 2 || s.height % 2 || s.width > MAX_REGION_W ||
            s.height > MAX_REGION_H) {
            return false;
        }
    }
    return true;
}(), "Region sizes are positive, even (whole NV12 chroma samples) and at most MAX_REGION_W x MAX_REGION_H");
static_assert(MAX_REGION_W + 2 * REGION_MARGIN <= FRAME_W && MAX_REGION_H + 2 * REGION_MARGIN <= FRAME_H,
              "The regions must fit in the frame with their margins");
static_assert(REGION_MARGIN % 2 == 0, "Region origins are even");
static_assert(RES_W >= RES_H, "The output is letterboxed to a square with rows on top and bottom");

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

struct Region {
    int x, y, width, height;
};

// The regions of iterations 0 to iterations() - 1, MAX_BATCH per iteration (batch N uses the first N), generated at
// construction from REGION_SEED: for every iteration in order and every region in order, an even x and an even y
// drawn uniformly so that a MAX_REGION_W x MAX_REGION_H box at (x, y) is at least REGION_MARGIN pixels from the frame
// edges, and the size regionSize(t, i). The regions of iteration t are the same whatever the number of iterations.
class RegionSets {
public:
    explicit RegionSets(size_t iterations) : m_iterations(iterations), m_regions(iterations * MAX_BATCH) {
        std::mt19937 gen(REGION_SEED);
        for (size_t t = 0; t < iterations; ++t) {
            for (size_t i = 0; i < MAX_BATCH; ++i) {
                const int x = 2 * uniformInt(gen, REGION_MARGIN / 2, (FRAME_W - MAX_REGION_W - REGION_MARGIN) / 2);
                const int y = 2 * uniformInt(gen, REGION_MARGIN / 2, (FRAME_H - MAX_REGION_H - REGION_MARGIN) / 2);
                const RegionSize s = regionSize(t, i);
                m_regions[t * MAX_BATCH + i] = Region{x, y, s.width, s.height};
            }
        }
    }
    size_t iterations() const { return m_iterations; }
    // Region i of iteration t
    const Region& at(size_t t, size_t i) const { return m_regions.at(t * MAX_BATCH + i); }

private:
    size_t m_iterations;
    std::vector<Region> m_regions;
};

// The batch sizes, separated by sep (by spaces: what the --batches mode of the GPU benchmarks prints for the scripts)
inline std::string batchList(const char* sep = " ") {
    std::ostringstream os;
    for (size_t i = 0; i < BATCHES.size(); ++i) {
        os << (i ? sep : "") << BATCHES[i];
    }
    return os.str();
}

// The region sizes of this build, as "20x10, 32x16, 48x24"
inline std::string regionSizeList() {
    std::ostringstream os;
    for (size_t s = 0; s < REGION_SIZE_COUNT; ++s) {
        os << (s ? ", " : "") << REGION_SIZES[s].width << "x" << REGION_SIZES[s].height;
    }
    return os.str();
}

// The workload, for the platform files of every benchmark
inline std::string describe() {
    std::ostringstream os;
    os << FRAME_W << "x" << FRAME_H << " NV12 frame (uniform random limited-range values, Y in [" << LUMA_MIN << ", "
       << LUMA_MAX << "], U and V in [" << CHROMA_MIN << ", " << CHROMA_MAX << "], seed " << FRAME_SEED
       << "), NV12->RGB BT.601 limited range, N in {" << batchList(", ") << "} regions per iteration, new every "
       << "iteration (warm-up and timed) and precomputed before the loop (seed " << REGION_SEED
       << ", batch N uses the first N of every iteration): ";
    if (MIXED_SIZES) {
        os << "sizes " << regionSizeList() << " by turns (region i of iteration t has the size (i + t) mod "
           << REGION_SIZE_COUNT << " of this list)";
    } else {
        os << "every region " << REGION_SIZES[0].width << "x" << REGION_SIZES[0].height;
    }
    os << ", at uniform random even positions at least " << REGION_MARGIN << " pixels from the frame edges (drawn for a "
       << MAX_REGION_W << "x" << MAX_REGION_H << " box, the same positions for every size), bilinear resize to " << RES_W
       << "x" << RES_H << " (aspect ratio ignored), " << BORDER_TOP << " black rows added on top and " << BORDER_BOTTOM
       << " at the bottom, output " << OUT_W << "x" << OUT_H << " per region (CMake option BENCH_REGIONS="
       << (MIXED_SIZES ? std::string("mixed") : regionSizeList()) << ")";
    return os.str();
}

// Statistics of the regions of iterations [first, first + count) for batch N: how many regions of each size, the mean
// area of a region and the smallest, mean and largest total area of the N regions of an iteration (source pixels)
struct RegionStats {
    size_t batch{0}, iterations{0};
    std::array<size_t, PAPER_REGION_SIZES.size()> perSize{};
    double meanRegionArea{0.}, meanIterationArea{0.};
    long long minIterationArea{0}, maxIterationArea{0};
};

inline RegionStats regionStats(const RegionSets& sets, size_t batch, size_t first, size_t count) {
    RegionStats s{batch, count};
    long long total = 0;
    for (size_t t = first; t < first + count; ++t) {
        long long area = 0;
        for (size_t i = 0; i < batch; ++i) {
            const Region& r = sets.at(t, i);
            area += static_cast<long long>(r.width) * r.height;
            for (size_t k = 0; k < PAPER_REGION_SIZES.size(); ++k) {
                s.perSize[k] += r.width == PAPER_REGION_SIZES[k].width && r.height == PAPER_REGION_SIZES[k].height;
            }
        }
        s.minIterationArea = t == first ? area : std::min(s.minIterationArea, area);
        s.maxIterationArea = t == first ? area : std::max(s.maxIterationArea, area);
        total += area;
    }
    s.meanIterationArea = count ? static_cast<double>(total) / static_cast<double>(count) : 0.;
    s.meanRegionArea = count ? s.meanIterationArea / static_cast<double>(batch) : 0.;
    return s;
}

// regions.csv of a results folder: the statistics of the timed iterations [first, first + count) for every batch size
inline void writeRegionStats(const std::string& path, const RegionSets& sets, size_t first, size_t count) {
    std::ofstream csv(path);
    csv << "N,iterations,first_iteration";
    for (const RegionSize s : PAPER_REGION_SIZES) {
        csv << ",regions_" << s.width << "x" << s.height;
    }
    csv << ",mean_region_area_px,mean_iteration_area_px,min_iteration_area_px,max_iteration_area_px\n";
    for (const size_t n : BATCHES) {
        const RegionStats s = regionStats(sets, n, first, count);
        csv << n << "," << count << "," << first;
        for (const size_t c : s.perSize) {
            csv << "," << c;
        }
        csv << "," << s.meanRegionArea << "," << s.meanIterationArea << "," << s.minIterationArea << ","
            << s.maxIterationArea << "\n";
    }
}

// The MAX_BATCH regions of iteration t as CSV (iteration, region, x, y, width, height): what the --regions mode of the
// GPU benchmarks prints, for the traffic model of the Nsight Compute profiles
inline std::string regionsCsv(const RegionSets& sets, size_t t) {
    std::ostringstream os;
    os << "iteration,region,x,y,width,height\n";
    for (size_t i = 0; i < MAX_BATCH; ++i) {
        const Region& r = sets.at(t, i);
        os << t << "," << i << "," << r.x << "," << r.y << "," << r.width << "," << r.height << "\n";
    }
    return os.str();
}

} // namespace bench::geometry

#endif // BVF_BENCH_GEOMETRY_H
