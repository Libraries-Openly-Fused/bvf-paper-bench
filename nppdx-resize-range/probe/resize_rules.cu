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

// NPPDx's own rules for the resize_probe.cu pipeline (ingest -> BoxBlur<5, 5> -> Resize<J, K, Method> -> exgest),
// evaluated at run time over every tile and ratio. Host code only: it calls the constexpr functions NPPDx uses to
// build the types and to check them, so it gives the tiles, the shared memory and the checks of any point without
// compiling it. sweep.py compiles a subset of the points with resize_probe.cu and compares the two.
//
// The checks of a point, as in NPPDx v0.1.2:
// - tile: is_supported_v of the ingest, the box blur and the resize descriptions (nppdx_execution.hpp), i.e.
//   is_valid_tile_v(TileSize) (cell_config.hpp): a multiple of the 4x2 cell and 3 float channels within the shared
//   memory of the SM in commondx's table (device_info_shared_memory.hpp);
// - downscale: the first static_assert of op_impl<Resize> (function_operation_implementation.hpp), 2 * tile * K >= J;
// - output: is_valid_tile_v of the resize's output tile (op_impl<Resize>, and is_supported_v of the exgest, whose
//   TileSize is the output tile);
// - intermediate: is_valid_tile_v of the intermediate tile, output width x (tile height + 2 resize halos).
//
// Usage:
//   resize_rules summary <max J and K> <output directory>   every coprime J:K up to the maximum, every method, SM
//                                                            890 and 900, every tile: rules_by_ratio.csv and
//                                                            rules_summary.csv
//   resize_rules points <points file> <output csv>          one line per point "sm method J K TX TY"

#include <nppdx.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

using nppdx::interpolation_method;
using uint2 = nppdx::uint2; // NPPDx's own vector type, not CUDA's

constexpr unsigned int BOX = 5;
constexpr unsigned int MAX_BLOCK_THREADS = 1024; // every CUDA device
// cudaDevAttrMaxSharedMemoryPerBlockOptin: measured on the RTX 4090 (resize_probe --launch prints it); for SM 900 the
// value of NPPDx's own table (227 KB), which is also the documented limit of the H100, not measured here
unsigned int sharedMemoryLimit(int sm) {
    return sm == 890 ? 101376u : nppdx::detail::sm_shared_memory_size(sm);
}

const char* methodName(interpolation_method m) {
    switch (m) {
    case interpolation_method::nearest: return "nearest";
    case interpolation_method::bilinear: return "bilinear";
    case interpolation_method::bicubic: return "bicubic";
    case interpolation_method::lanczos3: return "lanczos3";
    default: return "unknown";
    }
}

bool parseMethod(const std::string& s, interpolation_method& m) {
    for (auto c : {interpolation_method::nearest, interpolation_method::bilinear, interpolation_method::bicubic,
                   interpolation_method::lanczos3}) {
        if (s == methodName(c)) {
            m = c;
            return true;
        }
    }
    return false;
}

struct Point {
    int sm;
    interpolation_method method;
    unsigned int J, K, tx, ty;
    // NPPDx's values
    unsigned int boxHalo = 0, resizeHalo = 0;
    uint2 input{0, 0}, blurred{0, 0}, intermediate{0, 0}, output{0, 0};
    unsigned int smem = 0, block = 0;
    // NPPDx's checks
    bool tileOk = false, downscaleOk = false, outputOk = false, intermediateOk = false;

    bool compiles() const { return tileOk && downscaleOk && outputOk && intermediateOk; }
    bool fitsSmem() const { return smem <= sharedMemoryLimit(sm); }
    bool fitsBlock() const { return block <= MAX_BLOCK_THREADS; }
    bool dividesTile() const { return tx % J == 0 && ty % J == 0; }
    std::string failures() const {
        std::string f;
        auto add = [&](bool ok, const char* name) {
            if (!ok) f += (f.empty() ? "" : "+") + std::string(name);
        };
        add(tileOk, "tile");
        add(downscaleOk, "downscale");
        add(outputOk, "output");
        add(intermediateOk, "intermediate");
        return f.empty() ? "none" : f;
    }
};

unsigned int tileBytes(uint2 size) {
    return nppdx::shared_memory::compute_channels_storage(size.x, size.y, nppdx::detail::storage_num_channels,
                                                          sizeof(float), 4u);
}

Point evaluate(int sm, interpolation_method method, unsigned int J, unsigned int K, unsigned int tx, unsigned int ty) {
    Point p{sm, method, J, K, tx, ty};
    const nppdx::ResizeData resize{J, K, method};
    const nppdx::Halo4 resizeHalo = nppdx::compute_local_halo(resize);
    const nppdx::Halo4 boxHalo = nppdx::compute_local_halo(uint2{BOX, BOX});
    const nppdx::Halo4 totalHalo = boxHalo + resizeHalo;
    p.boxHalo = unsigned(boxHalo.left_top.x);
    p.resizeHalo = unsigned(resizeHalo.left_top.x);

    const uint2 tile{tx, ty};
    // The ingest and the box blur: memory tiles of the tile plus both halos (make_unary_storage_config)
    const nppdx::layout_props ingestLayout{tile, totalHalo, totalHalo};
    p.input = ingestLayout.memory_size();
    p.blurred = ingestLayout.memory_size();
    // The resize: memory halo of both, cumulative halo of the resize (fused_resize.cu)
    const nppdx::layout_props resizeLayout{tile, totalHalo, resizeHalo};
    p.tileOk = nppdx::detail::is_valid_tile_v(tile, sm, nppdx::packing_format::internal);
    p.downscaleOk = 2u * tx * K >= J && 2u * ty * K >= J;
    const nppdx::detail::storage_config storage = nppdx::compute_storage_config(resize, resizeLayout);
    p.output = storage.output.size;
    p.intermediate = storage.temp.size;
    p.outputOk = p.downscaleOk && nppdx::detail::is_valid_tile_v(p.output, sm, nppdx::packing_format::internal);
    p.intermediateOk =
        p.downscaleOk && nppdx::detail::is_valid_tile_v(p.intermediate, sm, nppdx::packing_format::internal);
    p.smem = tileBytes(p.input) + tileBytes(p.blurred) + tileBytes(p.intermediate) + tileBytes(p.output);
    // fused_resize.cu's block: the cells of the ingest's tile for a downscale, of the exgest's otherwise
    const uint2 cell = nppdx::detail::get_cell_config(sm, nppdx::packing_format::internal).size;
    p.block = nppdx::detail::get_number_of_cells(J > K ? tile : p.output, cell);
    return p;
}

std::string fmt(uint2 v) {
    return std::to_string(v.x) + "x" + std::to_string(v.y);
}

void writePointHeader(std::ostream& os) {
    os << "sm,method,J,K,tile,box_halo,resize_halo,input_storage,blurred_storage,intermediate_storage,"
          "output_storage,smem_bytes,block,predicted_compiles,failing_checks,shared_memory_limit,fits_smem,"
          "example_block_fits,j_divides_tile\n";
}

void writePoint(std::ostream& os, const Point& p) {
    os << p.sm << ',' << methodName(p.method) << ',' << p.J << ',' << p.K << ',' << p.tx << 'x' << p.ty << ','
       << p.boxHalo << ',' << p.resizeHalo << ',' << fmt(p.input) << ',' << fmt(p.blurred) << ','
       << fmt(p.intermediate) << ',' << fmt(p.output) << ',' << p.smem << ',' << p.block << ','
       << (p.compiles() ? "yes" : "no") << ',' << p.failures() << ',' << sharedMemoryLimit(p.sm) << ','
       << (p.fitsSmem() ? "yes" : "no") << ',' << (p.fitsBlock() ? "yes" : "no") << ','
       << (p.dividesTile() ? "yes" : "no") << '\n';
}

// Every tile NPPDx accepts as a TileSize on this SM: multiples of the 4x2 cell within the SM's shared memory
std::vector<uint2> validTiles(int sm) {
    std::vector<uint2> tiles;
    for (unsigned int x = 4;; x += 4) {
        if (!nppdx::detail::is_valid_tile_v(uint2{x, 2}, sm, nppdx::packing_format::internal)) break;
        for (unsigned int y = 2;; y += 2) {
            if (!nppdx::detail::is_valid_tile_v(uint2{x, y}, sm, nppdx::packing_format::internal)) break;
            tiles.push_back(uint2{x, y});
        }
    }
    return tiles;
}

// The best tile of a ratio at each level: the one with the least shared memory (ties: the smallest tile)
struct Best {
    bool found = false;
    Point point{};
    size_t count = 0;
    void offer(const Point& p) {
        ++count;
        if (!found || p.smem < point.smem || (p.smem == point.smem && p.tx * p.ty < point.tx * point.ty)) {
            point = p;
            found = true;
        }
    }
};

// The levels of a ratio. A point that compiles can run on the device if its shared memory fits the opt-in limit and
// its block has at most 1024 threads. Every shared-memory step of the pipeline is strided over the block's threads,
// so the block can also be smaller than fused_resize.cu's one thread per cell: the levels without "block" take any
// block of at most 1024 threads. NPPDx assumes that J divides the tile (op_impl<Resize>), which the levels with
// "j_divides" require.
struct Level {
    const char* name;
    bool smem, block, divides;
};
constexpr Level LEVELS[] = {
    {"compiles", false, false, false},
    {"compiles_j_divides", false, false, true},
    {"fits_smem", true, false, false},
    {"fits_smem_j_divides", true, false, true},
    {"fits_smem_example_block", true, true, false},
    {"fits_smem_example_block_j_divides", true, true, true},
};
constexpr int NUM_LEVELS = sizeof(LEVELS) / sizeof(LEVELS[0]);

bool atLevel(const Point& p, const Level& l) {
    return p.compiles() && (!l.smem || p.smem <= sharedMemoryLimit(p.sm)) &&
           (!l.block || p.block <= MAX_BLOCK_THREADS) && (!l.divides || p.dividesTile());
}

int summary(unsigned int maxJK, const std::string& dir) {
    const interpolation_method methods[] = {interpolation_method::nearest, interpolation_method::bilinear,
                                            interpolation_method::bicubic, interpolation_method::lanczos3};
    const int sms[] = {890, 900};
    std::ofstream byRatio(dir + "/rules_by_ratio.csv");
    byRatio << "sm,method,J,K,valid_tiles";
    for (const Level& l : LEVELS) {
        byRatio << ',' << l.name << "_tiles," << l.name << "_best_tile," << l.name << "_best_smem," << l.name
                << "_best_output," << l.name << "_best_block";
    }
    byRatio << '\n';
    std::ofstream summaryCsv(dir + "/rules_summary.csv");
    summaryCsv << "sm,method,direction,level,max_factor,J,K,tile,smem_bytes,output_storage,intermediate_storage,"
                  "block,shared_memory_limit,max_J_and_K_searched\n";

    for (int sm : sms) {
        const std::vector<uint2> tiles = validTiles(sm);
        std::printf("SM %d: %zu valid tiles, shared memory limit %u bytes\n", sm, tiles.size(), sharedMemoryLimit(sm));
        for (auto method : methods) {
            Point maxUp[NUM_LEVELS]{}, maxDown[NUM_LEVELS]{};
            bool hasUp[NUM_LEVELS]{}, hasDown[NUM_LEVELS]{};
            for (unsigned int J = 1; J <= maxJK; ++J) {
                for (unsigned int K = 1; K <= maxJK; ++K) {
                    if (std::gcd(J, K) != 1) continue;
                    Best best[NUM_LEVELS];
                    for (const uint2& t : tiles) {
                        const Point p = evaluate(sm, method, J, K, t.x, t.y);
                        if (!p.compiles()) continue;
                        for (int l = 0; l < NUM_LEVELS; ++l) {
                            if (atLevel(p, LEVELS[l])) best[l].offer(p);
                        }
                    }
                    byRatio << sm << ',' << methodName(method) << ',' << J << ',' << K << ',' << tiles.size();
                    for (int l = 0; l < NUM_LEVELS; ++l) {
                        byRatio << ',' << best[l].count;
                        if (best[l].found) {
                            const Point& p = best[l].point;
                            byRatio << ',' << p.tx << 'x' << p.ty << ',' << p.smem << ',' << fmt(p.output) << ','
                                    << p.block;
                        } else {
                            byRatio << ",,,,";
                        }
                    }
                    byRatio << '\n';
                    for (int l = 0; l < NUM_LEVELS; ++l) {
                        if (!best[l].found) continue;
                        const Point& p = best[l].point;
                        if (K > J && (!hasUp[l] || double(K) / J > double(maxUp[l].K) / maxUp[l].J)) {
                            maxUp[l] = p;
                            hasUp[l] = true;
                        }
                        if (J > K && (!hasDown[l] || double(J) / K > double(maxDown[l].J) / maxDown[l].K)) {
                            maxDown[l] = p;
                            hasDown[l] = true;
                        }
                    }
                }
            }
            for (int d = 0; d < 2; ++d) {
                for (int l = 0; l < NUM_LEVELS; ++l) {
                    const bool has = d == 0 ? hasUp[l] : hasDown[l];
                    if (!has) continue;
                    const Point& p = d == 0 ? maxUp[l] : maxDown[l];
                    const double f = d == 0 ? double(p.K) / p.J : double(p.J) / p.K;
                    summaryCsv << sm << ',' << methodName(method) << ',' << (d == 0 ? "upscale" : "downscale") << ','
                               << LEVELS[l].name << ',' << f << ',' << p.J << ',' << p.K << ',' << p.tx << 'x' << p.ty
                               << ',' << p.smem << ',' << fmt(p.output) << ',' << fmt(p.intermediate) << ','
                               << p.block << ',' << sharedMemoryLimit(sm) << ',' << maxJK << '\n';
                    std::printf("  %-8s %-9s %-34s %7.3fx  %u:%u tile %ux%u smem %u block %u\n", methodName(method),
                                d == 0 ? "upscale" : "downscale", LEVELS[l].name, f, p.J, p.K, p.tx, p.ty, p.smem,
                                p.block);
                }
            }
        }
    }
    return 0;
}

int points(const std::string& in, const std::string& out) {
    std::ifstream is(in);
    std::ofstream os(out);
    writePointHeader(os);
    std::string line;
    while (std::getline(is, line)) {
        std::istringstream ls(line);
        int sm;
        std::string m;
        unsigned int J, K, tx, ty;
        if (!(ls >> sm >> m >> J >> K >> tx >> ty)) continue;
        interpolation_method method;
        if (!parseMethod(m, method)) {
            std::fprintf(stderr, "unknown method %s\n", m.c_str());
            return 1;
        }
        writePoint(os, evaluate(sm, method, J, K, tx, ty));
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "summary") return summary(unsigned(std::atoi(argv[2])), argv[3]);
    if (argc == 4 && std::string(argv[1]) == "points") return points(argv[2], argv[3]);
    std::fprintf(stderr, "usage: %s summary <max J and K> <output directory> | points <points file> <output csv>\n",
                 argv[0]);
    return 1;
}
