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

// What the two NPPDx implementations of the Figure 1 pipeline share (nppdx_figure1.h: one kernel with a hand written
// end; nppdx_npp_figure1.h: an NPPDx-only kernel followed by NPP): the NPPDx operations of one tile configuration
// (NV12 ingest of the crop, limited to full range, color conversion BT.601, bilinear resize, 8 bit RGB write) and the
// tile and block configurations NPPDx supports for the sizes of common/bench_geometry.h.

#ifndef FIGURE1_NPPDX_COMMON_H
#define FIGURE1_NPPDX_COMMON_H

#include <fused_kernel/core/utils/utils.h> // gpuErrchk

#include <nppdx.hpp>

#include "bench_geometry.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace figure1 {

using namespace bench::geometry;

// ImageNet normalization of 8 bit RGB
struct Normalization {
    float mul[3]{1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    float sub[3]{0.485f, 0.456f, 0.406f};
    float div[3]{0.229f, 0.224f, 0.225f};
};

template <size_t N>
struct CropOrigins {
    int2 origin[N];
};

// Explicit 1D block size for the NPPDx operations (BlockDim<X>), or 0 for NPPDx's recommended configuration
template <unsigned BlockX>
using NppdxBlock = std::conditional_t<BlockX == 0, decltype(nppdx::Block()), decltype(nppdx::Block() + nppdx::BlockDim<BlockX>())>;

// NPPDx tiles are made of 4x2 pixel cells, and the resize maps whole groups of J source pixels, so a tile is a multiple
// of these steps; it must also split the crop exactly
constexpr unsigned TILE_STEP_X = std::lcm(4u, RESIZE_J);
constexpr unsigned TILE_STEP_Y = std::lcm(2u, RESIZE_J);
constexpr bool tileSplitsCrop(unsigned tileX, unsigned tileY) {
    return tileX > 0 && tileY > 0 && CROP_W % tileX == 0 && CROP_H % tileY == 0 && tileX % TILE_STEP_X == 0 &&
           tileY % TILE_STEP_Y == 0;
}

// The NPPDx operations before the output for TileX x TileY input tiles (output tiles TileX x TileY * K / J) and block
// size BlockX, and the shared memory of their tiles
template <int Arch, unsigned BlockX, unsigned TileX, unsigned TileY>
struct NppdxOps {
    static constexpr unsigned J = RESIZE_J, K = RESIZE_K;
    static_assert(tileSplitsCrop(TileX, TileY), "Tiles must split the crop in 4x2 pixel cells and J pixel groups");
    using ResizePre = decltype(nppdx::Function<nppdx::function::resize>() +
                               nppdx::Resize<J, K, nppdx::interpolation_method::bilinear>() +
                               nppdx::TileSize<TileX, TileY>() + nppdx::SM<Arch>() + NppdxBlock<BlockX>());
    static constexpr auto halo = ResizePre::local_halo;
    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::nv12>() + nppdx::TileSize<TileX, TileY>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() + NPPDX_MAKE_HALO(nppdx::CumulativeHalo, halo)() +
                            nppdx::SM<Arch>() + NppdxBlock<BlockX>());
    // NPPDx's color conversion is full range; limited range input is first expanded with NPPDx's own range operations
    using FromLimitedY = decltype(nppdx::Function<nppdx::function::affine_channel_map>() +
                                  nppdx::FromLimitedLuma<nppdx::bit_depth::bpp_8u>() + nppdx::TileSize<TileX, TileY>() +
                                  NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() +
                                  NPPDX_MAKE_HALO(nppdx::CumulativeHalo, halo)() + nppdx::SM<Arch>() +
                                  NppdxBlock<BlockX>());
    using FromLimitedUV = decltype(nppdx::Function<nppdx::function::affine_channel_map>() +
                                   nppdx::FromLimitedChroma<nppdx::bit_depth::bpp_8u>() +
                                   nppdx::TileSize<TileX, TileY>() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() +
                                   NPPDX_MAKE_HALO(nppdx::CumulativeHalo, halo)() + nppdx::SM<Arch>() +
                                   NppdxBlock<BlockX>());
    using Convert = decltype(nppdx::Function<nppdx::function::color_convert>() +
                             nppdx::ColorConvert<nppdx::color_space::yuv_bt601, nppdx::color_space::rgb,
                                                 nppdx::bit_depth::bpp_8u, nppdx::bit_depth::bpp_8u>() +
                             nppdx::TileSize<TileX, TileY>() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() +
                             NPPDX_MAKE_HALO(nppdx::CumulativeHalo, halo)() + nppdx::SM<Arch>() + NppdxBlock<BlockX>());
    using Resize = decltype(ResizePre() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() +
                            NPPDX_MAKE_HALO(nppdx::CumulativeHalo, halo)());
    using InputTile = nppdx::input_storage_of_t<Ingest>;
    using OutputTile = nppdx::output_storage_of_t<Resize>;
    using TempTile = nppdx::temp_storage_of_t<Resize>;
    static constexpr unsigned outTileX = OutputTile::ChannelSliceType::width;
    static constexpr unsigned outTileY = OutputTile::ChannelSliceType::height;
    static_assert(outTileX * J == TileX * K && outTileY * J == TileY * K, "Unexpected output tile size");
    static_assert(nppdx::is_supported_v<Ingest, Arch> && nppdx::is_supported_v<FromLimitedY, Arch> &&
                  nppdx::is_supported_v<FromLimitedUV, Arch> && nppdx::is_supported_v<Convert, Arch> &&
                  nppdx::is_supported_v<Resize, Arch>);
    static constexpr size_t smem = nppdx::shared_memory::compute_total_tile_storage<InputTile, OutputTile, TempTile>();
    // The largest recommended block of the operations before the write: a kernel runs all of them with one block size
    // (with BlockDim<X> they are all X). Ingest alone would not be enough: its recommended block has one thread per 4x2
    // cell of the tile, but it loads the tile plus the resize halo, which has more cells.
    static constexpr unsigned blockBeforeWrite =
        std::max({Ingest::block_dim.x, FromLimitedY::block_dim.x, FromLimitedUV::block_dim.x, Convert::block_dim.x,
                  Resize::block_dim.x});
    // One block per output tile of every crop
    static constexpr dim3 gridOf(size_t n) { return dim3{RES_W / outTileX, RES_H / outTileY, static_cast<unsigned>(n)}; }
};

// ------------------------------------------------------------------------------------------------
// The NPPDx configurations of the benchmarks, derived from the sizes: every candidate tile that NPPDx supports, with
// NPPDx's recommended block and with every block size of NPPDX_BLOCKS. The benchmarks validate and time all of them.
// ------------------------------------------------------------------------------------------------

// One NPPDx configuration: the requested block size (BlockDim<block>, 0 for NPPDx's recommended configuration) and the
// input tile of one thread block
struct NppdxConfig {
    unsigned block, tileX, tileY;
};

constexpr std::array<unsigned, 5> NPPDX_BLOCKS{0, 64, 128, 256, 512};

struct Tile {
    unsigned x, y;
};

// Candidate tiles: the whole crop, then the longer side halved (the height on ties, the other side when the longer
// one cannot be halved) while the tile still splits the crop, down to 64 pixels (a crop of fewer pixels is its only
// candidate). 24 entries hold every halving of any crop that fits in the frame.
constexpr unsigned MIN_TILE_PIXELS = 64;
struct TileList {
    std::array<Tile, 24> tiles{};
    size_t count{0};
};
constexpr TileList candidateTiles() {
    static_assert(tileSplitsCrop(CROP_W, CROP_H), "NPPDx needs crops made of 4x2 pixel cells and J pixel groups");
    TileList list;
    Tile t{CROP_W, CROP_H};
    while (list.count < list.tiles.size() && (list.count == 0 || t.x * t.y >= MIN_TILE_PIXELS)) {
        list.tiles[list.count++] = t;
        const Tile halveX{t.x / 2, t.y}, halveY{t.x, t.y / 2};
        const Tile first = t.x > t.y ? halveX : halveY, second = t.x > t.y ? halveY : halveX;
        if (tileSplitsCrop(first.x, first.y)) {
            t = first;
        } else if (tileSplitsCrop(second.x, second.y)) {
            t = second;
        } else {
            break;
        }
    }
    return list;
}
constexpr TileList CANDIDATE_TILES = candidateTiles();

// Whether NPPDx runs the operations with TileX x TileY tiles on SM Arch, checked without instantiating an unsupported
// configuration (that is a compile error inside NPPDx): nppdx::is_supported_v of the ingest and the resize (their
// input tile fits in shared memory), with RgbWrite also of the 8 bit RGB write of the output tile, the checks the resize
// makes of its output tile and of its intermediate tile (output width x input height with the halo), and the shared
// memory of all the tiles of the kernel together
template <int Arch, unsigned TileX, unsigned TileY, bool RgbWrite>
constexpr bool tileSupported() {
    using IngestDescription = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                                       nppdx::InputFormat<nppdx::packing_format::nv12>() +
                                       nppdx::TileSize<TileX, TileY>() + nppdx::SM<Arch>());
    using ResizeDescription = decltype(nppdx::Function<nppdx::function::resize>() +
                                       nppdx::Resize<RESIZE_J, RESIZE_K, nppdx::interpolation_method::bilinear>() +
                                       nppdx::TileSize<TileX, TileY>() + nppdx::SM<Arch>());
    if constexpr (!nppdx::is_supported_v<IngestDescription, Arch> || !nppdx::is_supported_v<ResizeDescription, Arch>) {
        return false;
    } else {
        constexpr auto halo = decltype(ResizeDescription() + nppdx::Block())::local_halo;
        constexpr unsigned outX = TileX * RESIZE_K / RESIZE_J, outY = TileY * RESIZE_K / RESIZE_J;
        constexpr unsigned intermediateY = TileY + halo.left_top.y + halo.right_bottom.y;
        constexpr auto internal = nppdx::packing_format::internal;
        using ExgestDescription = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                                           nppdx::OutputFormat<nppdx::packing_format::rgb24>() +
                                           nppdx::TileSize<outX, outY>() + nppdx::SM<Arch>());
        if constexpr (!nppdx::detail::is_valid_tile_v(nppdx::uint2{outX, outY}, Arch, internal) ||
                      !nppdx::detail::is_valid_tile_v(nppdx::uint2{outX, intermediateY}, Arch, internal) ||
                      (RgbWrite && !nppdx::is_supported_v<ExgestDescription, Arch>)) {
            return false;
        } else {
            return NppdxOps<Arch, 0, TileX, TileY>::smem <= commondx::sm_shared_memory_size(Arch);
        }
    }
}

template <int Arch, bool RgbWrite>
constexpr size_t supportedTileCount() {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return (size_t{0} + ... +
                (tileSupported<Arch, CANDIDATE_TILES.tiles[Is].x, CANDIDATE_TILES.tiles[Is].y, RgbWrite>() ? 1 : 0));
    }(std::make_index_sequence<CANDIDATE_TILES.count>{});
}

// Every supported tile with every block size, tiles from the largest
template <int Arch, bool RgbWrite>
constexpr auto nppdxConfigs() {
    std::array<NppdxConfig, supportedTileCount<Arch, RgbWrite>() * NPPDX_BLOCKS.size()> configs{};
    static_assert(configs.size() > 0, "NPPDx supports none of the candidate tiles of these crops");
    size_t next = 0;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        const auto add = [&](const Tile t, const bool supported) {
            for (size_t b = 0; supported && b < NPPDX_BLOCKS.size(); ++b) {
                configs[next++] = NppdxConfig{NPPDX_BLOCKS[b], t.x, t.y};
            }
        };
        (add(CANDIDATE_TILES.tiles[Is],
             tileSupported<Arch, CANDIDATE_TILES.tiles[Is].x, CANDIDATE_TILES.tiles[Is].y, RgbWrite>()),
         ...);
    }(std::make_index_sequence<CANDIDATE_TILES.count>{});
    return configs;
}

// Calls f.template operator()<C>() with the configuration CONFIGS[index]
template <const auto& CONFIGS, typename F>
void withConfig(const size_t index, F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((Is == index ? f.template operator()<CONFIGS[Is]>() : void()), ...);
    }(std::make_index_sequence<CONFIGS.size()>{});
}

} // namespace figure1

#endif // FIGURE1_NPPDX_COMMON_H
