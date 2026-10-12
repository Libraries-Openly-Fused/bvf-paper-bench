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

// The NPPDx implementation of the Figure 1 pipeline: one CUDA kernel, one or more thread blocks per crop (one per tile
// of the crop), built from NPPDx operations (NV12 ingest, color conversion, resize), with the border, the normalization
// and the float3 write written by hand. The sizes are those of common/bench_geometry.h. Also: the NPPDx configurations
// (tile and block size) that are valid for these sizes. Used by benchmark_figure1_bvf.cu and compile_time_pipeline.cu.

#ifndef FIGURE1_NPPDX_FIGURE1_H
#define FIGURE1_NPPDX_FIGURE1_H

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

template <typename Ingest, typename Convert, typename Resize, typename InputTile, typename OutputTile, typename TempTile,
          size_t N>
__global__ void nppdxFigure1Kernel(const uint8_t* y, const uint8_t* uv, const int pitch,
                                   const CropOrigins<N> crops, const Normalization n, float* out) {
    extern __shared__ unsigned char smem[];
    auto tiles = nppdx::shared_memory::slice_into_tile_storage<InputTile, OutputTile, TempTile>(smem);
    auto inputTile = std::get<0>(tiles);
    auto resizedTile = std::get<1>(tiles);
    auto tempTile = std::get<2>(tiles);

    // ReadNV12 + Crop: the crop is ingested as a CROP_W x CROP_H NV12 image inside the frame; NPPDx takes the position
    // of this block's tile in the crop from blockIdx.x and blockIdx.y, and blockIdx.z is the crop
    const int2 o = crops.origin[blockIdx.z];
    const nppdx::ImageLayout<const uint8_t, 2> crop{{{y + static_cast<size_t>(o.y) * pitch + o.x,
                                                      uv + static_cast<size_t>(o.y / 2) * pitch + o.x}},
                                                    {{pitch, pitch}}};
    Ingest().execute(crop, inputTile, CROP_W, CROP_H);
    // YUVtoRGB, in place on the shared memory tile
    Convert().execute(inputTile, CROP_W, CROP_H);
    // Resize, J input pixels to K output pixels in both dimensions
    Resize().execute(inputTile, tempTile, resizedTile, CROP_W, CROP_H);
    __syncthreads();

    // AddBorder + Mul + Sub + Div + Write float3 (no NPPDx operations for these). Each block writes its
    // resized tile, plus the top border rows if it is in the first tile row and the bottom ones if in the last
    constexpr int OTW = OutputTile::ChannelSliceType::width;
    constexpr int OTH = OutputTile::ChannelSliceType::height;
    const int ox = blockIdx.x * OTW;
    const int oy = blockIdx.y * OTH;
    const int rowBegin = blockIdx.y == 0 ? 0 : BORDER_TOP + oy;
    const int rowEnd = blockIdx.y == gridDim.y - 1 ? OUT_H : BORDER_TOP + oy + OTH;
    float* plane = out + static_cast<size_t>(blockIdx.z) * OUT_W * OUT_H * 3;
    const int numThreads = blockDim.x * blockDim.y * blockDim.z;
    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    for (int idx = tid; idx < (rowEnd - rowBegin) * OTW; idx += numThreads) {
        const int tx = idx % OTW;
        const int outY = rowBegin + idx / OTW;
        const bool inside = outY >= BORDER_TOP && outY < BORDER_TOP + RES_H;
        float* pixel = plane + (static_cast<size_t>(outY) * OUT_W + ox + tx) * 3;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const float value = inside ? resizedTile.channel(c)(outY - BORDER_TOP - oy, tx) : 0.f;
            pixel[c] = ((value * n.mul[c]) - n.sub[c]) / n.div[c];
        }
    }
}

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

// TileX x TileY is the input tile of one thread block; its output tile is TileX x TileY * K / J
template <int Arch, size_t N, unsigned BlockX, unsigned TileX, unsigned TileY>
struct NppdxFigure1 {
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
    using Convert = decltype(nppdx::Function<nppdx::function::color_convert>() +
                             nppdx::ColorConvert<nppdx::color_space::yuv_bt2020, nppdx::color_space::rgb,
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
    static_assert(nppdx::is_supported_v<Ingest, Arch> && nppdx::is_supported_v<Convert, Arch> &&
                  nppdx::is_supported_v<Resize, Arch>);
    static constexpr size_t smem = nppdx::shared_memory::compute_total_tile_storage<InputTile, OutputTile, TempTile>();
    // All the operations run in the same kernel, so it is launched with the largest of their block sizes
    // (with BlockDim<X> they are all X). Ingest alone would not be enough: its recommended block has one thread
    // per 4x2 cell of the tile, but it loads the tile plus the resize halo, which has more cells.
    static constexpr dim3 block{std::max({Ingest::block_dim.x, Convert::block_dim.x, Resize::block_dim.x}), 1, 1};
    static constexpr dim3 grid{RES_W / outTileX, RES_H / outTileY, N};

    static constexpr auto kernel = nppdxFigure1Kernel<Ingest, Convert, Resize, InputTile, OutputTile, TempTile, N>;

    static void init() {
        if constexpr (smem > 48 * 1024) {
            gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        }
    }

    static void run(cudaStream_t stream, const uint8_t* y, const uint8_t* uv, int pitch, const CropOrigins<N>& crops,
                    const Normalization& n, float* out) {
        kernel<<<grid, block, smem, stream>>>(y, uv, pitch, crops, n, out);
        gpuErrchk(cudaGetLastError());
    }
};

// ------------------------------------------------------------------------------------------------
// The NPPDx configurations of the benchmarks, derived from the sizes: every candidate tile that NPPDx supports, with
// NPPDx's recommended block and with every block size of NPPDX_BLOCKS. The benchmark validates and times all of them.
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
// one cannot be halved) while the tile still splits the crop, down to 64 pixels
constexpr unsigned MIN_TILE_PIXELS = 64;
struct TileList {
    std::array<Tile, 8> tiles{};
    size_t count{0};
};
constexpr TileList candidateTiles() {
    static_assert(tileSplitsCrop(CROP_W, CROP_H), "NPPDx needs crops made of 4x2 pixel cells and J pixel groups");
    TileList list;
    Tile t{CROP_W, CROP_H};
    while (list.count < list.tiles.size() && t.x * t.y >= MIN_TILE_PIXELS) {
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

// Whether NPPDx runs the pipeline with TileX x TileY tiles on SM Arch, checked without instantiating an unsupported
// configuration (that is a compile error inside NPPDx): nppdx::is_supported_v of the ingest and the resize (their
// input tile fits in shared memory), the checks the resize makes of its output tile and of its intermediate tile
// (output width x input height with the halo), and the shared memory of all the tiles of the kernel together
template <int Arch, unsigned TileX, unsigned TileY>
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
        if constexpr (!nppdx::detail::is_valid_tile_v(nppdx::uint2{outX, outY}, Arch, internal) ||
                      !nppdx::detail::is_valid_tile_v(nppdx::uint2{outX, intermediateY}, Arch, internal)) {
            return false;
        } else {
            return NppdxFigure1<Arch, 1, 0, TileX, TileY>::smem <= commondx::sm_shared_memory_size(Arch);
        }
    }
}

template <int Arch>
constexpr size_t supportedTileCount() {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return (size_t{0} + ... + (tileSupported<Arch, CANDIDATE_TILES.tiles[Is].x, CANDIDATE_TILES.tiles[Is].y>() ? 1 : 0));
    }(std::make_index_sequence<CANDIDATE_TILES.count>{});
}

// Every supported tile with every block size, tiles from the largest
template <int Arch>
constexpr auto nppdxConfigs() {
    std::array<NppdxConfig, supportedTileCount<Arch>() * NPPDX_BLOCKS.size()> configs{};
    static_assert(configs.size() > 0, "NPPDx supports none of the candidate tiles of these crops");
    size_t next = 0;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        const auto add = [&](const Tile t, const bool supported) {
            for (size_t b = 0; supported && b < NPPDX_BLOCKS.size(); ++b) {
                configs[next++] = NppdxConfig{NPPDX_BLOCKS[b], t.x, t.y};
            }
        };
        (add(CANDIDATE_TILES.tiles[Is], tileSupported<Arch, CANDIDATE_TILES.tiles[Is].x, CANDIDATE_TILES.tiles[Is].y>()),
         ...);
    }(std::make_index_sequence<CANDIDATE_TILES.count>{});
    return configs;
}

// The NPPDx kernel of one configuration
template <int Arch, size_t N, NppdxConfig C>
using NppdxKernel = NppdxFigure1<Arch, N, C.block, C.tileX, C.tileY>;

// Calls f.template operator()<C>() with the configuration CONFIGS[index]
template <const auto& CONFIGS, typename F>
void withConfig(const size_t index, F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((Is == index ? f.template operator()<CONFIGS[Is]>() : void()), ...);
    }(std::make_index_sequence<CONFIGS.size()>{});
}

} // namespace figure1

#endif // FIGURE1_NPPDX_FIGURE1_H
