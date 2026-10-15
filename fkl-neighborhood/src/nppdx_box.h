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

// The NPPDx kernel of the neighborhood experiment (common/bench_neighborhood.h): NPPDx's operations on one tile of a
// CROP_W x CROP_H crop, in the order of the pipeline, NV12 ingest of the crop with the halo of the box, limited to full
// range, ColorConvert BT.601, BoxBlur<BOX_K, BOX_K>, then the write. The crop is given to NPPDx as the image, so its
// ingest replicates the edge of the crop into the halo, as every other implementation. One thread block per tile of a
// crop; blockIdx.z selects the crop. Two writes:
//   FLOAT_WRITE  the normalization (Mul, Sub, Div) and the float3 write, hand written as in fkl-vs-nppdx's NPPDx
//                kernel (nppdx_figure1.h): NPPDx has no float output and no such operations
//   8 bit        NPPDx's RGB24 exgest (rounded half up, clamped), no hand written pixel code
// and the tile and block configurations NPPDx supports for the crops, which the programs validate and time.

#ifndef BVF_BENCH_NPPDX_BOX_H
#define BVF_BENCH_NPPDX_BOX_H

#include <fused_kernel/core/utils/utils.h> // gpuErrchk

#include <nppdx.hpp>

#include "bench_neighborhood.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace bench::neighborhood::nppdx_box {

constexpr int HALO = BOX_K / 2;
constexpr size_t PLANE_BYTES = PLANE_FLOATS; // an 8 bit RGB plane

// The normalization of the float write, ImageNet's on 8 bit RGB (common/bench_gpu.h's MUL, SUB, DIV)
struct Normalization {
    float mul[3];
    float sub[3];
    float div[3];
};

template <size_t N>
struct CropOrigins {
    int2 origin[N];
};

// Explicit 1D block size for the NPPDx operations (BlockDim<X>), or 0 for NPPDx's recommended configuration
template <unsigned BlockX>
using NppdxBlock = std::conditional_t<BlockX == 0, decltype(nppdx::Block()),
                                      decltype(nppdx::Block() + nppdx::BlockDim<BlockX>())>;

// The NPPDx operations of one tile configuration: TileX x TileY tiles of the crop with a halo of HALO pixels on every
// side, block size BlockX
template <int Arch, unsigned BlockX, unsigned TileX, unsigned TileY>
struct BoxOps {
    static constexpr nppdx::Halo4 halo = nppdx::Halo4::make_uniform(HALO);
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
    // The box consumes the halo: its output is the TileX x TileY tile, at (HALO, HALO) in its memory tile
    using Blur = decltype(nppdx::Function<nppdx::function::box_blur>() + nppdx::BoxBlur<BOX_K, BOX_K>() +
                          nppdx::TileSize<TileX, TileY>() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() +
                          NPPDX_MAKE_HALO(nppdx::CumulativeHalo, halo)() + nppdx::SM<Arch>() + NppdxBlock<BlockX>());
    using Exgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                            nppdx::OutputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<TileX, TileY>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() +
                            NPPDX_MAKE_HALO(nppdx::CumulativeHalo, nppdx::Halo4::make_empty())() + nppdx::SM<Arch>() +
                            NppdxBlock<BlockX>());
    using Tile = nppdx::input_storage_of_t<Ingest>;
    static_assert(std::is_same_v<Tile, nppdx::output_storage_of_t<Blur>>, "The box writes a tile like its input's");
    // The input tile and the blurred tile
    static constexpr size_t smem = nppdx::shared_memory::compute_total_tile_storage<Tile, Tile>();
    // A kernel runs every operation with one block size: the largest recommended one (with BlockDim<X> all are X)
    static constexpr unsigned blockFloatWrite = std::max({Ingest::block_dim.x, FromLimitedY::block_dim.x,
                                                          FromLimitedUV::block_dim.x, Convert::block_dim.x,
                                                          Blur::block_dim.x});
    static constexpr unsigned blockExgest = std::max(blockFloatWrite, Exgest::block_dim.x);
    static constexpr dim3 gridOf(size_t n) { return dim3{CROP_W / TileX, CROP_H / TileY, static_cast<unsigned>(n)}; }
};

template <typename Ops, size_t N, bool FLOAT_WRITE, unsigned TileX, unsigned TileY>
__global__ void nppdxBoxKernel(const uint8_t* y, const uint8_t* uv, const int pitch, const CropOrigins<N> crops,
                               const Normalization n, float* outF, uint8_t* out8) {
    extern __shared__ unsigned char smem[];
    auto tiles = nppdx::shared_memory::slice_into_tile_storage<typename Ops::Tile, typename Ops::Tile>(smem);
    auto inputTile = std::get<0>(tiles);
    auto blurredTile = std::get<1>(tiles);

    // ReadNV12 + Crop: the crop is ingested as a CROP_W x CROP_H NV12 image inside the frame, so the halo replicates
    // its edge; NPPDx takes the position of this block's tile in the crop from blockIdx.x and blockIdx.y
    const int2 o = crops.origin[blockIdx.z];
    const nppdx::ImageLayout<const uint8_t, 2> crop{{{y + static_cast<size_t>(o.y) * pitch + o.x,
                                                      uv + static_cast<size_t>(o.y / 2) * pitch + o.x}},
                                                    {{pitch, pitch}}};
    typename Ops::Ingest().execute(crop, inputTile, CROP_W, CROP_H);
    // YUVtoRGB, BT.601 limited range, in place on the shared memory tile
    typename Ops::FromLimitedY().execute(inputTile, CROP_W, CROP_H);
    typename Ops::FromLimitedUV().execute(inputTile, CROP_W, CROP_H);
    typename Ops::Convert().execute(inputTile, CROP_W, CROP_H);
    // Box, from the tile with its halo into the blurred tile
    typename Ops::Blur().execute(inputTile, blurredTile, CROP_W, CROP_H);
    if constexpr (FLOAT_WRITE) {
        __syncthreads();
        // Mul + Sub + Div + Write float3 (no NPPDx operations for these): each block writes its tile
        float* plane = outF + static_cast<size_t>(blockIdx.z) * PLANE_FLOATS;
        const int numThreads = blockDim.x * blockDim.y * blockDim.z;
        const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
        for (int idx = tid; idx < static_cast<int>(TileX * TileY); idx += numThreads) {
            const int tx = idx % TileX;
            const int ty = idx / TileX;
            float* pixel =
                plane + (static_cast<size_t>(blockIdx.y * TileY + ty) * CROP_W + blockIdx.x * TileX + tx) * 3;
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                pixel[c] = ((blurredTile.channel(c)(ty + HALO, tx + HALO) * n.mul[c]) - n.sub[c]) / n.div[c];
            }
        }
    } else {
        // NPPDx's 8 bit RGB write of the tile, into the crop's packed plane
        typename Ops::Exgest().execute(blurredTile, out8 + static_cast<size_t>(blockIdx.z) * PLANE_BYTES, CROP_W,
                                       CROP_H);
    }
}

// The kernel of one configuration: TileX x TileY tiles, block size BlockX (0: NPPDx's recommended one)
template <int Arch, size_t N, bool FLOAT_WRITE, unsigned BlockX, unsigned TileX, unsigned TileY>
struct NppdxBox {
    using Ops = BoxOps<Arch, BlockX, TileX, TileY>;
    static constexpr size_t smem = Ops::smem;
    static constexpr dim3 block{FLOAT_WRITE ? Ops::blockFloatWrite : Ops::blockExgest, 1, 1};
    static constexpr dim3 grid = Ops::gridOf(N);
    static constexpr auto kernel = nppdxBoxKernel<Ops, N, FLOAT_WRITE, TileX, TileY>;

    static void init() {
        if constexpr (smem > 48 * 1024) {
            gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        }
    }

    static void run(cudaStream_t stream, const uint8_t* y, const uint8_t* uv, int pitch, const CropOrigins<N>& crops,
                    const Normalization& n, float* outF, uint8_t* out8) {
        kernel<<<grid, block, smem, stream>>>(y, uv, pitch, crops, n, outF, out8);
        gpuErrchk(cudaGetLastError());
    }
};

// ------------------------------------------------------------------------------------------------
// The configurations: every candidate tile that NPPDx supports, with NPPDx's recommended block and with every block
// size of NPPDX_BLOCKS (as fkl-vs-nppdx)
// ------------------------------------------------------------------------------------------------

struct NppdxConfig {
    unsigned block, tileX, tileY;
};

constexpr std::array<unsigned, 5> NPPDX_BLOCKS{0, 64, 128, 256, 512};

struct TileSize {
    unsigned x, y;
};

// NPPDx tiles are made of 4x2 pixel cells and must split the crop
constexpr bool tileSplitsCrop(unsigned tileX, unsigned tileY) {
    return tileX > 0 && tileY > 0 && CROP_W % tileX == 0 && CROP_H % tileY == 0 && tileX % 4 == 0 && tileY % 2 == 0;
}

// Candidate tiles: the whole crop, then the longer side halved (the height on ties) while the tile still splits the
// crop, down to 64 pixels: 64x32, 32x32, 32x16, 16x16, 16x8, 8x8
constexpr unsigned MIN_TILE_PIXELS = 64;
struct TileList {
    std::array<TileSize, 16> tiles{};
    size_t count{0};
};
constexpr TileList candidateTiles() {
    TileList list;
    TileSize t{CROP_W, CROP_H};
    while (list.count < list.tiles.size() && tileSplitsCrop(t.x, t.y) && t.x * t.y >= MIN_TILE_PIXELS) {
        list.tiles[list.count++] = t;
        t = t.x > t.y ? TileSize{t.x / 2, t.y} : TileSize{t.x, t.y / 2};
    }
    return list;
}
constexpr TileList CANDIDATE_TILES = candidateTiles();

// Whether NPPDx runs the operations with TileX x TileY tiles on SM Arch (their tiles fit in shared memory)
template <int Arch, unsigned TileX, unsigned TileY, bool FLOAT_WRITE>
constexpr bool tileSupported() {
    using Ops = BoxOps<Arch, 0, TileX, TileY>;
    return nppdx::is_supported_v<typename Ops::Ingest, Arch> && nppdx::is_supported_v<typename Ops::Blur, Arch> &&
           (FLOAT_WRITE || nppdx::is_supported_v<typename Ops::Exgest, Arch>) &&
           Ops::smem <= commondx::sm_shared_memory_size(Arch);
}

template <int Arch, bool FLOAT_WRITE>
constexpr size_t supportedTileCount() {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return (size_t{0} + ... +
                (tileSupported<Arch, CANDIDATE_TILES.tiles[Is].x, CANDIDATE_TILES.tiles[Is].y, FLOAT_WRITE>() ? 1 : 0));
    }(std::make_index_sequence<CANDIDATE_TILES.count>{});
}

// Every supported tile with every block size, tiles from the largest
template <int Arch, bool FLOAT_WRITE>
constexpr auto nppdxConfigs() {
    std::array<NppdxConfig, supportedTileCount<Arch, FLOAT_WRITE>() * NPPDX_BLOCKS.size()> configs{};
    static_assert(configs.size() > 0, "NPPDx supports none of the candidate tiles of the crops");
    size_t next = 0;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        const auto add = [&](const TileSize t, const bool supported) {
            for (size_t b = 0; supported && b < NPPDX_BLOCKS.size(); ++b) {
                configs[next++] = NppdxConfig{NPPDX_BLOCKS[b], t.x, t.y};
            }
        };
        (add(CANDIDATE_TILES.tiles[Is],
             tileSupported<Arch, CANDIDATE_TILES.tiles[Is].x, CANDIDATE_TILES.tiles[Is].y, FLOAT_WRITE>()),
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

} // namespace bench::neighborhood::nppdx_box

#endif // BVF_BENCH_NPPDX_BOX_H
