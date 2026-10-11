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

// The NPPDx implementation of the Figure 1 pipeline: one CUDA kernel, one thread block per crop (or per tile of a
// crop), built from NPPDx operations (NV12 ingest, color conversion, resize), with the border, the normalization and
// the float3 write written by hand. Used by benchmark_figure1_bvf.cu and compile_time_pipeline.cu.

#ifndef FIGURE1_NPPDX_FIGURE1_H
#define FIGURE1_NPPDX_FIGURE1_H

#include <fused_kernel/core/utils/utils.h> // gpuErrchk

#include <nppdx.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace figure1 {

constexpr int CROP_W = 20;
constexpr int CROP_H = 10;
constexpr int RES_W = 64;
constexpr int RES_H = 32;
constexpr int BORDER_TOP = 16;
constexpr int BORDER_BOTTOM = 16;
constexpr int OUT_W = RES_W;
constexpr int OUT_H = RES_H + BORDER_TOP + BORDER_BOTTOM;

// Resize ratio as NPPDx expresses it: J input samples become K output samples (20 -> 64 and 10 -> 32)
constexpr unsigned RESIZE_J = 5;
constexpr unsigned RESIZE_K = 16;
static_assert(CROP_W * RESIZE_K == RES_W * RESIZE_J && CROP_H * RESIZE_K == RES_H * RESIZE_J);

// Sizes of one benchmark: CropW x CropH regions resized (J:K in both axes) to ResW x ResH, plus the border rows
template <int CropW, int CropH, int ResW, int ResH, unsigned J, unsigned K>
struct Geometry {
    static constexpr int cropW = CropW, cropH = CropH, resW = ResW, resH = ResH;
    static constexpr int outW = ResW, outH = ResH + BORDER_TOP + BORDER_BOTTOM;
    static constexpr unsigned resizeJ = J, resizeK = K;
    static_assert(CropW * K == ResW * J && CropH * K == ResH * J, "The resize must be J:K in both dimensions");
};
// The main benchmark: 20x10 crops resized to 64x32
using MainGeometry = Geometry<CROP_W, CROP_H, RES_W, RES_H, RESIZE_J, RESIZE_K>;

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

template <typename G, typename Ingest, typename Convert, typename Resize, typename InputTile, typename OutputTile,
          typename TempTile, size_t N>
__global__ void nppdxFigure1Kernel(const uint8_t* y, const uint8_t* uv, const int pitch,
                                   const CropOrigins<N> crops, const Normalization n, float* out) {
    extern __shared__ unsigned char smem[];
    auto tiles = nppdx::shared_memory::slice_into_tile_storage<InputTile, OutputTile, TempTile>(smem);
    auto inputTile = std::get<0>(tiles);
    auto resizedTile = std::get<1>(tiles);
    auto tempTile = std::get<2>(tiles);

    // ReadNV12 + Crop: the crop is ingested as a cropW x cropH NV12 image inside the frame; NPPDx takes
    // the position of this block's tile in the crop from blockIdx.x and blockIdx.y, and blockIdx.z is the crop
    const int2 o = crops.origin[blockIdx.z];
    const nppdx::ImageLayout<const uint8_t, 2> crop{{{y + static_cast<size_t>(o.y) * pitch + o.x,
                                                      uv + static_cast<size_t>(o.y / 2) * pitch + o.x}},
                                                    {{pitch, pitch}}};
    Ingest().execute(crop, inputTile, G::cropW, G::cropH);
    // YUVtoRGB, in place on the shared memory tile
    Convert().execute(inputTile, G::cropW, G::cropH);
    // Resize, J input pixels to K output pixels in both dimensions
    Resize().execute(inputTile, tempTile, resizedTile, G::cropW, G::cropH);
    __syncthreads();

    // AddBorder + Mul + Sub + Div + Write float3 (no NPPDx operations for these). Each block writes its
    // resized tile, plus the top border rows if it is in the first tile row and the bottom ones if in the last
    constexpr int OTW = OutputTile::ChannelSliceType::width;
    constexpr int OTH = OutputTile::ChannelSliceType::height;
    const int ox = blockIdx.x * OTW;
    const int oy = blockIdx.y * OTH;
    const int rowBegin = blockIdx.y == 0 ? 0 : BORDER_TOP + oy;
    const int rowEnd = blockIdx.y == gridDim.y - 1 ? G::outH : BORDER_TOP + oy + OTH;
    float* plane = out + static_cast<size_t>(blockIdx.z) * G::outW * G::outH * 3;
    const int numThreads = blockDim.x * blockDim.y * blockDim.z;
    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    for (int idx = tid; idx < (rowEnd - rowBegin) * OTW; idx += numThreads) {
        const int tx = idx % OTW;
        const int outY = rowBegin + idx / OTW;
        const bool inside = outY >= BORDER_TOP && outY < BORDER_TOP + G::resH;
        float* pixel = plane + (static_cast<size_t>(outY) * G::outW + ox + tx) * 3;
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

// TileX x TileY is the input tile of one thread block; its output tile is TileX x TileY * K / J
template <int Arch, size_t N, unsigned BlockX = 0, unsigned TileX = CROP_W, unsigned TileY = CROP_H,
          typename G = MainGeometry>
struct NppdxFigure1 {
    static constexpr unsigned J = G::resizeJ, K = G::resizeK;
    static_assert(G::cropW % TileX == 0 && G::cropH % TileY == 0 && TileX % J == 0 && TileY % J == 0,
                  "Tiles must split the crop and be multiples of the resize input ratio");
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
    static constexpr dim3 grid{G::resW / outTileX, G::resH / outTileY, N};

    static constexpr auto kernel = nppdxFigure1Kernel<G, Ingest, Convert, Resize, InputTile, OutputTile, TempTile, N>;

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

} // namespace figure1

#endif // FIGURE1_NPPDX_FIGURE1_H
