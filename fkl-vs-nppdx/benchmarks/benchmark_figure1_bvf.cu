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

// Figure 1 pipeline of the Backwards Vertical Fusion paper, as one kernel in each library:
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// One 3840x2160 NV12 frame in device memory, N crops of 20x10, NV12->RGB BT.2020 full range,
// bilinear resize to 64x32, 16 black rows added on top and bottom, then (x * mul - sub) / div,
// written as float3 into a tensor of N planes of 64x64.
//
//   NPPDx: one CUDA kernel built from NPPDx operations, one thread block per crop (blockIdx.z):
//          NV12 ingest of the crop -> ColorConvert BT.2020 -> RGB -> Resize 5:16 bilinear in shared
//          memory, then AddBorder, Mul, Sub, Div and the float3 write written by hand in the same kernel,
//          because NPPDx has no border operation, its affine map only takes integer constants, and it
//          has no float output format.
//   FKL:   one executeOperations call (Backwards Vertical Fusion), the paper's Listing 2.
//
// --region-sweep runs the same chain with N = 16 regions of growing size and an exact 2x resize (NPPDx ratio 1:2),
// from 32x16 regions up to 960x524 ones that cover 97% of the frame (see namespace region below).

#include <benchmarks/benchmark_common.h>
#include "bench_platform.h"

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <nppdx.hpp>

#include <nvtx3/nvToolsExt.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace figure1 {

constexpr int FRAME_W = 3840;
constexpr int FRAME_H = 2160;
constexpr int CROP_W = 20;
constexpr int CROP_H = 10;
constexpr int RES_W = 64;
constexpr int RES_H = 32;
constexpr int BORDER_TOP = 16;
constexpr int BORDER_BOTTOM = 16;
constexpr int OUT_W = RES_W;
constexpr int OUT_H = RES_H + BORDER_TOP + BORDER_BOTTOM;
constexpr int MAX_BATCH = 100;
constexpr std::array<size_t, 5> BATCHES{1, 4, 16, 50, 100};
constexpr unsigned FRAME_SEED = 1;
constexpr unsigned CROP_SEED = 42;
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
constexpr Normalization norm{};

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Full, fk::ColorPrimitives::bt2020>;

struct Workload {
    cudaStream_t stream{};
    std::unique_ptr<fk::Stream> fkStream;
    std::unique_ptr<NV12Image> frameDevice;
    std::unique_ptr<NV12Image> frameHost;
    const uint8_t* y{nullptr};
    const uint8_t* uv{nullptr};
    int pitch{0};
    // Random positions snapped to even coordinates, so that both libraries read the same chroma samples:
    // NPPDx takes the crop's chroma at (x / 2, y / 2) relative to the crop origin, FKL relative to the frame
    std::array<fk::Rect, MAX_BATCH> crops{};
};

void initWorkload(Workload& w) {
    gpuErrchk(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    w.fkStream = std::make_unique<fk::Stream>(w.stream);
    w.frameDevice = std::make_unique<NV12Image>(FRAME_W, FRAME_H);
    w.frameHost = std::make_unique<NV12Image>(FRAME_W, FRAME_H, fk::MemType::Host);
    auto deviceData = w.frameDevice->getData();
    auto hostData = w.frameHost->getData();
    std::mt19937 gen(FRAME_SEED);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int y = 0; y < FRAME_H * 3 / 2; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const auto value = static_cast<uint8_t>(byte(gen));
            deviceData.at(x, y) = value;
            hostData.at(x, y) = value;
        }
    }
    w.frameDevice->upload(*w.fkStream);
    gpuErrchk(cudaStreamSynchronize(w.stream));
    const auto raw = deviceData.ptr();
    w.pitch = static_cast<int>(raw.dims.pitch);
    w.y = raw.data;
    w.uv = raw.data + static_cast<size_t>(FRAME_H) * w.pitch;

    std::mt19937 cropGen(CROP_SEED);
    std::uniform_int_distribution<int> distX(0, FRAME_W - CROP_W);
    std::uniform_int_distribution<int> distY(0, FRAME_H - CROP_H);
    for (auto& crop : w.crops) {
        const int x = distX(cropGen) & ~1;
        const int y = distY(cropGen) & ~1;
        crop = fk::Rect(x, y, CROP_W, CROP_H);
    }
}

template <size_t N>
std::array<fk::Rect, N> firstCrops(const Workload& w) {
    std::array<fk::Rect, N> crops{};
    std::copy_n(w.crops.begin(), N, crops.begin());
    return crops;
}

// ------------------------------------------------------------------------------------------------
// FKL: Backwards Vertical Fusion, one kernel
// ------------------------------------------------------------------------------------------------
template <size_t N, typename G = MainGeometry>
void runFKL(Workload& w, const std::array<fk::Rect, N>& crops, fk::Tensor<fk::float3>& out) {
    fk::executeOperations<fk::TransformDPP<>>(*w.fkStream,
        fk::ReadYUV<fk::PixelFormat::NV12>::build(*w.frameDevice),
        YUVToRGB::build(),
        fk::Crop<>::build(crops),
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(G::resW, G::resH)),
        fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, fk::float3{0.f, 0.f, 0.f}),
        fk::Mul<fk::float3>::build(fk::float3{norm.mul[0], norm.mul[1], norm.mul[2]}),
        fk::Sub<fk::float3>::build(fk::float3{norm.sub[0], norm.sub[1], norm.sub[2]}),
        fk::Div<fk::float3>::build(fk::float3{norm.div[0], norm.div[1], norm.div[2]}),
        fk::TensorWrite<fk::float3>::build(out.ptr()));
}

// ------------------------------------------------------------------------------------------------
// NPPDx: one kernel, one thread block per crop
// ------------------------------------------------------------------------------------------------
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

    static void run(const Workload& w, const CropOrigins<N>& crops, float* out) {
        kernel<<<grid, block, smem, w.stream>>>(w.y, w.uv, w.pitch, crops, norm, out);
        gpuErrchk(cudaGetLastError());
    }
};

// ------------------------------------------------------------------------------------------------
// Reference on FKL's CPU backend
// ------------------------------------------------------------------------------------------------
template <size_t N, typename G = MainGeometry>
std::vector<float> cpuReference(const Workload& w, const std::array<fk::Rect, N>& crops) {
    fk::Stream_<fk::ParArch::CPU> cpuStream;
    fk::Tensor<fk::float3> out(G::outW, G::outH, N, 1, fk::MemType::Host);
    fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
        fk::ReadYUV<fk::PixelFormat::NV12>::build(*w.frameHost),
        YUVToRGB::build(),
        fk::Crop<>::build(crops),
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(G::resW, G::resH)),
        fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, fk::float3{0.f, 0.f, 0.f}),
        fk::Mul<fk::float3>::build(fk::float3{norm.mul[0], norm.mul[1], norm.mul[2]}),
        fk::Sub<fk::float3>::build(fk::float3{norm.sub[0], norm.sub[1], norm.sub[2]}),
        fk::Div<fk::float3>::build(fk::float3{norm.div[0], norm.div[1], norm.div[2]}),
        fk::TensorWrite<fk::float3>::build(out.ptr()));
    cpuStream.sync();
    const float* data = reinterpret_cast<const float*>(out.ptr().data);
    return std::vector<float>(data, data + N * G::outW * G::outH * 3);
}

struct Error {
    float max{0.f};
    double mean{0.};
};

Error compare(const std::vector<float>& output, const std::vector<float>& reference) {
    Error e;
    for (size_t i = 0; i < output.size(); ++i) {
        const float d = std::abs(output[i] - reference[i]);
        e.max = std::isfinite(d) ? std::max(e.max, d) : INFINITY;
        e.mean += d;
    }
    e.mean /= static_cast<double>(output.size());
    return e;
}

std::vector<float> download(const float* data, size_t floats) {
    std::vector<float> host(floats);
    gpuErrchk(cudaMemcpy(host.data(), data, floats * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
}

// Host reference with selectable semantics, to find out what each library computes:
// optional clamp of RGB to [0, 255] after the color conversion, and the bilinear sampling convention
enum class Sampling { DstTimesScale, PixelCenters, NppMinusQuarter };

template <typename G = MainGeometry>
std::vector<float> hostVariant(const Workload& w, const fk::Rect& r, bool clampRGB, Sampling sampling) {
    constexpr int CW = G::cropW, CH = G::cropH, OW = G::outW, OH = G::outH;
    std::vector<float> out(static_cast<size_t>(OW) * OH * 3);
    std::vector<float> cropPixels(static_cast<size_t>(CW) * CH * 3);
    const auto crop = [&](int y, int x, int c) -> float& { return cropPixels[(static_cast<size_t>(y) * CW + x) * 3 + c]; };
    for (int y = 0; y < CH; ++y) {
        for (int x = 0; x < CW; ++x) {
            const fk::float3 rgb = YUVToRGB::exec(w.frameHost->readAt(fk::Point{static_cast<int>(r.x) + x, static_cast<int>(r.y) + y, 0}));
            const float c3[3] = {rgb.x, rgb.y, rgb.z};
            for (int c = 0; c < 3; ++c) {
                crop(y, x, c) = clampRGB ? std::min(255.f, std::max(0.f, c3[c])) : c3[c];
            }
        }
    }
    const float sx = static_cast<float>(CW) / G::resW, sy = static_cast<float>(CH) / G::resH;
    const auto map = [&](int d, float scale) {
        switch (sampling) {
            case Sampling::DstTimesScale: return d * scale;
            case Sampling::PixelCenters: return (d + 0.5f) * scale - 0.5f;
            default: return d * scale - 0.25f;
        }
    };
    for (int y = 0; y < OH; ++y) {
        for (int x = 0; x < OW; ++x) {
            const int ry = y - BORDER_TOP;
            float v[3] = {0.f, 0.f, 0.f};
            if (ry >= 0 && ry < G::resH) {
                const float fxs = std::min<float>(CW - 1, std::max(0.f, map(x, sx)));
                const float fys = std::min<float>(CH - 1, std::max(0.f, map(ry, sy)));
                const int x0 = static_cast<int>(fxs), y0 = static_cast<int>(fys);
                const int x1 = std::min(x0 + 1, CW - 1), y1 = std::min(y0 + 1, CH - 1);
                const float ax = fxs - x0, ay = fys - y0;
                for (int c = 0; c < 3; ++c) {
                    v[c] = (crop(y0, x0, c) * (1.f - ax) + crop(y0, x1, c) * ax) * (1.f - ay) +
                           (crop(y1, x0, c) * (1.f - ax) + crop(y1, x1, c) * ax) * ay;
                }
            }
            for (int c = 0; c < 3; ++c) {
                out[(static_cast<size_t>(y) * OW + x) * 3 + c] = ((v[c] * norm.mul[c]) - norm.sub[c]) / norm.div[c];
            }
        }
    }
    return out;
}

// Ingest + ColorConvert only: writes the whole input tile (with its halo) so the conversion can be checked alone
template <typename Ingest, typename Convert, typename InputTile>
__global__ void nppdxConvertOnlyKernel(const uint8_t* y, const uint8_t* uv, const int pitch, const int2 o, float* out) {
    extern __shared__ unsigned char smem[];
    auto tiles = nppdx::shared_memory::slice_into_tile_storage<InputTile>(smem);
    auto inputTile = std::get<0>(tiles);
    const nppdx::ImageLayout<const uint8_t, 2> crop{{{y + static_cast<size_t>(o.y) * pitch + o.x,
                                                      uv + static_cast<size_t>(o.y / 2) * pitch + o.x}},
                                                    {{pitch, pitch}}};
    Ingest().execute(crop, inputTile, CROP_W, CROP_H);
    Convert().execute(inputTile, CROP_W, CROP_H);
    __syncthreads();
    constexpr int TW = InputTile::ChannelSliceType::width, TH = InputTile::ChannelSliceType::height;
    for (int idx = threadIdx.x; idx < TW * TH; idx += blockDim.x) {
        for (int c = 0; c < 3; ++c) {
            out[idx * 3 + c] = inputTile.channel(c)(idx / TW, idx % TW);
        }
    }
}

template <unsigned BlockX>
void analyze(Workload& w) {
    using Nppdx = NppdxFigure1<NPPDX_SM, 1, BlockX>;
    Nppdx::init();
    std::cout << "Recommended block sizes: Ingest " << Nppdx::Ingest::block_dim.x << ", Convert "
              << Nppdx::Convert::block_dim.x << ", Resize " << Nppdx::Resize::block_dim.x << std::endl;
    std::cout << "NPPDx kernel: block " << Nppdx::block.x << "x" << Nppdx::block.y << "x" << Nppdx::block.z
              << ", dynamic shared memory " << Nppdx::smem << " bytes, resize halo (l,r,t,b) "
              << Nppdx::halo.left_top.x << "," << Nppdx::halo.right_bottom.x << "," << Nppdx::halo.left_top.y << ","
              << Nppdx::halo.right_bottom.y << std::endl;
    CropOrigins<1> origins{{int2{static_cast<int>(w.crops[0].x), static_cast<int>(w.crops[0].y)}}};
    float* out = nullptr;
    gpuErrchk(cudaMalloc(&out, OUT_W * OUT_H * 3 * sizeof(float)));
    Nppdx::run(w, origins, out);
    gpuErrchk(cudaStreamSynchronize(w.stream));
    const auto nppdx = download(out, OUT_W * OUT_H * 3);
    gpuErrchk(cudaFree(out));
    const char* names[] = {"dst*scale (FKL)", "pixel centers", "dst*scale-0.25 (NPP)"};
    for (bool clampRGB : {false, true}) {
        for (Sampling s : {Sampling::DstTimesScale, Sampling::PixelCenters, Sampling::NppMinusQuarter}) {
            const Error e = compare(nppdx, hostVariant(w, w.crops[0], clampRGB, s));
            std::cout << "NPPDx vs host reference, " << (clampRGB ? "RGB clamped to [0,255], " : "RGB not clamped,      ")
                      << names[static_cast<int>(s)] << ": max " << e.max << ", mean " << e.mean << std::endl;
        }
    }
    {
        using In = typename Nppdx::InputTile;
        constexpr int TW = In::ChannelSliceType::width, TH = In::ChannelSliceType::height;
        std::cout << "Input tile storage " << TW << "x" << TH << ", output tile storage "
                  << Nppdx::OutputTile::ChannelSliceType::width << "x" << Nppdx::OutputTile::ChannelSliceType::height
                  << ", temp tile storage " << Nppdx::TempTile::ChannelSliceType::width << "x"
                  << Nppdx::TempTile::ChannelSliceType::height << std::endl;
        float* tile = nullptr;
        gpuErrchk(cudaMalloc(&tile, TW * TH * 3 * sizeof(float)));
        constexpr size_t smemIn = nppdx::shared_memory::compute_total_tile_storage<In>();
        nppdxConvertOnlyKernel<typename Nppdx::Ingest, typename Nppdx::Convert, In>
            <<<1, Nppdx::block, smemIn, w.stream>>>(w.y, w.uv, w.pitch, origins.origin[0], tile);
        gpuErrchk(cudaStreamSynchronize(w.stream));
        const auto t = download(tile, TW * TH * 3);
        gpuErrchk(cudaFree(tile));
        // Compare tile pixel (tx, ty) with crop pixel (tx - hl, ty - ht), clamped to the crop
        const int hl = Nppdx::halo.left_top.x, ht = Nppdx::halo.left_top.y;
        float maxRaw = 0.f, maxClamped = 0.f;
        const fk::Rect& r = w.crops[0];
        for (int ty = 0; ty < TH; ++ty) {
            for (int tx = 0; tx < TW; ++tx) {
                const int cx = std::min(CROP_W - 1, std::max(0, tx - hl)), cy = std::min(CROP_H - 1, std::max(0, ty - ht));
                const fk::float3 rgb = YUVToRGB::exec(w.frameHost->readAt(fk::Point{static_cast<int>(r.x) + cx, static_cast<int>(r.y) + cy, 0}));
                const float c3[3] = {rgb.x, rgb.y, rgb.z};
                for (int c = 0; c < 3; ++c) {
                    const float v = t[(ty * TW + tx) * 3 + c];
                    maxRaw = std::max(maxRaw, std::abs(v - c3[c]));
                    maxClamped = std::max(maxClamped, std::abs(v - std::min(255.f, std::max(0.f, c3[c]))));
                }
            }
        }
        std::cout << "Ingest + ColorConvert tile vs FKL conversion (8 bit levels): max diff not clamped " << maxRaw
                  << ", clamped to [0,255] " << maxClamped << std::endl;
    }
    const auto ref = hostVariant(w, w.crops[0], false, Sampling::DstTimesScale);
    for (const int y : {0, 16, 17, 30, 47, 48}) {
        for (const int x : {0, 1, 31, 63}) {
            const size_t i = (y * OUT_W + x) * 3;
            std::cout << "  (" << x << "," << y << ") NPPDx " << nppdx[i] << " " << nppdx[i + 1] << " " << nppdx[i + 2]
                      << " | FKL reference " << ref[i] << " " << ref[i + 1] << " " << ref[i + 2] << std::endl;
        }
    }
}

// NPPDx configurations tried for every batch: block size (0 is NPPDx's recommended configuration) and input
// tile per thread block
struct NppdxConfig {
    unsigned block, tileX, tileY;
};
// NPPDx tiles must be multiples of its 4x2 pixel cells, and the 5:16 resize needs multiples of 5, so for a 20x10
// crop the only valid tile is the whole crop: one thread block per crop
constexpr std::array<NppdxConfig, 5> NPPDX_CONFIGS{{{0, 20, 10}, {64, 20, 10}, {128, 20, 10}, {256, 20, 10},
                                                    {512, 20, 10}}};
constexpr float TOLERANCE = 1e-3f;

// NPPDx configuration (index into NPPDX_CONFIGS) chosen for every batch size in the CPU + GPU benchmark, reused by
// the GPU only benchmark
std::map<size_t, size_t> chosenConfig;

struct BlockSweep {
    size_t batch{0};
    unsigned requestedBlock{0}, tileX{0}, tileY{0}, actualBlock{0}, blocksPerCrop{0};
    benchmark::TimeStats time;
    Error vsPixelCenters;
};

struct Validation {
    size_t batch{0};
    unsigned nppdxBlock{0}, tileX{0}, tileY{0};
    Error nppdxVsFkl, nppdxVsPixelCenters, fkl;
};

// NPPDx samples the bilinear resize at pixel centers, FKL at dst * scale (see --analyze); this reference
// is the same chain as FKL's CPU reference but with pixel-center sampling
template <size_t N, typename G = MainGeometry>
std::vector<float> pixelCenterReference(const Workload& w, const std::array<fk::Rect, N>& crops) {
    std::vector<float> out;
    out.reserve(N * G::outW * G::outH * 3);
    for (size_t i = 0; i < N; ++i) {
        const auto plane = hostVariant<G>(w, crops[i], false, Sampling::PixelCenters);
        out.insert(out.end(), plane.begin(), plane.end());
    }
    return out;
}

template <size_t N, NppdxConfig C>
void sweepBlock(Workload& w, const CropOrigins<N>& origins, float* out, const std::vector<float>& reference,
                std::vector<BlockSweep>& sweep) {
    using Nppdx = NppdxFigure1<NPPDX_SM, N, C.block, C.tileX, C.tileY>;
    Nppdx::init();
    constexpr size_t outFloats = N * OUT_W * OUT_H * 3;
    gpuErrchk(cudaMemset(out, 0xFF, outFloats * sizeof(float)));
    Nppdx::run(w, origins, out);
    gpuErrchk(cudaStreamSynchronize(w.stream));
    BlockSweep b{N, C.block, C.tileX, C.tileY, Nppdx::block.x, Nppdx::grid.x * Nppdx::grid.y, {},
                 compare(download(out, outFloats), reference)};
    b.time = benchmark::runOneExecution(w.stream, [&]() { Nppdx::run(w, origins, out); });
    sweep.push_back(b);
}

template <size_t N, NppdxConfig C>
void compareWithFKL(Workload& w, const CropOrigins<N>& origins, const std::array<fk::Rect, N>& crops, float* nppdxOut,
                    fk::Tensor<fk::float3>& fklOut, const std::vector<float>& fklReference,
                    const std::vector<float>& pixelCenters, std::vector<benchmark::TwoExecutionsResult>& results,
                    std::vector<Validation>& checks) {
    using Nppdx = NppdxFigure1<NPPDX_SM, N, C.block, C.tileX, C.tileY>;
    constexpr size_t outFloats = N * OUT_W * OUT_H * 3;
    gpuErrchk(cudaMemset(nppdxOut, 0xFF, outFloats * sizeof(float)));
    gpuErrchk(cudaMemset(fklOut.ptr().data, 0xFF, outFloats * sizeof(float)));
    Nppdx::run(w, origins, nppdxOut);
    runFKL<N>(w, crops, fklOut);
    gpuErrchk(cudaStreamSynchronize(w.stream));
    const auto nppdx = download(nppdxOut, outFloats);
    const auto fkl = download(reinterpret_cast<const float*>(fklOut.ptr().data), outFloats);
    const Validation v{N, Nppdx::block.x, C.tileX, C.tileY, compare(nppdx, fklReference), compare(nppdx, pixelCenters),
                       compare(fkl, fklReference)};
    checks.push_back(v);
    const auto r = benchmark::runTwoExecutions(
        w.stream, static_cast<int>(N), [&]() { Nppdx::run(w, origins, nppdxOut); },
        [&]() { runFKL<N>(w, crops, fklOut); });
    results.push_back(r);
    std::cout << "Batch " << N << ": NPPDx (block " << Nppdx::block.x << ", tile " << C.tileX << "x" << C.tileY
              << ") median " << r.first.median * 1000.f
              << " us, FKL median " << r.second.median * 1000.f << " us, mean speedup " << r.meanSpeedup
              << " | max abs error: NPPDx vs pixel-center reference " << v.nppdxVsPixelCenters.max
              << ", NPPDx vs FKL reference " << v.nppdxVsFkl.max << ", FKL vs FKL reference " << v.fkl.max << std::endl;
}

template <size_t N>
void runBatch(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results, std::vector<Validation>& checks,
              std::vector<BlockSweep>& sweep) {
    const auto crops = firstCrops<N>(w);
    CropOrigins<N> origins{};
    for (size_t i = 0; i < N; ++i) {
        origins.origin[i] = int2{static_cast<int>(crops[i].x), static_cast<int>(crops[i].y)};
    }
    constexpr size_t outFloats = N * OUT_W * OUT_H * 3;
    fk::Tensor<fk::float3> fklOut(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    float* nppdxOut = nullptr;
    gpuErrchk(cudaMalloc(&nppdxOut, outFloats * sizeof(float)));
    const auto fklReference = cpuReference<N>(w, crops);
    const auto pixelCenters = pixelCenterReference<N>(w, crops);

    // NPPDx block size sweep; the fastest configuration that is correct is compared with FKL
    const size_t first = sweep.size();
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (sweepBlock<N, NPPDX_CONFIGS[Is]>(w, origins, nppdxOut, pixelCenters, sweep), ...);
    }(std::make_index_sequence<NPPDX_CONFIGS.size()>{});
    size_t best = NPPDX_CONFIGS.size();
    float bestMedian = INFINITY;
    for (size_t i = first; i < sweep.size(); ++i) {
        if (sweep[i].vsPixelCenters.max <= TOLERANCE && sweep[i].time.median < bestMedian) {
            bestMedian = sweep[i].time.median;
            best = i - first;
        }
    }
    if (best == NPPDX_CONFIGS.size()) {
        throw std::runtime_error("No NPPDx configuration produced a correct result for batch " + std::to_string(N));
    }
    chosenConfig[N] = best;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((Is == best ? compareWithFKL<N, NPPDX_CONFIGS[Is]>(w, origins, crops, nppdxOut, fklOut, fklReference,
                                                            pixelCenters, results, checks)
                     : void()),
         ...);
    }(std::make_index_sequence<NPPDX_CONFIGS.size()>{});
    gpuErrchk(cudaFree(nppdxOut));
}

// GPU time only: the same comparison, timing each execution by the GPU execution time of its kernel (CUPTI)
template <size_t N, NppdxConfig C>
void compareGpuOnly(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results,
                    const benchmark::L2Flusher* flusher = nullptr) {
    using Nppdx = NppdxFigure1<NPPDX_SM, N, C.block, C.tileX, C.tileY>;
    Nppdx::init();
    const auto crops = firstCrops<N>(w);
    CropOrigins<N> origins{};
    for (size_t i = 0; i < N; ++i) {
        origins.origin[i] = int2{static_cast<int>(crops[i].x), static_cast<int>(crops[i].y)};
    }
    constexpr size_t outFloats = N * OUT_W * OUT_H * 3;
    fk::Tensor<fk::float3> fklOut(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    float* nppdxOut = nullptr;
    gpuErrchk(cudaMalloc(&nppdxOut, outFloats * sizeof(float)));
    const auto r = benchmark::runTwoExecutionsGpuOnly(
        w.stream, static_cast<int>(N), [&]() { Nppdx::run(w, origins, nppdxOut); },
        [&]() { runFKL<N>(w, crops, fklOut); }, flusher);
    results.push_back(r);
    std::cout << (flusher ? "GPU only, L2 flushed, batch " : "GPU only, batch ") << N << ": NPPDx (block " << Nppdx::block.x << ") median "
              << r.first.median * 1000.f << " us, FKL median " << r.second.median * 1000.f << " us, mean speedup "
              << r.meanSpeedup << std::endl;
    gpuErrchk(cudaFree(nppdxOut));
}

template <size_t N>
void gpuOnlyBatch(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results) {
    const size_t chosen = chosenConfig.at(N);
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((Is == chosen ? compareGpuOnly<N, NPPDX_CONFIGS[Is]>(w, results) : void()), ...);
    }(std::make_index_sequence<NPPDX_CONFIGS.size()>{});
}

// Target for Nsight Compute: warms up NPPDx (block 256, the configuration chosen for every batch size on the
// RTX 4090) or FKL, then runs it once more inside the NVTX range "measure", for ncu --nvtx --nvtx-include "measure/"
template <size_t N>
void ncuTarget(Workload& w, const std::string& name) {
    using Nppdx = NppdxFigure1<NPPDX_SM, N, 256, CROP_W, CROP_H>;
    Nppdx::init();
    const auto crops = firstCrops<N>(w);
    CropOrigins<N> origins{};
    for (size_t i = 0; i < N; ++i) {
        origins.origin[i] = int2{static_cast<int>(crops[i].x), static_cast<int>(crops[i].y)};
    }
    fk::Tensor<fk::float3> fklOut(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    float* nppdxOut = nullptr;
    gpuErrchk(cudaMalloc(&nppdxOut, N * OUT_W * OUT_H * 3 * sizeof(float)));
    std::function<void()> run;
    if (name == "nppdx") {
        run = [&]() { Nppdx::run(w, origins, nppdxOut); };
    } else if (name == "fkl_bvf") {
        run = [&]() { runFKL<N>(w, crops, fklOut); };
    } else {
        throw std::runtime_error("Unknown implementation " + name + ", use nppdx or fkl_bvf");
    }
    for (int i = 0; i < 3; ++i) {
        run();
    }
    gpuErrchk(cudaStreamSynchronize(w.stream));
    nvtxRangePushA("measure");
    run();
    gpuErrchk(cudaStreamSynchronize(w.stream));
    nvtxRangePop();
    gpuErrchk(cudaFree(nppdxOut));
}

template <size_t... Is>
void ncuTargetDispatch(Workload& w, const std::string& name, size_t batch, std::index_sequence<Is...>) {
    bool found = false;
    ((BATCHES[Is] == batch ? (ncuTarget<BATCHES[Is]>(w, name), found = true) : false), ...);
    if (!found) {
        throw std::runtime_error("Batch size " + std::to_string(batch) + " is not one of the benchmarked ones");
    }
}

// GPU only with the L2 flushed before every kernel, with NPPDx in the block 256 configuration (the one chosen for every
// batch size on the RTX 4090)
template <size_t... Is>
void gpuOnlyFlushedAllBatches(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results,
                              const benchmark::L2Flusher& flusher, std::index_sequence<Is...>) {
    (compareGpuOnly<BATCHES[Is], NppdxConfig{256, CROP_W, CROP_H}>(w, results, &flusher), ...);
}

template <size_t... Is>
void gpuOnlyAllBatches(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results, std::index_sequence<Is...>) {
    (gpuOnlyBatch<BATCHES[Is]>(w, results), ...);
}

template <size_t... Is>
void runAllBatches(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results, std::vector<Validation>& checks,
                   std::vector<BlockSweep>& sweep, std::index_sequence<Is...>) {
    (runBatch<BATCHES[Is]>(w, results, checks, sweep), ...);
}

// ------------------------------------------------------------------------------------------------
// Region size sweep: the same chain with N = 16 regions of growing size, resized exactly 2x in both dimensions
// (NPPDx ratio 1:2), to see how the advantage of fusion depends on how much of the frame the regions use
// ------------------------------------------------------------------------------------------------
namespace region {

constexpr size_t N = 16;
// A fixed 4x4 grid of 960x540 cells; region i = row * 4 + column is at the origin of its cell, so regions never
// overlap and the largest ones (960x524) nearly tile the frame. The origins are even, as NV12 chroma needs.
constexpr int GRID_COLS = 4;
constexpr int CELL_W = FRAME_W / GRID_COLS;
constexpr int CELL_H = FRAME_H / 4;
static_assert(GRID_COLS * 4 == N && CELL_W % 2 == 0 && CELL_H % 2 == 0);
constexpr std::array<unsigned, 3> BLOCKS{0, 128, 256};

// Tiles x block sizes tried for one region size. NPPDx tiles must split the region, be multiples of its 4x2 pixel
// cells, and their input, temporary and output tiles must fit in the shared memory of one block (101376 bytes on
// sm_89): a tile with the whole region is only possible for the smallest size; for the others NPPDx rejects it with
// the static_assert "This operation configuration is not supported".
constexpr std::array<NppdxConfig, 9> configsFor(const std::array<std::pair<unsigned, unsigned>, 3>& tiles) {
    std::array<NppdxConfig, 9> configs{};
    for (size_t t = 0; t < tiles.size(); ++t) {
        for (size_t b = 0; b < BLOCKS.size(); ++b) {
            configs[t * BLOCKS.size() + b] = NppdxConfig{BLOCKS[b], tiles[t].first, tiles[t].second};
        }
    }
    return configs;
}

template <char Name, int CropW, int CropH>
struct Case : Geometry<CropW, CropH, 2 * CropW, 2 * CropH, 1, 2> {
    static constexpr char name = Name;
    static constexpr int pixels = CropW * CropH;
    static constexpr size_t outputFloats = N * (2 * CropW) * (2 * CropH + BORDER_TOP + BORDER_BOTTOM) * 3;
    static_assert(CropW <= CELL_W && CropH <= CELL_H, "Regions must fit in their grid cell");
};
struct CaseA : Case<'A', 32, 16> {
    static constexpr auto configs = configsFor({{{32, 16}, {16, 16}, {16, 8}}});
    static constexpr const char* note = "a 32x16 tile holds the whole region: one block per region is possible";
};
struct CaseB : Case<'B', 112, 96> {
    static constexpr auto configs = configsFor({{{28, 32}, {28, 24}, {16, 16}}});
    static constexpr const char* note = "a whole-region 112x96 tile is rejected by NPPDx (its 224x192 output tile does "
                                        "not fit in shared memory): regions are split in tiles";
};
struct CaseC : Case<'C', 320, 304> {
    static constexpr auto configs = configsFor({{{64, 16}, {32, 16}, {20, 38}}});
    static constexpr const char* note = "a whole-region 320x304 tile is rejected by NPPDx (its 640x608 output tile does "
                                        "not fit in shared memory): regions are split in tiles";
};
struct CaseD : Case<'D', 960, 524> {
    static constexpr auto configs = configsFor({{{192, 4}, {64, 4}, {4, 262}}});
    static constexpr const char* note = "a whole-region 960x524 tile is rejected by NPPDx (its 1920x1048 output tile does "
                                        "not fit in shared memory): regions are split in tiles; 524 = 4 x 131 so tile "
                                        "heights can only be 2; 4; 262 or 524";
};

template <typename C>
std::array<fk::Rect, N> regions() {
    std::array<fk::Rect, N> r{};
    for (size_t i = 0; i < N; ++i) {
        r[i] = fk::Rect(static_cast<int>(i % GRID_COLS) * CELL_W, static_cast<int>(i / GRID_COLS) * CELL_H, C::cropW,
                        C::cropH);
    }
    return r;
}

CropOrigins<N> originsOf(const std::array<fk::Rect, N>& r) {
    CropOrigins<N> origins{};
    for (size_t i = 0; i < N; ++i) {
        origins.origin[i] = int2{static_cast<int>(r[i].x), static_cast<int>(r[i].y)};
    }
    return origins;
}

struct ConfigResult {
    char caseName{};
    int pixels{0};
    NppdxConfig config{};
    unsigned actualBlock{0}, blocksPerRegion{0};
    size_t smem{0};
    Error vsPixelCenters;
    bool timed{false}, chosen{false};
    benchmark::TimeStats time;
};

struct CaseResult {
    char name{};
    int cropW{0}, cropH{0}, resW{0}, resH{0}, outW{0}, outH{0};
    const char* note{""};
    bool nppdxFeasible{false};
    NppdxConfig chosen{};
    size_t chosenIndex{0};
    unsigned actualBlock{0};
    size_t smem{0};
    dim3 grid{};
    Error nppdxVsPixelCenters, nppdxVsFkl, fkl;
    size_t nppdxKernels{0}, fklKernels{0};
};

struct Results {
    std::vector<ConfigResult> configs;
    std::vector<CaseResult> cases;
    std::vector<benchmark::TwoExecutionsResult> cpuGpu, gpuOnly, gpuOnlyFlushed;
};

size_t kernelsPerExecution(cudaStream_t stream, const std::function<void()>& run) {
    return benchmark::cupti::trace(stream, run, 1).size();
}

template <typename C, NppdxConfig Cfg>
using Nppdx = NppdxFigure1<NPPDX_SM, N, Cfg.block, Cfg.tileX, Cfg.tileY, C>;

template <typename C, NppdxConfig Cfg>
void tryConfig(Workload& w, const CropOrigins<N>& origins, float* out, const std::vector<float>& pixelCenters,
               const bool timing, std::vector<ConfigResult>& results) {
    using K = Nppdx<C, Cfg>;
    K::init();
    gpuErrchk(cudaMemset(out, 0xFF, C::outputFloats * sizeof(float)));
    K::run(w, origins, out);
    gpuErrchk(cudaStreamSynchronize(w.stream));
    ConfigResult r{C::name, C::pixels, Cfg, K::block.x, K::grid.x * K::grid.y, K::smem,
                   compare(download(out, C::outputFloats), pixelCenters)};
    if (timing && r.vsPixelCenters.max <= TOLERANCE) {
        r.timed = true;
        r.time = benchmark::runOneExecution(w.stream, [&]() { K::run(w, origins, out); });
    }
    results.push_back(r);
}

// Calls f.template operator()<Cfg>() with the configuration C::configs[index]
template <typename C, typename F>
void withConfig(const size_t index, F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((Is == index ? f.template operator()<C::configs[Is]>() : void()), ...);
    }(std::make_index_sequence<C::configs.size()>{});
}

// Validation and choice of the NPPDx configuration, then the CPU + GPU comparison when timing
template <typename C>
void firstPass(Workload& w, const bool timing, Results& res) {
    const auto crops = regions<C>();
    const auto origins = originsOf(crops);
    float* nppdxOut = nullptr;
    gpuErrchk(cudaMalloc(&nppdxOut, C::outputFloats * sizeof(float)));
    fk::Tensor<fk::float3> fklOut(C::outW, C::outH, N, 1, fk::MemType::Device);
    const auto fklReference = cpuReference<N, C>(w, crops);
    const auto pixelCenters = pixelCenterReference<N, C>(w, crops);

    const size_t first = res.configs.size();
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (tryConfig<C, C::configs[Is]>(w, origins, nppdxOut, pixelCenters, timing, res.configs), ...);
    }(std::make_index_sequence<C::configs.size()>{});
    // The fastest correct configuration (without timing, the first correct one)
    size_t best = C::configs.size();
    float bestMedian = INFINITY;
    for (size_t i = first; i < res.configs.size(); ++i) {
        const ConfigResult& r = res.configs[i];
        if (r.vsPixelCenters.max <= TOLERANCE && (timing ? r.time.median < bestMedian : best == C::configs.size())) {
            bestMedian = r.time.median;
            best = i - first;
        }
    }
    CaseResult c{C::name, C::cropW, C::cropH, C::resW, C::resH, C::outW, C::outH, C::note};
    const auto runFkl = [&]() { runFKL<N, C>(w, crops, fklOut); };
    gpuErrchk(cudaMemset(fklOut.ptr().data, 0xFF, C::outputFloats * sizeof(float)));
    runFkl();
    gpuErrchk(cudaStreamSynchronize(w.stream));
    c.fkl = compare(download(reinterpret_cast<const float*>(fklOut.ptr().data), C::outputFloats), fklReference);
    if (best < C::configs.size()) {
        c.nppdxFeasible = true;
        c.chosenIndex = best;
        res.configs[first + best].chosen = true;
        withConfig<C>(best, [&]<NppdxConfig Cfg>() {
            using K = Nppdx<C, Cfg>;
            c.chosen = Cfg;
            c.actualBlock = K::block.x;
            c.smem = K::smem;
            c.grid = K::grid;
            gpuErrchk(cudaMemset(nppdxOut, 0xFF, C::outputFloats * sizeof(float)));
            K::run(w, origins, nppdxOut);
            gpuErrchk(cudaStreamSynchronize(w.stream));
            const auto nppdx = download(nppdxOut, C::outputFloats);
            c.nppdxVsPixelCenters = compare(nppdx, pixelCenters);
            c.nppdxVsFkl = compare(nppdx, fklReference);
            if (timing) {
                const auto r = benchmark::runTwoExecutions(w.stream, C::pixels, [&]() { K::run(w, origins, nppdxOut); },
                                                           runFkl);
                res.cpuGpu.push_back(r);
                std::cout << "Region " << C::name << " (" << C::cropW << "x" << C::cropH << "): NPPDx (tile "
                          << Cfg.tileX << "x" << Cfg.tileY << ", block " << K::block.x << ") median "
                          << r.first.median * 1000.f << " us, FKL median " << r.second.median * 1000.f
                          << " us, mean speedup " << r.meanSpeedup << std::endl;
            } else {
                c.nppdxKernels = kernelsPerExecution(w.stream, [&]() { K::run(w, origins, nppdxOut); });
                c.fklKernels = kernelsPerExecution(w.stream, runFkl);
            }
        });
    }
    std::cout << "Region " << C::name << " (" << C::cropW << "x" << C::cropH << " -> " << C::resW << "x" << C::resH
              << ", output " << C::outW << "x" << C::outH << "): ";
    if (c.nppdxFeasible) {
        std::cout << "NPPDx tile " << c.chosen.tileX << "x" << c.chosen.tileY << " block " << c.actualBlock << " ("
                  << c.grid.x * c.grid.y << " blocks per region, " << c.smem << " B shared memory) max abs error "
                  << c.nppdxVsPixelCenters.max << " vs pixel-center reference, " << c.nppdxVsFkl.max
                  << " vs FKL reference; ";
    } else {
        std::cout << "no NPPDx configuration matched the pixel-center reference; ";
    }
    std::cout << "FKL max abs error " << c.fkl.max << " vs FKL reference" << std::endl;
    res.cases.push_back(c);
    gpuErrchk(cudaFree(nppdxOut));
}

// GPU time only (CUPTI), with or without the L2 flushed before every kernel, with the configuration chosen in the
// first pass. Also counts the kernels of one execution of each library.
template <typename C>
void gpuOnlyPass(Workload& w, const benchmark::L2Flusher* flusher, Results& res) {
    auto c = std::find_if(res.cases.begin(), res.cases.end(), [](const CaseResult& r) { return r.name == C::name; });
    if (c == res.cases.end() || !c->nppdxFeasible) {
        return;
    }
    const auto crops = regions<C>();
    const auto origins = originsOf(crops);
    float* nppdxOut = nullptr;
    gpuErrchk(cudaMalloc(&nppdxOut, C::outputFloats * sizeof(float)));
    fk::Tensor<fk::float3> fklOut(C::outW, C::outH, N, 1, fk::MemType::Device);
    const auto runFkl = [&]() { runFKL<N, C>(w, crops, fklOut); };
    withConfig<C>(c->chosenIndex, [&]<NppdxConfig Cfg>() {
        using K = Nppdx<C, Cfg>;
        const auto runNppdx = [&]() { K::run(w, origins, nppdxOut); };
        if (!flusher) {
            c->nppdxKernels = kernelsPerExecution(w.stream, runNppdx);
            c->fklKernels = kernelsPerExecution(w.stream, runFkl);
        }
        const auto r = benchmark::runTwoExecutionsGpuOnly(w.stream, C::pixels, runNppdx, runFkl, flusher);
        (flusher ? res.gpuOnlyFlushed : res.gpuOnly).push_back(r);
        std::cout << (flusher ? "GPU only, L2 flushed, region " : "GPU only, region ") << C::name << ": NPPDx median "
                  << r.first.median * 1000.f << " us, FKL median " << r.second.median * 1000.f << " us, mean speedup "
                  << r.meanSpeedup << std::endl;
    });
    gpuErrchk(cudaFree(nppdxOut));
}

using Cases = std::tuple<CaseA, CaseB, CaseC, CaseD>;

template <typename F>
void forEachCase(F&& f) {
    std::apply([&](auto... c) { (f.template operator()<decltype(c)>(), ...); }, Cases{});
}

void writeCSVs(const std::filesystem::path& outDir, const Results& res, const bool timing) {
    const std::string name = "benchmark_figure1_bvf_region_sweep";
    const std::string parArch(fk::toStrView(fk::defaultParArch));
    const std::string variable = "Source pixels per region";
    if (timing) {
        benchmark::writeTwoExecutionsCSV(outDir, name, variable, "NPPDx", "FKL", res.cpuGpu);
        benchmark::writeTwoExecutionsCSV(outDir, name + "_gpu_only", variable, "NPPDx", "FKL", res.gpuOnly);
        benchmark::writeTwoExecutionsCSV(outDir, name + "_gpu_only_l2_flushed", variable, "NPPDx", "FKL",
                                         res.gpuOnlyFlushed);
    }
    std::filesystem::create_directories(outDir);
    std::ofstream validation(outDir / (name + "_validation_" + parArch + ".csv"));
    validation << "Region, " << variable << ", NPPDx Tile, NPPDx BlockSize, NPPDx MaxAbsError vs pixel-center reference, "
                  "NPPDx MeanAbsError vs pixel-center reference, NPPDx MaxAbsError vs FKL reference, "
                  "NPPDx MeanAbsError vs FKL reference, FKL MaxAbsError vs FKL reference, "
                  "FKL MeanAbsError vs FKL reference, Within tolerance 1e-3" << std::endl;
    for (const auto& c : res.cases) {
        validation << c.name << ", " << c.cropW * c.cropH << ", ";
        if (c.nppdxFeasible) {
            validation << c.chosen.tileX << "x" << c.chosen.tileY << ", " << c.actualBlock << ", "
                       << c.nppdxVsPixelCenters.max << ", " << c.nppdxVsPixelCenters.mean << ", " << c.nppdxVsFkl.max
                       << ", " << c.nppdxVsFkl.mean << ", ";
        } else {
            validation << "none, , , , , , ";
        }
        validation << c.fkl.max << ", " << c.fkl.mean << ", "
                   << (c.fkl.max <= TOLERANCE && c.nppdxFeasible && c.nppdxVsPixelCenters.max <= TOLERANCE ? "yes" : "no")
                   << std::endl;
    }
    std::ofstream configs(outDir / (name + "_nppdx_configs_" + parArch + ".csv"));
    configs << "Region, " << variable << ", Tile, Requested BlockDim (0 = NPPDx recommended), Actual BlockSize, "
               "Blocks per region, Dynamic shared memory bytes, MeanTime, MedianTime, P10Time, P90Time, "
               "MaxAbsError vs pixel-center reference, Chosen" << std::endl;
    for (const auto& r : res.configs) {
        configs << r.caseName << ", " << r.pixels << ", " << r.config.tileX << "x" << r.config.tileY << ", "
                << r.config.block << ", " << r.actualBlock << ", " << r.blocksPerRegion << ", " << r.smem << ", ";
        if (r.timed) {
            configs << r.time.mean << ", " << r.time.median << ", " << r.time.p10 << ", " << r.time.p90 << ", ";
        } else {
            configs << ", , , , ";
        }
        configs << r.vsPixelCenters.max << ", " << (r.chosen ? "yes" : "no") << std::endl;
    }
    std::ofstream summary(outDir / (name + "_summary_" + parArch + ".csv"));
    summary << "Region, Crop, Resize, Output, " << variable << ", Coverage of the frame by the 16 regions (%), "
               "NPPDx Tile, NPPDx BlockSize, NPPDx Grid, NPPDx Blocks per region, NPPDx Dynamic shared memory bytes, "
               "NPPDx Kernels per execution, FKL Kernels per execution, NPPDx Intermediate bytes in global memory, "
               "FKL Intermediate bytes in global memory, Output bytes, NPPDx note" << std::endl;
    for (const auto& c : res.cases) {
        const double coverage = 100. * N * c.cropW * c.cropH / (static_cast<double>(FRAME_W) * FRAME_H);
        const size_t outputBytes = N * static_cast<size_t>(c.outW) * c.outH * 3 * sizeof(float);
        summary << c.name << ", " << c.cropW << "x" << c.cropH << ", " << c.resW << "x" << c.resH << ", " << c.outW
                << "x" << c.outH << ", " << c.cropW * c.cropH << ", " << std::fixed << std::setprecision(4) << coverage
                << std::defaultfloat << ", ";
        if (c.nppdxFeasible) {
            summary << c.chosen.tileX << "x" << c.chosen.tileY << ", " << c.actualBlock << ", " << c.grid.x << "x"
                    << c.grid.y << "x" << c.grid.z << ", " << c.grid.x * c.grid.y << ", " << c.smem << ", "
                    << c.nppdxKernels << ", ";
        } else {
            summary << "none, , , , , , ";
        }
        summary << c.fklKernels << ", 0, 0, " << outputBytes << ", " << c.note << std::endl;
    }
}

void run(Workload& w, const std::filesystem::path& outDir, const bool timing) {
    Results res;
    forEachCase([&]<typename C>() { firstPass<C>(w, timing, res); });
    if (timing) {
        // GPU time only, after all the CPU + GPU measurements so that CUPTI cannot affect them
        forEachCase([&]<typename C>() { gpuOnlyPass<C>(w, nullptr, res); });
        const benchmark::L2Flusher flusher(w.stream);
        std::cout << "Flushing " << flusher.bytes() / (1 << 20) << " MiB (2x L2) before every kernel" << std::endl;
        forEachCase([&]<typename C>() { gpuOnlyPass<C>(w, &flusher, res); });
    }
    writeCSVs(outDir, res, timing);
    std::cout << "CSV files written to " << std::filesystem::absolute(outDir) << std::endl;
}

// Target for Nsight Compute: one execution of NPPDx (with the given tile and block, block 0 being NPPDx's recommended
// configuration) or FKL for one region size, inside the NVTX range "measure", after 3 warm-up executions
template <typename C>
bool ncuTarget(Workload& w, const std::string& impl, const char name, const unsigned tileX, const unsigned tileY,
               const unsigned block) {
    if (name != C::name) {
        return false;
    }
    const auto crops = regions<C>();
    const auto origins = originsOf(crops);
    float* nppdxOut = nullptr;
    gpuErrchk(cudaMalloc(&nppdxOut, C::outputFloats * sizeof(float)));
    fk::Tensor<fk::float3> fklOut(C::outW, C::outH, N, 1, fk::MemType::Device);
    std::function<void()> runOnce;
    if (impl == "fkl_bvf") {
        runOnce = [&]() { runFKL<N, C>(w, crops, fklOut); };
    } else if (impl == "nppdx") {
        size_t index = C::configs.size();
        for (size_t i = 0; i < C::configs.size(); ++i) {
            if (C::configs[i].tileX == tileX && C::configs[i].tileY == tileY && C::configs[i].block == block) {
                index = i;
            }
        }
        if (index < C::configs.size()) {
            withConfig<C>(index, [&]<NppdxConfig Cfg>() {
                Nppdx<C, Cfg>::init();
                runOnce = [&]() { Nppdx<C, Cfg>::run(w, origins, nppdxOut); };
            });
        } else {
            std::string valid;
            for (const auto& cfg : C::configs) {
                valid += " " + std::to_string(cfg.tileX) + "x" + std::to_string(cfg.tileY) + " " + std::to_string(cfg.block) + ",";
            }
            throw std::runtime_error(std::string("Region ") + name + " has no NPPDx configuration with that tile and block; valid:" + valid);
        }
    } else {
        throw std::runtime_error("Unknown implementation " + impl + ", use nppdx or fkl_bvf");
    }
    for (int i = 0; i < 3; ++i) {
        runOnce();
    }
    gpuErrchk(cudaStreamSynchronize(w.stream));
    nvtxRangePushA("measure");
    runOnce();
    gpuErrchk(cudaStreamSynchronize(w.stream));
    nvtxRangePop();
    gpuErrchk(cudaFree(nppdxOut));
    return true;
}

} // namespace region

} // namespace figure1

int main(int argc, char** argv) {
    using namespace figure1;
    try {
        const std::filesystem::path outDir = argc > 1 ? argv[1] : "results";
        int device = 0;
        gpuErrchk(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        gpuErrchk(cudaGetDeviceProperties(&prop, device));
        if (prop.major * 100 + prop.minor * 10 != NPPDX_SM) {
            std::cerr << "NPPDx operations were built for SM " << NPPDX_SM << " but the GPU is sm_" << prop.major
                      << prop.minor << ". Reconfigure with -DNPPDX_SM=" << prop.major * 100 + prop.minor * 10
                      << std::endl;
            return 1;
        }

        Workload w;
        initWorkload(w);
        if (argc > 2 && std::string(argv[2]) == "--gpu-only-l2-flushed") {
            const benchmark::L2Flusher flusher(w.stream);
            std::cout << "Flushing " << flusher.bytes() / (1 << 20) << " MiB (2x L2) before every kernel" << std::endl;
            std::vector<benchmark::TwoExecutionsResult> flushed;
            gpuOnlyFlushedAllBatches(w, flushed, flusher, std::make_index_sequence<BATCHES.size()>{});
            benchmark::writeTwoExecutionsCSV(outDir, "benchmark_figure1_bvf_gpu_only_l2_flushed", "Batch size", "NPPDx",
                                             "FKL", flushed);
            return 0;
        }
        if (argc > 2 && (std::string(argv[2]) == "--region-sweep" || std::string(argv[2]) == "--region-sweep-validate")) {
            const bool timing = std::string(argv[2]) == "--region-sweep";
            region::run(w, outDir, timing);
            if (timing) {
                int runtime = 0;
                gpuErrchk(cudaRuntimeGetVersion(&runtime));
                std::ofstream platform(outDir / "benchmark_figure1_bvf_region_sweep_platform.txt");
                platform << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ", " << prop.multiProcessorCount
                         << " SMs)\n"
                         << "NVIDIA driver: " << bench::shell("nvidia-smi --query-gpu=driver_version --format=csv,noheader") << "\n"
                         << "CUDA runtime: " << runtime / 1000 << "." << (runtime % 1000) / 10 << "\n"
                         << "Host compiler: " << bench::hostCompiler() << "\n"
                         << "FKL: " << FKL_VERSION << "\n"
                         << "NPPDx: " << NPPDX_GIT_VERSION << ", operations for SM " << NPPDX_SM << "\n"
                         << "CPU: " << bench::cpuName() << "\n"
                         << "OS: " << bench::osName() << "\n"
                         << "Region sweep: N = 16 regions on a 4x4 grid of 960x540 cells (region i = row * 4 + column at "
                            "the origin of its cell), sizes 32x16, 112x96, 320x304 and 960x524, resized exactly 2x "
                            "(NPPDx ratio 1:2), 16 black rows added top and bottom, same frame and normalization as the "
                            "main benchmark. For every size the NPPDx tiles and block sizes of "
                            "benchmark_figure1_bvf_region_sweep_nppdx_configs_" << fk::toStrView(fk::defaultParArch)
                         << ".csv are validated and timed alone, and the fastest correct one is compared with FKL. "
                            "Timing as in the main benchmark: " << benchmark::WARMUP_ITERS << " warm-up and "
                         << benchmark::ITERS << " timed iterations, CPU + GPU (CUDA events), GPU only (CUPTI) and "
                            "GPU only with the L2 flushed before every kernel; per-iteration speedups averaged; times in ms.\n";
            }
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--region-ncu-target") {
            // --region-ncu-target <nppdx|fkl_bvf> <A|B|C|D> [<tileX>x<tileY> <block>], the tile and block for nppdx
            if (argc < 5 || std::strlen(argv[4]) != 1) {
                throw std::runtime_error("Usage: --region-ncu-target <nppdx|fkl_bvf> <A|B|C|D> "
                                         "[<tileX>x<tileY> <block>] (the tile and block for nppdx)");
            }
            const std::string impl = argv[3];
            const char name = argv[4][0];
            unsigned tileX = 0, tileY = 0, block = 0;
            if (impl == "nppdx") {
                if (argc < 7 || std::sscanf(argv[5], "%ux%u", &tileX, &tileY) != 2) {
                    throw std::runtime_error("nppdx needs the tile and block: --region-ncu-target nppdx <A|B|C|D> "
                                             "<tileX>x<tileY> <block>");
                }
                block = static_cast<unsigned>(std::stoul(argv[6]));
            }
            bool found = false;
            region::forEachCase([&]<typename C>() {
                found = region::ncuTarget<C>(w, impl, name, tileX, tileY, block) || found;
            });
            if (!found) {
                throw std::runtime_error(std::string("Unknown region ") + name + ", use A, B, C or D");
            }
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--ncu-target") {
            if (argc < 5) {
                throw std::runtime_error("Usage: --ncu-target <nppdx|fkl_bvf> <N>");
            }
            ncuTargetDispatch(w, argv[3], std::stoul(argv[4]), std::make_index_sequence<BATCHES.size()>{});
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--analyze") {
            std::cout << "### NPPDx recommended block configuration" << std::endl;
            analyze<0>(w);
            std::cout << "### BlockDim<256>" << std::endl;
            analyze<256>(w);
            return 0;
        }
        // Any other argument is a mistake: running the default benchmark instead would overwrite its results in outDir
        if (argc > 2) {
            throw std::runtime_error(std::string("Unknown mode: ") + argv[2] + " (see the modes in README.md)");
        }
        std::vector<benchmark::TwoExecutionsResult> results;
        std::vector<Validation> checks;
        std::vector<BlockSweep> sweep;
        runAllBatches(w, results, checks, sweep, std::make_index_sequence<BATCHES.size()>{});
        // GPU time only, after all the CPU + GPU measurements so that CUPTI cannot affect them
        std::vector<benchmark::TwoExecutionsResult> gpuOnlyResults;
        gpuOnlyAllBatches(w, gpuOnlyResults, std::make_index_sequence<BATCHES.size()>{});

        const std::string name = "benchmark_figure1_bvf";
        benchmark::writeTwoExecutionsCSV(outDir, name, "Batch size", "NPPDx", "FKL", results);
        benchmark::writeTwoExecutionsCSV(outDir, name + "_gpu_only", "Batch size", "NPPDx", "FKL", gpuOnlyResults);
        const std::string parArch(fk::toStrView(fk::defaultParArch));
        std::ofstream validation(outDir / (name + "_validation_" + parArch + ".csv"));
        validation << "Batch size, NPPDx BlockSize, NPPDx Tile, NPPDx MaxAbsError vs pixel-center reference, "
                      "NPPDx MeanAbsError vs pixel-center reference, NPPDx MaxAbsError vs FKL reference, "
                      "NPPDx MeanAbsError vs FKL reference, FKL MaxAbsError vs FKL reference, "
                      "FKL MeanAbsError vs FKL reference" << std::endl;
        for (const auto& v : checks) {
            validation << v.batch << ", " << v.nppdxBlock << ", " << v.tileX << "x" << v.tileY << ", "
                       << v.nppdxVsPixelCenters.max << ", "
                       << v.nppdxVsPixelCenters.mean << ", " << v.nppdxVsFkl.max << ", " << v.nppdxVsFkl.mean << ", "
                       << v.fkl.max << ", " << v.fkl.mean << std::endl;
        }
        std::ofstream blocks(outDir / (name + "_nppdx_block_sweep_" + parArch + ".csv"));
        blocks << "Batch size, Requested BlockDim (0 = NPPDx recommended), Tile, Actual BlockSize, Blocks per crop, "
                  "MeanTime, MedianTime, P10Time, P90Time, MaxAbsError vs pixel-center reference" << std::endl;
        for (const auto& b : sweep) {
            blocks << b.batch << ", " << b.requestedBlock << ", " << b.tileX << "x" << b.tileY << ", " << b.actualBlock
                   << ", " << b.blocksPerCrop << ", " << b.time.mean << ", "
                   << b.time.median << ", " << b.time.p10 << ", " << b.time.p90 << ", " << b.vsPixelCenters.max
                   << std::endl;
        }
        int runtime = 0;
        gpuErrchk(cudaRuntimeGetVersion(&runtime));
        std::ofstream platform(outDir / (name + "_platform.txt"));
        platform << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ", " << prop.multiProcessorCount
                 << " SMs)\n"
                 << "NVIDIA driver: " << bench::shell("nvidia-smi --query-gpu=driver_version --format=csv,noheader") << "\n"
                 << "CUDA runtime: " << runtime / 1000 << "." << (runtime % 1000) / 10 << "\n"
                 << "Host compiler: " << bench::hostCompiler() << "\n"
                 << "FKL: " << FKL_VERSION << "\n"
                 << "NPPDx: " << NPPDX_GIT_VERSION << ", operations for SM " << NPPDX_SM << "\n"
                 << "CPU: " << bench::cpuName() << "\n"
                 << "OS: " << bench::osName() << "\n"
                 << "Timing, two benchmarks: " << benchmark::WARMUP_ITERS << " warm-up and " << benchmark::ITERS
                 << " timed iterations, in each one NPPDx and then FKL, with a synchronization after each. "
                    "CPU + GPU (" << name << "_" << parArch << ".csv): CUDA events around everything the host does "
                    "to enqueue the kernel plus its execution. GPU only (" << name << "_gpu_only_" << parArch
                 << ".csv): a second pass with CUPTI kernel tracing, the GPU execution time of the kernel. The speedup "
                    "is computed per iteration and averaged. Times in the CSV are in ms.\n"
                 << "nvcc flags: -O3 -DNDEBUG -std=c++20 --threads 0 -rdc=false -split-compile=0 -ftemplate-depth=1000 "
                    "-lineinfo --expt-relaxed-constexpr, CUDA_ARCHITECTURES " << CUDA_ARCHITECTURES_STR << "\n"
                 << "Correctness: FKL against FKL's CPU backend running the same chain (resize samples at dst * scale); "
                    "NPPDx against the same chain with pixel-center sampling ((dst + 0.5) * scale - 0.5), which is "
                    "the convention NPPDx's resize uses, and also against FKL's CPU backend.\n"
                 << "NPPDx configuration: the only valid tile for a 20x10 crop is 20x10 (tiles are multiples of 4x2 "
                    "pixel cells and of the 5:16 resize ratio), so one thread block per crop; block sizes swept in "
                    "benchmark_figure1_bvf_nppdx_block_sweep_" << parArch << ".csv and the fastest correct one is used.\n";
        std::cout << "CSV files written to " << std::filesystem::absolute(outDir) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
