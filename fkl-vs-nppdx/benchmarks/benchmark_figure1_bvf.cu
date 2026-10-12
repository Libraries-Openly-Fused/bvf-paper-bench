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
// One 3840x2160 NV12 frame in device memory and N crops, with the sizes and positions of common/bench_geometry.h
// (by default 32x16 crops resized to 64x32), NV12->RGB BT.2020 full range, bilinear resize, black rows added on top
// and bottom up to a square, then (x * mul - sub) / div, written as float3 into a tensor of N planes.
//
//   NPPDx: one CUDA kernel built from NPPDx operations, one or more thread blocks per crop (one per tile of the
//          crop, blockIdx.z is the crop): NV12 ingest of the tile -> ColorConvert BT.2020 -> RGB -> Resize J:K
//          bilinear in shared memory, then AddBorder, Mul, Sub, Div and the float3 write written by hand in the
//          same kernel, because NPPDx has no border operation, its affine map only takes integer constants, and it
//          has no float output format. Every tile and block size NPPDx supports for these crops is validated and
//          timed, and the fastest correct one is compared with FKL.
//   FKL:   one executeOperations call (Backwards Vertical Fusion), the paper's Listing 2.

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
#include <benchmarks/nppdx_figure1.h>

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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace figure1 {

// The frame, sizes, batch sizes and crop positions are those of common/bench_geometry.h (through nppdx_figure1.h)

// ImageNet normalization (Normalization is in nppdx_figure1.h)
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
    // makeCrops() positions, which are even, so that both libraries read the same chroma samples: NPPDx takes the
    // crop's chroma at (x / 2, y / 2) relative to the crop origin, FKL relative to the frame
    std::array<fk::Rect, MAX_BATCH> crops{};
};

void initWorkload(Workload& w) {
    gpuErrchk(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    w.fkStream = std::make_unique<fk::Stream>(w.stream);
    w.frameDevice = std::make_unique<NV12Image>(FRAME_W, FRAME_H);
    w.frameHost = std::make_unique<NV12Image>(FRAME_W, FRAME_H, fk::MemType::Host);
    auto deviceData = w.frameDevice->getData();
    auto hostData = w.frameHost->getData();
    generateFrame([&](int x, int y, uint8_t value) {
        deviceData.at(x, y) = value;
        hostData.at(x, y) = value;
    });
    w.frameDevice->upload(*w.fkStream);
    gpuErrchk(cudaStreamSynchronize(w.stream));
    const auto raw = deviceData.ptr();
    w.pitch = static_cast<int>(raw.dims.pitch);
    w.y = raw.data;
    w.uv = raw.data + static_cast<size_t>(FRAME_H) * w.pitch;

    const auto crops = makeCrops();
    for (size_t i = 0; i < MAX_BATCH; ++i) {
        w.crops[i] = fk::Rect(crops[i].x, crops[i].y, crops[i].width, crops[i].height);
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
template <size_t N>
void runFKL(Workload& w, const std::array<fk::Rect, N>& crops, fk::Tensor<fk::float3>& out) {
    fk::executeOperations<fk::TransformDPP<>>(*w.fkStream,
        fk::ReadYUV<fk::PixelFormat::NV12>::build(*w.frameDevice),
        YUVToRGB::build(),
        fk::Crop<>::build(crops),
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
        fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, fk::float3{0.f, 0.f, 0.f}),
        fk::Mul<fk::float3>::build(fk::float3{norm.mul[0], norm.mul[1], norm.mul[2]}),
        fk::Sub<fk::float3>::build(fk::float3{norm.sub[0], norm.sub[1], norm.sub[2]}),
        fk::Div<fk::float3>::build(fk::float3{norm.div[0], norm.div[1], norm.div[2]}),
        fk::TensorWrite<fk::float3>::build(out.ptr()));
}

// NPPDx: one kernel, one or more thread blocks per crop, in nppdx_figure1.h

// ------------------------------------------------------------------------------------------------
// Reference on FKL's CPU backend
// ------------------------------------------------------------------------------------------------
template <size_t N>
std::vector<float> cpuReference(const Workload& w, const std::array<fk::Rect, N>& crops) {
    fk::Stream_<fk::ParArch::CPU> cpuStream;
    fk::Tensor<fk::float3> out(OUT_W, OUT_H, N, 1, fk::MemType::Host);
    fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
        fk::ReadYUV<fk::PixelFormat::NV12>::build(*w.frameHost),
        YUVToRGB::build(),
        fk::Crop<>::build(crops),
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
        fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, fk::float3{0.f, 0.f, 0.f}),
        fk::Mul<fk::float3>::build(fk::float3{norm.mul[0], norm.mul[1], norm.mul[2]}),
        fk::Sub<fk::float3>::build(fk::float3{norm.sub[0], norm.sub[1], norm.sub[2]}),
        fk::Div<fk::float3>::build(fk::float3{norm.div[0], norm.div[1], norm.div[2]}),
        fk::TensorWrite<fk::float3>::build(out.ptr()));
    cpuStream.sync();
    const float* data = reinterpret_cast<const float*>(out.ptr().data);
    return std::vector<float>(data, data + N * OUT_FLOATS);
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

std::vector<float> hostVariant(const Workload& w, const fk::Rect& r, bool clampRGB, Sampling sampling) {
    constexpr int CW = CROP_W, CH = CROP_H, OW = OUT_W, OH = OUT_H;
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
    const float sx = static_cast<float>(CW) / RES_W, sy = static_cast<float>(CH) / RES_H;
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
            if (ry >= 0 && ry < RES_H) {
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

// Every NPPDx configuration the benchmark tries (nppdx_figure1.h): the candidate tiles NPPDx supports for these
// crops, each with NPPDx's recommended block (block 0) and with BlockDim 64, 128, 256 and 512
constexpr auto NPPDX_CONFIGS = nppdxConfigs<NPPDX_SM>();

// What NPPDx computes, with the first (largest) tile of NPPDX_CONFIGS and block size BlockX
template <unsigned BlockX>
void analyze(Workload& w) {
    using Nppdx = NppdxFigure1<NPPDX_SM, 1, BlockX, NPPDX_CONFIGS[0].tileX, NPPDX_CONFIGS[0].tileY>;
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
    Nppdx::run(w.stream, w.y, w.uv, w.pitch, origins, norm, out);
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
        constexpr auto convertOnly = nppdxConvertOnlyKernel<typename Nppdx::Ingest, typename Nppdx::Convert, In>;
        if constexpr (smemIn > 48 * 1024) {
            gpuErrchk(cudaFuncSetAttribute(convertOnly, cudaFuncAttributeMaxDynamicSharedMemorySize, smemIn));
        }
        convertOnly<<<1, Nppdx::block, smemIn, w.stream>>>(w.y, w.uv, w.pitch, origins.origin[0], tile);
        gpuErrchk(cudaGetLastError());
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
    // The first rows of the top border and of the resized crop, the middle, the last resized row and the first row of
    // the bottom border (when there is one)
    for (const int y : {0, BORDER_TOP, BORDER_TOP + 1, OUT_H / 2 - 2, BORDER_TOP + RES_H - 1, BORDER_TOP + RES_H}) {
        if (y >= OUT_H) {
            continue;
        }
        for (const int x : {0, 1, OUT_W / 2 - 1, OUT_W - 1}) {
            const size_t i = (y * OUT_W + x) * 3;
            std::cout << "  (" << x << "," << y << ") NPPDx " << nppdx[i] << " " << nppdx[i + 1] << " " << nppdx[i + 2]
                      << " | FKL reference " << ref[i] << " " << ref[i + 1] << " " << ref[i + 2] << std::endl;
        }
    }
}

constexpr float TOLERANCE = 1e-3f;

// NPPDx configuration (index into NPPDX_CONFIGS) chosen for every batch size in the CPU + GPU benchmark, reused by
// the GPU only benchmark and, through the configuration CSV, by --gpu-only-l2-flushed
std::map<size_t, size_t> chosenConfig;
// Configurations given with --configs <csv>, used instead of choosing one
std::map<size_t, size_t> pinnedConfig;

struct BlockSweep {
    size_t batch{0};
    unsigned requestedBlock{0}, tileX{0}, tileY{0}, actualBlock{0}, blocksPerCrop{0};
    size_t smem{0};
    benchmark::TimeStats time;
    Error vsPixelCenters;
    bool chosen{false};
};

struct Validation {
    size_t batch{0};
    unsigned nppdxBlock{0}, tileX{0}, tileY{0};
    Error nppdxVsFkl, nppdxVsPixelCenters, fkl;
};

// NPPDx samples the bilinear resize at pixel centers, FKL at dst * scale (see --analyze); this reference
// is the same chain as FKL's CPU reference but with pixel-center sampling
template <size_t N>
std::vector<float> pixelCenterReference(const Workload& w, const std::array<fk::Rect, N>& crops) {
    std::vector<float> out;
    out.reserve(N * OUT_FLOATS);
    for (size_t i = 0; i < N; ++i) {
        const auto plane = hostVariant(w, crops[i], false, Sampling::PixelCenters);
        out.insert(out.end(), plane.begin(), plane.end());
    }
    return out;
}

template <size_t N>
CropOrigins<N> originsOf(const std::array<fk::Rect, N>& crops) {
    CropOrigins<N> origins{};
    for (size_t i = 0; i < N; ++i) {
        origins.origin[i] = int2{static_cast<int>(crops[i].x), static_cast<int>(crops[i].y)};
    }
    return origins;
}

// Validates one NPPDx configuration against the pixel-center reference and times it alone (CPU + GPU)
template <size_t N, NppdxConfig C>
void sweepConfig(Workload& w, const CropOrigins<N>& origins, float* out, const std::vector<float>& reference,
                 std::vector<BlockSweep>& sweep) {
    using Nppdx = NppdxKernel<NPPDX_SM, N, C>;
    constexpr size_t outFloats = N * OUT_FLOATS;
    BlockSweep b{N, C.block, C.tileX, C.tileY, Nppdx::block.x, Nppdx::grid.x * Nppdx::grid.y, Nppdx::smem};
    gpuErrchk(cudaMemset(out, 0xFF, outFloats * sizeof(float)));
    try {
        Nppdx::init();
        Nppdx::run(w.stream, w.y, w.uv, w.pitch, origins, norm, out);
        gpuErrchk(cudaStreamSynchronize(w.stream));
    } catch (const std::exception& e) {
        // A configuration that cannot be launched (more than 1024 threads, too many registers or too much shared
        // memory for this GPU) is recorded as failed, with an infinite error, and never chosen
        std::cout << "Batch " << N << ": NPPDx tile " << C.tileX << "x" << C.tileY << " block " << C.block
                  << " cannot run here: " << e.what() << std::endl;
        b.vsPixelCenters.max = INFINITY;
        sweep.push_back(b);
        return;
    }
    b.vsPixelCenters = compare(download(out, outFloats), reference);
    b.time = benchmark::runOneExecution(w.stream, [&]() { Nppdx::run(w.stream, w.y, w.uv, w.pitch, origins, norm, out); });
    sweep.push_back(b);
}

template <size_t N, NppdxConfig C>
void compareWithFKL(Workload& w, const CropOrigins<N>& origins, const std::array<fk::Rect, N>& crops, float* nppdxOut,
                    fk::Tensor<fk::float3>& fklOut, const std::vector<float>& fklReference,
                    const std::vector<float>& pixelCenters, std::vector<benchmark::TwoExecutionsResult>& results,
                    std::vector<Validation>& checks) {
    using Nppdx = NppdxKernel<NPPDX_SM, N, C>;
    constexpr size_t outFloats = N * OUT_FLOATS;
    gpuErrchk(cudaMemset(nppdxOut, 0xFF, outFloats * sizeof(float)));
    gpuErrchk(cudaMemset(fklOut.ptr().data, 0xFF, outFloats * sizeof(float)));
    Nppdx::run(w.stream, w.y, w.uv, w.pitch, origins, norm, nppdxOut);
    runFKL<N>(w, crops, fklOut);
    gpuErrchk(cudaStreamSynchronize(w.stream));
    const auto nppdx = download(nppdxOut, outFloats);
    const auto fkl = download(reinterpret_cast<const float*>(fklOut.ptr().data), outFloats);
    const Validation v{N, Nppdx::block.x, C.tileX, C.tileY, compare(nppdx, fklReference), compare(nppdx, pixelCenters),
                       compare(fkl, fklReference)};
    checks.push_back(v);
    const auto r = benchmark::runTwoExecutions(
        w.stream, static_cast<int>(N), [&]() { Nppdx::run(w.stream, w.y, w.uv, w.pitch, origins, norm, nppdxOut); },
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
    const CropOrigins<N> origins = originsOf(crops);
    constexpr size_t outFloats = N * OUT_FLOATS;
    fk::Tensor<fk::float3> fklOut(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    float* nppdxOut = nullptr;
    gpuErrchk(cudaMalloc(&nppdxOut, outFloats * sizeof(float)));
    const auto fklReference = cpuReference<N>(w, crops);
    const auto pixelCenters = pixelCenterReference<N>(w, crops);

    // Every NPPDx configuration is validated and timed alone; the fastest correct one is compared with FKL (unless
    // --configs gave one)
    size_t best = NPPDX_CONFIGS.size();
    if (pinnedConfig.count(N)) {
        best = pinnedConfig.at(N);
    } else {
        const size_t first = sweep.size();
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (sweepConfig<N, NPPDX_CONFIGS[Is]>(w, origins, nppdxOut, pixelCenters, sweep), ...);
        }(std::make_index_sequence<NPPDX_CONFIGS.size()>{});
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
        sweep[first + best].chosen = true;
    }
    chosenConfig[N] = best;
    withConfig<NPPDX_CONFIGS>(best, [&]<NppdxConfig C>() {
        compareWithFKL<N, C>(w, origins, crops, nppdxOut, fklOut, fklReference, pixelCenters, results, checks);
    });
    gpuErrchk(cudaFree(nppdxOut));
}

// GPU time only: the same comparison, timing each execution by the GPU execution time of its kernel (CUPTI), with or
// without the L2 flushed before every kernel
template <size_t N, NppdxConfig C>
void compareGpuOnly(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results,
                    const benchmark::L2Flusher* flusher) {
    using Nppdx = NppdxKernel<NPPDX_SM, N, C>;
    Nppdx::init();
    const auto crops = firstCrops<N>(w);
    const CropOrigins<N> origins = originsOf(crops);
    constexpr size_t outFloats = N * OUT_FLOATS;
    fk::Tensor<fk::float3> fklOut(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    float* nppdxOut = nullptr;
    gpuErrchk(cudaMalloc(&nppdxOut, outFloats * sizeof(float)));
    const auto r = benchmark::runTwoExecutionsGpuOnly(
        w.stream, static_cast<int>(N), [&]() { Nppdx::run(w.stream, w.y, w.uv, w.pitch, origins, norm, nppdxOut); },
        [&]() { runFKL<N>(w, crops, fklOut); }, flusher);
    results.push_back(r);
    std::cout << (flusher ? "GPU only, L2 flushed, batch " : "GPU only, batch ") << N << ": NPPDx (block "
              << Nppdx::block.x << ", tile " << C.tileX << "x" << C.tileY << ") median " << r.first.median * 1000.f
              << " us, FKL median " << r.second.median * 1000.f << " us, mean speedup " << r.meanSpeedup << std::endl;
    gpuErrchk(cudaFree(nppdxOut));
}

// GPU only with the configuration chosen for batch N
template <size_t N>
void gpuOnlyBatch(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results,
                  const benchmark::L2Flusher* flusher = nullptr) {
    withConfig<NPPDX_CONFIGS>(chosenConfig.at(N), [&]<NppdxConfig C>() { compareGpuOnly<N, C>(w, results, flusher); });
}

// Index into NPPDX_CONFIGS of a tile and requested block
size_t configIndex(unsigned tileX, unsigned tileY, unsigned block) {
    std::string valid;
    for (size_t i = 0; i < NPPDX_CONFIGS.size(); ++i) {
        const NppdxConfig& c = NPPDX_CONFIGS[i];
        if (c.tileX == tileX && c.tileY == tileY && c.block == block) {
            return i;
        }
        valid += " " + std::to_string(c.tileX) + "x" + std::to_string(c.tileY) + " " + std::to_string(c.block) + ",";
    }
    throw std::runtime_error("No NPPDx configuration with tile " + std::to_string(tileX) + "x" + std::to_string(tileY) +
                             " and block " + std::to_string(block) + "; valid:" + valid);
}

// The NPPDx configuration of every batch size from a configuration CSV, as the CPU + GPU benchmark writes it
// (benchmark_figure1_bvf_nppdx_config_<ParArch>.csv: batch size, tile, requested block, ...)
std::map<size_t, size_t> readConfigs(const std::filesystem::path& path) {
    std::ifstream csv(path);
    if (!csv) {
        throw std::runtime_error("Cannot read " + path.string() + ": run the benchmark without a mode into the same "
                                 "directory first, it chooses the NPPDx configuration of every batch size");
    }
    std::map<size_t, size_t> configs;
    std::string line;
    std::getline(csv, line); // header
    while (std::getline(csv, line)) {
        size_t batch = 0;
        unsigned tileX = 0, tileY = 0, block = 0;
        if (std::sscanf(line.c_str(), "%zu, %ux%u, %u", &batch, &tileX, &tileY, &block) == 4) {
            configs[batch] = configIndex(tileX, tileY, block);
        }
    }
    for (const size_t n : BATCHES) {
        if (!configs.count(n)) {
            throw std::runtime_error(path.string() + " has no configuration for batch size " + std::to_string(n));
        }
    }
    return configs;
}

// Target for Nsight Compute: warms up NPPDx (in the configuration given by its index into NPPDX_CONFIGS) or FKL, then
// runs it once more inside the NVTX range "measure", for ncu --nvtx --nvtx-include "measure/"
template <size_t N>
void ncuTarget(Workload& w, const std::string& name, size_t config) {
    const auto crops = firstCrops<N>(w);
    const CropOrigins<N> origins = originsOf(crops);
    fk::Tensor<fk::float3> fklOut(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    float* nppdxOut = nullptr;
    gpuErrchk(cudaMalloc(&nppdxOut, N * OUT_FLOATS * sizeof(float)));
    std::function<void()> run;
    if (name == "nppdx") {
        withConfig<NPPDX_CONFIGS>(config, [&]<NppdxConfig C>() {
            NppdxKernel<NPPDX_SM, N, C>::init();
            run = [&]() { NppdxKernel<NPPDX_SM, N, C>::run(w.stream, w.y, w.uv, w.pitch, origins, norm, nppdxOut); };
        });
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
void ncuTargetDispatch(Workload& w, const std::string& name, size_t batch, size_t config, std::index_sequence<Is...>) {
    bool found = false;
    ((BATCHES[Is] == batch ? (ncuTarget<BATCHES[Is]>(w, name, config), found = true) : false), ...);
    if (!found) {
        throw std::runtime_error("Batch size " + std::to_string(batch) + " is not one of the benchmarked ones");
    }
}

template <size_t... Is>
void gpuOnlyAllBatches(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results,
                       const benchmark::L2Flusher* flusher, std::index_sequence<Is...>) {
    (gpuOnlyBatch<BATCHES[Is]>(w, results, flusher), ...);
}

template <size_t... Is>
void runAllBatches(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results, std::vector<Validation>& checks,
                   std::vector<BlockSweep>& sweep, std::index_sequence<Is...>) {
    (runBatch<BATCHES[Is]>(w, results, checks, sweep), ...);
}

} // namespace figure1

int main(int argc, char** argv) {
    using namespace figure1;
    try {
        if (argc == 2 && std::string(argv[1]) == "--batches") {
            // The batch sizes, for the Nsight Compute scripts
            for (size_t i = 0; i < BATCHES.size(); ++i) {
                std::cout << (i ? " " : "") << BATCHES[i];
            }
            std::cout << std::endl;
            return 0;
        }
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

        const std::string name = "benchmark_figure1_bvf";
        const std::string parArch(fk::toStrView(fk::defaultParArch));
        const std::filesystem::path configCsv = outDir / (name + "_nppdx_config_" + parArch + ".csv");
        Workload w;
        initWorkload(w);
        if (argc > 2 && std::string(argv[2]) == "--gpu-only-l2-flushed") {
            // NPPDx in the configuration the CPU + GPU benchmark chose for every batch size, from its CSV in outDir
            chosenConfig = readConfigs(configCsv);
            const benchmark::L2Flusher flusher(w.stream);
            std::cout << "Flushing " << flusher.bytes() / (1 << 20) << " MiB (2x L2) before every kernel" << std::endl;
            std::vector<benchmark::TwoExecutionsResult> flushed;
            gpuOnlyAllBatches(w, flushed, &flusher, std::make_index_sequence<BATCHES.size()>{});
            benchmark::writeTwoExecutionsCSV(outDir, name + "_gpu_only_l2_flushed", "Batch size", "NPPDx", "FKL",
                                             flushed);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--ncu-target") {
            // --ncu-target <nppdx|fkl_bvf> <N> [<tileX>x<tileY> <block>], the tile and requested block for nppdx
            if (argc < 5) {
                throw std::runtime_error("Usage: --ncu-target <nppdx|fkl_bvf> <N> [<tileX>x<tileY> <block>] (the tile "
                                         "and block for nppdx)");
            }
            size_t config = 0;
            if (std::string(argv[3]) == "nppdx") {
                unsigned tileX = 0, tileY = 0;
                if (argc < 7 || std::sscanf(argv[5], "%ux%u", &tileX, &tileY) != 2) {
                    throw std::runtime_error("nppdx needs the tile and block: --ncu-target nppdx <N> <tileX>x<tileY> "
                                             "<block>, for example the ones of " + configCsv.filename().string());
                }
                config = configIndex(tileX, tileY, static_cast<unsigned>(std::stoul(argv[6])));
            }
            ncuTargetDispatch(w, argv[3], std::stoul(argv[4]), config, std::make_index_sequence<BATCHES.size()>{});
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--analyze") {
            std::cout << "### NPPDx recommended block configuration" << std::endl;
            analyze<0>(w);
            std::cout << "### BlockDim<256>" << std::endl;
            analyze<256>(w);
            return 0;
        }
        // --configs <csv>: the default benchmark with the NPPDx configuration of every batch size given in the CSV (in
        // the format of the configuration CSV this benchmark writes) instead of the fastest of NPPDX_CONFIGS
        if (argc > 3 && std::string(argv[2]) == "--configs") {
            pinnedConfig = readConfigs(argv[3]);
        } else if (argc > 2) {
            // Any other argument is a mistake: running the default benchmark instead would overwrite its results
            throw std::runtime_error(std::string("Unknown mode: ") + argv[2] + " (see the modes in README.md)");
        }
        std::vector<benchmark::TwoExecutionsResult> results;
        std::vector<Validation> checks;
        std::vector<BlockSweep> sweep;
        runAllBatches(w, results, checks, sweep, std::make_index_sequence<BATCHES.size()>{});
        // GPU time only, after all the CPU + GPU measurements so that CUPTI cannot affect them
        std::vector<benchmark::TwoExecutionsResult> gpuOnlyResults;
        gpuOnlyAllBatches(w, gpuOnlyResults, nullptr, std::make_index_sequence<BATCHES.size()>{});

        benchmark::writeTwoExecutionsCSV(outDir, name, "Batch size", "NPPDx", "FKL", results);
        benchmark::writeTwoExecutionsCSV(outDir, name + "_gpu_only", "Batch size", "NPPDx", "FKL", gpuOnlyResults);
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
                  "Dynamic shared memory bytes, MeanTime, MedianTime, P10Time, P90Time, "
                  "MaxAbsError vs pixel-center reference, Chosen" << std::endl;
        for (const auto& b : sweep) {
            blocks << b.batch << ", " << b.requestedBlock << ", " << b.tileX << "x" << b.tileY << ", " << b.actualBlock
                   << ", " << b.blocksPerCrop << ", " << b.smem << ", " << b.time.mean << ", " << b.time.median << ", "
                   << b.time.p10 << ", " << b.time.p90 << ", " << b.vsPixelCenters.max << ", "
                   << (b.chosen ? "yes" : "no") << std::endl;
        }
        // The configuration of every batch size, read back by --gpu-only-l2-flushed, --configs, the Nsight Compute
        // scripts and the compile-time units
        std::ofstream chosen(configCsv);
        chosen << "Batch size, Tile, Requested BlockDim (0 = NPPDx recommended), Actual BlockSize, Blocks per crop, "
                  "Dynamic shared memory bytes" << std::endl;
        for (const auto& [batch, index] : chosenConfig) {
            withConfig<NPPDX_CONFIGS>(index, [&, batch = batch]<NppdxConfig C>() {
                using K = NppdxKernel<NPPDX_SM, 1, C>;
                chosen << batch << ", " << C.tileX << "x" << C.tileY << ", " << C.block << ", " << K::block.x << ", "
                       << K::grid.x * K::grid.y << ", " << K::smem << std::endl;
            });
        }
        std::string tiles;
        for (size_t i = 0; i < NPPDX_CONFIGS.size(); i += NPPDX_BLOCKS.size()) {
            tiles += (i ? ", " : "") + std::to_string(NPPDX_CONFIGS[i].tileX) + "x" + std::to_string(NPPDX_CONFIGS[i].tileY);
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
                 << "Workload: " << describe() << ". NV12->RGB BT.2020 full range, AddBorder (black), Mul (1/255), "
                    "Sub (0.485, 0.456, 0.406), Div (0.229, 0.224, 0.225), float3 tensor of N planes " << OUT_W << "x"
                 << OUT_H << "\n"
                 << "Timing, three benchmarks: " << benchmark::WARMUP_ITERS << " warm-up and " << benchmark::ITERS
                 << " timed iterations, in each one NPPDx and then FKL, with a synchronization after each. "
                    "CPU + GPU (" << name << "_" << parArch << ".csv): CUDA events around everything the host does "
                    "to enqueue the kernel plus its execution. GPU only (" << name << "_gpu_only_" << parArch
                 << ".csv): a second pass with CUPTI kernel tracing, the GPU execution time of the kernel. GPU only "
                    "with the L2 flushed before every kernel (" << name << "_gpu_only_l2_flushed_" << parArch
                 << ".csv, --gpu-only-l2-flushed, a separate run). The speedup is computed per iteration and averaged. "
                    "Times in the CSV are in ms.\n"
                 << "nvcc flags: -O3 -DNDEBUG -std=c++20 --threads 0 -rdc=false -split-compile=0 -ftemplate-depth=1000 "
                    "-lineinfo --expt-relaxed-constexpr, CUDA_ARCHITECTURES " << CUDA_ARCHITECTURES_STR << "\n"
                 << "Correctness: FKL against FKL's CPU backend running the same chain (resize samples at dst * scale); "
                    "NPPDx against the same chain with pixel-center sampling ((dst + 0.5) * scale - 0.5), which is "
                    "the convention NPPDx's resize uses, and also against FKL's CPU backend.\n"
                 << "NPPDx configuration: resize " << RESIZE_J << ":" << RESIZE_K << "; the tiles NPPDx supports for "
                 << CROP_W << "x" << CROP_H << " crops among the whole crop and its successive halvings down to "
                 << MIN_TILE_PIXELS << " pixels (multiples of NPPDx's 4x2 pixel cells and of J) are " << tiles
                 << ", each with NPPDx's recommended block and with BlockDim 64, 128, 256 and 512. "
                 << (pinnedConfig.empty()
                         ? "For every batch size every configuration is validated and timed alone (" + name +
                               "_nppdx_block_sweep_" + parArch + ".csv), and the fastest correct one (" +
                               configCsv.filename().string() + ") is compared with FKL in all three benchmarks.\n"
                         : "This run used the configurations given with --configs (" + configCsv.filename().string() +
                               ") instead of choosing them.\n");
        std::cout << "CSV files written to " << std::filesystem::absolute(outDir) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
