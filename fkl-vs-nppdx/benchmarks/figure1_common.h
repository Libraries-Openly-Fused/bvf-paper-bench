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

// What the two FKL vs NPPDx benchmarks share (benchmark_figure1_bvf.cu: NPPDx as one kernel with a hand written end;
// benchmark_figure1_nppdx_npp.cu: NPPDx and NPP without hand written pixel processing code): the workload, the FKL
// pipeline, the references, the validation, the choice of the NPPDx configuration, the three comparisons with FKL and
// the program itself. A benchmark describes its NPPDx pipeline with a Pipeline type:
//   implementation, label, name    its implementation name (Nsight Compute target), CSV label and file name prefix
//   configs                        the NPPDx configurations it tries (nppdx_common.h)
//   tolerance, referenceName       the tolerance against its reference, and the reference's description
//   description()                  its pipeline, for the platform file
//   Buffers<N>                     its buffers for N crops, with the float3 output in out
//   poison<N>(b, stream)           fills every buffer it writes with values a correct execution never leaves
//   reference<N>(w, crops)         its reference output
//   run<N, C>(w, origins, b, f)    one execution in configuration C; f() runs before each of its kernels but the
//                                  first (the L2 flush of the GPU only benchmark with the L2 flushed)
//   runKernel<N, C>(w, origins, b) its NPPDx kernel alone, timed to choose the configuration
//   kernel<N, C>                   the NPPDx kernel type of configuration C (block, grid, smem, init())
//   analyze(w)                     its --analyze mode

#ifndef FIGURE1_COMMON_H
#define FIGURE1_COMMON_H

#include <benchmarks/benchmark_common.h>
#include "bench_platform.h"
#include "bench_npp.h"
#include "bench_validation.h"

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <benchmarks/nppdx_common.h>

#include <nvtx3/nvToolsExt.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace figure1 {

// The frame, sizes, batch sizes and crop positions are those of common/bench_geometry.h (through nppdx_common.h)

// ImageNet normalization (Normalization is in nppdx_common.h)
constexpr Normalization norm{};

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
// FKL's NV12 -> RGB conversion: BT.601, limited range
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Limited, fk::ColorPrimitives::bt601>;

struct Workload {
    cudaStream_t stream{};
    std::unique_ptr<fk::Stream> fkStream;
    std::unique_ptr<NV12Image> frameDevice;
    std::unique_ptr<NV12Image> frameHost;
    const uint8_t* y{nullptr};
    const uint8_t* uv{nullptr};
    int pitch{0};
    NppStreamContext npp{};
    // makeCrops() positions, which are even, so that both libraries read the same chroma samples: NPPDx takes the
    // crop's chroma at (x / 2, y / 2) relative to the crop origin, FKL relative to the frame
    std::array<fk::Rect, MAX_BATCH> crops{};
};

inline void initWorkload(Workload& w) {
    gpuErrchk(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    w.fkStream = std::make_unique<fk::Stream>(w.stream);
    w.npp = bench::npp::streamContext(w.stream);
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

template <size_t N>
CropOrigins<N> originsOf(const std::array<fk::Rect, N>& crops) {
    CropOrigins<N> origins{};
    for (size_t i = 0; i < N; ++i) {
        origins.origin[i] = int2{static_cast<int>(crops[i].x), static_cast<int>(crops[i].y)};
    }
    return origins;
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

// ------------------------------------------------------------------------------------------------
// References and validation
// ------------------------------------------------------------------------------------------------

// FKL's reference: FKL's CPU backend running the same chain
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

// Tolerances, errors and the rule of an output within its tolerance, shared with the GPU benchmark
// (common/bench_validation.h): TOLERANCE against a float reference, TOLERANCE_8BIT against one rounded to 8 bits
using bench::validation::TOLERANCE;
using bench::validation::TOLERANCE_8BIT;
using Error = bench::validation::Errors;
using bench::validation::compare;
using bench::validation::withinTolerance;

template <typename T>
std::vector<T> download(const T* data, size_t count) {
    std::vector<T> host(count);
    gpuErrchk(cudaMemcpy(host.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

// Host reference of the pipeline for one crop, with selectable semantics to find out what each library computes:
// optional clamp of RGB to [0, 255] after the color conversion (BT.601 limited range, FKL's conversion), the bilinear
// sampling convention, and optional rounding of the resized RGB to 8 bits (what NPPDx's 8 bit write does: + 0.5, then
// clamped to [0, 255]). Gives the resized RGB (RES_W x RES_H) and the normalized output plane (OUT_W x OUT_H).
enum class Sampling { DstTimesScale, PixelCenters, NppMinusQuarter };

struct HostCrop {
    std::vector<float> resized, plane;
};

inline HostCrop hostVariant(const Workload& w, const fk::Rect& r, bool clampRGB, Sampling sampling, bool roundTo8bit) {
    constexpr int CW = CROP_W, CH = CROP_H;
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
    HostCrop out{std::vector<float>(static_cast<size_t>(RES_W) * RES_H * 3), std::vector<float>(OUT_FLOATS)};
    for (int y = 0; y < RES_H; ++y) {
        for (int x = 0; x < RES_W; ++x) {
            const float fxs = std::min<float>(CW - 1, std::max(0.f, map(x, sx)));
            const float fys = std::min<float>(CH - 1, std::max(0.f, map(y, sy)));
            const int x0 = static_cast<int>(fxs), y0 = static_cast<int>(fys);
            const int x1 = std::min(x0 + 1, CW - 1), y1 = std::min(y0 + 1, CH - 1);
            const float ax = fxs - x0, ay = fys - y0;
            for (int c = 0; c < 3; ++c) {
                float v = (crop(y0, x0, c) * (1.f - ax) + crop(y0, x1, c) * ax) * (1.f - ay) +
                          (crop(y1, x0, c) * (1.f - ax) + crop(y1, x1, c) * ax) * ay;
                if (roundTo8bit) {
                    v = std::min(255.f, std::max(0.f, std::floor(v + 0.5f)));
                }
                out.resized[(static_cast<size_t>(y) * RES_W + x) * 3 + c] = v;
            }
        }
    }
    for (int y = 0; y < OUT_H; ++y) {
        for (int x = 0; x < OUT_W; ++x) {
            const int ry = y - BORDER_TOP;
            for (int c = 0; c < 3; ++c) {
                const float v = ry >= 0 && ry < RES_H ? out.resized[(static_cast<size_t>(ry) * RES_W + x) * 3 + c] : 0.f;
                out.plane[(static_cast<size_t>(y) * OUT_W + x) * 3 + c] = ((v * norm.mul[c]) - norm.sub[c]) / norm.div[c];
            }
        }
    }
    return out;
}

// The output planes of N crops with hostVariant at pixel centers (NPPDx's convention), optionally rounded to 8 bits
template <size_t N>
std::vector<float> pixelCenterReference(const Workload& w, const std::array<fk::Rect, N>& crops, bool roundTo8bit) {
    std::vector<float> out;
    out.reserve(N * OUT_FLOATS);
    for (size_t i = 0; i < N; ++i) {
        const auto plane = hostVariant(w, crops[i], false, Sampling::PixelCenters, roundTo8bit).plane;
        out.insert(out.end(), plane.begin(), plane.end());
    }
    return out;
}

// ------------------------------------------------------------------------------------------------
// Choice of the NPPDx configuration and comparison with FKL
// ------------------------------------------------------------------------------------------------

// NPPDx configuration (index into Pipeline::configs) chosen for every batch size in the CPU + GPU benchmark, reused by
// the GPU only benchmark and, through the configuration CSV, by --gpu-only-l2-flushed
inline std::map<size_t, size_t> chosenConfig;
// Configurations given with --configs <csv>, used instead of choosing one
inline std::map<size_t, size_t> pinnedConfig;

struct BlockSweep {
    size_t batch{0};
    unsigned requestedBlock{0}, tileX{0}, tileY{0}, actualBlock{0}, blocksPerCrop{0};
    size_t smem{0};
    benchmark::TimeStats time;
    Error vsReference;
    bool chosen{false};
};

struct Validation {
    size_t batch{0};
    unsigned nppdxBlock{0}, tileX{0}, tileY{0};
    Error nppdxVsReference, nppdxVsFkl, fkl;
};

const auto noFlush = []() {};

// Validates one NPPDx configuration (the whole pipeline) against its reference and times its NPPDx kernel alone
// (CPU + GPU; anything after the kernel is the same for every configuration)
template <typename P, size_t N, NppdxConfig C>
void sweepConfig(Workload& w, const CropOrigins<N>& origins, const typename P::template Buffers<N>& b,
                 const std::vector<float>& reference, std::vector<BlockSweep>& sweep) {
    using K = typename P::template kernel<N, C>;
    BlockSweep s{N, C.block, C.tileX, C.tileY, K::block.x, K::grid.x * K::grid.y, K::smem};
    try {
        P::poison(b, w.stream);
        K::init();
        P::template run<N, C>(w, origins, b, noFlush);
        gpuErrchk(cudaStreamSynchronize(w.stream));
    } catch (const std::exception& e) {
        // A configuration that cannot be launched (more than 1024 threads, too many registers or too much shared
        // memory for this GPU) is recorded as failed, with an infinite error, and never chosen
        std::cout << "Batch " << N << ": NPPDx tile " << C.tileX << "x" << C.tileY << " block " << C.block
                  << " cannot run here: " << e.what() << std::endl;
        s.vsReference.maxPerChannel.fill(INFINITY);
        sweep.push_back(s);
        return;
    }
    s.vsReference = compare(download(b.out, N * OUT_FLOATS), reference);
    s.time = benchmark::runOneExecution(w.stream, [&]() { P::template runKernel<N, C>(w, origins, b); });
    sweep.push_back(s);
}

template <typename P, size_t N, NppdxConfig C>
void compareWithFKL(Workload& w, const CropOrigins<N>& origins, const std::array<fk::Rect, N>& crops,
                    const typename P::template Buffers<N>& b, fk::Tensor<fk::float3>& fklOut,
                    const std::vector<float>& fklReference, const std::vector<float>& reference,
                    std::vector<benchmark::TwoExecutionsResult>& results, std::vector<Validation>& checks) {
    using K = typename P::template kernel<N, C>;
    // The sweep initializes every configuration it tries, but --configs skips it
    K::init();
    constexpr size_t outFloats = N * OUT_FLOATS;
    P::poison(b, w.stream);
    gpuErrchk(cudaMemsetAsync(fklOut.ptr().data, 0xFF, outFloats * sizeof(float), w.stream));
    P::template run<N, C>(w, origins, b, noFlush);
    runFKL<N>(w, crops, fklOut);
    gpuErrchk(cudaStreamSynchronize(w.stream));
    const auto nppdx = download(b.out, outFloats);
    const auto fkl = download(reinterpret_cast<const float*>(fklOut.ptr().data), outFloats);
    const Validation v{N, K::block.x, C.tileX, C.tileY, compare(nppdx, reference), compare(nppdx, fklReference),
                       compare(fkl, fklReference)};
    checks.push_back(v);
    const auto r = benchmark::runTwoExecutions(
        w.stream, static_cast<int>(N), [&]() { P::template run<N, C>(w, origins, b, noFlush); },
        [&]() { runFKL<N>(w, crops, fklOut); });
    results.push_back(r);
    std::cout << "Batch " << N << ": " << P::label << " (block " << K::block.x << ", tile " << C.tileX << "x" << C.tileY
              << ") median " << r.first.median * 1000.f << " us, FKL median " << r.second.median * 1000.f
              << " us, mean speedup " << r.meanSpeedup << " | max abs error: " << P::label << " vs its reference "
              << v.nppdxVsReference.max() << " (" << v.nppdxVsReference.beyond << " values beyond 1e-3), vs FKL "
              << "reference " << v.nppdxVsFkl.max() << ", FKL vs FKL reference " << v.fkl.max() << std::endl;
}

template <typename P, size_t N>
void runBatch(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results, std::vector<Validation>& checks,
              std::vector<BlockSweep>& sweep) {
    const auto crops = firstCrops<N>(w);
    const CropOrigins<N> origins = originsOf(crops);
    fk::Tensor<fk::float3> fklOut(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const typename P::template Buffers<N> buffers;
    const auto fklReference = cpuReference<N>(w, crops);
    const auto reference = P::template reference<N>(w, crops);

    // Every NPPDx configuration is validated and timed alone; the fastest correct one is compared with FKL (unless
    // --configs gave one)
    size_t best = P::configs.size();
    if (pinnedConfig.count(N)) {
        best = pinnedConfig.at(N);
    } else {
        const size_t first = sweep.size();
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (sweepConfig<P, N, P::configs[Is]>(w, origins, buffers, reference, sweep), ...);
        }(std::make_index_sequence<P::configs.size()>{});
        float bestMedian = INFINITY;
        for (size_t i = first; i < sweep.size(); ++i) {
            if (withinTolerance(sweep[i].vsReference, P::tolerance) &&
                sweep[i].time.median < bestMedian) {
                bestMedian = sweep[i].time.median;
                best = i - first;
            }
        }
        if (best == P::configs.size()) {
            throw std::runtime_error("No NPPDx configuration produced a correct result for batch " + std::to_string(N));
        }
        sweep[first + best].chosen = true;
    }
    chosenConfig[N] = best;
    withConfig<P::configs>(best, [&]<NppdxConfig C>() {
        compareWithFKL<P, N, C>(w, origins, crops, buffers, fklOut, fklReference, reference, results, checks);
    });
}

// GPU time only: the same comparison, timing each execution by the GPU execution times of its kernels (CUPTI), with or
// without the L2 flushed before every kernel
template <typename P, size_t N, NppdxConfig C>
void compareGpuOnly(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results,
                    const benchmark::L2Flusher* flusher) {
    using K = typename P::template kernel<N, C>;
    K::init();
    const auto crops = firstCrops<N>(w);
    const CropOrigins<N> origins = originsOf(crops);
    fk::Tensor<fk::float3> fklOut(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const typename P::template Buffers<N> buffers;
    // The harness flushes before every execution; a pipeline with NPP kernels also flushes before each of them
    const auto flushBeforeNpp = [flusher]() {
        if (flusher) {
            flusher->flush();
        }
    };
    const auto r = benchmark::runTwoExecutionsGpuOnly(
        w.stream, static_cast<int>(N), [&]() { P::template run<N, C>(w, origins, buffers, flushBeforeNpp); },
        [&]() { runFKL<N>(w, crops, fklOut); }, flusher);
    results.push_back(r);
    std::cout << (flusher ? "GPU only, L2 flushed, batch " : "GPU only, batch ") << N << ": " << P::label << " (block "
              << K::block.x << ", tile " << C.tileX << "x" << C.tileY << ") median " << r.first.median * 1000.f
              << " us, FKL median " << r.second.median * 1000.f << " us, mean speedup " << r.meanSpeedup << std::endl;
}

// GPU only with the configuration chosen for batch N
template <typename P, size_t N>
void gpuOnlyBatch(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results,
                  const benchmark::L2Flusher* flusher = nullptr) {
    withConfig<P::configs>(chosenConfig.at(N), [&]<NppdxConfig C>() { compareGpuOnly<P, N, C>(w, results, flusher); });
}

// Index into the configurations of a tile and requested block
template <typename P>
size_t configIndex(unsigned tileX, unsigned tileY, unsigned block) {
    std::string valid;
    for (size_t i = 0; i < P::configs.size(); ++i) {
        const NppdxConfig& c = P::configs[i];
        if (c.tileX == tileX && c.tileY == tileY && c.block == block) {
            return i;
        }
        valid += " " + std::to_string(c.tileX) + "x" + std::to_string(c.tileY) + " " + std::to_string(c.block) + ",";
    }
    throw std::runtime_error("No NPPDx configuration with tile " + std::to_string(tileX) + "x" + std::to_string(tileY) +
                             " and block " + std::to_string(block) + "; valid:" + valid);
}

// The NPPDx configuration of every batch size from a configuration CSV, as the CPU + GPU benchmark writes it
// (<name>_nppdx_config_<ParArch>.csv: batch size, tile, requested block, ...)
template <typename P>
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
            configs[batch] = configIndex<P>(tileX, tileY, block);
        }
    }
    for (const size_t n : BATCHES) {
        if (!configs.count(n)) {
            throw std::runtime_error(path.string() + " has no configuration for batch size " + std::to_string(n));
        }
    }
    return configs;
}

// Target for Nsight Compute: warms up the NPPDx pipeline (its kernel in the configuration given by its index into the
// configurations) or FKL, then runs it once more inside the NVTX range "measure", for
// ncu --nvtx --nvtx-include "measure/"
template <typename P, size_t N>
void ncuTarget(Workload& w, const std::string& name, size_t config) {
    const auto crops = firstCrops<N>(w);
    const CropOrigins<N> origins = originsOf(crops);
    fk::Tensor<fk::float3> fklOut(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const typename P::template Buffers<N> buffers;
    std::function<void()> run;
    if (name == P::implementation) {
        withConfig<P::configs>(config, [&]<NppdxConfig C>() {
            P::template kernel<N, C>::init();
            run = [&]() { P::template run<N, C>(w, origins, buffers, noFlush); };
        });
    } else if (name == "fkl_bvf") {
        run = [&]() { runFKL<N>(w, crops, fklOut); };
    } else {
        throw std::runtime_error("Unknown implementation " + name + ", use " + P::implementation + " or fkl_bvf");
    }
    for (int i = 0; i < 3; ++i) {
        run();
    }
    gpuErrchk(cudaStreamSynchronize(w.stream));
    nvtxRangePushA("measure");
    run();
    gpuErrchk(cudaStreamSynchronize(w.stream));
    nvtxRangePop();
}

template <typename P, size_t... Is>
void ncuTargetDispatch(Workload& w, const std::string& name, size_t batch, size_t config, std::index_sequence<Is...>) {
    bool found = false;
    ((BATCHES[Is] == batch ? (ncuTarget<P, BATCHES[Is]>(w, name, config), found = true) : false), ...);
    if (!found) {
        throw std::runtime_error("Batch size " + std::to_string(batch) + " is not one of the benchmarked ones");
    }
}

template <typename P, size_t... Is>
void gpuOnlyAllBatches(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results,
                       const benchmark::L2Flusher* flusher, std::index_sequence<Is...>) {
    (gpuOnlyBatch<P, BATCHES[Is]>(w, results, flusher), ...);
}

template <typename P, size_t... Is>
void runAllBatches(Workload& w, std::vector<benchmark::TwoExecutionsResult>& results, std::vector<Validation>& checks,
                   std::vector<BlockSweep>& sweep, std::index_sequence<Is...>) {
    (runBatch<P, BATCHES[Is]>(w, results, checks, sweep), ...);
}

// ------------------------------------------------------------------------------------------------
// The program: <outDir> [--gpu-only-l2-flushed | --ncu-target <impl> <N> [<tileX>x<tileY> <block>] | --analyze |
// --configs <csv>], or --batches / --workload alone
// ------------------------------------------------------------------------------------------------
template <typename P>
int figure1Main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--batches") {
            // The batch sizes, for the Nsight Compute scripts
            std::cout << batchList() << std::endl;
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--workload") {
            // The workload this binary was built for, so that scripts can check that binaries and results match
            std::cout << describe() << std::endl;
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

        const std::string name = P::name;
        const std::string parArch(fk::toStrView(fk::defaultParArch));
        const std::filesystem::path configCsv = outDir / (name + "_nppdx_config_" + parArch + ".csv");
        Workload w;
        initWorkload(w);
        if (argc > 2 && std::string(argv[2]) == "--gpu-only-l2-flushed") {
            // NPPDx in the configuration the CPU + GPU benchmark chose for every batch size, from its CSV in outDir
            chosenConfig = readConfigs<P>(configCsv);
            const benchmark::L2Flusher flusher(w.stream);
            std::cout << "Flushing " << flusher.bytes() / (1 << 20) << " MiB (2x L2) before every kernel" << std::endl;
            std::vector<benchmark::TwoExecutionsResult> flushed;
            gpuOnlyAllBatches<P>(w, flushed, &flusher, std::make_index_sequence<BATCHES.size()>{});
            benchmark::writeTwoExecutionsCSV(outDir, name + "_gpu_only_l2_flushed", "Batch size", P::label, "FKL",
                                             flushed);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--ncu-target") {
            // --ncu-target <impl|fkl_bvf> <N> [<tileX>x<tileY> <block>], the tile and requested block for NPPDx
            if (argc < 5) {
                throw std::runtime_error(std::string("Usage: --ncu-target <") + P::implementation +
                                         "|fkl_bvf> <N> [<tileX>x<tileY> <block>] (the tile and block for " +
                                         P::implementation + ")");
            }
            size_t config = 0;
            if (std::string(argv[3]) == P::implementation) {
                unsigned tileX = 0, tileY = 0;
                if (argc < 7 || std::sscanf(argv[5], "%ux%u", &tileX, &tileY) != 2) {
                    throw std::runtime_error(std::string(P::implementation) + " needs the tile and block: --ncu-target " +
                                             P::implementation + " <N> <tileX>x<tileY> <block>, for example the ones of " +
                                             configCsv.filename().string());
                }
                config = configIndex<P>(tileX, tileY, static_cast<unsigned>(std::stoul(argv[6])));
            }
            ncuTargetDispatch<P>(w, argv[3], std::stoul(argv[4]), config, std::make_index_sequence<BATCHES.size()>{});
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--analyze") {
            P::analyze(w);
            return 0;
        }
        // --configs <csv>: the default benchmark with the NPPDx configuration of every batch size given in the CSV (in
        // the format of the configuration CSV this benchmark writes) instead of the fastest of the configurations
        std::filesystem::path configsInput; // the CSV given with --configs, recorded in the platform file
        if (argc > 3 && std::string(argv[2]) == "--configs") {
            configsInput = argv[3];
            pinnedConfig = readConfigs<P>(configsInput);
        } else if (argc > 2) {
            // Any other argument is a mistake: running the default benchmark instead would overwrite its results
            throw std::runtime_error(std::string("Unknown mode: ") + argv[2] + " (see the modes in README.md)");
        }
        std::vector<benchmark::TwoExecutionsResult> results;
        std::vector<Validation> checks;
        std::vector<BlockSweep> sweep;
        runAllBatches<P>(w, results, checks, sweep, std::make_index_sequence<BATCHES.size()>{});
        // GPU time only, after all the CPU + GPU measurements so that CUPTI cannot affect them
        std::vector<benchmark::TwoExecutionsResult> gpuOnlyResults;
        gpuOnlyAllBatches<P>(w, gpuOnlyResults, nullptr, std::make_index_sequence<BATCHES.size()>{});

        benchmark::writeTwoExecutionsCSV(outDir, name, "Batch size", P::label, "FKL", results);
        benchmark::writeTwoExecutionsCSV(outDir, name + "_gpu_only", "Batch size", P::label, "FKL", gpuOnlyResults);
        std::ofstream validation(outDir / (name + "_validation_" + parArch + ".csv"));
        validation << "Batch size, NPPDx BlockSize, NPPDx Tile, NPPDx MaxAbsError vs NPPDx reference, "
                      "NPPDx MeanAbsError vs NPPDx reference, NPPDx values beyond 1e-3 of NPPDx reference, "
                      "NPPDx MaxAbsError vs FKL reference, NPPDx MeanAbsError vs FKL reference, "
                      "FKL MaxAbsError vs FKL reference, FKL MeanAbsError vs FKL reference, Values, "
                      "NPPDx tolerance, NPPDx within tolerance, FKL within tolerance" << std::endl;
        for (const auto& v : checks) {
            validation << v.batch << ", " << v.nppdxBlock << ", " << v.tileX << "x" << v.tileY << ", "
                       << v.nppdxVsReference.max() << ", " << v.nppdxVsReference.mean << ", "
                       << v.nppdxVsReference.beyond << ", " << v.nppdxVsFkl.max() << ", " << v.nppdxVsFkl.mean << ", "
                       << v.fkl.max() << ", "
                       << v.fkl.mean << ", " << v.nppdxVsReference.values << ", " << P::tolerance << ", "
                       << (withinTolerance(v.nppdxVsReference, P::tolerance) ? "yes" : "no") << ", "
                       << (withinTolerance(v.fkl, TOLERANCE) ? "yes" : "no") << std::endl;
        }
        std::ofstream blocks(outDir / (name + "_nppdx_block_sweep_" + parArch + ".csv"));
        blocks << "Batch size, Requested BlockDim (0 = NPPDx recommended), Tile, Actual BlockSize, Blocks per crop, "
                  "Dynamic shared memory bytes, MeanTime, MedianTime, P10Time, P90Time, "
                  "MaxAbsError vs NPPDx reference, Chosen" << std::endl;
        for (const auto& b : sweep) {
            blocks << b.batch << ", " << b.requestedBlock << ", " << b.tileX << "x" << b.tileY << ", " << b.actualBlock
                   << ", " << b.blocksPerCrop << ", " << b.smem << ", " << b.time.mean << ", " << b.time.median << ", "
                   << b.time.p10 << ", " << b.time.p90 << ", " << b.vsReference.max() << ", "
                   << (b.chosen ? "yes" : "no") << std::endl;
        }
        // The configuration of every batch size, read back by --gpu-only-l2-flushed, --configs, the Nsight Compute
        // scripts and the compile-time units
        std::ofstream chosen(configCsv);
        chosen << "Batch size, Tile, Requested BlockDim (0 = NPPDx recommended), Actual BlockSize, Blocks per crop, "
                  "Dynamic shared memory bytes" << std::endl;
        for (const auto& [batch, index] : chosenConfig) {
            withConfig<P::configs>(index, [&, batch = batch]<NppdxConfig C>() {
                using K = typename P::template kernel<1, C>;
                chosen << batch << ", " << C.tileX << "x" << C.tileY << ", " << C.block << ", " << K::block.x << ", "
                       << K::grid.x * K::grid.y << ", " << K::smem << std::endl;
            });
        }
        std::string tiles;
        for (size_t i = 0; i < P::configs.size(); i += NPPDX_BLOCKS.size()) {
            tiles += (i ? ", " : "") + std::to_string(P::configs[i].tileX) + "x" + std::to_string(P::configs[i].tileY);
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
                 << "Workload: " << describe() << ". AddBorder (black), Mul (1/255), Sub (0.485, 0.456, 0.406), Div "
                    "(0.229, 0.224, 0.225), float3 tensor of N planes " << OUT_W << "x" << OUT_H << "\n"
                 << P::label << " pipeline: " << P::description() << "\n"
                 << "FKL pipeline: one executeOperations call (Backwards Vertical Fusion), one kernel.\n"
                 << "Timing, three benchmarks: " << benchmark::WARMUP_ITERS << " warm-up and " << benchmark::ITERS
                 << " timed iterations, in each one " << P::label << " and then FKL, with a synchronization after each. "
                    "CPU + GPU (" << name << "_" << parArch << ".csv): CUDA events around everything the host does "
                    "to enqueue the pipeline plus its execution. GPU only (" << name << "_gpu_only_" << parArch
                 << ".csv): a second pass with CUPTI kernel tracing, the sum of the GPU execution times of the "
                    "pipeline's kernels. GPU only with the L2 flushed before every kernel (" << name
                 << "_gpu_only_l2_flushed_" << parArch << ".csv, --gpu-only-l2-flushed, a separate run). The speedup is "
                    "computed per iteration and averaged. Times in the CSV are in ms.\n"
                 << "nvcc flags: -O3 -DNDEBUG -std=c++20 --threads 0 -rdc=false -split-compile=0 -ftemplate-depth=1000 "
                    "-lineinfo --expt-relaxed-constexpr, CUDA_ARCHITECTURES " << CUDA_ARCHITECTURES_STR << "\n"
                 << "Correctness: FKL against FKL's CPU backend running the same chain (resize samples at dst * scale), "
                    "tolerance " << TOLERANCE << "; " << P::label << " against " << P::referenceName << ", tolerance "
                 << P::tolerance << "; and also against FKL's CPU backend.\n"
                 << "NPPDx configuration: resize " << RESIZE_J << ":" << RESIZE_K << "; the tiles NPPDx supports for "
                 << CROP_W << "x" << CROP_H << " crops among the whole crop and its successive halvings down to "
                 << MIN_TILE_PIXELS << " pixels (multiples of NPPDx's 4x2 pixel cells and of J) are " << tiles
                 << ", each with NPPDx's recommended block and with BlockDim 64, 128, 256 and 512. "
                 << (pinnedConfig.empty()
                         ? "For every batch size every configuration is validated and its NPPDx kernel timed alone (" +
                               name + "_nppdx_block_sweep_" + parArch + ".csv), and the fastest correct one (" +
                               configCsv.filename().string() + ") is compared with FKL in all three benchmarks.\n"
                         : "This run used the configurations given with --configs " +
                               std::filesystem::absolute(configsInput).string() + " instead of choosing them (copied to " +
                               configCsv.filename().string() + ").\n");
        std::cout << "CSV files written to " << std::filesystem::absolute(outDir) << std::endl;
        // The sweep only chooses configurations within their tolerance, but one given with --configs may not be, and
        // FKL's output is checked too: the results are written either way, and the program fails
        bool failed = false;
        for (const auto& v : checks) {
            if (!withinTolerance(v.nppdxVsReference, P::tolerance) || !withinTolerance(v.fkl, TOLERANCE)) {
                std::cerr << "ERROR: batch " << v.batch << ": " << P::label << " (tile " << v.tileX << "x" << v.tileY
                          << ", block " << v.nppdxBlock << ") or FKL is not within its tolerance, see "
                          << name << "_validation_" << parArch << ".csv" << std::endl;
                failed = true;
            }
        }
        if (failed) {
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

} // namespace figure1

#endif // FIGURE1_COMMON_H
