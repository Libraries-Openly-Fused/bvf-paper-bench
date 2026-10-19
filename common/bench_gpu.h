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

// The harness of the GPU benchmarks that compare a library's pipeline with FKL's (fkl-vs-opencv-cuda, fkl-vs-npp,
// fkl-ablation): the workload in device memory, FKL BVF (the paper's Listing 2), the references and the validation,
// the timing, one implementation's loop after the other (CPU + GPU, GPU only, GPU only with the L2 flushed), the kernel
// counts and per-kernel times,
// the Nsight Compute target, the CSV and platform files, and main(). Figure 1 pipeline:
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// with the frame, region sizes, batch sizes and region positions of common/bench_geometry.h, and NV12 -> RGB in BT.601
// limited range in every implementation.
//
// The regions change every iteration: iteration t of each implementation's loop runs its pipeline on the regions of
// iteration t of common/bench_geometry.h, which are generated before any loop (ctx.regions). Each pipeline converts them into the
// arguments its library takes (FKL's rectangles, OpenCV's cv::Rect...) when it is created, also before the loops; what
// it does in run(t) is what the library needs to receive the regions of this iteration (FKL's build() calls and the
// fusion of the chain, NPP's descriptor upload, OpenCV's per-region views and calls), so that is timed.
//
// A benchmark describes what it compares with a Program type and calls gpuBenchMain<Program>(argc, argv):
//   title                                 what it compares, for the output and the platform file
//   implementations<N>(ctx)               the baseline pipelines for N regions; FKL BVF is added after them
//   describe(os, outDir)                  library versions and pipelines, for the platform file
//   diagnostic(ctx)                       anything worth recording about the libraries (or an empty string)
//   extraMode(ctx, mode, outDir)          handles a mode of its own (true) or not (false)
// and, for a pipeline other than Figure 1 (fkl-neighborhood/), optionally (without them, the defaults are Figure 1's):
//   fklImplementation<N>(ctx)             the FKL implementation the baselines are compared with (FKL BVF)
//   fklName, fklLabel, fklShortLabel      its name in the files ("fkl_bvf") and in the text ("FKL BVF", "BVF")
//   reference<N>(ctx, r, t)               the reference of conventions r on the regions of iteration t
//   workload(os)                          the pipeline and its output, for the platform file
//   batches                               the batch sizes to run, a subset of BATCHES in the same order
//   setup(outDir, mode)                   called before any run, with the mode argument (empty for the main run)
//   regionSize                            every region of this size (RegionSets' fixed-size constructor) instead of
//                                         the sizes of BENCH_REGIONS
//   perIterationTimes                     also write every timed iteration's time of every implementation
//                                         (per_iteration_<mode>.csv), for per-iteration speedups between any two
//                                         implementations (nppdx-wins/)

#ifndef BVF_BENCH_GPU_H
#define BVF_BENCH_GPU_H

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <cupti.h>
#include <nvtx3/nvToolsExt.h>

#include "bench_geometry.h"
#include "bench_npp.h"
#include "bench_platform.h"
#include "bench_timing.h"
#include "bench_validation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace bench {

// Frame, crop, resize, border and output sizes, batch sizes, seeds and crop positions, shared by every benchmark
using namespace geometry;
// BENCH_WARMUP_ITERATIONS and BENCH_ITERATIONS come from CMake: 100 and 1000 for the paper's results, 0 and 1 in the CI
constexpr int WARMUP_ITERS = BENCH_WARMUP_ITERATIONS;
constexpr int TIMED_ITERS = BENCH_ITERATIONS;
static_assert(WARMUP_ITERS >= 0 && TIMED_ITERS >= 1, "at least one timed iteration");
// --profile-kernels: PROFILE_WARMUP executions on the regions of iterations 0 to 9, then PROFILE_RUNS traced ones on
// those of iterations 10 to 109
constexpr int PROFILE_WARMUP = 10;
constexpr int PROFILE_RUNS = 100;
// --ncu-target: the profiled execution runs on the regions of iteration NCU_ITERATION, after NCU_WARMUP executions on
// those of the following iterations (other regions, as in the timed loop, where every iteration has new ones)
constexpr size_t NCU_ITERATION = 0;
constexpr int NCU_WARMUP = 3;
// The iterations whose regions are generated: every loop of the program uses the first ones
constexpr size_t REGION_ITERATIONS = std::max<size_t>({static_cast<size_t>(WARMUP_ITERS + TIMED_ITERS),
                                                       static_cast<size_t>(PROFILE_WARMUP + PROFILE_RUNS),
                                                       NCU_ITERATION + NCU_WARMUP + 1});
using validation::validatedIterations;
// Tolerances and error statistics shared with fkl-vs-nppdx (common/bench_validation.h)
using validation::TOLERANCE;
using validation::TOLERANCE_8BIT;
using ErrorStats = validation::Errors;
using validation::compare;

// Pointwise tail: ImageNet normalization of 8 bit RGB
constexpr float MUL[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
constexpr float SUB[3] = {0.485f, 0.456f, 0.406f};
constexpr float DIV[3] = {0.229f, 0.224f, 0.225f};
// 8 bit quantization of the RGB values for the reference of the 8 bit pipelines:
// normalize to [0, 1], add half a level so that the truncating denormalization rounds to nearest,
// saturate with __saturatef and denormalize to uchar3
constexpr float QUANTIZATION_OFFSET = 0.5f / 255.f;

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
// FKL's NV12 -> RGB conversion: BT.601, limited range
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Limited, fk::ColorPrimitives::bt601>;

inline const fk::float3 mulValue{MUL[0], MUL[1], MUL[2]};
inline const fk::float3 subValue{SUB[0], SUB[1], SUB[2]};
inline const fk::float3 divValue{DIV[0], DIV[1], DIV[2]};
inline const fk::float3 black{0.f, 0.f, 0.f};

struct Context {
    cudaStream_t stream{};
    std::unique_ptr<fk::Stream> fkStream;
    NppStreamContext npp{};
    std::unique_ptr<NV12Image> frameDevice;
    std::unique_ptr<NV12Image> frameHost;
    const Npp8u* y{nullptr};
    const Npp8u* uv{nullptr};
    int pitch{0};
    // The regions of every iteration (common/bench_geometry.h)
    std::unique_ptr<RegionSets> regions;
};

// regions: the regions of every iteration (regionSetsOf<P>), or nullptr for those of common/bench_geometry.h
inline void initContext(Context& ctx, std::unique_ptr<RegionSets> regions = nullptr) {
    gpuErrchk(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
    ctx.fkStream = std::make_unique<fk::Stream>(ctx.stream);
    ctx.npp = npp::streamContext(ctx.stream);

    // Synthetic NV12 frame (legal limited-range values), identical for every implementation
    ctx.frameDevice = std::make_unique<NV12Image>(FRAME_W, FRAME_H);
    ctx.frameHost = std::make_unique<NV12Image>(FRAME_W, FRAME_H, fk::MemType::Host);
    auto deviceData = ctx.frameDevice->getData();
    auto hostData = ctx.frameHost->getData();
    generateFrame([&](int x, int y, uint8_t value) {
        deviceData.at(x, y) = value;
        hostData.at(x, y) = value;
    });
    ctx.frameDevice->upload(*ctx.fkStream);
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    const auto raw = deviceData.ptr();
    cudaPointerAttributes attributes{};
    gpuErrchk(cudaPointerGetAttributes(&attributes, raw.data));
    if (attributes.type != cudaMemoryTypeDevice) {
        throw std::runtime_error("The NV12 frame is expected in device memory");
    }
    ctx.pitch = static_cast<int>(raw.dims.pitch);
    ctx.y = raw.data;
    ctx.uv = raw.data + static_cast<size_t>(FRAME_H) * ctx.pitch;

    // The regions of every iteration, generated before any loop; batch N uses the first N of every iteration
    ctx.regions = regions ? std::move(regions) : std::make_unique<RegionSets>(REGION_ITERATIONS);
}

// L2 flush before every kernel, for the GPU only benchmark with a flushed L2: simulates a GPU that runs other kernels
// in between, so that no intermediate result (nor the input frame) can be assumed to still be in L2. The flush kernel
// reads a buffer of twice the L2 size, which evicts every other line (writing back the dirty ones during the flush)
// and leaves L2 clean. Every library call of the pipelines launches exactly one kernel, and beforeKernel() is called
// right before each of them; it does nothing unless the flushed benchmark is running.
__global__ void flushL2Kernel(const int4* buffer, size_t count, int* sink) {
    int acc = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int4 v = __ldcg(buffer + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x7fffffff) {
        *sink = acc; // never true for the zeroed buffer, keeps the loads
    }
}

struct L2Flusher {
    int4* buffer{nullptr};
    size_t count{0};
    int* sink{nullptr};
    cudaStream_t stream{};
    int blocks{0};
};
inline L2Flusher* activeFlusher = nullptr;

inline void beforeKernel() {
    if (activeFlusher) {
        flushL2Kernel<<<activeFlusher->blocks, 256, 0, activeFlusher->stream>>>(activeFlusher->buffer,
                                                                              activeFlusher->count, activeFlusher->sink);
    }
}

// NPP's NV12 -> 8 bit RGB, BT.601 limited range (common/bench_npp.h): the first step of the pipelines with an 8 bit
// RGB frame, OpenCV's (OpenCV CUDA has no NV12 conversion) and NPP's
inline void nv12ToRgb8(const Context& ctx, Npp8u* dst, int dstStep) {
    beforeKernel();
    npp::nv12ToRgb8Bt601Limited(ctx.y, ctx.uv, ctx.pitch, dst, dstStep, NppiSize{FRAME_W, FRAME_H}, ctx.npp);
}

// The reference an implementation is checked against, with its tolerance: the Figure 1 chain on FKL's CPU backend in
// float (FKL BVF); the same with the RGB rounded to 8 bits, for the pipelines with an 8 bit RGB frame (OpenCV, FKL
// VF+HF), which resize at dst * scale like FKL; or that with NPP's resize sampling (NPP, nppConventionReference8bit).
// Fkl8bitFloorBox (the 8 bit RGB and a box filter that rounds down, NPP's 8 bit one) and FklFloatRounded (the float
// chain with the filtered RGB rounded half up to 8 bits, NPPDx's 8 bit write) are only for programs with their own
// reference() (fkl-neighborhood/). NppdxPixelCenterFloat (float, NPPDx's resize sampling at pixel centers) is for
// nppdx-wins/.
enum class Reference { FklFloat, Fkl8bit, NppSampling8bit, Fkl8bitFloorBox, FklFloatRounded, NppdxPixelCenterFloat };

inline const char* referenceName(Reference r) {
    switch (r) {
        case Reference::FklFloat: return "fkl_cpu_float";
        case Reference::Fkl8bit: return "fkl_cpu_8bit_rgb";
        case Reference::Fkl8bitFloorBox: return "fkl_cpu_8bit_rgb_floor_box";
        case Reference::FklFloatRounded: return "fkl_cpu_float_rounded_8bit";
        case Reference::NppdxPixelCenterFloat: return "nppdx_pixel_center_float";
        default: return "npp_sampling_8bit";
    }
}

// 1e-3 against the float references, one level of 8 bit RGB against the 8 bit ones
inline float toleranceOf(Reference r) {
    return r == Reference::FklFloat || r == Reference::NppdxPixelCenterFloat ? TOLERANCE : TOLERANCE_8BIT;
}

// Each implementation enqueues one complete pipeline on the regions of iteration t on ctx.stream with run(t), and
// returns the N x OUT_H x OUT_W x 3 float output with download(). fillNaN() poisons the output before validation.
using Stage = std::function<void(size_t)>;
struct Implementation {
    std::string name;
    Reference reference{Reference::FklFloat};
    Stage run;
    std::function<std::vector<float>()> download;
    std::function<void()> fillNaN;
    // Optional: the kernels of a multi-kernel pipeline, in order, for the per-kernel breakdown
    std::vector<std::pair<std::string, Stage>> stages;
};

inline std::vector<float> downloadDevice(const void* data, size_t floats) {
    std::vector<float> host(floats);
    gpuErrchk(cudaMemcpy(host.data(), data, floats * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
}

// The first N regions of iteration t as FKL's rectangles
template <size_t N>
std::array<fk::Rect, N> fklRects(const Context& ctx, size_t t) {
    std::array<fk::Rect, N> rects{};
    for (size_t i = 0; i < N; ++i) {
        const Region& r = ctx.regions->at(t, i);
        rects[i] = fk::Rect(r.x, r.y, r.width, r.height);
    }
    return rects;
}

// The rectangles of every iteration, converted before the loops: what FKL's Crop takes
template <size_t N>
using FklRegionSets = std::shared_ptr<const std::vector<std::array<fk::Rect, N>>>;

template <size_t N>
FklRegionSets<N> fklRegionSets(const Context& ctx) {
    auto sets = std::make_shared<std::vector<std::array<fk::Rect, N>>>(ctx.regions->iterations());
    for (size_t t = 0; t < sets->size(); ++t) {
        (*sets)[t] = fklRects<N>(ctx, t);
    }
    return sets;
}

// FKL with Backwards Vertical Fusion: one executeOperations call, one kernel for the whole chain (the paper's Listing 2).
// Its build() calls and the fusion of the chain take the regions of iteration t, in every execution.
template <size_t N>
Implementation makeFklBvf(Context& ctx) {
    auto out = std::make_shared<fk::Tensor<fk::float3>>(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const auto regions = fklRegionSets<N>(ctx);
    Implementation impl{"fkl_bvf", Reference::FklFloat};
    impl.run = [&ctx, out, regions](size_t t) {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
            YUVToRGB::build(),
            fk::Crop<>::build((*regions)[t]),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
            fk::Mul<fk::float3>::build(mulValue),
            fk::Sub<fk::float3>::build(subValue),
            fk::Div<fk::float3>::build(divValue),
            fk::TensorWrite<fk::float3>::build(out->ptr()));
    };
    impl.download = [out]() { return downloadDevice(out->ptr().data, N * OUT_FLOATS); };
    impl.fillNaN = [out]() { gpuErrchk(cudaMemset(out->ptr().data, 0xFF, N * OUT_FLOATS * sizeof(float))); };
    return impl;
}

// The stages of a multi-kernel pipeline, run one after the other
inline Stage runStages(const std::vector<std::pair<std::string, Stage>>& stages) {
    return [stages](size_t t) {
        for (const auto& stage : stages) {
            stage.second(t);
        }
    };
}

// References on FKL's CPU backend, for the regions of iteration t. The float reference is the Figure 1 chain itself. The
// 8 bit reference quantizes the RGB values to 8 bit right after the color conversion, like the 8 bit RGB frame
// pipelines do. Without LETTERBOX, the chain has no AddBorder and the output planes are RES_W x RES_H.
template <size_t N, bool EIGHT_BIT, bool LETTERBOX = true>
std::vector<float> cpuReference(const Context& ctx, size_t t) {
    const auto crops = fklRects<N>(ctx, t);
    constexpr int outW = LETTERBOX ? OUT_W : RES_W;
    constexpr int outH = LETTERBOX ? OUT_H : RES_H;
    constexpr size_t outFloats = static_cast<size_t>(outW) * outH * 3;
    fk::Stream_<fk::ParArch::CPU> cpuStream;
    fk::Tensor<fk::float3> out(outW, outH, N, 1, fk::MemType::Host);
    const auto tail = [&](const auto& rgbRead) {
        const auto resize =
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H));
        if constexpr (LETTERBOX) {
            fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
                rgbRead,
                fk::Crop<>::build(crops),
                resize,
                fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
                fk::Mul<fk::float3>::build(mulValue),
                fk::Sub<fk::float3>::build(subValue),
                fk::Div<fk::float3>::build(divValue),
                fk::TensorWrite<fk::float3>::build(out.ptr()));
        } else {
            fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
                rgbRead,
                fk::Crop<>::build(crops),
                resize,
                fk::Mul<fk::float3>::build(mulValue),
                fk::Sub<fk::float3>::build(subValue),
                fk::Div<fk::float3>::build(divValue),
                fk::TensorWrite<fk::float3>::build(out.ptr()));
        }
    };
    const auto rgbRead = fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameHost).then(YUVToRGB::build());
    if constexpr (EIGHT_BIT) {
        tail(rgbRead.then(fk::NormalizePixel<fk::float3, fk::ColorDepth::p8bit>::build())
                    .then(fk::Add<fk::float3>::build(fk::make_set<fk::float3>(QUANTIZATION_OFFSET)))
                    .then(fk::SaturateDenormalizePixel<fk::float3, fk::uchar3, fk::ColorDepth::p8bit>::build())
                    .then(fk::Cast<fk::uchar3, fk::float3>::build()));
    } else {
        tail(rgbRead);
    }
    cpuStream.sync();
    const float* data = reinterpret_cast<const float*>(out.ptr().data);
    return std::vector<float>(data, data + N * outFloats);
}

// Plain C++ reference with NPP's bilinear sampling convention. FKL and OpenCV CUDA sample at src = dst * scale.
// NPP's nppiResize / nppiResizeBatch 32f linear sample at src = dst * scale - 0.25, clamped to the image
// (measured with linear ramps at upscale factors 2, 3.2 and 5). That is the pixel-center convention,
// (dst + 0.5) * scale - 0.5, only for an exact 2x upscale (scale 0.5); at 3.2x pixel centers would be
// dst * scale - 0.34375. Same 8 bit RGB, border and pointwise tail as the 8 bit reference, for the regions of
// iteration t, each with its own size and scale. Only measured for upscales, which every region size of
// common/bench_geometry.h is. Without LETTERBOX, the output planes are RES_W x RES_H, without the border rows.
constexpr float NPP_SAMPLE_OFFSET = -0.25f;

template <size_t N, bool LETTERBOX = true>
std::vector<float> nppConventionReference8bit(const Context& ctx, size_t t) {
    const int outW = LETTERBOX ? OUT_W : RES_W, outH = LETTERBOX ? OUT_H : RES_H;
    const int borderTop = LETTERBOX ? BORDER_TOP : 0;
    std::vector<float> out(N * static_cast<size_t>(outW) * outH * 3);
    for (size_t i = 0; i < N; ++i) {
        const Region& r = ctx.regions->at(t, i);
        const int w = r.width, h = r.height;
        const float scaleX = static_cast<float>(w) / RES_W;
        const float scaleY = static_cast<float>(h) / RES_H;
        std::vector<float> crop(static_cast<size_t>(w) * h * 3);
        const auto at = [&](int y, int x, int c) -> float& {
            return crop[(static_cast<size_t>(y) * w + x) * 3 + c];
        };
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const fk::float3 rgb = YUVToRGB::exec(
                    ctx.frameHost->readAt(fk::Point{static_cast<int>(r.x) + x, static_cast<int>(r.y) + y, 0}));
                const float channels[3] = {rgb.x, rgb.y, rgb.z};
                for (int c = 0; c < 3; ++c) {
                    at(y, x, c) = std::floor(std::min(255.f, std::max(0.f, channels[c])) + 0.5f);
                }
            }
        }
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                float* dst = &out[((i * outH + y) * outW + x) * 3];
                const int ry = y - borderTop;
                float value[3] = {0.f, 0.f, 0.f};
                if (ry >= 0 && ry < RES_H) {
                    const float sx = std::min(static_cast<float>(w - 1), std::max(0.f, x * scaleX + NPP_SAMPLE_OFFSET));
                    const float sy = std::min(static_cast<float>(h - 1), std::max(0.f, ry * scaleY + NPP_SAMPLE_OFFSET));
                    const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
                    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
                    const float fx = sx - x0, fy = sy - y0;
                    for (int c = 0; c < 3; ++c) {
                        value[c] = (at(y0, x0, c) * (1.f - fx) + at(y0, x1, c) * fx) * (1.f - fy) +
                                   (at(y1, x0, c) * (1.f - fx) + at(y1, x1, c) * fx) * fy;
                    }
                }
                for (int c = 0; c < 3; ++c) {
                    dst[c] = (value[c] * MUL[c] - SUB[c]) / DIV[c];
                }
            }
        }
    }
    return out;
}

// The reference of batch N on the regions of iteration t with the given conventions
template <size_t N>
std::vector<float> referenceOutput(const Context& ctx, Reference r, size_t t) {
    switch (r) {
        case Reference::FklFloat: return cpuReference<N, false>(ctx, t);
        case Reference::Fkl8bit: return cpuReference<N, true>(ctx, t);
        case Reference::Fkl8bitFloorBox:
        case Reference::FklFloatRounded:
            throw std::runtime_error("This reference is only computed by programs with their own reference()");
        default: return nppConventionReference8bit<N>(ctx, t);
    }
}

// The optional members of a Program (see the top of this file), or the Figure 1 defaults
template <typename P, size_t N>
std::vector<float> referenceOf(const Context& ctx, Reference r, size_t t) {
    if constexpr (requires { P::template reference<N>(ctx, r, t); }) {
        return P::template reference<N>(ctx, r, t);
    } else {
        return referenceOutput<N>(ctx, r, t);
    }
}

template <typename P>
constexpr const char* fklNameOf() {
    if constexpr (requires { P::fklName; }) {
        return P::fklName;
    } else {
        return "fkl_bvf";
    }
}

template <typename P>
constexpr const char* fklLabelOf() {
    if constexpr (requires { P::fklLabel; }) {
        return P::fklLabel;
    } else {
        return "FKL BVF";
    }
}

// The FKL implementation in "baseline time / BVF time"
template <typename P>
constexpr const char* fklShortLabelOf() {
    if constexpr (requires { P::fklShortLabel; }) {
        return P::fklShortLabel;
    } else {
        return "BVF";
    }
}

// The regions of iterations 0 to iterations - 1: of size P::regionSize, or those of common/bench_geometry.h
template <typename P>
std::unique_ptr<RegionSets> regionSetsOf(size_t iterations) {
    if constexpr (requires { P::regionSize; }) {
        return std::make_unique<RegionSets>(iterations, P::regionSize);
    } else {
        return std::make_unique<RegionSets>(iterations);
    }
}

template <typename P>
constexpr auto batchesOf() {
    if constexpr (requires { P::batches; }) {
        return P::batches;
    } else {
        return BATCHES;
    }
}

// Compares NPP's 8 bit NV12->RGB with FKL's float conversion, to show how NPP rounds (for the pipelines that use it)
inline std::string nppConversionDiagnostic(const Context& ctx) {
    Npp8u* rgb8 = nullptr;
    size_t pitch = 0;
    gpuErrchk(cudaMallocPitch(reinterpret_cast<void**>(&rgb8), &pitch, FRAME_W * 3, FRAME_H));
    nv12ToRgb8(ctx, rgb8, static_cast<int>(pitch));
    std::vector<Npp8u> host(static_cast<size_t>(FRAME_W) * 3 * FRAME_H);
    gpuErrchk(cudaMemcpy2DAsync(host.data(), FRAME_W * 3, rgb8, pitch, FRAME_W * 3, FRAME_H, cudaMemcpyDeviceToHost,
                                ctx.stream));
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    gpuErrchk(cudaFree(rgb8));
    size_t diffRound = 0, diffTrunc = 0, total = 0;
    int maxRound = 0, maxTrunc = 0;
    for (int y = 0; y < FRAME_H; ++y) {
        const Npp8u* row = host.data() + static_cast<size_t>(y) * FRAME_W * 3;
        for (int x = 0; x < FRAME_W; ++x) {
            const fk::uchar3 yuv = ctx.frameHost->readAt(fk::Point{x, y, 0});
            const fk::float3 rgb = YUVToRGB::exec(yuv);
            const float channels[3] = {rgb.x, rgb.y, rgb.z};
            for (int c = 0; c < 3; ++c) {
                const float clamped = std::min(255.f, std::max(0.f, channels[c]));
                const int rounded = static_cast<int>(std::floor(clamped + 0.5f));
                const int truncated = static_cast<int>(clamped);
                const int npp = row[3 * x + c];
                diffRound += npp != rounded;
                diffTrunc += npp != truncated;
                maxRound = std::max(maxRound, std::abs(npp - rounded));
                maxTrunc = std::max(maxTrunc, std::abs(npp - truncated));
                total++;
            }
        }
    }
    std::ostringstream os;
    os << "NPP NV12->RGB (ColorTwist32f with the BT.601 limited range matrix of the standard, bench_npp.h) vs FKL's "
          "float BT.601 limited range conversion:\n"
       << "  vs round-to-nearest of FKL: " << diffRound << " of " << total << " channel values differ, max "
       << maxRound << " levels\n"
       << "  vs truncation of FKL:       " << diffTrunc << " of " << total << " channel values differ, max "
       << maxTrunc << " levels\n";
    return os.str();
}

// Latency of each complete pipeline, the way FKL's two-execution benchmarks measure, with the implementations never
// running at the same time: each implementation's complete loop (the warm-up iterations, then the timed ones, on the
// regions of each iteration) runs before the next one's, the baselines first and FKL BVF last. Each run is between two
// CUDA events on the same stream (everything the host does to enqueue it is included) and followed by a
// synchronization. The frame is already in device memory. Returns the TIMED_ITERS execution times of each
// implementation, in ms, in iteration order: the speedups pair the runs of the same iteration (identical inputs).
inline std::vector<std::vector<float>> measureSequential(const Context& ctx, const std::vector<Implementation>& impls) {
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    std::vector<std::vector<float>> ms(impls.size(), std::vector<float>(TIMED_ITERS));
    for (size_t k = 0; k < impls.size(); ++k) {
        for (int i = 0; i < WARMUP_ITERS + TIMED_ITERS; ++i) {
            gpuErrchk(cudaEventRecord(start, ctx.stream));
            impls[k].run(static_cast<size_t>(i));
            gpuErrchk(cudaEventRecord(stop, ctx.stream));
            gpuErrchk(cudaEventSynchronize(stop));
            if (i >= WARMUP_ITERS) {
                gpuErrchk(cudaEventElapsedTime(&ms[k][i - WARMUP_ITERS], start, stop));
            }
        }
    }
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return ms;
}

// Time of each kernel of a multi-kernel pipeline: CUDA events between its kernels on the same stream
struct StageTiming {
    std::string name;
    int batch{0};
    std::string stage;
    TimingStats timing;
};
inline std::vector<StageTiming> breakdown;

inline void measureStages(const Context& ctx, const Implementation& impl, int batch) {
    const size_t numStages = impl.stages.size();
    std::vector<cudaEvent_t> events(numStages + 1);
    for (auto& e : events) {
        gpuErrchk(cudaEventCreate(&e));
    }
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        impl.run(static_cast<size_t>(i));
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
    }
    std::vector<std::vector<float>> ms(numStages, std::vector<float>(TIMED_ITERS));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        gpuErrchk(cudaEventRecord(events[0], ctx.stream));
        for (size_t s = 0; s < numStages; ++s) {
            impl.stages[s].second(static_cast<size_t>(WARMUP_ITERS + i));
            gpuErrchk(cudaEventRecord(events[s + 1], ctx.stream));
        }
        gpuErrchk(cudaEventSynchronize(events[numStages]));
        for (size_t s = 0; s < numStages; ++s) {
            gpuErrchk(cudaEventElapsedTime(&ms[s][i], events[s], events[s + 1]));
        }
    }
    for (auto& e : events) {
        gpuErrchk(cudaEventDestroy(e));
    }
    std::vector<float> total(TIMED_ITERS, 0.f);
    for (size_t s = 0; s < numStages; ++s) {
        for (int i = 0; i < TIMED_ITERS; ++i) {
            total[i] += ms[s][i];
        }
        breakdown.push_back({impl.name, batch, impl.stages[s].first, computeStats(ms[s])});
    }
    breakdown.push_back({impl.name, batch, "total", computeStats(total)});
}

// Kernel, memset and memcpy counts of one pipeline, from CUPTI activity tracing (no hardware counters, so it works
// without admin rights). It runs in a separate pass after all the timings.
struct LaunchCounts {
    size_t kernels{0}, memsets{0}, memcpys{0};
};

// GPU execution of one kernel, from its CUPTI activity record (nanosecond timestamps)
struct KernelRecord {
    uint64_t start{0}, end{0};
    unsigned gridX{0}, gridY{0}, gridZ{0}, blockX{0}, blockY{0}, blockZ{0};
};

namespace cupti {
inline LaunchCounts counts;
inline bool keepKernels = false;
inline std::vector<KernelRecord> kernels;

#define CUPTI_CHECK(expr)                                                                                    \
    do {                                                                                                     \
        const CUptiResult result_ = (expr);                                                                  \
        if (result_ != CUPTI_SUCCESS) {                                                                      \
            const char* message_ = nullptr;                                                                  \
            cuptiGetResultString(result_, &message_);                                                        \
            throw std::runtime_error(std::string("CUPTI error ") + (message_ ? message_ : "") + " in " + #expr); \
        }                                                                                                    \
    } while (0)

inline void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *size = 8 << 20;
    // CUPTI needs 8 byte alignment, which malloc guarantees (std::aligned_alloc does not exist on Windows)
    *buffer = static_cast<uint8_t*>(std::malloc(*size));
    *maxNumRecords = 0;
}

inline void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            counts.kernels++;
            // CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL records are CUpti_ActivityKernel12 in CUDA 13.3 and 13 in 13.4; new
            // versions only append fields, so the ones read here have the same offsets in both (and in 9 to 11)
            const auto* k = reinterpret_cast<const CUpti_ActivityKernel12*>(record);
            const bool flushKernel = k->name && std::strstr(k->name, "flushL2Kernel");
            if (keepKernels && !flushKernel) {
                kernels.push_back({k->start, k->end, static_cast<unsigned>(k->gridX), static_cast<unsigned>(k->gridY),
                                   static_cast<unsigned>(k->gridZ), static_cast<unsigned>(k->blockX),
                                   static_cast<unsigned>(k->blockY), static_cast<unsigned>(k->blockZ)});
            }
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            counts.memsets++;
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            counts.memcpys++;
        }
    }
    std::free(buffer);
}

inline void init() {
    CUPTI_CHECK(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}
} // namespace cupti

// On the regions of the first timed iteration: the counts do not depend on the regions
inline LaunchCounts countLaunches(const Context& ctx, const Implementation& impl) {
    gpuErrchk(cudaDeviceSynchronize());
    CUPTI_CHECK(cuptiActivityFlushAll(0));
    cupti::counts = LaunchCounts{};
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    impl.run(WARMUP_ITERS);
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    return cupti::counts;
}

// What the GPU actually does in one pipeline, from CUPTI kernel records over PROFILE_RUNS runs (median per run), on the
// regions of iterations PROFILE_WARMUP to PROFILE_WARMUP + PROFILE_RUNS - 1 after PROFILE_WARMUP runs on the ones
// before: the sum of the kernel execution times (GPU busy), the time from the first kernel start to the last kernel
// end, and the execution time and launch configuration of every kernel of the pipeline

struct KernelProfile {
    unsigned gridX, gridY, gridZ, blockThreads;
    double medianUs;
};

struct PipelineProfile {
    std::string name;
    int batch{0};
    size_t kernelsPerRun{0};
    double busyUs{0.}, spanUs{0.};
    std::vector<KernelProfile> kernels;
};

inline double medianOf(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline PipelineProfile profilePipeline(const Context& ctx, const Implementation& impl, int batch) {
    for (int i = 0; i < PROFILE_WARMUP; ++i) {
        impl.run(static_cast<size_t>(i));
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
    }
    CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    cupti::kernels.clear();
    cupti::keepKernels = true;
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    for (int r = 0; r < PROFILE_RUNS; ++r) {
        impl.run(static_cast<size_t>(PROFILE_WARMUP + r));
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
    }
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    cupti::keepKernels = false;
    auto& k = cupti::kernels;
    std::sort(k.begin(), k.end(), [](const KernelRecord& a, const KernelRecord& b) { return a.start < b.start; });
    if (k.empty() || k.size() % PROFILE_RUNS != 0) {
        throw std::runtime_error("Unexpected number of kernel records for " + impl.name);
    }
    PipelineProfile profile{impl.name, batch, k.size() / PROFILE_RUNS};
    const size_t perRun = profile.kernelsPerRun;
    std::vector<double> busy, span;
    std::vector<std::vector<double>> perKernel(perRun);
    for (int r = 0; r < PROFILE_RUNS; ++r) {
        double b = 0.;
        for (size_t i = 0; i < perRun; ++i) {
            const KernelRecord& rec = k[r * perRun + i];
            b += (rec.end - rec.start) * 1e-3;
            perKernel[i].push_back((rec.end - rec.start) * 1e-3);
        }
        busy.push_back(b);
        span.push_back((k[r * perRun + perRun - 1].end - k[r * perRun].start) * 1e-3);
    }
    profile.busyUs = medianOf(busy);
    profile.spanUs = medianOf(span);
    for (size_t i = 0; i < perRun; ++i) {
        const KernelRecord& rec = k[i];
        profile.kernels.push_back({rec.gridX, rec.gridY, rec.gridZ, rec.blockX * rec.blockY * rec.blockZ,
                                   medianOf(perKernel[i])});
    }
    return profile;
}

inline std::vector<PipelineProfile> profiles;

// Every timed iteration's time of every implementation, one file per mode (per_iteration_latency.csv,
// per_iteration_gpu_only.csv, per_iteration_gpu_only_l2_flushed.csv), when the Program has perIterationTimes: the
// directory is set by gpuBenchMain; each file is truncated by the first batch of a process and appended by the others
inline std::filesystem::path perIterationDir;
inline void writePerIteration(const std::string& mode, const size_t batch, const std::vector<Implementation>& impls,
                              const std::vector<std::vector<float>>& ms) {
    if (perIterationDir.empty()) {
        return;
    }
    static std::map<std::string, bool> started;
    const auto file = perIterationDir / ("per_iteration_" + mode + ".csv");
    std::ofstream os(file, started[mode] ? std::ios::app : std::ios::trunc);
    if (!started[mode]) {
        os << "N,iteration,implementation,us\n";
        started[mode] = true;
    }
    for (size_t k = 0; k < impls.size(); ++k) {
        for (size_t i = 0; i < ms[k].size(); ++i) {
            os << batch << "," << WARMUP_ITERS + i << "," << impls[k].name << "," << std::fixed << std::setprecision(3)
               << ms[k][i] * 1000.f << "\n";
        }
    }
}

struct Result {
    std::string name;
    int batch{0};
    TimingStats timing;
    // Mean over the timed iterations of (this implementation's time / FKL BVF's time) in the same iteration
    double meanSpeedupOfBvf{NAN};
    // GPU time only: sum of the execution times of the pipeline's kernels (CUPTI), per execution
    TimingStats gpuTiming;
    double meanGpuSpeedupOfBvf{NAN};
    // GPU time only with L2 flushed before every kernel
    TimingStats gpuFlushedTiming;
    double meanGpuFlushedSpeedupOfBvf{NAN};
    // Over the validated iterations together, and per validated iteration (iteration, against its own reference,
    // against the float reference)
    ErrorStats errorVsOwnReference;
    ErrorStats errorVsFloatReference;
    std::vector<std::tuple<size_t, ErrorStats, ErrorStats>> errorsPerIteration;
    Reference reference{Reference::FklFloat};
    LaunchCounts launches;
    float tolerance() const { return toleranceOf(reference); }
    // Within the tolerance in every validated iteration
    bool withinTolerance() const {
        return std::all_of(errorsPerIteration.begin(), errorsPerIteration.end(), [&](const auto& e) {
            return validation::withinTolerance(std::get<1>(e), tolerance());
        });
    }
};

// The implementations of a program for batch N: its baselines, then FKL BVF, which has no intermediates in memory and
// keeps float3 from the color conversion to the write (or the program's own FKL implementation)
template <typename P, size_t N>
std::vector<Implementation> implementationsOf(Context& ctx) {
    auto impls = P::template implementations<N>(ctx);
    if constexpr (requires { P::template fklImplementation<N>(ctx); }) {
        impls.push_back(P::template fklImplementation<N>(ctx));
    } else {
        impls.push_back(makeFklBvf<N>(ctx));
    }
    return impls;
}

// The names of a program's baselines (every implementation but FKL BVF)
template <typename P>
std::vector<std::string> baselineNames(Context& ctx) {
    std::vector<std::string> names;
    for (const auto& impl : P::template implementations<batchesOf<P>()[0]>(ctx)) {
        names.push_back(impl.name);
    }
    return names;
}

// Validation on the validated iterations, timing (one implementation's loop after the other) with per-iteration
// speedups, and per kernel breakdown, for
// batch N
template <typename P, size_t N>
void runBatch(Context& ctx, std::vector<Result>& results) {
    std::cout << "=== N = " << N << std::endl;
    const auto impls = implementationsOf<P, N>(ctx);
    const size_t bvf = impls.size() - 1;
    std::vector<Result> batchResults(impls.size());
    for (size_t k = 0; k < impls.size(); ++k) {
        batchResults[k].name = impls[k].name;
        batchResults[k].batch = static_cast<int>(N);
        batchResults[k].reference = impls[k].reference;
    }
    // Before the timing, on the regions of the validated timed iterations (the timed loop runs the same code on the
    // same regions): every pipeline against the reference of its own conventions, and against the float one
    for (const size_t k : validatedIterations(TIMED_ITERS)) {
        const size_t t = WARMUP_ITERS + k;
        std::map<Reference, std::vector<float>> references;
        const auto referenceFor = [&](Reference r) -> const std::vector<float>& {
            if (!references.count(r)) {
                references[r] = referenceOf<P, N>(ctx, r, t);
            }
            return references.at(r);
        };
        for (size_t i = 0; i < impls.size(); ++i) {
            impls[i].fillNaN();
            // fillNaN's cudaMemset runs on the legacy default stream, which ctx.stream (non-blocking) does not wait for:
            // without this, the poison can land after the pipeline's writes
            gpuErrchk(cudaDeviceSynchronize());
            impls[i].run(t);
            gpuErrchk(cudaStreamSynchronize(ctx.stream));
            const auto output = impls[i].download();
            Result& r = batchResults[i];
            const ErrorStats own = compare(output, referenceFor(impls[i].reference));
            const ErrorStats vsFloat = compare(output, referenceFor(Reference::FklFloat));
            r.errorsPerIteration.emplace_back(t, own, vsFloat);
            r.errorVsOwnReference = r.errorsPerIteration.size() == 1 ? own : validation::combine(r.errorVsOwnReference, own);
            r.errorVsFloatReference =
                r.errorsPerIteration.size() == 1 ? vsFloat : validation::combine(r.errorVsFloatReference, vsFloat);
        }
    }
    const auto ms = measureSequential(ctx, impls);
    writePerIteration("latency", N, impls, ms);
    for (size_t k = 0; k < impls.size(); ++k) {
        Result& r = batchResults[k];
        r.timing = computeStats(ms[k]);
        if (k != bvf) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += ms[k][i] / ms[bvf][i];
            }
            r.meanSpeedupOfBvf = sum / TIMED_ITERS;
        }
        std::cout << std::left << std::setw(12) << r.name << std::right << std::fixed << std::setprecision(1)
                  << " median " << std::setw(7) << r.timing.median << " us  mean " << std::setw(7) << r.timing.mean
                  << " us  p10 " << std::setw(7) << r.timing.p10 << "  p90 " << std::setw(7) << r.timing.p90;
        if (k != bvf) {
            std::cout << "  " << fklLabelOf<P>() << " mean speedup " << std::setw(6) << std::setprecision(2)
                      << r.meanSpeedupOfBvf << "x";
        }
        std::cout << "  max err " << std::scientific << std::setprecision(2) << r.errorVsOwnReference.max() << " ("
                  << referenceName(r.reference) << ")" << std::defaultfloat << std::endl;
        results.push_back(r);
    }
    // Per stage breakdown of the multi-kernel pipelines
    for (const auto& impl : impls) {
        if (!impl.stages.empty()) {
            measureStages(ctx, impl, static_cast<int>(N));
        }
    }
}

// Kernel counts of every implementation for batch N, after all the timings
template <typename P, size_t N>
void countBatch(Context& ctx, std::vector<Result>& results) {
    for (const auto& impl : implementationsOf<P, N>(ctx)) {
        const LaunchCounts launches = countLaunches(ctx, impl);
        for (auto& r : results) {
            if (r.name == impl.name && r.batch == static_cast<int>(N)) {
                r.launches = launches;
            }
        }
        std::cout << std::left << std::setw(10) << impl.name << " N " << std::right << std::setw(3) << N
                  << "  kernels " << std::setw(4) << launches.kernels << "  memsets " << launches.memsets
                  << "  memcpys " << launches.memcpys << std::endl;
    }
}

template <typename P, size_t N>
void profileBatch(Context& ctx) {
    for (const auto& impl : implementationsOf<P, N>(ctx)) {
        profiles.push_back(profilePipeline(ctx, impl, static_cast<int>(N)));
        const auto& p = profiles.back();
        std::cout << std::left << std::setw(24) << p.name << " N " << std::right << std::setw(3) << N << "  kernels "
                  << std::setw(3) << p.kernelsPerRun << "  GPU busy " << std::fixed << std::setprecision(2)
                  << std::setw(7) << p.busyUs << " us  first start to last end " << std::setw(7) << p.spanUs << " us"
                  << std::defaultfloat << std::endl;
    }
}

// Target for Nsight Compute: runs one pipeline NCU_WARMUP times on the regions of iterations NCU_ITERATION + 1 onwards,
// then once on those of iteration NCU_ITERATION inside the NVTX range "measure", so that
// ncu --nvtx --nvtx-include "measure/" profiles exactly that execution, on the same regions in every benchmark
template <typename P, size_t N>
void ncuTarget(Context& ctx, const std::string& name) {
    const auto impls = implementationsOf<P, N>(ctx);
    const auto impl = std::find_if(impls.begin(), impls.end(), [&](const Implementation& i) { return i.name == name; });
    if (impl == impls.end()) {
        std::string valid;
        for (const auto& i : impls) {
            valid += " " + i.name;
        }
        throw std::runtime_error("Unknown implementation " + name + ", this benchmark has:" + valid);
    }
    for (int i = 1; i <= NCU_WARMUP; ++i) {
        impl->run(NCU_ITERATION + i);
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
    }
    nvtxRangePushA("measure");
    impl->run(NCU_ITERATION);
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    nvtxRangePop();
}

// GPU time only benchmark: the same loops as the CPU + GPU one, one implementation after the other, with CUPTI kernel
// tracing on during the timed iterations. The GPU time of one execution is the sum of the execution times of its
// kernels: no host work and no launch gaps.
template <typename P, size_t N>
void gpuOnlyBatch(Context& ctx, std::vector<Result>& results, L2Flusher* flusher = nullptr) {
    const auto impls = implementationsOf<P, N>(ctx);
    const size_t bvf = impls.size() - 1;
    std::vector<size_t> kernelsPerRun(impls.size());
    for (size_t k = 0; k < impls.size(); ++k) {
        kernelsPerRun[k] = countLaunches(ctx, impls[k]).kernels;
    }
    activeFlusher = flusher;
    // Each implementation's warm-up and traced timed iterations, then the next implementation's; the executions are
    // sequential, so in start time order the records of one implementation are iteration by iteration
    std::vector<std::vector<float>> ms(impls.size(), std::vector<float>(TIMED_ITERS));
    auto& records = cupti::kernels;
    for (size_t k = 0; k < impls.size(); ++k) {
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            impls[k].run(static_cast<size_t>(i));
            gpuErrchk(cudaStreamSynchronize(ctx.stream));
        }
        CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
        records.clear();
        cupti::keepKernels = true;
        CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        for (int i = 0; i < TIMED_ITERS; ++i) {
            impls[k].run(static_cast<size_t>(WARMUP_ITERS + i));
            gpuErrchk(cudaStreamSynchronize(ctx.stream));
            if (i % 100 == 99) {
                CUPTI_CHECK(cuptiActivityFlushAll(0));
            }
        }
        CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
        cupti::keepKernels = false;
        std::sort(records.begin(), records.end(),
                  [](const KernelRecord& a, const KernelRecord& b) { return a.start < b.start; });
        if (records.size() != kernelsPerRun[k] * TIMED_ITERS) {
            throw std::runtime_error("GPU only benchmark: expected " + std::to_string(kernelsPerRun[k] * TIMED_ITERS) +
                                     " kernel records of " + impls[k].name + ", got " + std::to_string(records.size()));
        }
        size_t idx = 0;
        for (int i = 0; i < TIMED_ITERS; ++i) {
            uint64_t ns = 0;
            for (size_t j = 0; j < kernelsPerRun[k]; ++j, ++idx) {
                ns += records[idx].end - records[idx].start;
            }
            ms[k][i] = static_cast<float>(ns * 1e-6);
        }
    }
    activeFlusher = nullptr;
    records.clear();
    writePerIteration(flusher ? "gpu_only_l2_flushed" : "gpu_only", N, impls, ms);
    for (size_t k = 0; k < impls.size(); ++k) {
        auto r = std::find_if(results.begin(), results.end(),
                              [&](const Result& x) { return x.name == impls[k].name && x.batch == static_cast<int>(N); });
        if (r == results.end()) {
            Result fresh;
            fresh.name = impls[k].name;
            fresh.batch = static_cast<int>(N);
            fresh.launches.kernels = kernelsPerRun[k];
            results.push_back(fresh);
            r = results.end() - 1;
        }
        TimingStats& timing = flusher ? r->gpuFlushedTiming : r->gpuTiming;
        double& speedup = flusher ? r->meanGpuFlushedSpeedupOfBvf : r->meanGpuSpeedupOfBvf;
        timing = computeStats(ms[k]);
        if (k != bvf) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += ms[k][i] / ms[bvf][i];
            }
            speedup = sum / TIMED_ITERS;
        }
        std::cout << (flusher ? "GPU only, L2 flushed " : "GPU only ") << std::left << std::setw(12) << r->name
                  << std::right << " N " << std::setw(3) << N << std::fixed << std::setprecision(2) << "  median "
                  << std::setw(7) << timing.median << " us  p10 " << std::setw(7) << timing.p10 << "  p90 "
                  << std::setw(7) << timing.p90;
        if (k != bvf) {
            std::cout << "  " << fklLabelOf<P>() << " mean speedup " << std::setw(6) << speedup << "x";
        }
        std::cout << std::defaultfloat << std::endl;
    }
}

// Calls f.template operator()<BATCHES[i]>() for every batch size, in order
template <typename F>
void forEachBatch(F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (f.template operator()<BATCHES[Is]>(), ...);
    }(std::make_index_sequence<BATCHES.size()>{});
}

// Calls f.template operator()<N>() for every batch size of program P (batchesOf), in order
template <typename P, typename F>
void forEachBatchOf(F&& f) {
    constexpr auto batches = batchesOf<P>();
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (f.template operator()<batches[Is]>(), ...);
    }(std::make_index_sequence<batches.size()>{});
}

// Calls f.template operator()<N>() for the batch size N == batch of program P, or throws
template <typename P, typename F>
void withBatch(size_t batch, F&& f) {
    bool found = false;
    forEachBatchOf<P>([&]<size_t N>() {
        if (N == batch) {
            f.template operator()<N>();
            found = true;
        }
    });
    if (!found) {
        throw std::runtime_error("Batch size " + std::to_string(batch) + " is not one of the benchmarked ones");
    }
}

inline L2Flusher makeL2Flusher(const Context& ctx) {
    int device = 0;
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    gpuErrchk(cudaGetDeviceProperties(&prop, device));
    L2Flusher flusher;
    const size_t bytes = 2 * static_cast<size_t>(prop.l2CacheSize);
    flusher.count = bytes / sizeof(int4);
    gpuErrchk(cudaMalloc(&flusher.buffer, bytes));
    gpuErrchk(cudaMemset(flusher.buffer, 0, bytes));
    gpuErrchk(cudaMalloc(&flusher.sink, sizeof(int)));
    flusher.stream = ctx.stream;
    flusher.blocks = prop.multiProcessorCount * 8;
    std::cout << "Flushing " << bytes / (1 << 20) << " MiB (2x L2) before every kernel" << std::endl;
    return flusher;
}

inline void freeL2Flusher(L2Flusher& flusher) {
    gpuErrchk(cudaFree(flusher.buffer));
    gpuErrchk(cudaFree(flusher.sink));
}

// One CSV per baseline in FKL's two-execution format: <baseline>_vs_<fkl>_<suffix><ParArch>.csv, with fkl the name of
// the FKL implementation (fkl_bvf)
template <typename Field>
void writeTwoExecutionFiles(const std::filesystem::path& outDir, const std::vector<std::string>& baselines,
                            const std::string& fkl, const std::vector<Result>& results, const std::string& suffix,
                            Field&& field) {
    const std::string parArch(fk::toStrView(fk::defaultParArch));
    for (const std::string& baseline : baselines) {
        std::ofstream two(outDir / (baseline + "_vs_" + fkl + "_" + suffix + parArch + ".csv"));
        writeTwoExecutionHeader(two, "Batch size", baseline, fkl);
        for (const auto& b : results) {
            if (b.name != baseline) {
                continue;
            }
            const auto f = std::find_if(results.begin(), results.end(), [&](const Result& r) {
                return r.name == fkl && r.batch == b.batch;
            });
            const auto [baseTiming, bvfTiming, speedup] = field(b, *f);
            writeTwoExecutionRow(two, b.batch, baseTiming, bvfTiming, speedup);
        }
    }
}

// System, versions, workload and protocol of a run
template <typename P>
void writePlatform(const std::filesystem::path& outDir, const std::vector<std::string>& baselines,
                   const std::string& diagnostic) {
    int device = 0, runtime = 0, driver = 0;
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    gpuErrchk(cudaGetDeviceProperties(&prop, device));
    gpuErrchk(cudaRuntimeGetVersion(&runtime));
    gpuErrchk(cudaDriverGetVersion(&driver));
    std::string names;
    for (const auto& b : baselines) {
        names += b + ", ";
    }
    char busId[64] = {};
    gpuErrchk(cudaDeviceGetPCIBusId(busId, sizeof(busId), device));
    std::ofstream os(outDir / "platform.txt");
    os << "Benchmark: " << P::title << "\n"
       << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ", " << prop.multiProcessorCount << " SMs, "
       << (prop.totalGlobalMem >> 20) << " MiB), CUDA device " << device << "; in nvidia-smi: "
       << bench::nvidiaSmi(busId, "index,pci.bus_id,uuid") << "\n"
       << "NVIDIA driver: " << bench::nvidiaSmi(busId, "driver_version") << "\n"
       << "CUDA runtime: " << runtime / 1000 << "." << (runtime % 1000) / 10 << ", CUDA driver API: " << driver / 1000
       << "." << (driver % 1000) / 10 << "\n"
       << "nvcc: " << BENCH_NVCC_VERSION << "\n"
       << "nvcc flags: " << BENCH_NVCC_FLAGS << "\n"
       << "Host compiler: " << bench::hostCompiler() << "\n"
       << "FKL: branch " << BENCH_FKL_BRANCH << ", commit " << BENCH_FKL_COMMIT << "\n";
    P::describe(os, outDir);
    os << "CPU: " << bench::cpuName() << "\n"
       << "OS: " << bench::osName() << ", kernel " << bench::kernelVersion() << "\n"
       << "GPU clocks: not locked (locking needs root); application clocks as reported by nvidia-smi: "
       << bench::nvidiaSmi(busId, "clocks.max.sm,clocks.max.mem") << "\n"
       << "\n";
    if constexpr (requires { P::workload(os); }) {
        P::workload(os);
    } else {
        os << "Workload: " << geometry::describe() << ". AddBorder (black), then Mul "
           << "(1/255), Sub (0.485, 0.456, 0.406), Div (0.229, 0.224, 0.225), float3 tensor of N planes " << OUT_W
           << "x" << OUT_H << "\n"
           << "FKL BVF: one executeOperations call, one kernel: no intermediates in memory, float3 from the color "
              "conversion to the write.\n";
    }
    os << "Regions: iteration t of each implementation's loop (the " << WARMUP_ITERS << " warm-up iterations are t in "
       << "[0, " << WARMUP_ITERS << "), the " << TIMED_ITERS << " timed ones t in [" << WARMUP_ITERS << ", "
       << WARMUP_ITERS + TIMED_ITERS << ")) runs that implementation's pipeline on the regions of "
       << "iteration t, generated before the loops and the same for every implementation, program and machine "
       << "(statistics of the timed ones in regions.csv). Each pipeline converts them into its library's arguments "
       << "before the loops; inside the timed region is what the library needs to receive the regions of the "
       << "iteration: FKL's build() calls and the fusion of the chain, and the per-region arguments and calls of "
       << "the baselines (see their pipelines).\n"
       << "Timing: the implementations never run at the same time: for every batch size, each one's complete loop ("
       << WARMUP_ITERS << " warm-up then " << TIMED_ITERS << " timed iterations) runs before the next one's, " << names
       << "then " << fklLabelOf<P>() << ", every run timed with CUDA events on one stream "
       << "around the complete pipeline (all host enqueue work included) and followed by a synchronization. "
       << fklLabelOf<P>() << "'s speedup "
       << "over each baseline is computed per iteration (baseline time / " << fklShortLabelOf<P>()
       << " time of the same iteration, identical inputs) and averaged (mean speedup, FKL's "
       << "two-execution convention). Median, p10, p90 with linear interpolation.\n"
       << "GPU time only: a second pass with the same loops and regions, one implementation after the other, and "
          "CUPTI kernel tracing on during the timed iterations; "
       << "the GPU time of one execution is the sum of the execution times of its kernels (no host work, no launch "
       << "gaps, no memory copies), and " << fklLabelOf<P>()
       << "'s speedups are again computed per execution and averaged.\n"
       << "Validation: before the timing, every pipeline on the regions of the first, every "
       << validation::VALIDATION_STRIDE << "th and the last timed iteration, against its reference (validation.csv, "
       << "one row per iteration): 1e-3 for " << fklLabelOf<P>()
       << " against the float reference, one level of 8 bit RGB for the 8 bit "
       << "pipelines against theirs.\n"
       << "Kernel counts: CUPTI activity tracing of one pipeline (regions of the first timed iteration), in a "
          "separate pass after all timings\n";
    if (!diagnostic.empty()) {
        os << "\n" << diagnostic;
    }
}

// The program: <outDir> [--gpu-only-l2-flushed | --profile-kernels | --ncu-target <impl> <N> | a mode of P], or
// --batches / --workload / --regions <iteration> alone
template <typename P>
int gpuBenchMain(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--batches") {
            // The batch sizes, for the Nsight Compute scripts
            const auto batches = batchesOf<P>();
            for (size_t i = 0; i < batches.size(); ++i) {
                std::cout << (i ? " " : "") << batches[i];
            }
            std::cout << std::endl;
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--workload") {
            // The workload this binary was built for, so that scripts can check that binaries and results match
            if constexpr (requires { P::workload(std::cout); }) {
                P::workload(std::cout);
            } else {
                std::cout << describe() << std::endl;
            }
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--regions") {
            // The regions of one iteration as CSV, for the traffic model of the Nsight Compute profiles
            const size_t t = std::stoul(argv[2]);
            std::cout << regionsCsv(*regionSetsOf<P>(t + 1), t);
            return 0;
        }
        std::filesystem::path outDir = argc > 1 ? argv[1] : "results";
        std::filesystem::create_directories(outDir);
        if constexpr (requires { P::perIterationTimes; }) {
            perIterationDir = outDir;
        }
        if constexpr (requires { P::setup(outDir, std::string()); }) {
            P::setup(outDir, argc > 2 ? std::string(argv[2]) : std::string());
        }

        Context ctx;
        initContext(ctx, regionSetsOf<P>(REGION_ITERATIONS));
        const auto baselines = baselineNames<P>(ctx);

        if (argc > 2 && std::string(argv[2]) == "--gpu-only-l2-flushed") {
            // GPU time only with the L2 flushed before every kernel of every pipeline (no intermediate result or
            // input can be assumed to be in L2). Separate from the main run, which it does not touch.
            L2Flusher flusher = makeL2Flusher(ctx);
            cupti::init();
            std::vector<Result> flushed;
            forEachBatchOf<P>([&]<size_t N>() { gpuOnlyBatch<P, N>(ctx, flushed, &flusher); });
            std::ofstream csv(outDir / "gpu_time_l2_flushed.csv");
            csv << "implementation,N,gpu_median_us,gpu_p10_us,gpu_p90_us,gpu_mean_us,iterations,kernels_launched,"
                << fklNameOf<P>() << "_mean_gpu_speedup\n";
            for (const auto& r : flushed) {
                csv << r.name << "," << r.batch << "," << std::fixed << std::setprecision(2) << r.gpuFlushedTiming.median
                    << "," << r.gpuFlushedTiming.p10 << "," << r.gpuFlushedTiming.p90 << ","
                    << r.gpuFlushedTiming.mean << "," << TIMED_ITERS << "," << r.launches.kernels << ",";
                if (!std::isnan(r.meanGpuFlushedSpeedupOfBvf)) {
                    csv << std::setprecision(3) << r.meanGpuFlushedSpeedupOfBvf;
                }
                csv << "\n";
            }
            writeTwoExecutionFiles(outDir, baselines, fklNameOf<P>(), flushed, "gpu_only_l2_flushed_",
                                   [](const Result& b, const Result& f) {
                return std::make_tuple(b.gpuFlushedTiming, f.gpuFlushedTiming, b.meanGpuFlushedSpeedupOfBvf);
            });
            freeL2Flusher(flusher);
            std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
            return 0;
        }
        if (argc > 4 && std::string(argv[2]) == "--ncu-target") {
            const std::string name = argv[3];
            withBatch<P>(std::stoul(argv[4]), [&]<size_t N>() { ncuTarget<P, N>(ctx, name); });
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--profile-kernels") {
            // GPU execution time of every kernel (CUPTI), without touching the latency results
            cupti::init();
            forEachBatchOf<P>([&]<size_t N>() { profileBatch<P, N>(ctx); });
            std::ofstream summary(outDir / "kernel_gpu_times.csv");
            summary << "implementation,N,kernels_per_pipeline,gpu_busy_us,first_kernel_start_to_last_kernel_end_us\n";
            std::ofstream details(outDir / "kernel_gpu_times_fkl.csv");
            details << "implementation,N,kernel,grid_x,grid_y,grid_z,threads_per_block,total_threads,gpu_time_us\n";
            for (const auto& p : profiles) {
                summary << p.name << "," << p.batch << "," << p.kernelsPerRun << "," << std::fixed
                        << std::setprecision(2) << p.busyUs << "," << p.spanUs << "\n";
                if (p.name.rfind("fkl_", 0) == 0) {
                    for (size_t i = 0; i < p.kernels.size(); ++i) {
                        const auto& k = p.kernels[i];
                        const size_t threads = static_cast<size_t>(k.gridX) * k.gridY * k.gridZ * k.blockThreads;
                        details << p.name << "," << p.batch << "," << i + 1 << "," << k.gridX << "," << k.gridY << ","
                                << k.gridZ << "," << k.blockThreads << "," << threads << "," << std::fixed
                                << std::setprecision(2) << k.medianUs << "\n";
                    }
                }
            }
            std::cout << "Kernel profiles written to " << std::filesystem::absolute(outDir) << std::endl;
            return 0;
        }
        if (argc > 2 && P::extraMode(ctx, std::string(argv[2]), outDir)) {
            return 0;
        }
        if (argc > 2) {
            throw std::runtime_error(std::string("Unknown mode or missing arguments: ") + argv[2] +
                                     " (see the modes in the README of this benchmark)");
        }

        std::cout << "=== " << P::title << std::endl;
        const std::string diagnostic = P::diagnostic(ctx);
        std::cout << diagnostic;

        std::vector<Result> results;
        forEachBatchOf<P>([&]<size_t N>() { runBatch<P, N>(ctx, results); });
        cupti::init();
        std::cout << "=== GPU time only (CUPTI kernel execution times, after all CPU + GPU timings)" << std::endl;
        forEachBatchOf<P>([&]<size_t N>() { gpuOnlyBatch<P, N>(ctx, results); });
        std::cout << "=== Kernel counts (CUPTI activity tracing, after all timings)" << std::endl;
        forEachBatchOf<P>([&]<size_t N>() { countBatch<P, N>(ctx, results); });

        std::ofstream latency(outDir / "latency.csv");
        latency << "implementation,N,median_us,p10_us,p90_us,iterations,max_abs_err,kernels_launched,mean_us,"
                << fklNameOf<P>() << "_mean_speedup\n";
        std::ofstream validationCsv(outDir / "validation.csv");
        validationCsv << "implementation,N,iteration,reference,max_abs_err_r,max_abs_err_g,max_abs_err_b,mean_abs_err,"
                      "non_finite_values,max_abs_err_vs_float_reference,values_beyond_1e-3,values,tolerance,"
                      "within_tolerance\n";
        std::ofstream kernels(outDir / "kernels.csv");
        kernels << "implementation,N,kernels,memsets,memcpys\n";
        bool failed = false;
        for (const auto& r : results) {
            latency << r.name << "," << r.batch << "," << std::fixed << std::setprecision(2) << r.timing.median << ","
                    << r.timing.p10 << "," << r.timing.p90 << "," << TIMED_ITERS << "," << std::scientific
                    << std::setprecision(3) << r.errorVsOwnReference.max() << "," << r.launches.kernels << ","
                    << std::fixed << std::setprecision(2) << r.timing.mean << ",";
            if (!std::isnan(r.meanSpeedupOfBvf)) {
                latency << std::setprecision(3) << r.meanSpeedupOfBvf;
            }
            latency << "\n";
            for (const auto& [iteration, e, vsFloat] : r.errorsPerIteration) {
                validationCsv << r.name << "," << r.batch << "," << iteration << "," << referenceName(r.reference) << ","
                           << std::scientific << std::setprecision(3) << e.maxPerChannel[0] << ","
                           << e.maxPerChannel[1] << "," << e.maxPerChannel[2] << "," << e.mean << "," << e.nonFinite
                           << "," << vsFloat.max() << "," << e.beyond << "," << e.values << "," << r.tolerance() << ","
                           << (validation::withinTolerance(e, r.tolerance()) ? "yes" : "no") << "\n";
            }
            kernels << r.name << "," << r.batch << "," << r.launches.kernels << "," << r.launches.memsets << ","
                    << r.launches.memcpys << "\n";
            // FKL's outputs must be within their tolerance; the libraries' are reported (they depend on the library
            // build and version of each machine)
            failed = failed || (r.name.rfind("fkl_", 0) == 0 && !r.withinTolerance());
        }
        // One CSV per baseline in FKL's two-execution format (times in ms), for both benchmarks:
        // <baseline>_vs_fkl_bvf_GPU_NVIDIA.csv (CPU + GPU) and <baseline>_vs_fkl_bvf_gpu_only_GPU_NVIDIA.csv
        writeTwoExecutionFiles(outDir, baselines, fklNameOf<P>(), results, "", [](const Result& b, const Result& f) {
            return std::make_tuple(b.timing, f.timing, b.meanSpeedupOfBvf);
        });
        writeTwoExecutionFiles(outDir, baselines, fklNameOf<P>(), results, "gpu_only_",
                               [](const Result& b, const Result& f) {
            return std::make_tuple(b.gpuTiming, f.gpuTiming, b.meanGpuSpeedupOfBvf);
        });
        std::ofstream gpuTime(outDir / "gpu_time.csv");
        gpuTime << "implementation,N,gpu_median_us,gpu_p10_us,gpu_p90_us,gpu_mean_us,iterations,kernels_launched,"
                << fklNameOf<P>() << "_mean_gpu_speedup\n";
        for (const auto& r : results) {
            gpuTime << r.name << "," << r.batch << "," << std::fixed << std::setprecision(2) << r.gpuTiming.median << ","
                    << r.gpuTiming.p10 << "," << r.gpuTiming.p90 << "," << r.gpuTiming.mean << "," << TIMED_ITERS << ","
                    << r.launches.kernels << ",";
            if (!std::isnan(r.meanGpuSpeedupOfBvf)) {
                gpuTime << std::setprecision(3) << r.meanGpuSpeedupOfBvf;
            }
            gpuTime << "\n";
        }
        std::ofstream stages(outDir / "kernel_breakdown.csv");
        stages << "implementation,N,kernel,median_us,p10_us,p90_us\n";
        for (const auto& b : breakdown) {
            stages << b.name << "," << b.batch << "," << b.stage << "," << std::fixed << std::setprecision(2)
                   << b.timing.median << "," << b.timing.p10 << "," << b.timing.p90 << "\n";
        }
        writePlatform<P>(outDir, baselines, diagnostic);
        writeRegionStats((outDir / "regions.csv").string(), *ctx.regions, WARMUP_ITERS, TIMED_ITERS);
        std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
        if (failed) {
            std::cerr << "ERROR: an FKL output is not within its tolerance, see validation.csv" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

} // namespace bench

#endif // BVF_BENCH_GPU_H
