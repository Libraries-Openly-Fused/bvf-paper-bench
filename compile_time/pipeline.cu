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

// One pipeline of the paper's Figure 1 per translation unit, for measuring nvcc compile time.
// Select with exactly one of:
//   -DIMPL_HEADERS_FKL  FKL headers only, no pipeline
//   -DIMPL_HEADERS_ALL  FKL + OpenCV + NPP headers, no pipeline
//   -DIMPL_BVF          FKL, one executeOperations call with Backwards Vertical Fusion (Listing 2)
//   -DIMPL_BVF_CWD      the same with the CONDITIONAL_WITH_DEFAULT plane policy: BATCH planes compiled, usedPlanes at
//                       run time
//   -DIMPL_VFHF         FKL, the four kernels of the VF+HF ablation (8 bit RGB frame)
//   -DIMPL_OPENCV       OpenCV CUDA calls, one per operation (NPP for the NV12 conversion)
//   -DIMPL_NPP          NPP calls, batched where they exist
// and -DCT_BATCH=<N> (number of crops). The program is compiled with the nvcc command line of bvf_bench (see run.sh) and
// is never executed: main() calls the pipeline only when it gets an argument, which keeps every template instantiated.
// The pipeline calls are those of src/bvf_bench.cu (makeFklBvf, makeFklVfHf<N, true>, makeOpenCV, makeNPP,
// nv12ToRgb8) without the benchmark scaffolding: same operations, template arguments and order.

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#ifndef IMPL_HEADERS_FKL
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <npp.h>
#endif

#include "bench_geometry.h"

#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CT_BATCH
#error "Define -DCT_BATCH=<number of crops> (not BATCH: FKL uses that name for template parameters)"
#endif

#ifndef IMPL_HEADERS_FKL
#define NPP_CHECK(expr)                                                                                     \
    do {                                                                                                    \
        const NppStatus status_ = (expr);                                                                   \
        if (status_ != NPP_SUCCESS) {                                                                       \
            throw std::runtime_error(std::string("NPP error ") + std::to_string(status_) + " in " + #expr); \
        }                                                                                                   \
    } while (0)
#endif

namespace {

// Frame, crop, resize, border and output sizes of the benchmarks
using namespace bench::geometry;
[[maybe_unused]] constexpr size_t N = CT_BATCH;
[[maybe_unused]] constexpr float MUL[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
[[maybe_unused]] constexpr float SUB[3] = {0.485f, 0.456f, 0.406f};
[[maybe_unused]] constexpr float DIV[3] = {0.229f, 0.224f, 0.225f};
[[maybe_unused]] constexpr float QUANTIZATION_OFFSET = 0.5f / 255.f;

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Full, fk::ColorPrimitives::bt2020>;
[[maybe_unused]] const fk::float3 mulValue{MUL[0], MUL[1], MUL[2]};
[[maybe_unused]] const fk::float3 subValue{SUB[0], SUB[1], SUB[2]};
[[maybe_unused]] const fk::float3 divValue{DIV[0], DIV[1], DIV[2]};
[[maybe_unused]] const fk::float3 black{0.f, 0.f, 0.f};

[[maybe_unused]] std::array<fk::Rect, N> someCrops() {
    std::array<fk::Rect, N> crops{};
    for (size_t i = 0; i < N; ++i) {
        crops[i] = fk::Rect(static_cast<int>(i), static_cast<int>(i), CROP_W, CROP_H);
    }
    return crops;
}

#if defined(IMPL_BVF) || defined(IMPL_BVF_CWD)
// ---------------------------------------------------------------- BVF: Listing 2 of the paper (makeFklBvf)
void runPipeline() {
    fk::Stream stream;
    NV12Image frame(FRAME_W, FRAME_H);
    fk::Tensor<fk::float3> out(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const auto crops = someCrops();
#if defined(IMPL_BVF_CWD)
    // CONDITIONAL_WITH_DEFAULT: the kernel is compiled for BATCH planes and told at run time how many are used
    const int usedPlanes = static_cast<int>(N);
    const auto crop = fk::Crop<>::build(usedPlanes, fk::float3{0.f, 0.f, 0.f}, crops);
#else
    const auto crop = fk::Crop<>::build(crops);
#endif
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::ReadYUV<fk::PixelFormat::NV12>::build(frame),
        YUVToRGB::build(),
        crop,
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
        fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
        fk::Mul<fk::float3>::build(mulValue),
        fk::Sub<fk::float3>::build(subValue),
        fk::Div<fk::float3>::build(divValue),
        fk::TensorWrite<fk::float3>::build(out.ptr()));
}

#elif defined(IMPL_VFHF)
// ---------------------------------------------------------------- VF+HF: four kernels, 8 bit RGB frame
// (makeFklVfHf<N, true>)
template <typename T>
std::array<fk::RawPtr<fk::ND::_2D, T>, N> planeViews(const fk::Tensor<T>& tensor, int width, int height) {
    const auto raw = tensor.ptr();
    std::array<fk::RawPtr<fk::ND::_2D, T>, N> views{};
    for (size_t i = 0; i < N; ++i) {
        views[i].data = raw.data + i * width * height;
        views[i].dims = fk::PtrDims<fk::ND::_2D>(width, height, width * sizeof(T));
    }
    return views;
}

void runPipeline() {
    using RGB = fk::uchar3;
    fk::Stream stream;
    NV12Image frame(FRAME_W, FRAME_H);
    fk::Ptr2D<RGB> rgb(FRAME_W, FRAME_H, 0, fk::MemType::Device);
    fk::Tensor<fk::float3> cropped(CROP_W, CROP_H, N, 1, fk::MemType::Device);
    fk::Tensor<fk::float3> resized(RES_W, RES_H, N, 1, fk::MemType::Device);
    fk::Tensor<fk::float3> out(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const auto crops = someCrops();
    const auto croppedViews = planeViews(cropped, CROP_W, CROP_H);
    const auto resizedViews = planeViews(resized, RES_W, RES_H);
    // 1: full frame NV12 -> 8 bit RGB, quantized as in the 8 bit reference
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::ReadYUV<fk::PixelFormat::NV12>::build(frame),
        YUVToRGB::build(),
        fk::NormalizePixel<fk::float3, fk::ColorDepth::p8bit>::build(),
        fk::Add<fk::float3>::build(fk::make_set<fk::float3>(QUANTIZATION_OFFSET)),
        fk::SaturateDenormalizePixel<fk::float3, fk::uchar3, fk::ColorDepth::p8bit>::build(),
        fk::PerThreadWrite<fk::ND::_2D, fk::uchar3>::build(rgb.ptr()));
    // 2: crop to float3 (HF over the N crops)
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::PerThreadRead<fk::ND::_2D, RGB>::build(rgb.ptr()),
        fk::Crop<>::build(crops),
        fk::Cast<fk::uchar3, fk::float3>::build(),
        fk::TensorWrite<fk::float3>::build(cropped.ptr()));
    // 3: resize
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(croppedViews),
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
        fk::TensorWrite<fk::float3>::build(resized.ptr()));
    // 4: border + normalization + write
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(resizedViews),
        fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
        fk::Mul<fk::float3>::build(mulValue),
        fk::Sub<fk::float3>::build(subValue),
        fk::Div<fk::float3>::build(divValue),
        fk::TensorWrite<fk::float3>::build(out.ptr()));
}

#elif defined(IMPL_OPENCV) || defined(IMPL_NPP)
// ---------------------------------------------------------------- the libraries
// The frame pointers, pitch, color twist and NPP stream context of the benchmark's Context
struct LibState {
    const Npp8u* y{nullptr};
    const Npp8u* uv{nullptr};
    int pitch{0};
    Npp32f twist[3][4]{};
    NppStreamContext npp{};
    cv::cuda::Stream cvStream;
};

void nv12ToRgb8(const LibState& st, Npp8u* dst, int dstStep) {
    const Npp8u* src[2] = {st.y, st.uv};
    int srcSteps[2] = {st.pitch, st.pitch};
    NPP_CHECK(nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(src, srcSteps, dst, dstStep, NppiSize{FRAME_W, FRAME_H},
                                                       st.twist, st.npp));
}

#if defined(IMPL_OPENCV)
// makeOpenCV
void runPipeline() {
    LibState st;
    cv::cuda::GpuMat rgb8(FRAME_H, FRAME_W, CV_8UC3);
    std::vector<cv::Rect> rects;
    std::vector<cv::cuda::GpuMat> cropped, resized;
    cv::cuda::GpuMat tensor;
    for (size_t i = 0; i < N; ++i) {
        rects.emplace_back(static_cast<int>(i), static_cast<int>(i), CROP_W, CROP_H);
        cropped.emplace_back(CROP_H, CROP_W, CV_32FC3);
        resized.emplace_back(RES_H, RES_W, CV_32FC3);
    }
    cv::cuda::createContinuous(static_cast<int>(N) * OUT_H, OUT_W, CV_32FC3, tensor);
    nv12ToRgb8(st, rgb8.data, static_cast<int>(rgb8.step));
    for (size_t i = 0; i < N; ++i) {
        rgb8(rects[i]).convertTo(cropped[i], CV_32F, st.cvStream);
        cv::cuda::resize(cropped[i], resized[i], cv::Size(RES_W, RES_H), 0., 0., cv::INTER_LINEAR, st.cvStream);
        cv::cuda::GpuMat plane = tensor.rowRange(static_cast<int>(i) * OUT_H, static_cast<int>(i + 1) * OUT_H);
        cv::cuda::copyMakeBorder(resized[i], plane, BORDER_TOP, BORDER_BOTTOM, 0, 0, cv::BORDER_CONSTANT,
                                 cv::Scalar(0., 0., 0.), st.cvStream);
    }
    cv::cuda::multiply(tensor, cv::Scalar(MUL[0], MUL[1], MUL[2]), tensor, 1., -1, st.cvStream);
    cv::cuda::subtract(tensor, cv::Scalar(SUB[0], SUB[1], SUB[2]), tensor, cv::noArray(), -1, st.cvStream);
    cv::cuda::divide(tensor, cv::Scalar(DIV[0], DIV[1], DIV[2]), tensor, 1., -1, st.cvStream);
}
#else
// makeNPP: the same allocations (gpuErrchk is FKL's) and calls
void runPipeline() {
    LibState st;
    const int cropStep = CROP_W * 3 * sizeof(Npp32f);
    const int outStep = OUT_W * 3 * sizeof(Npp32f);
    const size_t cropFloats = static_cast<size_t>(CROP_W) * CROP_H * 3;
    const size_t resizedFloats = static_cast<size_t>(RES_W) * RES_H * 3;
    const size_t outFloats = static_cast<size_t>(OUT_W) * OUT_H * 3;
    Npp8u* rgb8 = nullptr;
    Npp32f* cropped = nullptr;
    Npp32f* resized = nullptr;
    Npp32f* tensor = nullptr;
    NppiImageDescriptor* srcDescriptors = nullptr;
    NppiImageDescriptor* dstDescriptors = nullptr;
    NppiResizeBatchROI_Advanced* rois = nullptr;
    size_t rgbPitch = 0;
    gpuErrchk(cudaMallocPitch(reinterpret_cast<void**>(&rgb8), &rgbPitch, FRAME_W * 3, FRAME_H));
    const int rgbStep = static_cast<int>(rgbPitch);
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&cropped), N * cropFloats * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&resized), N * resizedFloats * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&tensor), N * outFloats * sizeof(Npp32f)));
    std::vector<NppiImageDescriptor> src(N), dst(N);
    std::vector<NppiResizeBatchROI_Advanced> roisHost(N);
    const auto crops = someCrops();
    for (size_t i = 0; i < N; ++i) {
        src[i] = NppiImageDescriptor{cropped + i * cropFloats, cropStep, NppiSize{CROP_W, CROP_H}};
        dst[i] = NppiImageDescriptor{resized + i * resizedFloats, outStep, NppiSize{RES_W, RES_H}};
        roisHost[i] = NppiResizeBatchROI_Advanced{NppiRect{0, 0, CROP_W, CROP_H}, NppiRect{0, 0, RES_W, RES_H}};
    }
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&srcDescriptors), N * sizeof(NppiImageDescriptor)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dstDescriptors), N * sizeof(NppiImageDescriptor)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&rois), N * sizeof(NppiResizeBatchROI_Advanced)));
    gpuErrchk(cudaMemcpy(srcDescriptors, src.data(), N * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dstDescriptors, dst.data(), N * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(rois, roisHost.data(), N * sizeof(NppiResizeBatchROI_Advanced), cudaMemcpyHostToDevice));

    // 1_nv12_to_rgb8_full_frame
    nv12ToRgb8(st, rgb8, rgbStep);
    // 2_per_crop_convert_8u32f
    for (size_t i = 0; i < N; ++i) {
        const fk::Rect& r = crops[i];
        const Npp8u* cropSrc = rgb8 + static_cast<size_t>(r.y) * rgbStep + r.x * 3;
        NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(cropSrc, rgbStep, cropped + i * cropFloats, cropStep,
                                            NppiSize{CROP_W, CROP_H}, st.npp));
    }
    // 3_resize_batch
    if constexpr (N == 1) {
        // nppiResizeBatch rejects a batch of one image with NPP_BAD_ARGUMENT_ERROR
        NPP_CHECK(nppiResize_32f_C3R_Ctx(cropped, cropStep, NppiSize{CROP_W, CROP_H},
                                         NppiRect{0, 0, CROP_W, CROP_H}, resized, outStep,
                                         NppiSize{RES_W, RES_H}, NppiRect{0, 0, RES_W, RES_H},
                                         NPPI_INTER_LINEAR, st.npp));
    } else {
        NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(RES_W, RES_H, srcDescriptors, dstDescriptors, rois,
                                                       static_cast<unsigned int>(N), NPPI_INTER_LINEAR, st.npp));
    }
    // 4_per_crop_copy_const_border
    static const Npp32f zero[3] = {0.f, 0.f, 0.f};
    for (size_t i = 0; i < N; ++i) {
        NPP_CHECK(nppiCopyConstBorder_32f_C3R_Ctx(resized + i * resizedFloats, outStep, NppiSize{RES_W, RES_H},
                                                  tensor + i * outFloats, outStep, NppiSize{OUT_W, OUT_H},
                                                  BORDER_TOP, 0, zero, st.npp));
    }
    // 5_mulc_subc_divc
    const NppiSize tensorSize{OUT_W, static_cast<int>(N) * OUT_H};
    NPP_CHECK(nppiMulC_32f_C3IR_Ctx(MUL, tensor, outStep, tensorSize, st.npp));
    NPP_CHECK(nppiSubC_32f_C3IR_Ctx(SUB, tensor, outStep, tensorSize, st.npp));
    NPP_CHECK(nppiDivC_32f_C3IR_Ctx(DIV, tensor, outStep, tensorSize, st.npp));
    cudaFree(rgb8);
    cudaFree(cropped);
    cudaFree(resized);
    cudaFree(tensor);
    cudaFree(srcDescriptors);
    cudaFree(dstDescriptors);
    cudaFree(rois);
}
#endif

#else
// ---------------------------------------------------------------- headers only
void runPipeline() {}
#endif

} // namespace

int main(int argc, char**) {
    if (argc > 1) {
        runPipeline(); // never reached in the compile-time benchmark
    }
    std::printf("compile-time TU, CT_BATCH=%d\n", static_cast<int>(CT_BATCH));
    return 0;
}
