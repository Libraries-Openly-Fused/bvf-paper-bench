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

// The compile-time units of FKL's ablation (scripts/compile_time.sh, SET=ablation; common/compile_time_unit.h), with
// FKL's headers only:
//   -DIMPL_HEADERS  the headers, no pipeline
//   -DIMPL_BVF      FKL BVF, one executeOperations call (Listing 2)
//   -DIMPL_BVF_CWD  the same with the CONDITIONAL_WITH_DEFAULT plane policy
//   -DIMPL_VFHF     FKL VF+HF, the four executeOperations calls of makeFklVfHf in benchmark_fkl_ablation.cu

#include "compile_time_unit.h"

#if defined(IMPL_VFHF)
// ---------------------------------------------------------------- VF+HF: four kernels, 8 bit RGB frame
template <typename T>
std::array<fk::RawPtr<fk::ND::_2D, T>, ct::N> planeViews(const fk::Tensor<T>& tensor, int planeWidth, int planeHeight,
                                                         const std::array<fk::Size, ct::N>& sizes) {
    const auto raw = tensor.ptr();
    std::array<fk::RawPtr<fk::ND::_2D, T>, ct::N> views{};
    for (size_t i = 0; i < ct::N; ++i) {
        views[i].data = raw.data + i * planeWidth * planeHeight;
        views[i].dims = fk::PtrDims<fk::ND::_2D>(sizes[i].width, sizes[i].height, planeWidth * sizeof(T));
    }
    return views;
}

void runPipeline() {
    using namespace ct;
    using RGB = fk::uchar3;
    fk::Stream stream;
    NV12Image frame(FRAME_W, FRAME_H);
    fk::Ptr2D<RGB> rgb(FRAME_W, FRAME_H, 0, fk::MemType::Device);
    fk::Tensor<fk::float3> cropped(MAX_REGION_W, MAX_REGION_H, N, 1, fk::MemType::Device);
    fk::Tensor<fk::float3> resized(RES_W, RES_H, N, 1, fk::MemType::Device);
    fk::Tensor<fk::float3> out(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const auto crops = someCrops();
    std::array<fk::Size, N> cropSizes{}, resizedSizes{};
    for (size_t i = 0; i < N; ++i) {
        cropSizes[i] = fk::Size(crops[i].width, crops[i].height);
        resizedSizes[i] = fk::Size(RES_W, RES_H);
    }
    const auto croppedViews = planeViews(cropped, MAX_REGION_W, MAX_REGION_H, cropSizes);
    const auto resizedViews = planeViews(resized, RES_W, RES_H, resizedSizes);
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
#elif !defined(IMPL_HEADERS) && !defined(IMPL_BVF) && !defined(IMPL_BVF_CWD)
#error "Define one of IMPL_HEADERS, IMPL_BVF, IMPL_BVF_CWD, IMPL_VFHF"
#endif
