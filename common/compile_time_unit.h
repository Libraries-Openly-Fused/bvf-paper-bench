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

// The common part of the compile-time units (<benchmark>/src/compile_time.cu, measured by scripts/compile_time.sh):
// one pipeline of the paper's Figure 1 per translation unit, selected with -DIMPL_<NAME> and -DCT_BATCH=<N> (not BATCH,
// which FKL uses as a template parameter name). A unit includes its library's headers, then this file, then defines
// runPipeline() for its own IMPL_ values; this file defines it for
//   -DIMPL_BVF      FKL BVF, the one executeOperations call of makeFklBvf in common/bench_gpu.h (Listing 2)
//   -DIMPL_BVF_CWD  the same with the CONDITIONAL_WITH_DEFAULT plane policy: CT_BATCH planes compiled, the number used
//                   given at run time
// A unit is compiled with the nvcc command line of its benchmark and never executed: main() calls the pipeline only
// when it gets an argument, which keeps every template instantiated. The pipelines are those of the benchmarks without
// their scaffolding: same operations, template arguments and order.

#ifndef BVF_COMPILE_TIME_UNIT_H
#define BVF_COMPILE_TIME_UNIT_H

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

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

namespace ct {

// Frame, region, resize, border and output sizes of the benchmarks
using namespace bench::geometry;
[[maybe_unused]] constexpr size_t N = CT_BATCH;
[[maybe_unused]] constexpr float MUL[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
[[maybe_unused]] constexpr float SUB[3] = {0.485f, 0.456f, 0.406f};
[[maybe_unused]] constexpr float DIV[3] = {0.229f, 0.224f, 0.225f};
[[maybe_unused]] constexpr float QUANTIZATION_OFFSET = 0.5f / 255.f;

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Limited, fk::ColorPrimitives::bt601>;
[[maybe_unused]] inline const fk::float3 mulValue{MUL[0], MUL[1], MUL[2]};
[[maybe_unused]] inline const fk::float3 subValue{SUB[0], SUB[1], SUB[2]};
[[maybe_unused]] inline const fk::float3 divValue{DIV[0], DIV[1], DIV[2]};
[[maybe_unused]] inline const fk::float3 black{0.f, 0.f, 0.f};

// N rectangles with the region sizes of iteration 0 (the code does not depend on the values)
[[maybe_unused]] inline std::array<fk::Rect, N> someCrops() {
    std::array<fk::Rect, N> crops{};
    for (size_t i = 0; i < N; ++i) {
        const RegionSize s = regionSize(0, i);
        crops[i] = fk::Rect(static_cast<int>(i), static_cast<int>(i), s.width, s.height);
    }
    return crops;
}

} // namespace ct

// The pipeline of this unit
void runPipeline();

#if defined(IMPL_BVF) || defined(IMPL_BVF_CWD)
// ---------------------------------------------------------------- BVF: Listing 2 of the paper (makeFklBvf)
void runPipeline() {
    using namespace ct;
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
#elif defined(IMPL_HEADERS)
// ---------------------------------------------------------------- headers only
void runPipeline() {}
#endif

int main(int argc, char**) {
    if (argc > 1) {
        runPipeline(); // never reached in the compile-time benchmark
    }
    std::printf("compile-time TU, CT_BATCH=%d\n", static_cast<int>(CT_BATCH));
    return 0;
}

#endif // BVF_COMPILE_TIME_UNIT_H
