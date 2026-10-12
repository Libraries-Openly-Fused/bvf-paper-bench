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

// One pipeline of the paper's Figure 1 per translation unit, for measuring nvcc compile time (compile_time/run.sh with
// SET=nppdx). Select with exactly one of:
//   -DIMPL_HEADERS  the headers of the other two variants (FKL, NPPDx, nppdx_figure1.h, NVTX: the ones of
//                   benchmark_figure1_bvf.cu without its benchmark scaffolding), no pipeline
//   -DIMPL_NPPDX    the NPPDx kernel in the configuration benchmark_figure1_bvf chose for this N, given with
//                   -DCT_NPPDX_TILE_X, -DCT_NPPDX_TILE_Y and -DCT_NPPDX_BLOCK (the requested block, 0 for NPPDx's
//                   recommended one; run.sh reads them from benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv);
//                   CropOrigins<N> is a kernel parameter, so it is instantiated for every N
//   -DIMPL_BVF      FKL, the runFKL<N> call of benchmark_figure1_bvf.cu (in figure1_common.h)
// and -DCT_BATCH=<N> (number of crops). It is compiled with the nvcc command line of benchmark_figure1_bvf (NPPDX_SM,
// NPPDx's include path, the geometry definitions) and never executed: main() calls the pipeline only when it gets an
// argument.

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
#include <cstdint>
#include <cstdio>

#ifndef CT_BATCH
#error "Define -DCT_BATCH=<number of crops> (not BATCH: FKL uses that name for template parameters)"
#endif

namespace figure1 {
namespace {

[[maybe_unused]] constexpr size_t N = CT_BATCH;
// ImageNet normalization (Normalization is in nppdx_common.h)
[[maybe_unused]] constexpr Normalization norm{};

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Limited, fk::ColorPrimitives::bt601>;

#if defined(IMPL_BVF)
// runFKL of benchmark_figure1_bvf.cu (figure1_common.h), with the stream and the frame of its Workload as arguments
template <size_t N>
void runFKL(fk::Stream& stream, const NV12Image& frame, const std::array<fk::Rect, N>& crops,
            fk::Tensor<fk::float3>& out) {
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::ReadYUV<fk::PixelFormat::NV12>::build(frame),
        YUVToRGB::build(),
        fk::Crop<>::build(crops),
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
        fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, fk::float3{0.f, 0.f, 0.f}),
        fk::Mul<fk::float3>::build(fk::float3{norm.mul[0], norm.mul[1], norm.mul[2]}),
        fk::Sub<fk::float3>::build(fk::float3{norm.sub[0], norm.sub[1], norm.sub[2]}),
        fk::Div<fk::float3>::build(fk::float3{norm.div[0], norm.div[1], norm.div[2]}),
        fk::TensorWrite<fk::float3>::build(out.ptr()));
}

void runPipeline() {
    fk::Stream stream;
    NV12Image frame(FRAME_W, FRAME_H);
    fk::Tensor<fk::float3> out(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    std::array<fk::Rect, N> crops{};
    for (size_t i = 0; i < N; ++i) {
        crops[i] = fk::Rect(static_cast<int>(2 * i), static_cast<int>(2 * i), CROP_W, CROP_H);
    }
    runFKL<N>(stream, frame, crops, out);
}

#elif defined(IMPL_NPPDX)
#if !defined(CT_NPPDX_TILE_X) || !defined(CT_NPPDX_TILE_Y) || !defined(CT_NPPDX_BLOCK)
#error "Define -DCT_NPPDX_TILE_X, -DCT_NPPDX_TILE_Y and -DCT_NPPDX_BLOCK (the configuration chosen for this N)"
#endif
// The NPPDx kernel in the configuration the benchmark chose for this N, launched as the benchmark does
void runPipeline() {
    using Nppdx = NppdxFigure1<NPPDX_SM, N, CT_NPPDX_BLOCK, CT_NPPDX_TILE_X, CT_NPPDX_TILE_Y>;
    Nppdx::init();
    const uint8_t* y = nullptr;
    const uint8_t* uv = nullptr;
    float* out = nullptr;
    CropOrigins<N> origins{};
    Nppdx::run(cudaStream_t{}, y, uv, FRAME_W, origins, norm, out);
}

#else
// headers only
void runPipeline() {}
#endif

} // namespace
} // namespace figure1

int main(int argc, char**) {
    if (argc > 1) {
        figure1::runPipeline(); // never reached in the compile-time benchmark
    }
    std::printf("compile-time TU, CT_BATCH=%d\n", static_cast<int>(CT_BATCH));
    return 0;
}
