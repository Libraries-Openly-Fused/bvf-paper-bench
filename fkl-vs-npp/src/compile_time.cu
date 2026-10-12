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

// The compile-time units of FKL vs NPP (scripts/compile_time.sh, SET=npp; common/compile_time_unit.h), all with the
// headers of FKL and NPP:
//   -DIMPL_HEADERS  the headers, no pipeline
//   -DIMPL_NPP      the NPP calls of makeNPP in benchmark_npp.cu (gpuErrchk is FKL's)
//   -DIMPL_BVF      FKL BVF, one executeOperations call (Listing 2)

#include <npp.h>

#include "bench_npp.h"

#include "compile_time_unit.h"

#if defined(IMPL_NPP)
// ---------------------------------------------------------------- makeNPP: the same allocations and calls
void runPipeline() {
    using namespace ct;
    const Npp8u* y = nullptr;
    const Npp8u* uv = nullptr;
    const int pitch = 0;
    const NppStreamContext npp{};
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
    bench::npp::nv12ToRgb8Bt601Limited(y, uv, pitch, rgb8, rgbStep, NppiSize{FRAME_W, FRAME_H}, npp);
    // 2_per_crop_convert_8u32f
    for (size_t i = 0; i < N; ++i) {
        const fk::Rect& r = crops[i];
        const Npp8u* cropSrc = rgb8 + static_cast<size_t>(r.y) * rgbStep + r.x * 3;
        NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(cropSrc, rgbStep, cropped + i * cropFloats, cropStep,
                                            NppiSize{CROP_W, CROP_H}, npp));
    }
    // 3_resize_batch
    if constexpr (N == 1) {
        // nppiResizeBatch rejects a batch of one image with NPP_BAD_ARGUMENT_ERROR
        NPP_CHECK(nppiResize_32f_C3R_Ctx(cropped, cropStep, NppiSize{CROP_W, CROP_H},
                                         NppiRect{0, 0, CROP_W, CROP_H}, resized, outStep,
                                         NppiSize{RES_W, RES_H}, NppiRect{0, 0, RES_W, RES_H},
                                         NPPI_INTER_LINEAR, npp));
    } else {
        NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(RES_W, RES_H, srcDescriptors, dstDescriptors, rois,
                                                       static_cast<unsigned int>(N), NPPI_INTER_LINEAR, npp));
    }
    // 4_per_crop_copy_const_border
    static const Npp32f zero[3] = {0.f, 0.f, 0.f};
    for (size_t i = 0; i < N; ++i) {
        NPP_CHECK(nppiCopyConstBorder_32f_C3R_Ctx(resized + i * resizedFloats, outStep, NppiSize{RES_W, RES_H},
                                                  tensor + i * outFloats, outStep, NppiSize{OUT_W, OUT_H},
                                                  BORDER_TOP, 0, zero, npp));
    }
    // 5_mulc_subc_divc
    const NppiSize tensorSize{OUT_W, static_cast<int>(N) * OUT_H};
    NPP_CHECK(nppiMulC_32f_C3IR_Ctx(MUL, tensor, outStep, tensorSize, npp));
    NPP_CHECK(nppiSubC_32f_C3IR_Ctx(SUB, tensor, outStep, tensorSize, npp));
    NPP_CHECK(nppiDivC_32f_C3IR_Ctx(DIV, tensor, outStep, tensorSize, npp));
    cudaFree(rgb8);
    cudaFree(cropped);
    cudaFree(resized);
    cudaFree(tensor);
    cudaFree(srcDescriptors);
    cudaFree(dstDescriptors);
    cudaFree(rois);
}
#elif !defined(IMPL_HEADERS) && !defined(IMPL_BVF)
#error "Define one of IMPL_HEADERS, IMPL_NPP, IMPL_BVF"
#endif
