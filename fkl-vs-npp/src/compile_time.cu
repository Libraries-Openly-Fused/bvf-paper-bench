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
    const int cropStep = MAX_REGION_W * 3 * sizeof(Npp32f);
    const int outStep = OUT_W * 3 * sizeof(Npp32f);
    const size_t cropFloats = static_cast<size_t>(MAX_REGION_W) * MAX_REGION_H * 3;
    const size_t resizedFloats = static_cast<size_t>(RES_W) * RES_H * 3;
    const size_t outFloats = static_cast<size_t>(OUT_W) * OUT_H * 3;
    Npp8u* rgb8 = nullptr;
    Npp32f* cropped = nullptr;
    Npp32f* resized = nullptr;
    Npp32f* tensor = nullptr;
    void* batchDevice = nullptr;
    void* batchHost = nullptr;
    NppiImageDescriptor* dstDescriptors = nullptr;
    size_t rgbPitch = 0;
    gpuErrchk(cudaMallocPitch(reinterpret_cast<void**>(&rgb8), &rgbPitch, FRAME_W * 3, FRAME_H));
    const int rgbStep = static_cast<int>(rgbPitch);
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&cropped), N * cropFloats * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&resized), N * resizedFloats * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&tensor), N * outFloats * sizeof(Npp32f)));
    const size_t descriptorBytes = N * sizeof(NppiImageDescriptor);
    const size_t batchBytes = descriptorBytes + N * sizeof(NppiResizeBatchROI_Advanced);
    gpuErrchk(cudaMalloc(&batchDevice, batchBytes));
    gpuErrchk(cudaMallocHost(&batchHost, batchBytes));
    std::vector<NppiImageDescriptor> dst(N);
    for (size_t i = 0; i < N; ++i) {
        dst[i] = NppiImageDescriptor{resized + i * resizedFloats, outStep, NppiSize{RES_W, RES_H}};
    }
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dstDescriptors), descriptorBytes));
    gpuErrchk(cudaMemcpy(dstDescriptors, dst.data(), descriptorBytes, cudaMemcpyHostToDevice));
    const auto crops = someCrops();

    // 1_nv12_to_rgb8_full_frame
    bench::npp::nv12ToRgb8Bt601Limited(y, uv, pitch, rgb8, rgbStep, NppiSize{FRAME_W, FRAME_H}, npp);
    // 2_per_crop_convert_8u32f
    for (size_t i = 0; i < N; ++i) {
        const fk::Rect& r = crops[i];
        const Npp8u* cropSrc = rgb8 + static_cast<size_t>(r.y) * rgbStep + r.x * 3;
        NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(cropSrc, rgbStep, cropped + i * cropFloats, cropStep,
                                            NppiSize{r.width, r.height}, npp));
    }
    // 3_resize_batch
    if constexpr (N == 1) {
        // nppiResizeBatch rejects a batch of one image with NPP_BAD_ARGUMENT_ERROR
        const fk::Rect& r = crops[0];
        NPP_CHECK(nppiResize_32f_C3R_Ctx(cropped, cropStep, NppiSize{r.width, r.height},
                                         NppiRect{0, 0, r.width, r.height}, resized, outStep,
                                         NppiSize{RES_W, RES_H}, NppiRect{0, 0, RES_W, RES_H},
                                         NPPI_INTER_LINEAR, npp));
    } else {
        auto* src = static_cast<NppiImageDescriptor*>(batchHost);
        auto* rois = reinterpret_cast<NppiResizeBatchROI_Advanced*>(static_cast<char*>(batchHost) + descriptorBytes);
        for (size_t i = 0; i < N; ++i) {
            const fk::Rect& r = crops[i];
            src[i] = NppiImageDescriptor{cropped + i * cropFloats, cropStep, NppiSize{r.width, r.height}};
            rois[i] = NppiResizeBatchROI_Advanced{NppiRect{0, 0, r.width, r.height}, NppiRect{0, 0, RES_W, RES_H}};
        }
        gpuErrchk(cudaMemcpyAsync(batchDevice, batchHost, batchBytes, cudaMemcpyHostToDevice, npp.hStream));
        NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(
            RES_W, RES_H, static_cast<NppiImageDescriptor*>(batchDevice), dstDescriptors,
            reinterpret_cast<NppiResizeBatchROI_Advanced*>(static_cast<char*>(batchDevice) + descriptorBytes),
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
    cudaFree(batchDevice);
    cudaFreeHost(batchHost);
    cudaFree(dstDescriptors);
}
#elif !defined(IMPL_HEADERS) && !defined(IMPL_BVF)
#error "Define one of IMPL_HEADERS, IMPL_NPP, IMPL_BVF"
#endif
