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

// Experiment 3, part (b): FKL's MorphologyDPP against NPP on the pipeline of the neighborhood experiment
// (common/bench_neighborhood.h), with a MORPH_K x MORPH_K mask and the operation NEIGHBORHOOD_OP (0: erode, 1: open;
// one build each, fkl-neighborhood/CMakeLists.txt). The harness is common/bench_gpu.h.
//
//   npp        the NPP pipeline of Figure 1 (fkl-vs-npp) up to the resize: NV12->RGB of the whole frame, per region
//              8u->32f, one batched resize into float3 64x32 planes; then per region the morphology with NPP's
//              rectangular minimum and maximum filters (nppiFilterMinBorder_32f_C3R for the erosion;
//              nppiFilterMinBorder into a plane buffer, then nppiFilterMaxBorder, for the opening), replicating the
//              border of the plane, into its output plane; then MulC, SubC and DivC on the whole output:
//              2N + 5 kernels (erode), 3N + 5 (open)
//   npp_morph  the same with NPP's morphology functions (nppiErodeBorder_32f_C3R with a K x K mask of ones;
//              nppiMorphOpenBorder_32f_C3R, which runs its two passes as two kernels with a scratch buffer)
//   fkl_dpp    FKL's MorphologyDPP, one kernel (neighborhood_common.h)

#include "neighborhood_common.h"

#ifndef NEIGHBORHOOD_OP
#error "Define NEIGHBORHOOD_OP, 0 (erode) or 1 (open) (fkl-neighborhood/CMakeLists.txt does)"
#endif

namespace bench::neighborhood {

constexpr MorphOp OP = NEIGHBORHOOD_OP == 0 ? MorphOp::Erode : MorphOp::Open;
static_assert(NEIGHBORHOOD_OP == 0 || NEIGHBORHOOD_OP == 1, "0 (erode) or 1 (open)");
constexpr int K = MORPH_K;

struct NPPState {
    Npp8u* rgb8{nullptr};
    int rgbStep{0};
    Npp32f* cropped{nullptr};
    Npp32f* resized{nullptr};
    // The result of the first pass of the opening with the filters
    Npp32f* firstPass{nullptr};
    Npp32f* tensor{nullptr};
    // The K x K mask of ones of nppiErodeBorder / nppiMorphOpenBorder, and the scratch buffer of the latter
    Npp8u* mask{nullptr};
    Npp8u* scratch{nullptr};
    // The batched resize's source descriptors and ROIs of one iteration, one after the other, in device memory and in
    // the pinned host buffer they are uploaded from; the destination descriptors do not change
    void* batchDevice{nullptr};
    void* batchHost{nullptr};
    NppiImageDescriptor* dstDescriptors{nullptr};
    // The regions of every iteration, N per iteration (rects[t * N + i])
    std::vector<fk::Rect> rects;
    ~NPPState() {
        cudaFree(rgb8);
        cudaFree(cropped);
        cudaFree(resized);
        cudaFree(firstPass);
        cudaFree(tensor);
        cudaFree(mask);
        cudaFree(scratch);
        cudaFree(batchDevice);
        cudaFreeHost(batchHost);
        cudaFree(dstDescriptors);
    }
};

// MORPH_FUNCTIONS false: npp (filters), true: npp_morph (morphology functions)
template <size_t N, bool MORPH_FUNCTIONS>
Implementation makeNPP(Context& ctx) {
    // One float3 crop of the largest region size per region; a region of another size uses its top-left corner
    const int cropStep = MAX_REGION_W * 3 * sizeof(Npp32f);
    const int planeStep = PLANE_W * 3 * sizeof(Npp32f);
    const size_t cropFloats = static_cast<size_t>(MAX_REGION_W) * MAX_REGION_H * 3;
    const NppiSize plane{PLANE_W, PLANE_H};
    auto st = std::make_shared<NPPState>();
    size_t rgbPitch = 0;
    gpuErrchk(cudaMallocPitch(reinterpret_cast<void**>(&st->rgb8), &rgbPitch, FRAME_W * 3, FRAME_H));
    st->rgbStep = static_cast<int>(rgbPitch);
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->cropped), N * cropFloats * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->resized), N * PLANE_FLOATS * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->firstPass), N * PLANE_FLOATS * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->tensor), N * PLANE_FLOATS * sizeof(Npp32f)));
    const std::vector<Npp8u> ones(static_cast<size_t>(K) * K, 1);
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->mask), ones.size()));
    gpuErrchk(cudaMemcpy(st->mask, ones.data(), ones.size(), cudaMemcpyHostToDevice));
    int scratchBytes = 0;
    NPP_CHECK(nppiMorphGetBufferSize_32f_C3R(plane, &scratchBytes));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->scratch), static_cast<size_t>(std::max(scratchBytes, 1))));
    const size_t descriptorBytes = N * sizeof(NppiImageDescriptor);
    const size_t batchBytes = descriptorBytes + N * sizeof(NppiResizeBatchROI_Advanced);
    static_assert(sizeof(NppiImageDescriptor) % alignof(NppiResizeBatchROI_Advanced) == 0, "ROIs after descriptors");
    gpuErrchk(cudaMalloc(&st->batchDevice, batchBytes));
    gpuErrchk(cudaMallocHost(&st->batchHost, batchBytes));
    std::vector<NppiImageDescriptor> dst(N);
    for (size_t i = 0; i < N; ++i) {
        dst[i] = NppiImageDescriptor{st->resized + i * PLANE_FLOATS, planeStep, plane};
    }
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->dstDescriptors), descriptorBytes));
    gpuErrchk(cudaMemcpy(st->dstDescriptors, dst.data(), descriptorBytes, cudaMemcpyHostToDevice));
    for (size_t t = 0; t < ctx.regions->iterations(); ++t) {
        const auto rects = fklRects<N>(ctx, t);
        st->rects.insert(st->rects.end(), rects.begin(), rects.end());
    }

    Implementation impl{MORPH_FUNCTIONS ? "npp_morph" : "npp", Reference::NppSampling8bit};
    impl.stages.emplace_back("1_nv12_to_rgb8_full_frame", [&ctx, st](size_t) {
        nv12ToRgb8(ctx, st->rgb8, st->rgbStep);
    });
    impl.stages.emplace_back("2_per_crop_convert_8u32f", [&ctx, st, cropFloats, cropStep](size_t t) {
        for (size_t i = 0; i < N; ++i) {
            const fk::Rect& r = st->rects[t * N + i];
            const Npp8u* cropSrc = st->rgb8 + static_cast<size_t>(r.y) * st->rgbStep + r.x * 3;
            beforeKernel();
            NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(cropSrc, st->rgbStep, st->cropped + i * cropFloats, cropStep,
                                                NppiSize{r.width, r.height}, ctx.npp));
        }
    });
    impl.stages.emplace_back("3_resize_batch", [&ctx, st, cropFloats, cropStep, planeStep, plane, descriptorBytes,
                                                batchBytes](size_t t) {
        if constexpr (N == 1) {
            // nppiResizeBatch rejects a batch of one image with NPP_BAD_ARGUMENT_ERROR
            const fk::Rect& r = st->rects[t];
            beforeKernel();
            NPP_CHECK(nppiResize_32f_C3R_Ctx(st->cropped, cropStep, NppiSize{r.width, r.height},
                                             NppiRect{0, 0, r.width, r.height}, st->resized, planeStep, plane,
                                             NppiRect{0, 0, PLANE_W, PLANE_H}, NPPI_INTER_LINEAR, ctx.npp));
        } else {
            // The source descriptors and ROIs of this iteration's regions, uploaded with one copy
            auto* src = static_cast<NppiImageDescriptor*>(st->batchHost);
            auto* rois = reinterpret_cast<NppiResizeBatchROI_Advanced*>(static_cast<char*>(st->batchHost) +
                                                                         descriptorBytes);
            for (size_t i = 0; i < N; ++i) {
                const fk::Rect& r = st->rects[t * N + i];
                src[i] = NppiImageDescriptor{st->cropped + i * cropFloats, cropStep, NppiSize{r.width, r.height}};
                rois[i] = NppiResizeBatchROI_Advanced{NppiRect{0, 0, r.width, r.height},
                                                      NppiRect{0, 0, PLANE_W, PLANE_H}};
            }
            gpuErrchk(cudaMemcpyAsync(st->batchDevice, st->batchHost, batchBytes, cudaMemcpyHostToDevice,
                                      ctx.stream));
            beforeKernel();
            NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(
                PLANE_W, PLANE_H, static_cast<NppiImageDescriptor*>(st->batchDevice), st->dstDescriptors,
                reinterpret_cast<NppiResizeBatchROI_Advanced*>(static_cast<char*>(st->batchDevice) + descriptorBytes),
                static_cast<unsigned int>(N), NPPI_INTER_LINEAR, ctx.npp));
        }
    });
    // The morphology of every plane: oSrcSize is the plane and oSrcOffset 0, so NPP replicates the border of the plane
    const NppiSize mask{K, K};
    const NppiPoint anchor{K / 2, K / 2};
    const NppiPoint origin{0, 0};
    std::string morphStage = MORPH_FUNCTIONS ? (OP == MorphOp::Erode ? "4_per_region_erode_border"
                                                                     : "4_per_region_morph_open_border")
                                             : (OP == MorphOp::Erode ? "4_per_region_filter_min_border"
                                                                     : "4_per_region_filter_min_then_max_border");
    impl.stages.emplace_back(morphStage, [&ctx, st, planeStep, plane, mask, anchor, origin](size_t) {
        for (size_t i = 0; i < N; ++i) {
            const Npp32f* src = st->resized + i * PLANE_FLOATS;
            Npp32f* out = st->tensor + i * PLANE_FLOATS;
            if constexpr (MORPH_FUNCTIONS && OP == MorphOp::Erode) {
                beforeKernel();
                NPP_CHECK(nppiErodeBorder_32f_C3R_Ctx(src, planeStep, plane, origin, out, planeStep, plane, st->mask,
                                                      mask, anchor, NPP_BORDER_REPLICATE, ctx.npp));
            } else if constexpr (MORPH_FUNCTIONS) {
                // Two kernels (erosion, dilation) in one call: the L2 flush runs before the call only
                beforeKernel();
                NPP_CHECK(nppiMorphOpenBorder_32f_C3R_Ctx(src, planeStep, plane, origin, out, planeStep, plane,
                                                          st->mask, mask, anchor, st->scratch, NPP_BORDER_REPLICATE,
                                                          ctx.npp));
            } else if constexpr (OP == MorphOp::Erode) {
                beforeKernel();
                NPP_CHECK(nppiFilterMinBorder_32f_C3R_Ctx(src, planeStep, plane, origin, out, planeStep, plane, mask,
                                                          anchor, NPP_BORDER_REPLICATE, ctx.npp));
            } else {
                Npp32f* firstPass = st->firstPass + i * PLANE_FLOATS;
                beforeKernel();
                NPP_CHECK(nppiFilterMinBorder_32f_C3R_Ctx(src, planeStep, plane, origin, firstPass, planeStep, plane,
                                                          mask, anchor, NPP_BORDER_REPLICATE, ctx.npp));
                beforeKernel();
                NPP_CHECK(nppiFilterMaxBorder_32f_C3R_Ctx(firstPass, planeStep, plane, origin, out, planeStep, plane,
                                                          mask, anchor, NPP_BORDER_REPLICATE, ctx.npp));
            }
        }
    });
    impl.stages.emplace_back("5_mulc_subc_divc", [&ctx, st, planeStep](size_t) {
        const NppiSize tensorSize{PLANE_W, static_cast<int>(N) * PLANE_H};
        beforeKernel();
        NPP_CHECK(nppiMulC_32f_C3IR_Ctx(MUL, st->tensor, planeStep, tensorSize, ctx.npp));
        beforeKernel();
        NPP_CHECK(nppiSubC_32f_C3IR_Ctx(SUB, st->tensor, planeStep, tensorSize, ctx.npp));
        beforeKernel();
        NPP_CHECK(nppiDivC_32f_C3IR_Ctx(DIV, st->tensor, planeStep, tensorSize, ctx.npp));
    });
    impl.run = runStages(impl.stages);
    impl.download = [st]() { return downloadDevice(st->tensor, N * PLANE_FLOATS); };
    impl.fillNaN = [st]() { gpuErrchk(cudaMemset(st->tensor, 0xFF, N * PLANE_FLOATS * sizeof(Npp32f))); };
    return impl;
}

struct NppVsFklDpp {
    static constexpr const char* title = "FKL MorphologyDPP vs NPP, neighborhood experiment (fkl-neighborhood)";
    static constexpr const char* fklName = "fkl_dpp";
    static constexpr const char* fklLabel = "FKL DPP";
    static constexpr const char* fklShortLabel = "DPP";

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        impls.push_back(makeNPP<N, false>(ctx));
        impls.push_back(makeNPP<N, true>(ctx));
        return impls;
    }

    template <size_t N>
    static Implementation fklImplementation(Context& ctx) {
        return fklDpp<N, K, OP>(ctx);
    }

    template <size_t N>
    static std::vector<float> reference(const Context& ctx, Reference r, size_t t) {
        return morphReference<N>(ctx, r, t, K, OP);
    }

    static void setup(const std::filesystem::path& outDir, const std::string& mode) { setupTileChoice(outDir, mode); }

    static void workload(std::ostream& os) { describeWorkload(os, K, OP, BATCHES); }

    static void describe(std::ostream& os, const std::filesystem::path&) {
        const NppLibraryVersion* npp = nppGetLibVersion();
        const bool erode = OP == MorphOp::Erode;
        os << "NPP: " << npp->major << "." << npp->minor << "." << npp->build << "\n"
           << "NPP pipelines: as fkl-vs-npp up to the resize (nppiNV12ToRGB_8u_ColorTwist32f_P2C3R of the whole frame "
              "into an 8 bit RGB frame, per region nppiConvert_8u32f_C3R, one nppiResizeBatch_32f_C3R_Advanced, or "
              "nppiResize_32f_C3R for N = 1, into float3 " << PLANE_W << "x" << PLANE_H << " planes, its descriptors "
              "and ROIs uploaded with one cudaMemcpyAsync per execution inside the timed region), then per region the "
              "morphology into its output plane (oSrcSize the plane, oSrcOffset 0, NPP_BORDER_REPLICATE: the border of "
              "the plane is replicated), then nppiMulC, nppiSubC and nppiDivC_32f_C3IR on the whole float3 output.\n"
           << "  npp: "
           << (erode ? "nppiFilterMinBorder_32f_C3R (the rectangular minimum filter): 2N + 5 kernels"
                     : "nppiFilterMinBorder_32f_C3R into a plane buffer, then nppiFilterMaxBorder_32f_C3R: 3N + 5 "
                       "kernels")
           << "\n"
           << "  npp_morph: "
           << (erode ? "nppiErodeBorder_32f_C3R with a K x K mask of ones: 2N + 5 kernels"
                     : "nppiMorphOpenBorder_32f_C3R with a K x K mask of ones and its scratch buffer "
                       "(nppiMorphGetBufferSize_32f_C3R), which launches an erosion and a dilation kernel: 3N + 5 "
                       "kernels; with the L2 flushed, the flush runs before the call, not between its two kernels")
           << "\n"
           << "NPP samples its bilinear resize at dst * scale - 0.25 (measured, see fkl-vs-npp), so both are checked "
              "against a reference with that sampling and the 8 bit RGB of NPP's conversion.\n";
    }

    static std::string diagnostic(const Context& ctx) { return nppConversionDiagnostic(ctx); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench::neighborhood

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::neighborhood::NppVsFklDpp>(argc, argv);
}
