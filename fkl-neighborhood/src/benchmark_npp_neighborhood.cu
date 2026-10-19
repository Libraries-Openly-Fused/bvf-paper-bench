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

// Experiment 3, part (b): FKL's MorphologyDPP box blur against NPP on the pipeline of the neighborhood experiment
// (common/bench_neighborhood.h): BOX_K x BOX_K box, one pass. The harness is common/bench_gpu.h. NPP has no batched
// box filter, so it runs one call per crop; nppiFilterBoxBorder with oSrcSize the crop and oSrcOffset 0 replicates the
// edge of the crop (NPP_BORDER_REPLICATE, the only border it accepts). Every call is one kernel at any offset.
//
//   npp            NV12->RGB of the whole frame (8 bit); per crop nppiFilterBoxBorder_8u_C3R from the crop of the
//                  8 bit frame into its packed 8 bit plane (floor(sum / 25)); one nppiConvert_8u32f_C3R of the N
//                  planes; MulC, SubC, DivC on the whole output: N + 5 kernels
//   npp_32f        NV12->RGB; per crop nppiConvert_8u32f_C3R of the crop into a float crop, then
//                  nppiFilterBoxBorder_32f_C3R into its output plane; MulC, SubC, DivC: 2N + 4 kernels
//   npp_32f_frame  NV12->RGB; one nppiConvert_8u32f_C3R of the whole frame; per crop nppiFilterBoxBorder_32f_C3R from
//                  the crop of the float frame into its output plane; MulC, SubC, DivC: N + 5 kernels
//   npp_crop_nv12, npp_32f_crop_nv12
//                  npp and npp_32f with the NV12->RGB of each crop only (the same color twist with the source
//                  pointers at the crop's origin and the crop's size, writing at the crop's position in the 8 bit
//                  frame, which the rest of the pipeline reads as before): 2N + 4 and 3N + 3 kernels. npp_32f_frame
//                  converts the whole 8 bit frame to float, so it keeps the conversion of the whole frame.
//   fkl_dpp        FKL's MorphologyDPP box blur, one kernel (neighborhood_common.h)
// Per batch size, of each pipeline's two conversion forms (npp and npp_crop_nv12, npp_32f and npp_32f_crop_nv12), each
// view (latency, GPU only, GPU only with the L2 flushed) reports the one with the lower median in that view.

#include "neighborhood_common.h"

namespace bench::neighborhood {

constexpr int K = BOX_K;

enum class NppVariant { Box8u, Convert32fPerCrop, Convert32fFrame };
// The two forms of NPP's NV12 -> RGB: the whole frame, or each crop only
enum class Nv12Form { Frame, Crop };

struct NPPState {
    Npp8u* rgb8{nullptr};
    int rgbStep{0};
    // The float frame (Convert32fFrame)
    Npp32f* rgb32f{nullptr};
    int rgb32fStep{0};
    // The packed 8 bit planes (Box8u) or the float crop of every crop (Convert32fPerCrop)
    Npp8u* planes8{nullptr};
    Npp32f* crops32f{nullptr};
    Npp32f* tensor{nullptr};
    // The crops of every iteration, N per iteration (rects[t * N + i])
    std::vector<fk::Rect> rects;
    ~NPPState() {
        cudaFree(rgb8);
        cudaFree(rgb32f);
        cudaFree(planes8);
        cudaFree(crops32f);
        cudaFree(tensor);
    }
};

inline const char* nppVariantName(const NppVariant v, const Nv12Form f) {
    switch (v) {
        case NppVariant::Box8u: return f == Nv12Form::Frame ? "npp" : "npp_crop_nv12";
        case NppVariant::Convert32fPerCrop: return f == Nv12Form::Frame ? "npp_32f" : "npp_32f_crop_nv12";
        default: return "npp_32f_frame";
    }
}

template <size_t N, NppVariant V, Nv12Form F = Nv12Form::Frame>
Implementation makeNPP(Context& ctx) {
    static_assert(F == Nv12Form::Frame || V != NppVariant::Convert32fFrame,
                  "npp_32f_frame converts the whole 8 bit frame to float: it converts the whole NV12 frame");
    const int plane8Step = PLANE_W * 3;
    const int planeStep = PLANE_W * 3 * sizeof(Npp32f);
    const NppiSize crop{CROP_W, CROP_H};
    const NppiSize mask{K, K};
    const NppiPoint anchor{K / 2, K / 2};
    const NppiPoint origin{0, 0};
    auto st = std::make_shared<NPPState>();
    size_t pitch = 0;
    gpuErrchk(cudaMallocPitch(reinterpret_cast<void**>(&st->rgb8), &pitch, FRAME_W * 3, FRAME_H));
    st->rgbStep = static_cast<int>(pitch);
    if constexpr (V == NppVariant::Convert32fFrame) {
        gpuErrchk(cudaMallocPitch(reinterpret_cast<void**>(&st->rgb32f), &pitch, FRAME_W * 3 * sizeof(Npp32f),
                                  FRAME_H));
        st->rgb32fStep = static_cast<int>(pitch);
    }
    if constexpr (V == NppVariant::Box8u) {
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->planes8), N * PLANE_FLOATS));
    }
    if constexpr (V == NppVariant::Convert32fPerCrop) {
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->crops32f), N * PLANE_FLOATS * sizeof(Npp32f)));
    }
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->tensor), N * PLANE_FLOATS * sizeof(Npp32f)));
    for (size_t t = 0; t < ctx.regions->iterations(); ++t) {
        const auto rects = fklRects<N>(ctx, t);
        st->rects.insert(st->rects.end(), rects.begin(), rects.end());
    }

    Implementation impl{nppVariantName(V, F), V == NppVariant::Box8u ? Reference::Fkl8bitFloorBox : Reference::Fkl8bit};
    if constexpr (F == Nv12Form::Frame) {
        impl.stages.emplace_back("1_nv12_to_rgb8_full_frame", [&ctx, st](size_t) {
            nv12ToRgb8(ctx, st->rgb8, st->rgbStep);
        });
    } else {
        // Written at each crop's position in the 8 bit RGB frame: the frame is written only where the crops of the
        // iteration land, and the rest of the pipeline reads only those pixels
        impl.stages.emplace_back("1_nv12_to_rgb8_per_crop", [&ctx, st](size_t t) {
            for (size_t i = 0; i < N; ++i) {
                nv12ToRgb8Crop(ctx, st->rgb8, st->rgbStep, st->rects[t * N + i]);
            }
        });
    }
    if constexpr (V == NppVariant::Box8u) {
        impl.stages.emplace_back("2_per_crop_filter_box_border_8u", [&ctx, st, plane8Step, crop, mask, anchor,
                                                                     origin](size_t t) {
            for (size_t i = 0; i < N; ++i) {
                const fk::Rect& r = st->rects[t * N + i];
                beforeKernel();
                NPP_CHECK(nppiFilterBoxBorder_8u_C3R_Ctx(
                    st->rgb8 + static_cast<size_t>(r.y) * st->rgbStep + r.x * 3, st->rgbStep, crop, origin,
                    st->planes8 + i * PLANE_FLOATS, plane8Step, crop, mask, anchor, NPP_BORDER_REPLICATE, ctx.npp));
            }
        });
        impl.stages.emplace_back("3_convert_8u32f_planes", [&ctx, st, plane8Step, planeStep](size_t) {
            beforeKernel();
            NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(st->planes8, plane8Step, st->tensor, planeStep,
                                                NppiSize{PLANE_W, static_cast<int>(N) * PLANE_H}, ctx.npp));
        });
    } else if constexpr (V == NppVariant::Convert32fPerCrop) {
        impl.stages.emplace_back("2_per_crop_convert_8u32f_and_filter_box_border_32f",
                                 [&ctx, st, planeStep, crop, mask, anchor, origin](size_t t) {
            for (size_t i = 0; i < N; ++i) {
                const fk::Rect& r = st->rects[t * N + i];
                Npp32f* crop32f = st->crops32f + i * PLANE_FLOATS;
                beforeKernel();
                NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(st->rgb8 + static_cast<size_t>(r.y) * st->rgbStep + r.x * 3,
                                                    st->rgbStep, crop32f, planeStep, crop, ctx.npp));
                beforeKernel();
                NPP_CHECK(nppiFilterBoxBorder_32f_C3R_Ctx(crop32f, planeStep, crop, origin,
                                                          st->tensor + i * PLANE_FLOATS, planeStep, crop, mask,
                                                          anchor, NPP_BORDER_REPLICATE, ctx.npp));
            }
        });
    } else {
        impl.stages.emplace_back("2_convert_8u32f_full_frame", [&ctx, st](size_t) {
            beforeKernel();
            NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(st->rgb8, st->rgbStep, st->rgb32f, st->rgb32fStep,
                                                NppiSize{FRAME_W, FRAME_H}, ctx.npp));
        });
        impl.stages.emplace_back("3_per_crop_filter_box_border_32f", [&ctx, st, planeStep, crop, mask, anchor,
                                                                      origin](size_t t) {
            for (size_t i = 0; i < N; ++i) {
                const fk::Rect& r = st->rects[t * N + i];
                const Npp32f* src = reinterpret_cast<const Npp32f*>(reinterpret_cast<const Npp8u*>(st->rgb32f) +
                                                                    static_cast<size_t>(r.y) * st->rgb32fStep) +
                                    r.x * 3;
                beforeKernel();
                NPP_CHECK(nppiFilterBoxBorder_32f_C3R_Ctx(src, st->rgb32fStep, crop, origin,
                                                          st->tensor + i * PLANE_FLOATS, planeStep, crop, mask,
                                                          anchor, NPP_BORDER_REPLICATE, ctx.npp));
            }
        });
    }
    impl.stages.emplace_back("4_mulc_subc_divc", [&ctx, st, planeStep](size_t) {
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
    impl.fillNaN = [st]() {
        gpuErrchk(cudaMemset(st->tensor, 0xFF, N * PLANE_FLOATS * sizeof(Npp32f)));
        if constexpr (F == Nv12Form::Crop) {
            // The 8 bit frame too, which keeps the crops of earlier runs: a pixel the crops of the validated iteration
            // do not convert, if the pipeline read it, would be 0 instead of a converted value
            gpuErrchk(cudaMemset(st->rgb8, 0, static_cast<size_t>(st->rgbStep) * FRAME_H));
        }
    };
    return impl;
}

struct NppVsFklDpp {
    static constexpr const char* title = "FKL MorphologyDPP box blur vs NPP, neighborhood experiment (fkl-neighborhood)";
    static constexpr const char* fklName = "fkl_dpp";
    static constexpr const char* fklLabel = "FKL DPP";
    static constexpr const char* fklShortLabel = "DPP";
    static constexpr geometry::RegionSize regionSize = CROP_SIZE;

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        impls.push_back(makeNPP<N, NppVariant::Box8u>(ctx));
        impls.push_back(makeNPP<N, NppVariant::Convert32fPerCrop>(ctx));
        impls.push_back(makeNPP<N, NppVariant::Convert32fFrame>(ctx));
        impls.push_back(makeNPP<N, NppVariant::Box8u, Nv12Form::Crop>(ctx));
        impls.push_back(makeNPP<N, NppVariant::Convert32fPerCrop, Nv12Form::Crop>(ctx));
        return impls;
    }

    template <size_t N>
    static Implementation fklImplementation(Context& ctx) {
        return fklDpp<N, K, 1>(ctx);
    }

    template <size_t N>
    static std::vector<float> reference(const Context& ctx, Reference r, size_t t) {
        return boxReferenceOf<N>(ctx, r, t, K, 1);
    }

    static void setup(const std::filesystem::path& outDir, const std::string& mode) { setupTileChoice(outDir, mode); }

    static void workload(std::ostream& os) { describeWorkload(os, K, 1, BATCHES); }

    static void describe(std::ostream& os, const std::filesystem::path&) {
        const NppLibraryVersion* npp = nppGetLibVersion();
        os << "NPP: " << npp->major << "." << npp->minor << "." << npp->build << "\n"
           << "NPP pipelines: nppiNV12ToRGB_8u_ColorTwist32f_P2C3R of the whole frame into an 8 bit RGB frame (as "
              "fkl-vs-npp), the box per crop with nppiFilterBoxBorder (NPP has no batched box filter; oSrcSize the "
              "crop and oSrcOffset 0, NPP_BORDER_REPLICATE: the edge of the crop is replicated; " << K << "x" << K
           << " mask, anchor (" << K / 2 << ", " << K / 2 << ")), and nppiMulC, nppiSubC and nppiDivC_32f_C3IR on "
              "the whole float3 output. One kernel per call.\n"
           << "  npp: per crop nppiFilterBoxBorder_8u_C3R from the crop of the 8 bit frame into its packed 8 bit plane "
              "(floor(sum / " << K * K << ")), then one nppiConvert_8u32f_C3R of the N planes: N + 5 kernels\n"
           << "  npp_32f: per crop nppiConvert_8u32f_C3R of the crop into a float crop, then "
              "nppiFilterBoxBorder_32f_C3R into its output plane: 2N + 4 kernels\n"
           << "  npp_32f_frame: one nppiConvert_8u32f_C3R of the whole frame, then per crop "
              "nppiFilterBoxBorder_32f_C3R from the crop of the float frame into its output plane: N + 5 kernels\n"
           << "  npp_crop_nv12 and npp_32f_crop_nv12: npp and npp_32f with the NV12->RGB of each crop only: per crop "
              "one nppiNV12ToRGB_8u_ColorTwist32f_P2C3R with the same matrix, its Y and UV source pointers at the "
              "crop's origin (row y of the luma, row y / 2 of the chroma; every crop has even coordinates) and "
              "oSizeROI the crop's size, writing the 8 bit RGB at the crop's position in the frame, which the rest of "
              "the pipeline reads as npp and npp_32f do: 2N + 4 and 3N + 3 kernels. The frame is written only where "
              "the crops of the iteration land (overlapping crops are converted twice), and only those pixels are "
              "read; before every validated run, the 8 bit frame is filled with 0 as well as the output with NaN. "
              "npp_32f_frame converts the whole 8 bit frame to float, so it keeps the conversion of the whole frame. "
              "Per batch size, of each pipeline's two conversion forms (npp and npp_crop_nv12, npp_32f and "
              "npp_32f_crop_nv12), each view (latency, GPU only, GPU only with the L2 flushed) reports the one with "
              "the lower median in that view.\n"
           << "References: the 8 bit RGB of NPP's conversion (FKL's conversion rounded to nearest, see the diagnostic "
              "below), with the box rounded down for npp and npp_crop_nv12 and in float for npp_32f, npp_32f_frame and "
              "npp_32f_crop_nv12.\n";
    }

    static std::string diagnostic(const Context& ctx) { return nppConversionDiagnostic(ctx); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench::neighborhood

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::neighborhood::NppVsFklDpp>(argc, argv);
}
