// FKL vs NPP, Figure 1 pipeline of the Backwards Vertical Fusion paper:
// ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// with the frame, sizes, batch sizes and crop positions of common/bench_geometry.h and NV12 -> RGB in BT.601 limited
// range. The harness (validation, timing, modes, output files) is common/bench_gpu.h.
//
//   npp            NV12->RGB of the whole frame (color twist with the BT.601 limited range matrix,
//                  common/bench_npp.h), per crop 8u->32f, one batched resize (nppiResize for N = 1, since
//                  nppiResizeBatch rejects a batch of one), per crop CopyConstBorder (NPP has no batched border), then
//                  MulC, SubC, DivC on the whole output tensor: 2N + 5 kernels, with an 8 bit RGB frame, float3 crops,
//                  float3 resized crops and the float3 output in memory
//   npp_crop_nv12  the same pipeline with NV12->RGB of each crop only (the same color twist with the source pointers at
//                  the crop's origin and the crop's size, writing at the crop's position in the 8 bit RGB frame, which
//                  the rest of the pipeline reads as before): 3N + 4 kernels
//   fkl_bvf        one FKL kernel for the whole chain (BVF)
// The two NPP forms each have their own loop and speedup in every mode; per batch size, each view (latency, GPU only,
// GPU only with the L2 flushed) reports the form with the lower median in that view (the results report says which).

#include <npp.h>

#include "bench_gpu.h"

namespace bench {

struct NPPState {
    Npp8u* rgb8{nullptr};
    int rgbStep{0};
    Npp32f* cropped{nullptr};
    Npp32f* resized{nullptr};
    Npp32f* tensor{nullptr};
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
        cudaFree(tensor);
        cudaFree(batchDevice);
        cudaFreeHost(batchHost);
        cudaFree(dstDescriptors);
    }
};

// The two forms of NPP's NV12 -> RGB: the whole frame, or each crop only
enum class Nv12Form { Frame, Crop };

template <size_t N, Nv12Form F>
Implementation makeNPP(Context& ctx) {
    // One float3 crop of the largest region size per region; a region of another size uses its top-left corner
    const int cropStep = MAX_REGION_W * 3 * sizeof(Npp32f);
    const int outStep = OUT_W * 3 * sizeof(Npp32f);
    const size_t cropFloats = static_cast<size_t>(MAX_REGION_W) * MAX_REGION_H * 3;
    const size_t resizedFloats = static_cast<size_t>(RES_W) * RES_H * 3;
    const size_t outFloats = OUT_FLOATS;
    auto st = std::make_shared<NPPState>();
    size_t rgbPitch = 0;
    gpuErrchk(cudaMallocPitch(reinterpret_cast<void**>(&st->rgb8), &rgbPitch, FRAME_W * 3, FRAME_H));
    st->rgbStep = static_cast<int>(rgbPitch);
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->cropped), N * cropFloats * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->resized), N * resizedFloats * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->tensor), N * outFloats * sizeof(Npp32f)));
    const size_t descriptorBytes = N * sizeof(NppiImageDescriptor);
    const size_t batchBytes = descriptorBytes + N * sizeof(NppiResizeBatchROI_Advanced);
    static_assert(sizeof(NppiImageDescriptor) % alignof(NppiResizeBatchROI_Advanced) == 0, "ROIs after descriptors");
    gpuErrchk(cudaMalloc(&st->batchDevice, batchBytes));
    gpuErrchk(cudaMallocHost(&st->batchHost, batchBytes));
    std::vector<NppiImageDescriptor> dst(N);
    for (size_t i = 0; i < N; ++i) {
        dst[i] = NppiImageDescriptor{st->resized + i * resizedFloats, outStep, NppiSize{RES_W, RES_H}};
    }
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->dstDescriptors), descriptorBytes));
    gpuErrchk(cudaMemcpy(st->dstDescriptors, dst.data(), descriptorBytes, cudaMemcpyHostToDevice));
    for (size_t t = 0; t < ctx.regions->iterations(); ++t) {
        const auto rects = fklRects<N>(ctx, t);
        st->rects.insert(st->rects.end(), rects.begin(), rects.end());
    }

    Implementation impl{F == Nv12Form::Frame ? "npp" : "npp_crop_nv12", Reference::NppSampling8bit};
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
    impl.stages.emplace_back("2_per_crop_convert_8u32f", [&ctx, st, cropFloats, cropStep](size_t t) {
        for (size_t i = 0; i < N; ++i) {
            const fk::Rect& r = st->rects[t * N + i];
            const Npp8u* cropSrc = st->rgb8 + static_cast<size_t>(r.y) * st->rgbStep + r.x * 3;
            beforeKernel();
            NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(cropSrc, st->rgbStep, st->cropped + i * cropFloats, cropStep,
                                                NppiSize{r.width, r.height}, ctx.npp));
        }
    });
    impl.stages.emplace_back("3_resize_batch", [&ctx, st, cropFloats, cropStep, outStep, descriptorBytes,
                                                batchBytes](size_t t) {
        if constexpr (N == 1) {
            // nppiResizeBatch rejects a batch of one image with NPP_BAD_ARGUMENT_ERROR
            const fk::Rect& r = st->rects[t];
            beforeKernel();
            NPP_CHECK(nppiResize_32f_C3R_Ctx(st->cropped, cropStep, NppiSize{r.width, r.height},
                                             NppiRect{0, 0, r.width, r.height}, st->resized, outStep,
                                             NppiSize{RES_W, RES_H}, NppiRect{0, 0, RES_W, RES_H},
                                             NPPI_INTER_LINEAR, ctx.npp));
        } else {
            // The source descriptors and ROIs of this iteration's regions, uploaded with one copy
            auto* src = static_cast<NppiImageDescriptor*>(st->batchHost);
            auto* rois = reinterpret_cast<NppiResizeBatchROI_Advanced*>(static_cast<char*>(st->batchHost) +
                                                                         descriptorBytes);
            for (size_t i = 0; i < N; ++i) {
                const fk::Rect& r = st->rects[t * N + i];
                src[i] = NppiImageDescriptor{st->cropped + i * cropFloats, cropStep, NppiSize{r.width, r.height}};
                rois[i] = NppiResizeBatchROI_Advanced{NppiRect{0, 0, r.width, r.height}, NppiRect{0, 0, RES_W, RES_H}};
            }
            gpuErrchk(cudaMemcpyAsync(st->batchDevice, st->batchHost, batchBytes, cudaMemcpyHostToDevice,
                                      ctx.stream));
            beforeKernel();
            NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(
                RES_W, RES_H, static_cast<NppiImageDescriptor*>(st->batchDevice), st->dstDescriptors,
                reinterpret_cast<NppiResizeBatchROI_Advanced*>(static_cast<char*>(st->batchDevice) + descriptorBytes),
                static_cast<unsigned int>(N), NPPI_INTER_LINEAR, ctx.npp));
        }
    });
    impl.stages.emplace_back("4_per_crop_copy_const_border", [&ctx, st, resizedFloats, outFloats, outStep](size_t) {
        static const Npp32f zero[3] = {0.f, 0.f, 0.f};
        for (size_t i = 0; i < N; ++i) {
            beforeKernel();
            NPP_CHECK(nppiCopyConstBorder_32f_C3R_Ctx(st->resized + i * resizedFloats, outStep, NppiSize{RES_W, RES_H},
                                                      st->tensor + i * outFloats, outStep, NppiSize{OUT_W, OUT_H},
                                                      BORDER_TOP, 0, zero, ctx.npp));
        }
    });
    impl.stages.emplace_back("5_mulc_subc_divc", [&ctx, st, outStep](size_t) {
        const NppiSize tensorSize{OUT_W, static_cast<int>(N) * OUT_H};
        beforeKernel();
        NPP_CHECK(nppiMulC_32f_C3IR_Ctx(MUL, st->tensor, outStep, tensorSize, ctx.npp));
        beforeKernel();
        NPP_CHECK(nppiSubC_32f_C3IR_Ctx(SUB, st->tensor, outStep, tensorSize, ctx.npp));
        beforeKernel();
        NPP_CHECK(nppiDivC_32f_C3IR_Ctx(DIV, st->tensor, outStep, tensorSize, ctx.npp));
    });
    impl.run = runStages(impl.stages);
    impl.download = [st, outFloats]() { return downloadDevice(st->tensor, N * outFloats); };
    impl.fillNaN = [st, outFloats]() {
        gpuErrchk(cudaMemset(st->tensor, 0xFF, N * outFloats * sizeof(Npp32f)));
        if constexpr (F == Nv12Form::Crop) {
            // The 8 bit frame too, which keeps the crops of earlier runs: a pixel the crops of the validated iteration
            // do not convert, if the pipeline read it, would be 0 instead of a converted value
            gpuErrchk(cudaMemset(st->rgb8, 0, static_cast<size_t>(st->rgbStep) * FRAME_H));
        }
    };
    return impl;
}

struct NppVsFkl {
    static constexpr const char* title = "FKL vs NPP (fkl-vs-npp)";

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        impls.push_back(makeNPP<N, Nv12Form::Frame>(ctx));
        impls.push_back(makeNPP<N, Nv12Form::Crop>(ctx));
        return impls;
    }

    static void describe(std::ostream& os, const std::filesystem::path&) {
        const NppLibraryVersion* npp = nppGetLibVersion();
        os << "NPP: " << npp->major << "." << npp->minor << "." << npp->build << "\n"
           << "NPP pipeline: nppiNV12ToRGB_8u_ColorTwist32f_P2C3R (BT.601 limited range matrix) of the whole frame into "
              "an 8 bit RGB frame, then per region nppiConvert_8u32f_C3R (float3 crop, into a buffer of the largest "
              "region size), one nppiResizeBatch_32f_C3R_Advanced (nppiResize_32f_C3R for N = 1) into float3 resized "
              "crops, per region nppiCopyConstBorder_32f_C3R (NPP has no batched border) into its output plane, then "
              "nppiMulC, nppiSubC and nppiDivC_32f_C3IR on the whole float3 output: 2N + 5 kernels. The batched "
              "resize takes per-image sizes and ROIs from descriptors in device memory: every iteration, inside the "
              "timed region, the source descriptors and ROIs of its regions are written into a pinned host buffer and "
              "uploaded with one cudaMemcpyAsync on the pipeline's stream (one memory copy per execution for N > 1; "
              "the destination descriptors are the same in every iteration and are uploaded once, before the loops). "
              "NPP samples its bilinear resize at dst * scale - 0.25 (measured), so it is checked against a reference "
              "with that sampling.\n"
           << "npp_crop_nv12: the same pipeline with the NV12->RGB of each crop only: per region one "
              "nppiNV12ToRGB_8u_ColorTwist32f_P2C3R with the same matrix, its Y and UV source pointers at the crop's "
              "origin (row y of the luma, row y / 2 of the chroma; every region has even coordinates and sizes) and "
              "oSizeROI the crop's size, writing the 8 bit RGB at the crop's position in the frame, which the rest of "
              "the pipeline reads with the same views and calls as npp: 3N + 4 kernels. The frame is written only "
              "where the crops of the iteration land (overlapping crops are converted twice), and only those pixels "
              "are read; before every validated run, the 8 bit frame is filled with 0 as well as the output with "
              "NaN.\n"
           << "The two NPP forms each run their own loop (npp, then npp_crop_nv12, then FKL BVF) and have their own "
              "speedup in every mode; per batch size, each view (latency, GPU only, GPU only with the L2 flushed) "
              "reports the form with the lower median in that view.\n";
    }

    static std::string diagnostic(const Context& ctx) { return nppConversionDiagnostic(ctx); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::NppVsFkl>(argc, argv);
}
