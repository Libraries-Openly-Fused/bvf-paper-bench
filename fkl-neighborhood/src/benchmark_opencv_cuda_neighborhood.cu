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

// Experiment 3, part (b): FKL's MorphologyDPP against OpenCV CUDA on the pipeline of the neighborhood experiment
// (common/bench_neighborhood.h), with a MORPH_K x MORPH_K mask and the operation NEIGHBORHOOD_OP (0: erode, 1: open;
// one build each, fkl-neighborhood/CMakeLists.txt). The harness is common/bench_gpu.h.
//
// OpenCV CUDA's morphology filters (cudafilters) take 8UC1, 8UC4, 32FC1 and 32FC4 images, not 3 channels, so the
// pipeline of Figure 1 (fkl-vs-opencv-cuda: per region convertTo to float, resize) works on 4 channels here, and the
// alpha channel is dropped at the end:
//   opencv_cuda                  NPP NV12->RGB of the whole frame, cvtColor RGB2RGBA of the whole frame, then per
//                                region convertTo 32FC4, resize and the morphology filter (createMorphologyFilter,
//                                MORPH_ERODE or MORPH_OPEN, into its output plane), then multiply, subtract and divide
//                                on the whole 4 channel output and one cvtColor RGBA2RGB into the float3 output
//   opencv_cuda_rgba_per_region  the same with cvtColor RGB2RGBA per region (of the region of the RGB frame) instead
//                                of the whole frame
//   fkl_dpp                      FKL's MorphologyDPP, one kernel (neighborhood_common.h)

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include "neighborhood_common.h"

#ifndef NEIGHBORHOOD_OP
#error "Define NEIGHBORHOOD_OP, 0 (erode) or 1 (open) (fkl-neighborhood/CMakeLists.txt does)"
#endif

namespace bench::neighborhood {

constexpr MorphOp OP = NEIGHBORHOOD_OP == 0 ? MorphOp::Erode : MorphOp::Open;
static_assert(NEIGHBORHOOD_OP == 0 || NEIGHBORHOOD_OP == 1, "0 (erode) or 1 (open)");
constexpr int K = MORPH_K;

struct OpenCVState {
    cv::cuda::Stream stream;
    cv::cuda::GpuMat rgb8;
    // The whole frame with 4 channels (opencv_cuda), or one 8UC4 region of the largest size per region
    // (opencv_cuda_rgba_per_region)
    cv::cuda::GpuMat rgba8;
    std::vector<cv::cuda::GpuMat> rgba8Regions;
    // The regions of every iteration, N per iteration (rects[t * N + i])
    std::vector<cv::Rect> rects;
    // One 32FC4 crop of the largest region size per region; a region of another size uses its top-left corner
    std::vector<cv::cuda::GpuMat> cropped;
    std::vector<cv::cuda::GpuMat> resized;
    cv::Ptr<cv::cuda::Filter> morphology;
    // The 4 channel output of the morphology and the normalization, and the float3 output
    cv::cuda::GpuMat tensor4;
    cv::cuda::GpuMat tensor;
};

template <size_t N, bool RGBA_PER_REGION>
Implementation makeOpenCV(Context& ctx) {
    auto st = std::make_shared<OpenCVState>();
    st->stream = cv::cuda::StreamAccessor::wrapStream(ctx.stream);
    st->rgb8.create(FRAME_H, FRAME_W, CV_8UC3);
    if constexpr (!RGBA_PER_REGION) {
        st->rgba8.create(FRAME_H, FRAME_W, CV_8UC4);
    }
    for (size_t t = 0; t < ctx.regions->iterations(); ++t) {
        for (size_t i = 0; i < N; ++i) {
            const Region& r = ctx.regions->at(t, i);
            st->rects.emplace_back(r.x, r.y, r.width, r.height);
        }
    }
    for (size_t i = 0; i < N; ++i) {
        if constexpr (RGBA_PER_REGION) {
            st->rgba8Regions.emplace_back(MAX_REGION_H, MAX_REGION_W, CV_8UC4);
        }
        st->cropped.emplace_back(MAX_REGION_H, MAX_REGION_W, CV_32FC4);
        st->resized.emplace_back(PLANE_H, PLANE_W, CV_32FC4);
    }
    st->morphology = cv::cuda::createMorphologyFilter(OP == MorphOp::Erode ? cv::MORPH_ERODE : cv::MORPH_OPEN,
                                                      CV_32FC4, cv::Mat::ones(K, K, CV_8U));
    cv::cuda::createContinuous(static_cast<int>(N) * PLANE_H, PLANE_W, CV_32FC4, st->tensor4);
    cv::cuda::createContinuous(static_cast<int>(N) * PLANE_H, PLANE_W, CV_32FC3, st->tensor);

    Implementation impl{RGBA_PER_REGION ? "opencv_cuda_rgba_per_region" : "opencv_cuda", Reference::Fkl8bit};
    impl.stages.emplace_back("1_nv12_to_rgb8_full_frame_npp", [&ctx, st](size_t) {
        nv12ToRgb8(ctx, st->rgb8.data, static_cast<int>(st->rgb8.step));
    });
    if constexpr (!RGBA_PER_REGION) {
        impl.stages.emplace_back("2_rgb_to_rgba_full_frame", [st](size_t) {
            beforeKernel();
            cv::cuda::cvtColor(st->rgb8, st->rgba8, cv::COLOR_RGB2RGBA, 0, st->stream);
        });
    }
    const std::string perRegion = RGBA_PER_REGION ? "3_per_crop_cvtColor_convertTo_resize_morphology"
                                                  : "3_per_crop_convertTo_resize_morphology";
    impl.stages.emplace_back(perRegion, [st](size_t t) {
        for (size_t i = 0; i < N; ++i) {
            const cv::Rect& r = st->rects[t * N + i];
            const cv::Rect regionSize(0, 0, r.width, r.height);
            cv::cuda::GpuMat rgba;
            if constexpr (RGBA_PER_REGION) {
                rgba = st->rgba8Regions[i](regionSize);
                beforeKernel();
                cv::cuda::cvtColor(st->rgb8(r), rgba, cv::COLOR_RGB2RGBA, 0, st->stream);
            } else {
                rgba = st->rgba8(r);
            }
            cv::cuda::GpuMat cropped = st->cropped[i](regionSize);
            beforeKernel();
            rgba.convertTo(cropped, CV_32F, st->stream);
            beforeKernel();
            cv::cuda::resize(cropped, st->resized[i], cv::Size(PLANE_W, PLANE_H), 0., 0., cv::INTER_LINEAR, st->stream);
            cv::cuda::GpuMat plane =
                st->tensor4.rowRange(static_cast<int>(i) * PLANE_H, static_cast<int>(i + 1) * PLANE_H);
            // A copyMakeBorder and an NPP kernel per pass in one call: with the L2 flushed, the flush runs before the
            // call only
            beforeKernel();
            st->morphology->apply(st->resized[i], plane, st->stream);
        }
    });
    impl.stages.emplace_back("4_multiply_subtract_divide", [st](size_t) {
        // The alpha channel is multiplied and divided by 1 and has 0 subtracted
        beforeKernel();
        cv::cuda::multiply(st->tensor4, cv::Scalar(MUL[0], MUL[1], MUL[2], 1.), st->tensor4, 1., -1, st->stream);
        beforeKernel();
        cv::cuda::subtract(st->tensor4, cv::Scalar(SUB[0], SUB[1], SUB[2], 0.), st->tensor4, cv::noArray(), -1,
                           st->stream);
        beforeKernel();
        cv::cuda::divide(st->tensor4, cv::Scalar(DIV[0], DIV[1], DIV[2], 1.), st->tensor4, 1., -1, st->stream);
    });
    impl.stages.emplace_back("5_rgba_to_rgb", [st](size_t) {
        beforeKernel();
        cv::cuda::cvtColor(st->tensor4, st->tensor, cv::COLOR_RGBA2RGB, 0, st->stream);
    });
    impl.run = runStages(impl.stages);
    impl.download = [st]() { return downloadDevice(st->tensor.data, N * PLANE_FLOATS); };
    impl.fillNaN = [st]() { gpuErrchk(cudaMemset(st->tensor.data, 0xFF, N * PLANE_FLOATS * sizeof(float))); };
    return impl;
}

struct OpenCVCudaVsFklDpp {
    static constexpr const char* title =
        "FKL MorphologyDPP vs OpenCV CUDA, neighborhood experiment (fkl-neighborhood)";
    static constexpr const char* fklName = "fkl_dpp";
    static constexpr const char* fklLabel = "FKL DPP";
    static constexpr const char* fklShortLabel = "DPP";

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        impls.push_back(makeOpenCV<N, false>(ctx));
        impls.push_back(makeOpenCV<N, true>(ctx));
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

    static void describe(std::ostream& os, const std::filesystem::path& outDir) {
        const std::string opencvBuild = cv::getBuildInformation();
        std::ofstream(outDir / "opencv_build_information.txt") << opencvBuild;
        const NppLibraryVersion* npp = nppGetLibVersion();
        const bool erode = OP == MorphOp::Erode;
        os << "OpenCV: " << cv::getVersionString() << " (opencv_contrib "
           << bench::buildInfoValue(opencvBuild, "Version control (extra):") << ", CUDA_ARCH_BIN "
           << bench::buildInfoValue(opencvBuild, "NVIDIA GPU arch:") << ", C++ compiler "
           << bench::buildInfoValue(opencvBuild, "C++ Compiler:")
           << "; full configuration in opencv_build_information.txt)\n"
           << "NPP: " << npp->major << "." << npp->minor << "." << npp->build
           << " (the NV12 -> RGB of the OpenCV pipelines: OpenCV CUDA has no NV12 conversion)\n"
           << "OpenCV CUDA pipelines: OpenCV CUDA's morphology filters take 8UC1, 8UC4, 32FC1 and 32FC4 images (not "
              "3 channels), so the pipeline of fkl-vs-opencv-cuda works on 4 channels: NPP NV12->RGB of the whole "
              "frame into an 8 bit RGB frame, then per region convertTo 32FC4 (into a buffer of the largest region "
              "size through a view of the region's size), resize and the morphology filter (createMorphologyFilter("
           << (erode ? "MORPH_ERODE" : "MORPH_OPEN") << ", CV_32FC4, " << K << "x" << K << " ones), whose apply "
              "copies the plane with copyMakeBorder (BORDER_REFLECT_101, which gives the same minimum and maximum as "
              "replicating the border for a centered rectangular mask) and runs nppiErode / nppiDilate_32f_C4R on it, "
              "one copy and one NPP kernel per pass) into its plane of a 4 channel output, then multiply, subtract and "
              "divide on the whole 4 channel output and one cvtColor RGBA2RGB into the float3 output.\n"
           << "  opencv_cuda: cvtColor RGB2RGBA of the whole 8 bit frame before the per-region calls: "
           << (erode ? "4N + 6" : "6N + 6") << " kernels\n"
           << "  opencv_cuda_rgba_per_region: cvtColor RGB2RGBA of the region of the 8 bit frame, per region, into a "
              "buffer of the largest region size: " << (erode ? "5N + 5" : "7N + 5") << " kernels\n"
           << "With the L2 flushed, the flush runs before every OpenCV call, not between the kernels of one call (the "
              "morphology filter launches " << (erode ? 2 : 4) << " per call).\n";
    }

    static std::string diagnostic(const Context& ctx) { return nppConversionDiagnostic(ctx); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench::neighborhood

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::neighborhood::OpenCVCudaVsFklDpp>(argc, argv);
}
