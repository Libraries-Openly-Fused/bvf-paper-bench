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

// Experiment 3, part (b): FKL's MorphologyDPP box blur against OpenCV CUDA on the pipeline of the neighborhood
// experiment (common/bench_neighborhood.h): BOX_K x BOX_K box, one pass. The harness is common/bench_gpu.h.
//
// OpenCV CUDA's box filter (createBoxFilter) takes 8UC1, 8UC4 and 32FC1 only, so the box is a linear filter: the
// separable filter takes 3 channels, the 2D one (createLinearFilter) 1 or 4. Its filters build their border from the
// image they are given (a view of the crop), so REPLICATE replicates the edge of the crop.
//   opencv_cuda       NPP NV12->RGB of the whole frame (OpenCV CUDA has no NV12 conversion); per crop convertTo
//                     32FC3 of the crop of the 8 bit frame, then createSeparableLinearFilter (CV_32FC3, row and column
//                     taps 1/5, BORDER_REPLICATE; a row and a column kernel) into its plane of the float3 output; then
//                     multiply, subtract and divide on the whole output: 3N + 4 kernels
//   opencv_cuda_rgba  NPP NV12->RGB; cvtColor RGB2RGBA of the whole frame; per crop convertTo 32FC4, then
//                     createLinearFilter (CV_32FC4, 5x5 taps 1/25, BORDER_REPLICATE; one kernel) into its plane of a
//                     4 channel output; multiply, subtract and divide on it; cvtColor RGBA2RGB into the float3 output:
//                     2N + 6 kernels
//   fkl_dpp           FKL's MorphologyDPP box blur, one kernel (neighborhood_common.h)

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>

#include "neighborhood_common.h"

namespace bench::neighborhood {

constexpr int K = BOX_K;

struct OpenCVState {
    cv::cuda::Stream stream;
    cv::cuda::GpuMat rgb8;
    // The whole frame with 4 channels (opencv_cuda_rgba)
    cv::cuda::GpuMat rgba8;
    // The crops of every iteration, N per iteration (rects[t * N + i])
    std::vector<cv::Rect> rects;
    // One float crop per crop
    std::vector<cv::cuda::GpuMat> cropped;
    cv::Ptr<cv::cuda::Filter> box;
    // The 4 channel output of the box and the normalization (opencv_cuda_rgba), and the float3 output
    cv::cuda::GpuMat tensor4;
    cv::cuda::GpuMat tensor;
};

template <size_t N, bool RGBA>
Implementation makeOpenCV(Context& ctx) {
    auto st = std::make_shared<OpenCVState>();
    st->stream = cv::cuda::StreamAccessor::wrapStream(ctx.stream);
    st->rgb8.create(FRAME_H, FRAME_W, CV_8UC3);
    if constexpr (RGBA) {
        st->rgba8.create(FRAME_H, FRAME_W, CV_8UC4);
    }
    for (size_t t = 0; t < ctx.regions->iterations(); ++t) {
        for (size_t i = 0; i < N; ++i) {
            const Region& r = ctx.regions->at(t, i);
            st->rects.emplace_back(r.x, r.y, r.width, r.height);
        }
    }
    const int floatType = RGBA ? CV_32FC4 : CV_32FC3;
    for (size_t i = 0; i < N; ++i) {
        st->cropped.emplace_back(CROP_H, CROP_W, floatType);
    }
    if constexpr (RGBA) {
        st->box = cv::cuda::createLinearFilter(CV_32FC4, CV_32FC4,
                                               cv::Mat(K, K, CV_32F, cv::Scalar(1. / (K * K))), cv::Point(-1, -1),
                                               cv::BORDER_REPLICATE);
        cv::cuda::createContinuous(static_cast<int>(N) * PLANE_H, PLANE_W, CV_32FC4, st->tensor4);
    } else {
        const cv::Mat taps(1, K, CV_32F, cv::Scalar(1. / K));
        st->box = cv::cuda::createSeparableLinearFilter(CV_32FC3, CV_32FC3, taps, taps, cv::Point(-1, -1),
                                                        cv::BORDER_REPLICATE, cv::BORDER_REPLICATE);
    }
    cv::cuda::createContinuous(static_cast<int>(N) * PLANE_H, PLANE_W, CV_32FC3, st->tensor);

    Implementation impl{RGBA ? "opencv_cuda_rgba" : "opencv_cuda", Reference::Fkl8bit};
    impl.stages.emplace_back("1_nv12_to_rgb8_full_frame_npp", [&ctx, st](size_t) {
        nv12ToRgb8(ctx, st->rgb8.data, static_cast<int>(st->rgb8.step));
    });
    if constexpr (RGBA) {
        impl.stages.emplace_back("2_rgb_to_rgba_full_frame", [st](size_t) {
            beforeKernel();
            cv::cuda::cvtColor(st->rgb8, st->rgba8, cv::COLOR_RGB2RGBA, 0, st->stream);
        });
    }
    impl.stages.emplace_back(RGBA ? "3_per_crop_convertTo_linear_filter" : "3_per_crop_convertTo_separable_filter",
                             [st](size_t t) {
        cv::cuda::GpuMat& output = RGBA ? st->tensor4 : st->tensor;
        for (size_t i = 0; i < N; ++i) {
            const cv::Rect& r = st->rects[t * N + i];
            beforeKernel();
            (RGBA ? st->rgba8 : st->rgb8)(r).convertTo(st->cropped[i], CV_32F, st->stream);
            cv::cuda::GpuMat plane = output.rowRange(static_cast<int>(i) * PLANE_H, static_cast<int>(i + 1) * PLANE_H);
            // The separable filter is a row and a column kernel in one call: with the L2 flushed, the flush runs
            // before the call only
            beforeKernel();
            st->box->apply(st->cropped[i], plane, st->stream);
        }
    });
    impl.stages.emplace_back("4_multiply_subtract_divide", [st](size_t) {
        if constexpr (RGBA) {
            // The alpha channel is multiplied and divided by 1 and has 0 subtracted
            beforeKernel();
            cv::cuda::multiply(st->tensor4, cv::Scalar(MUL[0], MUL[1], MUL[2], 1.), st->tensor4, 1., -1, st->stream);
            beforeKernel();
            cv::cuda::subtract(st->tensor4, cv::Scalar(SUB[0], SUB[1], SUB[2], 0.), st->tensor4, cv::noArray(), -1,
                               st->stream);
            beforeKernel();
            cv::cuda::divide(st->tensor4, cv::Scalar(DIV[0], DIV[1], DIV[2], 1.), st->tensor4, 1., -1, st->stream);
        } else {
            beforeKernel();
            cv::cuda::multiply(st->tensor, cv::Scalar(MUL[0], MUL[1], MUL[2]), st->tensor, 1., -1, st->stream);
            beforeKernel();
            cv::cuda::subtract(st->tensor, cv::Scalar(SUB[0], SUB[1], SUB[2]), st->tensor, cv::noArray(), -1,
                               st->stream);
            beforeKernel();
            cv::cuda::divide(st->tensor, cv::Scalar(DIV[0], DIV[1], DIV[2]), st->tensor, 1., -1, st->stream);
        }
    });
    if constexpr (RGBA) {
        impl.stages.emplace_back("5_rgba_to_rgb", [st](size_t) {
            beforeKernel();
            cv::cuda::cvtColor(st->tensor4, st->tensor, cv::COLOR_RGBA2RGB, 0, st->stream);
        });
    }
    impl.run = runStages(impl.stages);
    impl.download = [st]() { return downloadDevice(st->tensor.data, N * PLANE_FLOATS); };
    impl.fillNaN = [st]() { gpuErrchk(cudaMemset(st->tensor.data, 0xFF, N * PLANE_FLOATS * sizeof(float))); };
    return impl;
}

struct OpenCVCudaVsFklDpp {
    static constexpr const char* title =
        "FKL MorphologyDPP box blur vs OpenCV CUDA, neighborhood experiment (fkl-neighborhood)";
    static constexpr const char* fklName = "fkl_dpp";
    static constexpr const char* fklLabel = "FKL DPP";
    static constexpr const char* fklShortLabel = "DPP";
    static constexpr geometry::RegionSize regionSize = CROP_SIZE;

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        impls.push_back(makeOpenCV<N, false>(ctx));
        impls.push_back(makeOpenCV<N, true>(ctx));
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

    static void describe(std::ostream& os, const std::filesystem::path& outDir) {
        const std::string opencvBuild = cv::getBuildInformation();
        std::ofstream(outDir / "opencv_build_information.txt") << opencvBuild;
        const NppLibraryVersion* npp = nppGetLibVersion();
        os << "OpenCV: " << cv::getVersionString() << " (opencv_contrib "
           << bench::buildInfoValue(opencvBuild, "Version control (extra):") << ", CUDA_ARCH_BIN "
           << bench::buildInfoValue(opencvBuild, "NVIDIA GPU arch:") << ", C++ compiler "
           << bench::buildInfoValue(opencvBuild, "C++ Compiler:")
           << "; full configuration in opencv_build_information.txt)\n"
           << "NPP: " << npp->major << "." << npp->minor << "." << npp->build
           << " (the NV12 -> RGB of the OpenCV pipelines: OpenCV CUDA has no NV12 conversion)\n"
           << "OpenCV CUDA pipelines: NPP NV12->RGB of the whole frame into an 8 bit RGB frame, then per crop "
              "convertTo of the crop of the frame (a view of it) into a float crop and the box as a linear filter on "
              "it, written into the crop's plane of the output (OpenCV CUDA's createBoxFilter takes 8UC1, 8UC4 and "
              "32FC1 only). The filters build their border from the float crop, so BORDER_REPLICATE replicates the "
              "edge of the crop.\n"
           << "  opencv_cuda: convertTo 32FC3 and createSeparableLinearFilter(CV_32FC3, CV_32FC3, " << K << " taps of "
              "1/" << K << " for the rows and the columns, BORDER_REPLICATE), a row and a column kernel per call, "
              "then multiply, subtract and divide on the float3 output: 3N + 4 kernels\n"
           << "  opencv_cuda_rgba: cvtColor RGB2RGBA of the whole 8 bit frame, convertTo 32FC4 and "
              "createLinearFilter(CV_32FC4, CV_32FC4, " << K << "x" << K << " taps of 1/" << K * K << ", "
              "BORDER_REPLICATE), one kernel per call, then multiply, subtract and divide on the 4 channel output and "
              "one cvtColor RGBA2RGB into the float3 output: 2N + 6 kernels\n"
           << "With the L2 flushed, the flush runs before every OpenCV call, not between the two kernels of the "
              "separable filter.\n"
           << "References: the 8 bit RGB of NPP's conversion (FKL's conversion rounded to nearest, see the diagnostic "
              "below) with the box in float.\n";
    }

    static std::string diagnostic(const Context& ctx) { return nppConversionDiagnostic(ctx); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench::neighborhood

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::neighborhood::OpenCVCudaVsFklDpp>(argc, argv);
}
