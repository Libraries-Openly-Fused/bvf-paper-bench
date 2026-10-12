// FKL vs OpenCV CUDA, Figure 1 pipeline of the Backwards Vertical Fusion paper:
// ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// with the frame, sizes, batch sizes and crop positions of common/bench_geometry.h and NV12 -> RGB in BT.601 limited
// range. The harness (validation, timing, modes, output files) is common/bench_gpu.h.
//
//   opencv_cuda   NPP NV12->RGB of the whole frame (OpenCV CUDA has no NV12 conversion), then per crop cv::cuda
//                 convertTo, resize and copyMakeBorder, then multiply, subtract and divide on the whole output tensor:
//                 3N + 4 kernels, with an 8 bit RGB frame, float3 crops, float3 resized crops and the float3 output
//                 in memory
//   fkl_bvf       one FKL kernel for the whole chain (BVF)

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include "bench_gpu.h"

namespace bench {

struct OpenCVState {
    cv::cuda::Stream stream;
    cv::cuda::GpuMat rgb8;
    std::vector<cv::Rect> rects;
    std::vector<cv::cuda::GpuMat> cropped;
    std::vector<cv::cuda::GpuMat> resized;
    cv::cuda::GpuMat tensor;
};

template <size_t N>
Implementation makeOpenCV(Context& ctx) {
    auto st = std::make_shared<OpenCVState>();
    st->stream = cv::cuda::StreamAccessor::wrapStream(ctx.stream);
    st->rgb8.create(FRAME_H, FRAME_W, CV_8UC3);
    for (size_t i = 0; i < N; ++i) {
        const fk::Rect& r = ctx.crops[i];
        st->rects.emplace_back(r.x, r.y, r.width, r.height);
        st->cropped.emplace_back(CROP_H, CROP_W, CV_32FC3);
        st->resized.emplace_back(RES_H, RES_W, CV_32FC3);
    }
    cv::cuda::createContinuous(static_cast<int>(N) * OUT_H, OUT_W, CV_32FC3, st->tensor);
    Implementation impl{"opencv_cuda", Reference::Fkl8bit};
    impl.stages.emplace_back("1_nv12_to_rgb8_full_frame_npp", [&ctx, st]() {
        nv12ToRgb8(ctx, st->rgb8.data, static_cast<int>(st->rgb8.step));
    });
    impl.stages.emplace_back("2_per_crop_convertTo_resize_copyMakeBorder", [st]() {
        for (size_t i = 0; i < N; ++i) {
            beforeKernel();
            st->rgb8(st->rects[i]).convertTo(st->cropped[i], CV_32F, st->stream);
            beforeKernel();
            cv::cuda::resize(st->cropped[i], st->resized[i], cv::Size(RES_W, RES_H), 0., 0., cv::INTER_LINEAR,
                             st->stream);
            cv::cuda::GpuMat plane = st->tensor.rowRange(static_cast<int>(i) * OUT_H, static_cast<int>(i + 1) * OUT_H);
            beforeKernel();
            cv::cuda::copyMakeBorder(st->resized[i], plane, BORDER_TOP, BORDER_BOTTOM, 0, 0, cv::BORDER_CONSTANT,
                                     cv::Scalar(0., 0., 0.), st->stream);
        }
    });
    impl.stages.emplace_back("3_multiply_subtract_divide", [st]() {
        beforeKernel();
        cv::cuda::multiply(st->tensor, cv::Scalar(MUL[0], MUL[1], MUL[2]), st->tensor, 1., -1, st->stream);
        beforeKernel();
        cv::cuda::subtract(st->tensor, cv::Scalar(SUB[0], SUB[1], SUB[2]), st->tensor, cv::noArray(), -1, st->stream);
        beforeKernel();
        cv::cuda::divide(st->tensor, cv::Scalar(DIV[0], DIV[1], DIV[2]), st->tensor, 1., -1, st->stream);
    });
    impl.run = runStages(impl.stages);
    impl.download = [st]() { return downloadDevice(st->tensor.data, N * OUT_FLOATS); };
    impl.fillNaN = [st]() { gpuErrchk(cudaMemset(st->tensor.data, 0xFF, N * OUT_FLOATS * sizeof(float))); };
    return impl;
}

struct OpenCVCudaVsFkl {
    static constexpr const char* title = "FKL vs OpenCV CUDA (fkl-vs-opencv-cuda)";

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        impls.push_back(makeOpenCV<N>(ctx));
        return impls;
    }

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
           << " (the NV12 -> RGB of the OpenCV pipeline: OpenCV CUDA has no NV12 conversion)\n"
           << "OpenCV CUDA pipeline: NPP NV12->RGB of the whole frame into an 8 bit RGB frame (NPP's NV12 conversion "
              "only writes 8 or 16 bit integers), then per crop cv::cuda convertTo (float3 crop), resize (float3 "
              "resized crop) and copyMakeBorder (into its output plane), then multiply, subtract and divide on the "
              "whole float3 output: 3N + 4 kernels.\n";
    }

    static std::string diagnostic(const Context& ctx) { return nppConversionDiagnostic(ctx); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::OpenCVCudaVsFkl>(argc, argv);
}
