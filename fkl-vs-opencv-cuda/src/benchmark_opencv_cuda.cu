// FKL vs OpenCV CUDA, Figure 1 pipeline of the Backwards Vertical Fusion paper:
// ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// with the frame, sizes, batch sizes and crop positions of common/bench_geometry.h and NV12 -> RGB in BT.601 limited
// range. The harness (validation, timing, modes, output files) is common/bench_gpu.h.
//
//   opencv_cuda   OpenCV's NV12->RGB of the whole frame (cudacodec's NVSurfaceToColorConverter, which converts whole
//                 surfaces only: common/bench_opencv_cuda.h), then per crop cv::cuda convertTo, resize and
//                 copyMakeBorder, then multiply, subtract and divide on the whole output tensor: 3N + 4 kernels, with
//                 an 8 bit RGB frame, float3 crops, float3 resized crops and the float3 output in memory
//   fkl_bvf       one FKL kernel for the whole chain (BVF)

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include "bench_gpu.h"
#include "bench_opencv_cuda.h"

namespace bench {

struct OpenCVState {
    cv::cuda::Stream stream;
    opencv::Nv12Converter converter;
    cv::cuda::GpuMat rgb8;
    // The regions of every iteration, N per iteration (rects[t * N + i])
    std::vector<cv::Rect> rects;
    // One float3 crop of the largest region size per region; a region of another size uses its top-left corner
    std::vector<cv::cuda::GpuMat> cropped;
    std::vector<cv::cuda::GpuMat> resized;
    cv::cuda::GpuMat tensor;
};

// Per region: the region of the 8 bit RGB frame and the part of its crop buffer of the region's size are views
// (GpuMat headers) made in the iteration, then convertTo, resize and copyMakeBorder take them
template <size_t N>
Implementation makeOpenCV(Context& ctx) {
    auto st = std::make_shared<OpenCVState>();
    st->stream = cv::cuda::StreamAccessor::wrapStream(ctx.stream);
    st->rgb8.create(FRAME_H, FRAME_W, CV_8UC3);
    for (size_t t = 0; t < ctx.regions->iterations(); ++t) {
        for (size_t i = 0; i < N; ++i) {
            const Region& r = ctx.regions->at(t, i);
            st->rects.emplace_back(r.x, r.y, r.width, r.height);
        }
    }
    for (size_t i = 0; i < N; ++i) {
        st->cropped.emplace_back(MAX_REGION_H, MAX_REGION_W, CV_32FC3);
        st->resized.emplace_back(RES_H, RES_W, CV_32FC3);
    }
    cv::cuda::createContinuous(static_cast<int>(N) * OUT_H, OUT_W, CV_32FC3, st->tensor);
    Implementation impl{"opencv_cuda", Reference::Fkl8bitTruncated};
    impl.stages.emplace_back("1_nv12_to_rgb8_full_frame", [&ctx, st](size_t) {
        st->converter.convert(ctx, st->rgb8, cv::cudacodec::ColorFormat::RGB, st->stream);
    });
    impl.stages.emplace_back("2_per_crop_convertTo_resize_copyMakeBorder", [st](size_t t) {
        for (size_t i = 0; i < N; ++i) {
            const cv::Rect& r = st->rects[t * N + i];
            cv::cuda::GpuMat cropped = st->cropped[i](cv::Rect(0, 0, r.width, r.height));
            beforeKernel();
            st->rgb8(r).convertTo(cropped, CV_32F, st->stream);
            beforeKernel();
            cv::cuda::resize(cropped, st->resized[i], cv::Size(RES_W, RES_H), 0., 0., cv::INTER_LINEAR, st->stream);
            cv::cuda::GpuMat plane = st->tensor.rowRange(static_cast<int>(i) * OUT_H, static_cast<int>(i + 1) * OUT_H);
            beforeKernel();
            cv::cuda::copyMakeBorder(st->resized[i], plane, BORDER_TOP, BORDER_BOTTOM, 0, 0, cv::BORDER_CONSTANT,
                                     cv::Scalar(0., 0., 0.), st->stream);
        }
    });
    impl.stages.emplace_back("3_multiply_subtract_divide", [st](size_t) {
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
        os << "OpenCV: " << cv::getVersionString() << " (opencv_contrib "
           << bench::buildInfoValue(opencvBuild, "Version control (extra):") << ", CUDA_ARCH_BIN "
           << bench::buildInfoValue(opencvBuild, "NVIDIA GPU arch:") << ", C++ compiler "
           << bench::buildInfoValue(opencvBuild, "C++ Compiler:")
           << "; full configuration in opencv_build_information.txt)\n"
           << "OpenCV CUDA pipeline: OpenCV's NV12->RGB of the whole frame into an 8 bit RGB frame "
              "(cudacodec::createNVSurfaceToColorConverter(ColorSpaceStandard::BT601, videoFullRangeFlag false) and "
              "convert(SF_NV12, ColorFormat::RGB, BitDepth::EIGHT, interleaved): one kernel, on the raw NV12 frame "
              "without a decoder session; it converts whole surfaces only, so the whole frame is converted; it "
              "truncates to 8 bits, and the pipeline is checked against the truncated 8 bit reference), then per "
              "region cv::cuda convertTo (float3 crop, into a buffer of the "
              "largest region size through a view of the region's size), resize (float3 resized crop) and "
              "copyMakeBorder (into its output plane), then multiply, subtract and divide on the whole float3 output: "
              "3N + 4 kernels. Per iteration, inside the timed region: the GpuMat views of every region (of the RGB "
              "frame and of its crop buffer) and the per-region calls with them.\n";
    }

    static std::string diagnostic(const Context& ctx) { return opencv::conversionDiagnostic(ctx); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::OpenCVCudaVsFkl>(argc, argv);
}
