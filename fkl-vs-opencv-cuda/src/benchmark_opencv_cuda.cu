// FKL vs OpenCV CUDA, Figure 1 pipeline of the Backwards Vertical Fusion paper:
// ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// with the frame, sizes, batch sizes and crop positions of common/bench_geometry.h and NV12 -> RGB in BT.601 limited
// range. The harness (validation, timing, modes, output files) is common/bench_gpu.h.
//
//   opencv_cuda   OpenCV's NV12->RGB of the whole frame (cudacodec's NVSurfaceToColorConverter,
//                 common/bench_opencv_cuda.h), then per crop cv::cuda convertTo, resize and copyMakeBorder, then
//                 multiply, subtract and divide on the whole output tensor: 3N + 4 kernels, with an 8 bit RGB frame,
//                 float3 crops, float3 resized crops and the float3 output in memory
//   opencv_cuda_crop_nv12
//                 the same pipeline with the NV12->RGB of each crop's columns only: per crop the converter on the
//                 full-height strip of the frame's columns [x, x + w) (FRAME_H x w: the converter cannot restrict the
//                 rows), written at those columns of the 8 bit RGB frame, which the rest reads as before: 4N + 3
//                 kernels
//   fkl_bvf       one FKL kernel for the whole chain (BVF)
// The two OpenCV forms each have their own loop and speedup in every mode; per batch size, the one with the lower
// latency median is OpenCV's, in every mode (the paper's rule for a library's variants).

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
using opencv::Nv12Form;

template <size_t N, Nv12Form F>
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
    Implementation impl{F == Nv12Form::Frame ? "opencv_cuda" : "opencv_cuda_crop_nv12", Reference::Fkl8bitTruncated};
    if constexpr (F == Nv12Form::Frame) {
        impl.stages.emplace_back("1_nv12_to_rgb8_full_frame", [&ctx, st](size_t) {
            st->converter.convert(ctx, st->rgb8, cv::cudacodec::ColorFormat::RGB, st->stream);
        });
    } else {
        // Written at each crop's columns of the 8 bit RGB frame, every row: the rest of the pipeline reads only the
        // crops' pixels
        impl.stages.emplace_back("1_nv12_to_rgb8_per_crop_column_strip", [&ctx, st](size_t t) {
            for (size_t i = 0; i < N; ++i) {
                const cv::Rect& r = st->rects[t * N + i];
                st->converter.convertStrip(ctx, st->rgb8, fk::Rect(r.x, r.y, r.width, r.height),
                                           cv::cudacodec::ColorFormat::RGB, st->stream);
            }
        });
    }
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
    impl.fillNaN = [st]() {
        gpuErrchk(cudaMemset(st->tensor.data, 0xFF, N * OUT_FLOATS * sizeof(float)));
        if constexpr (F == Nv12Form::CropStrip) {
            // The 8 bit frame too, which keeps the strips of earlier runs: a pixel the strips of the validated
            // iteration do not convert, if the pipeline read it, would be 0 instead of a converted value
            gpuErrchk(cudaMemset(st->rgb8.data, 0, st->rgb8.step * FRAME_H));
        }
    };
    return impl;
}

struct OpenCVCudaVsFkl {
    static constexpr const char* title = "FKL vs OpenCV CUDA (fkl-vs-opencv-cuda)";

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        impls.push_back(makeOpenCV<N, Nv12Form::Frame>(ctx));
        impls.push_back(makeOpenCV<N, Nv12Form::CropStrip>(ctx));
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
           << "OpenCV CUDA pipeline (opencv_cuda): OpenCV's NV12->RGB of the whole frame into an 8 bit RGB frame "
              "(cudacodec::createNVSurfaceToColorConverter(ColorSpaceStandard::BT601, videoFullRangeFlag false) and "
              "convert(SF_NV12, ColorFormat::RGB, BitDepth::EIGHT, interleaved): one kernel, on the raw NV12 frame "
              "without a decoder session; it truncates to 8 bits, and the pipeline is checked against the truncated "
              "8 bit reference), then per region cv::cuda convertTo (float3 crop, into a buffer of the "
              "largest region size through a view of the region's size), resize (float3 resized crop) and "
              "copyMakeBorder (into its output plane), then multiply, subtract and divide on the whole float3 output: "
              "3N + 4 kernels. Per iteration, inside the timed region: the GpuMat views of every region (of the RGB "
              "frame and of its crop buffer) and the per-region calls with them.\n"
           << "opencv_cuda_crop_nv12: the same pipeline with the conversion of each region's columns only. The "
              "converter cannot restrict the rows (its kernel reads the chroma of a luma row at a distance fixed by "
              "the surface's height), but it can the columns: per region, one convert of the full-height strip of "
              "the frame's columns [x, x + w) (a GpuMat view of FRAME_H x w at column x with the frame's pitch, whose "
              "chroma is where the kernel reads it, x being even) into the same columns of the 8 bit RGB frame, "
              "which the rest of the pipeline reads as opencv_cuda: 4N + 3 kernels. The strip of a region is "
              "FRAME_H x w pixels (w its width); the frame is written only at the columns of the iteration's regions "
              "(overlapping strips are converted twice), and only pixels of the regions are read; before every "
              "validated run, the 8 bit frame is filled with 0 as well as the output with NaN.\n"
           << "The two OpenCV forms each have their own loop and speedup in every mode; per batch size, the one "
              "with the lower latency median is OpenCV's, in every mode.\n";
    }

    static std::string diagnostic(const Context& ctx) { return opencv::conversionDiagnostic(ctx); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::OpenCVCudaVsFkl>(argc, argv);
}
