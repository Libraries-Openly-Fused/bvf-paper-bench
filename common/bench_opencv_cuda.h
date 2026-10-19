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

// OpenCV's own NV12 -> RGB conversion on the GPU, for the OpenCV CUDA pipelines (fkl-vs-opencv-cuda, fkl-neighborhood):
// cudacodec's NVSurfaceToColorConverter, which converts a raw NV12 surface without a decoder session (it needs
// OpenCV's CUDA build only, not the Video Codec SDK). It takes BT.601 in limited range (videoFullRangeFlag false) and
// writes 8 bit RGB or RGBA, interleaved: one kernel per call, which computes in float with the standard's coefficients
// (255 / 219 for luma, 255 / 224 for chroma) and truncates to 8 bits (opencv_contrib
// modules/cudacodec/src/cuda/ColorSpace.cu), so the pipelines are checked against the truncated 8 bit reference
// (Reference::Fkl8bitTruncated).
//
// It converts whole surfaces only: the surface is one 2-D image of FRAME_H * 3 / 2 rows (the luma rows, then the
// interleaved chroma rows, with one pitch), and the kernel reads the chroma of luma row y nHeight - y / 2 rows below
// it, nHeight the surface's luma height. A crop's chroma is not at that distance from the crop's luma (it is at frame
// row FRAME_H + y / 2), so a crop cannot be converted on its own: the OpenCV pipelines convert the whole frame.

#ifndef BVF_BENCH_OPENCV_CUDA_H
#define BVF_BENCH_OPENCV_CUDA_H

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudacodec.hpp>

#include "bench_gpu.h"

#include <string>
#include <vector>

namespace bench::opencv {

class Nv12Converter {
public:
    Nv12Converter()
        : m_converter(
              cv::cudacodec::createNVSurfaceToColorConverter(cv::cudacodec::ColorSpaceStandard::BT601, false)) {}

    // The whole NV12 frame of ctx into dst (FRAME_H x FRAME_W, CV_8UC3 for RGB or CV_8UC4 for RGBA, allocated before
    // the loops so that the converter writes into it), on stream: one kernel
    void convert(const Context& ctx, cv::cuda::GpuMat& dst, cv::cudacodec::ColorFormat format,
                 cv::cuda::Stream& stream) const {
        beforeKernel();
        m_converter->convert(surface(ctx), dst, cv::cudacodec::SurfaceFormat::SF_NV12, format,
                             cv::cudacodec::BitDepth::EIGHT, false, stream);
    }

    // The device frame of ctx as the converter takes it, without a copy (initContext puts the chroma right after the
    // luma rows, with the same pitch)
    static cv::cuda::GpuMat surface(const Context& ctx) {
        return cv::cuda::GpuMat(FRAME_H * 3 / 2, FRAME_W, CV_8UC1, const_cast<Npp8u*>(ctx.y),
                                static_cast<size_t>(ctx.pitch));
    }

private:
    cv::Ptr<cv::cudacodec::NVSurfaceToColorConverter> m_converter;
};

// OpenCV's 8 bit NV12->RGB of the whole frame against FKL's float conversion (rounded and truncated)
inline std::string conversionDiagnostic(const Context& ctx) {
    Nv12Converter converter;
    cv::cuda::Stream stream = cv::cuda::StreamAccessor::wrapStream(ctx.stream);
    cv::cuda::GpuMat rgb(FRAME_H, FRAME_W, CV_8UC3);
    converter.convert(ctx, rgb, cv::cudacodec::ColorFormat::RGB, stream);
    std::vector<Npp8u> host(static_cast<size_t>(FRAME_W) * 3 * FRAME_H);
    gpuErrchk(cudaMemcpy2DAsync(host.data(), FRAME_W * 3, rgb.data, rgb.step, FRAME_W * 3, FRAME_H,
                                cudaMemcpyDeviceToHost, ctx.stream));
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    return bench::conversionDiagnostic(ctx, host, "OpenCV NV12->RGB (cudacodec::NVSurfaceToColorConverter, BT601, "
                                                  "limited range, 8 bit)");
}

} // namespace bench::opencv

#endif // BVF_BENCH_OPENCV_CUDA_H
