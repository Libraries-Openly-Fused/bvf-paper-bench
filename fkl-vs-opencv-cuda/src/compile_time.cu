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

// The compile-time units of FKL vs OpenCV CUDA (scripts/compile_time.sh, SET=opencv-cuda; common/compile_time_unit.h),
// all with the headers of FKL and OpenCV CUDA (cudacodec for the NV12 conversion):
//   -DIMPL_HEADERS  the headers, no pipeline
//   -DIMPL_OPENCV   the OpenCV CUDA calls of makeOpenCV in benchmark_opencv_cuda.cu
//   -DIMPL_BVF      FKL BVF, one executeOperations call (Listing 2)

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudawarping.hpp>

#include "compile_time_unit.h"

#if defined(IMPL_OPENCV)
// ---------------------------------------------------------------- makeOpenCV
void runPipeline() {
    using namespace ct;
    const cv::cuda::GpuMat nv12(FRAME_H * 3 / 2, FRAME_W, CV_8UC1);
    const auto converter =
        cv::cudacodec::createNVSurfaceToColorConverter(cv::cudacodec::ColorSpaceStandard::BT601, false);
    cv::cuda::Stream stream;
    cv::cuda::GpuMat rgb8(FRAME_H, FRAME_W, CV_8UC3);
    std::vector<cv::Rect> rects;
    std::vector<cv::cuda::GpuMat> cropped, resized;
    cv::cuda::GpuMat tensor;
    for (const fk::Rect& r : someCrops()) {
        rects.emplace_back(static_cast<int>(r.x), static_cast<int>(r.y), r.width, r.height);
        cropped.emplace_back(MAX_REGION_H, MAX_REGION_W, CV_32FC3);
        resized.emplace_back(RES_H, RES_W, CV_32FC3);
    }
    cv::cuda::createContinuous(static_cast<int>(N) * OUT_H, OUT_W, CV_32FC3, tensor);
    converter->convert(nv12, rgb8, cv::cudacodec::SurfaceFormat::SF_NV12, cv::cudacodec::ColorFormat::RGB,
                       cv::cudacodec::BitDepth::EIGHT, false, stream);
    for (size_t i = 0; i < N; ++i) {
        cv::cuda::GpuMat crop = cropped[i](cv::Rect(0, 0, rects[i].width, rects[i].height));
        rgb8(rects[i]).convertTo(crop, CV_32F, stream);
        cv::cuda::resize(crop, resized[i], cv::Size(RES_W, RES_H), 0., 0., cv::INTER_LINEAR, stream);
        cv::cuda::GpuMat plane = tensor.rowRange(static_cast<int>(i) * OUT_H, static_cast<int>(i + 1) * OUT_H);
        cv::cuda::copyMakeBorder(resized[i], plane, BORDER_TOP, BORDER_BOTTOM, 0, 0, cv::BORDER_CONSTANT,
                                 cv::Scalar(0., 0., 0.), stream);
    }
    cv::cuda::multiply(tensor, cv::Scalar(MUL[0], MUL[1], MUL[2]), tensor, 1., -1, stream);
    cv::cuda::subtract(tensor, cv::Scalar(SUB[0], SUB[1], SUB[2]), tensor, cv::noArray(), -1, stream);
    cv::cuda::divide(tensor, cv::Scalar(DIV[0], DIV[1], DIV[2]), tensor, 1., -1, stream);
}
#elif !defined(IMPL_HEADERS) && !defined(IMPL_BVF)
#error "Define one of IMPL_HEADERS, IMPL_OPENCV, IMPL_BVF"
#endif
