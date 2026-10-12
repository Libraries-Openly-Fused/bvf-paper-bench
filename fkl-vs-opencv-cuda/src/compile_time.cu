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
// all with the headers of FKL, OpenCV CUDA and NPP (for the NV12 conversion):
//   -DIMPL_HEADERS  the headers, no pipeline
//   -DIMPL_OPENCV   the OpenCV CUDA calls of makeOpenCV in benchmark_opencv_cuda.cu
//   -DIMPL_BVF      FKL BVF, one executeOperations call (Listing 2)

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <npp.h>

#include "bench_npp.h"

#include "compile_time_unit.h"

#if defined(IMPL_OPENCV)
// ---------------------------------------------------------------- makeOpenCV
void runPipeline() {
    using namespace ct;
    const Npp8u* y = nullptr;
    const Npp8u* uv = nullptr;
    const int pitch = 0;
    const NppStreamContext npp{};
    cv::cuda::Stream stream;
    cv::cuda::GpuMat rgb8(FRAME_H, FRAME_W, CV_8UC3);
    std::vector<cv::Rect> rects;
    std::vector<cv::cuda::GpuMat> cropped, resized;
    cv::cuda::GpuMat tensor;
    for (size_t i = 0; i < N; ++i) {
        rects.emplace_back(static_cast<int>(i), static_cast<int>(i), CROP_W, CROP_H);
        cropped.emplace_back(CROP_H, CROP_W, CV_32FC3);
        resized.emplace_back(RES_H, RES_W, CV_32FC3);
    }
    cv::cuda::createContinuous(static_cast<int>(N) * OUT_H, OUT_W, CV_32FC3, tensor);
    bench::npp::nv12ToRgb8Bt601Limited(y, uv, pitch, rgb8.data, static_cast<int>(rgb8.step),
                                       NppiSize{FRAME_W, FRAME_H}, npp);
    for (size_t i = 0; i < N; ++i) {
        rgb8(rects[i]).convertTo(cropped[i], CV_32F, stream);
        cv::cuda::resize(cropped[i], resized[i], cv::Size(RES_W, RES_H), 0., 0., cv::INTER_LINEAR, stream);
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
