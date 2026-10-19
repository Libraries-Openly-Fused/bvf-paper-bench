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

// Independent host reference of the Figure 1 pipeline: the chain in plain C++ on the regions of one iteration, with
// FKL's NV12 -> RGB conversion (BT.601 limited range, chroma replicated) and the conventions of common/bench_pipeline.h
// (FKL's, PyTorch's pixel centers or OpenCV's), the black rows on top and bottom, then (x * MUL - SUB) / DIV, into an
// N x OUT_H x OUT_W x 3 float tensor. It includes FKL: in the CPU benchmarks only common/bench_fkl_cpu.cpp includes it
// (FKL's headers define variable template specializations that are not inline, so one program can include them in one
// translation unit only); fkl-vs-pytorch's GPU program includes it in src/fkl_gpu.cu.

#ifndef BVF_BENCH_REFERENCE_H
#define BVF_BENCH_REFERENCE_H

#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/image.h>

#include "bench_geometry.h"
#include "bench_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bench::reference {

using namespace geometry;
using namespace pipeline;

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
// FKL's NV12 -> RGB conversion: BT.601, limited range, as in every benchmark of the repository
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Limited, fk::ColorPrimitives::bt601>;

// FKL's conversion as a matrix and offsets
inline Conversion conversion() {
    constexpr fk::M3x3Float m = fk::ccMatrix<fk::ColorRange::Limited, fk::ColorPrimitives::bt601,
                                             fk::ColorConversionDir::YCbCr2RGB, fk::ColorDepth::p8bit>;
    constexpr auto offsets = fk::subCoefficients<fk::ColorDepth::p8bit>;
    return Conversion{{{m.x.x, m.x.y, m.x.z}, {m.y.x, m.y.y, m.y.z}, {m.z.x, m.z.y, m.z.z}}, offsets.luma,
                      offsets.chroma};
}

// The first n regions of iteration t of a host NV12 frame
inline std::vector<float> hostReference(const NV12Image& frame, const RegionSets& regions, size_t n, size_t t,
                                        Conventions conventions) {
    const bool centers = conventions != Conventions::Fkl;
    const bool openCV = conventions == Conventions::OpenCV;
    std::vector<float> out(n * OUT_FLOATS);
    for (size_t i = 0; i < n; ++i) {
        const Region& r = regions.at(t, i);
        const int cw = r.width, ch = r.height;
        const float sx = static_cast<float>(cw) / RES_W;
        const float sy = static_cast<float>(ch) / RES_H;
        std::vector<float> cropPixels(static_cast<size_t>(cw) * ch * 3);
        const auto crop = [&](int y, int x, int c) -> float& {
            return cropPixels[(static_cast<size_t>(y) * cw + x) * 3 + c];
        };
        for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
                fk::uchar3 yuv = frame.readAt(fk::Point{r.x + x, r.y + y, 0});
                if (openCV) {
                    // OpenCV's NV12 -> RGB clamps Y below 16 (limited range black) to 16 before converting
                    yuv.x = std::max<unsigned char>(yuv.x, 16);
                }
                const fk::float3 rgb = YUVToRGB::exec(yuv);
                const float c3[3] = {rgb.x, rgb.y, rgb.z};
                for (int c = 0; c < 3; ++c) {
                    crop(y, x, c) = openCV ? std::floor(std::clamp(c3[c], 0.f, 255.f) + 0.5f) : c3[c];
                }
            }
        }
        for (int y = 0; y < OUT_H; ++y) {
            for (int x = 0; x < OUT_W; ++x) {
                const int ry = y - BORDER_TOP;
                float v[3] = {0.f, 0.f, 0.f};
                if (ry >= 0 && ry < RES_H) {
                    float fx = centers ? (x + 0.5f) * sx - 0.5f : x * sx;
                    float fy = centers ? (ry + 0.5f) * sy - 0.5f : ry * sy;
                    fx = std::clamp(fx, 0.f, static_cast<float>(cw - 1));
                    fy = std::clamp(fy, 0.f, static_cast<float>(ch - 1));
                    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
                    const int x1 = std::min(x0 + 1, cw - 1), y1 = std::min(y0 + 1, ch - 1);
                    const float ax = fx - x0, ay = fy - y0;
                    for (int c = 0; c < 3; ++c) {
                        v[c] = (crop(y0, x0, c) * (1.f - ax) + crop(y0, x1, c) * ax) * (1.f - ay) +
                               (crop(y1, x0, c) * (1.f - ax) + crop(y1, x1, c) * ax) * ay;
                    }
                }
                float* dst = &out[((i * OUT_H + y) * OUT_W + x) * 3];
                for (int c = 0; c < 3; ++c) {
                    dst[c] = ((v[c] * MUL[c]) - SUB[c]) / DIV[c];
                }
            }
        }
    }
    return out;
}

} // namespace bench::reference

#endif // BVF_BENCH_REFERENCE_H
