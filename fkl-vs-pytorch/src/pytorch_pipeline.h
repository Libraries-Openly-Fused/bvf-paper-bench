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

// The Figure 1 pipeline in eager PyTorch (LibTorch's C++ API, no Python, no TorchScript, no torch.compile, no CUDA
// graphs), the same code on the CPU and on the GPU (CUDA, or ROCm, where the device is still torch::kCUDA):
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// Per iteration, with the regions of the iteration (host integers, so nothing waits for the device):
//   - the border rows of the output tensor are set to 0 (two zero_ on strided views of all the planes);
//   - per region: views of the luma and chroma planes of the region (slices, no copy); the chroma converted to float
//     and multiplied by FKL's chroma coefficients, with every offset of the conversion folded into the bias, once per
//     chroma sample (addmm); the luma converted to float and added with its coefficient, the chroma replicated to its
//     2x2 luma pixels by broadcasting (add): float RGB, no clamping, FKL's BT.601 limited range formula; bilinear
//     resize to 64x32 (torch::nn::functional::interpolate, align_corners=false); copy into the rows of its output
//     plane between the borders;
//   - multiply, subtract and divide in place on the whole N x 64 x 64 x 3 output tensor.
// The conversion matrix and offsets are FKL's (common/bench_pipeline.h's Conversion, filled by FKL's side):
// (R, G, B) = M (Y - 16, U - 128, V - 128) = M_Y Y + (M_U U + M_V V + bias), with bias = -16 M_Y - 128 (M_U + M_V).
//
// PyTorch's interpolate samples at pixel centers, src = (dst + 0.5) * (in / out) - 0.5 clamped at 0, while FKL samples
// at src = dst * (in / out): the output is checked against a reference with PyTorch's convention
// (common/bench_reference.h, Sampling::PixelCenters), and its difference to FKL's output is reported.

#ifndef BVF_BENCH_PYTORCH_PIPELINE_H
#define BVF_BENCH_PYTORCH_PIPELINE_H

#include <torch/torch.h>

#include "bench_geometry.h"
#include "bench_pipeline.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bench::pytorch {

using namespace geometry;
using namespace pipeline;

class Pipeline {
public:
    // n regions per iteration; the NV12 frame (FRAME_H * 3 / 2 rows of FRAME_W bytes, pitch bytes apart) is copied to
    // the device
    Pipeline(const uint8_t* nv12, size_t pitch, torch::Device device, size_t n, const Conversion& k)
        : m_n(n), m_device(device) {
        const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor frame = torch::empty({FRAME_H * 3 / 2, FRAME_W}, torch::TensorOptions().dtype(torch::kUInt8));
        for (int row = 0; row < FRAME_H * 3 / 2; ++row) {
            std::memcpy(frame[row].data_ptr<uint8_t>(), nv12 + static_cast<size_t>(row) * pitch, FRAME_W);
        }
        frame = frame.to(device);
        m_luma = frame.narrow(0, 0, FRAME_H);
        m_chroma = frame.narrow(0, FRAME_H, FRAME_H / 2).view({FRAME_H / 2, FRAME_W / 2, 2});
        // The luma coefficient is the same for R, G and B in every YCbCr -> RGB matrix
        m_lumaScale = k.matrix[0][0];
        if (k.matrix[1][0] != m_lumaScale || k.matrix[2][0] != m_lumaScale) {
            throw std::runtime_error("The luma coefficients of the conversion differ between channels");
        }
        float chromaMatrix[2][3], bias[3];
        for (int c = 0; c < 3; ++c) {
            chromaMatrix[0][c] = k.matrix[c][1];
            chromaMatrix[1][c] = k.matrix[c][2];
            bias[c] = -k.lumaOffset * k.matrix[c][0] - k.chromaOffset * (k.matrix[c][1] + k.matrix[c][2]);
        }
        m_chromaMatrix = torch::from_blob(chromaMatrix, {2, 3}, f32).to(device);
        m_bias = torch::from_blob(bias, {3}, f32).to(device);
        m_mul = torch::from_blob(const_cast<float*>(MUL), {3}, f32).to(device);
        m_sub = torch::from_blob(const_cast<float*>(SUB), {3}, f32).to(device);
        m_div = torch::from_blob(const_cast<float*>(DIV), {3}, f32).to(device);
        m_out = torch::empty({static_cast<int64_t>(n), OUT_H, OUT_W, 3}, f32.device(device));
        if (device.is_cpu()) {
            // from_blob does not copy: the CPU tensors must own their values
            m_chromaMatrix = m_chromaMatrix.clone();
            m_bias = m_bias.clone();
            m_mul = m_mul.clone();
            m_sub = m_sub.clone();
            m_div = m_div.clone();
        }
    }

    // The pipeline on the first n regions of an iteration, enqueued on the device's current stream
    void run(const Region* regions) {
        namespace F = torch::nn::functional;
        c10::InferenceMode inference;
        const auto resizeOptions = F::InterpolateFuncOptions()
                                       .size(std::vector<int64_t>{RES_H, RES_W})
                                       .mode(torch::kBilinear)
                                       .align_corners(false);
        m_out.narrow(1, 0, BORDER_TOP).zero_();
        m_out.narrow(1, BORDER_TOP + RES_H, BORDER_BOTTOM).zero_();
        for (size_t i = 0; i < m_n; ++i) {
            const Region& r = regions[i];
            const int64_t h = r.height, w = r.width;
            const torch::Tensor luma = m_luma.slice(0, r.y, r.y + h).slice(1, r.x, r.x + w);
            const torch::Tensor chroma = m_chroma.slice(0, r.y / 2, (r.y + h) / 2).slice(1, r.x / 2, (r.x + w) / 2);
            // M_U U + M_V V + bias per chroma sample: (h / 2) * (w / 2) x 3
            const torch::Tensor chromaTerms =
                torch::addmm(m_bias, chroma.to(torch::kFloat32).reshape({-1, 2}), m_chromaMatrix);
            // + M_Y Y, each chroma sample broadcast to its 2x2 luma pixels: h x w x 3 float RGB
            const torch::Tensor rgb = torch::add(chromaTerms.view({h / 2, 1, w / 2, 1, 3}),
                                                 luma.to(torch::kFloat32).reshape({h / 2, 2, w / 2, 2, 1}),
                                                 m_lumaScale)
                                          .view({h, w, 3});
            // 1 x 3 x h x w (a view of the h x w x 3 values) -> 1 x 3 x 32 x 64 (NCHW), transposed by the copy
            const torch::Tensor resized = F::interpolate(rgb.permute({2, 0, 1}).unsqueeze(0), resizeOptions);
            m_out.select(0, static_cast<int64_t>(i)).narrow(0, BORDER_TOP, RES_H).copy_(resized[0].permute({1, 2, 0}));
        }
        m_out.mul_(m_mul).sub_(m_sub).div_(m_div);
    }

    // The N x OUT_H x OUT_W x 3 output of the last run, in host memory (waits for it)
    std::vector<float> download() const {
        const torch::Tensor host = m_out.to(torch::kCPU).contiguous();
        const float* data = host.data_ptr<float>();
        return std::vector<float>(data, data + host.numel());
    }

    // Poisons the output before a validated run
    void fillNaN() { m_out.fill_(std::numeric_limits<float>::quiet_NaN()); }

private:
    size_t m_n;
    torch::Device m_device;
    torch::Tensor m_luma, m_chroma, m_chromaMatrix, m_bias, m_mul, m_sub, m_div, m_out;
    float m_lumaScale{0.f};
};

} // namespace bench::pytorch

#endif // BVF_BENCH_PYTORCH_PIPELINE_H
