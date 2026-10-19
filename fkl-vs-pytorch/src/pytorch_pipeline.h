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
//   - per region: views of the luma and chroma planes of the region (slices, no copy); the chroma multiplied by FKL's
//     chroma coefficients, with every offset of the conversion folded into the bias, once per chroma sample (addmm on
//     the chroma converted to float, or two addcmul on the 8-bit U and V views: ChromaTerms); the 8-bit luma
//     added with its coefficient, the chroma replicated to its 2x2 luma pixels by broadcasting (add): float RGB, no
//     clamping, FKL's BT.601 limited range formula; bilinear
//     resize to 64x32 (torch::nn::functional::interpolate, align_corners=false); copy into the rows of its output
//     plane between the borders;
//   - multiply, subtract and divide in place on the whole N x 64 x 64 x 3 output tensor.
// Or, with the batched variants (ResizeInput::Batched), without a host loop over the regions: the region parameters
// uploaded in one copy, the 48 x 24 window at every region's origin (the largest region size; the geometry draws every
// position for that box) gathered with one take per plane, converted in one pass, and resized with one
// grid_sample whose grid holds each region's pixel-center coordinates, clamped to the region, so that the window's
// pixels outside the region get no weight: interpolate's sampling, to float rounding.
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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench::pytorch {

using namespace geometry;
using namespace pipeline;

// The two formulations of the chroma terms, M_U U + M_V V + bias per chroma sample: addmm (the chroma converted to
// float, then a 2 x 3 matrix product with the bias, cuBLASLt on the GPU and MKL on the CPU) and addcmul (two addcmul on
// the 8-bit U and V views). Which is faster can depend on the device, the thread count and the batch size, so the
// programs time PyTorch's complete loop with each before every comparison and compare FKL with the faster one, as FKL
// vs NPPDx chooses NPPDx's configuration.
enum class ChromaTerms { Addmm, Addcmul };
// The view interpolate gets of the h x w x 3 RGB crop: permute then unsqueeze, which PyTorch's layout check
// (suggest_memory_format) treats as NCHW, so interpolate writes NCHW and the copy into the output plane transposes it;
// or unsqueeze then permute, which it treats as channels-last, so interpolate writes channels-last and the copy is a
// straight one (a device to device memcpy on the GPU). Or Batched: no per-region calls at all; every region's
// MAX_REGION_W x MAX_REGION_H window is gathered into one N x 24 x 48 batch, converted, and resized with one
// grid_sample (Pipeline::runBatched). The sweep chooses between them as between the chroma terms.
enum class ResizeInput { Nchw, ChannelsLast, Batched };
// One way of writing PyTorch's pipeline: the sweep times every one before each comparison
struct Variant {
    ChromaTerms chromaTerms;
    ResizeInput resizeInput;
};
inline constexpr std::array<Variant, 6> VARIANTS{Variant{ChromaTerms::Addmm, ResizeInput::Nchw},
                                                 Variant{ChromaTerms::Addmm, ResizeInput::ChannelsLast},
                                                 Variant{ChromaTerms::Addcmul, ResizeInput::Nchw},
                                                 Variant{ChromaTerms::Addcmul, ResizeInput::ChannelsLast},
                                                 Variant{ChromaTerms::Addmm, ResizeInput::Batched},
                                                 Variant{ChromaTerms::Addcmul, ResizeInput::Batched}};
inline const char* chromaTermsName(ChromaTerms f) {
    return f == ChromaTerms::Addmm ? "addmm" : "addcmul";
}
inline const char* resizeInputName(ResizeInput r) {
    return r == ResizeInput::Nchw ? "nchw" : r == ResizeInput::ChannelsLast ? "channels_last" : "batched";
}
inline std::string variantName(Variant v) {
    return std::string(chromaTermsName(v.chromaTerms)) + " " + resizeInputName(v.resizeInput);
}

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
        m_chromaU = m_chromaMatrix.select(0, 0);
        m_chromaV = m_chromaMatrix.select(0, 1);
        // The offsets of a window's values from its first one, in the luma plane (bytes) and in the chroma plane (UV
        // pairs, int16): row * pitch + column
        const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
        m_lumaWindow = (torch::arange(MAX_REGION_H, i64).view({MAX_REGION_H, 1}) * FRAME_W +
                        torch::arange(MAX_REGION_W, i64)).contiguous();
        m_chromaWindow = (torch::arange(MAX_REGION_H / 2, i64).view({MAX_REGION_H / 2, 1}) * (FRAME_W / 2) +
                          torch::arange(MAX_REGION_W / 2, i64)).contiguous();
        m_chroma16 = frame.narrow(0, FRAME_H, FRAME_H / 2).view(torch::kInt16);
        if (device.is_cpu()) {
            m_cpuParameters = torch::empty({static_cast<int64_t>(n * PARAMETER_BYTES)},
                                           torch::TensorOptions().dtype(torch::kUInt8));
        }
    }

    // The variant of the next runs
    void setVariant(Variant v) { m_variant = v; }

    // The pipeline on the first n regions of an iteration, enqueued on the device's current stream
    void run(const Region* regions) {
        if (m_variant.resizeInput == ResizeInput::Batched) {
            runBatched(regions);
            return;
        }
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
            // M_U U + M_V V + bias per chroma sample, (h / 2) x (w / 2) x 3 (ChromaTerms: addcmul is 2 kernels on the
            // GPU, addmm a conversion and cuBLASLt's split-K GEMM, 3 kernels)
            const torch::Tensor chromaTerms =
                m_variant.chromaTerms == ChromaTerms::Addcmul
                    ? torch::addcmul(m_bias, chroma.narrow(2, 0, 1), m_chromaU).addcmul_(chroma.narrow(2, 1, 1), m_chromaV)
                    : torch::addmm(m_bias, chroma.to(torch::kFloat32).reshape({-1, 2}), m_chromaMatrix);
            // + M_Y Y, each chroma sample broadcast to its 2x2 luma pixels: h x w x 3 float RGB. The 8-bit luma view goes
            // in as it is: the add promotes it to float (on the GPU in its own kernel; on the CPU TensorIterator makes a
            // float copy of it first)
            const torch::Tensor rgb = torch::add(chromaTerms.view({h / 2, 1, w / 2, 1, 3}),
                                                 luma.reshape({h / 2, 2, w / 2, 2, 1}), m_lumaScale)
                                          .view({h, w, 3});
            // 1 x 3 x h x w, a view of the h x w x 3 values (ResizeInput) -> 1 x 3 x 32 x 64, NCHW (transposed by the
            // copy) or channels-last (copied straight)
            const torch::Tensor input = m_variant.resizeInput == ResizeInput::ChannelsLast
                                            ? rgb.unsqueeze(0).permute({0, 3, 1, 2})
                                            : rgb.permute({2, 0, 1}).unsqueeze(0);
            const torch::Tensor resized = F::interpolate(input, resizeOptions);
            m_out.select(0, static_cast<int64_t>(i)).narrow(0, BORDER_TOP, RES_H).copy_(resized[0].permute({1, 2, 0}));
        }
        m_out.mul_(m_mul).sub_(m_sub).div_(m_div);
    }

    // The batched variants: no host loop over the regions in device work, a fixed number of operator calls whatever N
    void runBatched(const Region* regions) {
        namespace F = torch::nn::functional;
        c10::InferenceMode inference;
        const int64_t n = static_cast<int64_t>(m_n);
        constexpr int64_t WH = MAX_REGION_H, WW = MAX_REGION_W;
        m_out.narrow(1, 0, BORDER_TOP).zero_();
        m_out.narrow(1, BORDER_TOP + RES_H, BORDER_BOTTOM).zero_();
        // The parameters of the iteration's regions, written on the host and uploaded in one copy: per region the
        // index of its window's first value in the luma plane and in the chroma plane (int64), then its grid
        // coordinates, RES_W x and RES_H y, normalized to the window (float). On the GPU the host buffer is pinned
        // memory from PyTorch's caching host allocator, which keeps it until the copy is done.
        const torch::Tensor host =
            m_device.is_cpu()
                ? m_cpuParameters
                : torch::empty({n * PARAMETER_BYTES}, torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
        int64_t* starts = host.narrow(0, 0, n * START_BYTES).view(torch::kInt64).data_ptr<int64_t>();
        float* coordinates =
            host.narrow(0, n * START_BYTES, n * COORDINATE_BYTES).view(torch::kFloat32).data_ptr<float>();
        for (int64_t i = 0; i < n; ++i) {
            const Region& r = regions[i];
            if (r.width > WW || r.height > WH || r.x + WW > FRAME_W || r.y + WH > FRAME_H || r.x % 2 || r.y % 2) {
                throw std::runtime_error("The batched variants need every region's 48 x 24 window inside the frame");
            }
            starts[2 * i + 0] = static_cast<int64_t>(r.y) * FRAME_W + r.x;
            starts[2 * i + 1] = static_cast<int64_t>(r.y / 2) * (FRAME_W / 2) + r.x / 2;
            // interpolate's source coordinate, (dst + 0.5) * (in / out) - 0.5 clamped at 0, also clamped at the
            // region's last pixel (interpolate's second tap there is that pixel itself), then normalized for
            // grid_sample with align_corners=false: (2 src + 1) / window size - 1
            float* gx = coordinates + i * (RES_W + RES_H);
            float* gy = gx + RES_W;
            const float sx = static_cast<float>(r.width) / RES_W, sy = static_cast<float>(r.height) / RES_H;
            for (int j = 0; j < RES_W; ++j) {
                const float src = std::min(std::max((j + 0.5f) * sx - 0.5f, 0.f), static_cast<float>(r.width - 1));
                gx[j] = (2.f * src + 1.f) / WW - 1.f;
            }
            for (int j = 0; j < RES_H; ++j) {
                const float src = std::min(std::max((j + 0.5f) * sy - 0.5f, 0.f), static_cast<float>(r.height - 1));
                gy[j] = (2.f * src + 1.f) / WH - 1.f;
            }
        }
        const torch::Tensor parameters = m_device.is_cpu() ? host : host.to(m_device, /*non_blocking=*/true);
        const torch::Tensor first = parameters.narrow(0, 0, n * START_BYTES).view(torch::kInt64).view({n, 2, 1, 1});
        const torch::Tensor grid1d =
            parameters.narrow(0, n * START_BYTES, n * COORDINATE_BYTES).view(torch::kFloat32).view({n, RES_W + RES_H});
        // n x 24 x 48 luma and n x 12 x 24 x 2 chroma windows, 8-bit, each plane in one take (the values at the
        // windows' indices: first value + the window's offsets)
        const torch::Tensor luma = torch::take(m_luma, first.select(1, 0) + m_lumaWindow);
        const torch::Tensor chroma = torch::take(m_chroma16, first.select(1, 1) + m_chromaWindow)
                                         .view(torch::kUInt8)
                                         .view({n, WH / 2, WW / 2, 2});
        const torch::Tensor chromaTerms =
            m_variant.chromaTerms == ChromaTerms::Addcmul
                ? torch::addcmul(m_bias, chroma.narrow(3, 0, 1), m_chromaU).addcmul_(chroma.narrow(3, 1, 1), m_chromaV)
                : torch::addmm(m_bias, chroma.to(torch::kFloat32).reshape({-1, 2}), m_chromaMatrix);
        const torch::Tensor rgb = torch::add(chromaTerms.view({n, WH / 2, 1, WW / 2, 1, 3}),
                                             luma.view({n, WH / 2, 2, WW / 2, 2, 1}), m_lumaScale)
                                      .view({n, WH, WW, 3});
        // n x 32 x 64 x 2: x from the region's column coordinates, y from its row coordinates
        const torch::Tensor grid = torch::stack({grid1d.narrow(1, 0, RES_W).unsqueeze(1).expand({n, RES_H, RES_W}),
                                                 grid1d.narrow(1, RES_W, RES_H).unsqueeze(2).expand({n, RES_H, RES_W})},
                                                3);
        // n x 3 x 32 x 64 (NCHW), copied into the rows of the output planes between the borders
        const torch::Tensor resized = F::grid_sample(
            rgb.permute({0, 3, 1, 2}), grid,
            F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(false));
        m_out.narrow(1, BORDER_TOP, RES_H).copy_(resized.permute({0, 2, 3, 1}));
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
    // The bytes of one region's parameters in the batched variants: 2 int64 window starts, RES_W + RES_H floats
    static constexpr int64_t START_BYTES = 2 * sizeof(int64_t);
    static constexpr int64_t COORDINATE_BYTES = (RES_W + RES_H) * sizeof(float);
    static constexpr int64_t PARAMETER_BYTES = START_BYTES + COORDINATE_BYTES;

    size_t m_n;
    torch::Device m_device;
    torch::Tensor m_luma, m_chroma, m_chromaMatrix, m_bias, m_mul, m_sub, m_div, m_out;
    torch::Tensor m_chromaU, m_chromaV;
    torch::Tensor m_chroma16, m_lumaWindow, m_chromaWindow, m_cpuParameters;
    Variant m_variant{VARIANTS[0]};
    float m_lumaScale{0.f};
};

} // namespace bench::pytorch

#endif // BVF_BENCH_PYTORCH_PIPELINE_H
