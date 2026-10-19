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

// FKL's side of the CPU benchmarks (common/bench_cpu.h): the frame, FKL's pipelines on its CPU backend (one thread) and
// CPU_MULTITHREAD backend (FKL's thread pool) for every batch size, the plain C++ reference and FKL's conversion
// coefficients. This is the only translation unit of a CPU benchmark that includes FKL, and add_cpu_bench_target
// (common/bench_cpu.cmake) compiles it first, so that FKL's code is the same, and placed the same way, in every CPU
// benchmark, whatever the compared library adds.

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/execution_model/cpu_multithread.h>

#include "bench_cpu.h"
#include "bench_reference.h"

#include <array>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace bench::cpu {

namespace {

using reference::NV12Image;
using reference::YUVToRGB;

// The frame, generated once
const NV12Image& frameImage() {
    static const NV12Image frame = []() {
        NV12Image image(FRAME_W, FRAME_H, fk::MemType::Host);
        auto data = image.getData();
        generateFrame([&](int x, int y, uint8_t value) { data.at(x, y) = value; });
        return image;
    }();
    return frame;
}

template <size_t N, fk::ParArch PA>
class FklCpuPipeline final : public FklCpu {
public:
    explicit FklCpuPipeline(const Workload& workload) : m_crops(workload.regions->iterations()) {
        for (size_t t = 0; t < m_crops.size(); ++t) {
            for (size_t i = 0; i < N; ++i) {
                const Region& r = workload.regions->at(t, i);
                m_crops[t][i] = fk::Rect(r.x, r.y, r.width, r.height);
            }
        }
    }

    void run(size_t t) override {
        fk::executeOperations<fk::TransformDPP<PA>>(m_stream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(m_frame),
            YUVToRGB::build(),
            fk::Crop<>::build(m_crops[t]),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, fk::float3{0.f, 0.f, 0.f}),
            fk::Mul<fk::float3>::build(fk::float3{MUL[0], MUL[1], MUL[2]}),
            fk::Sub<fk::float3>::build(fk::float3{SUB[0], SUB[1], SUB[2]}),
            fk::Div<fk::float3>::build(fk::float3{DIV[0], DIV[1], DIV[2]}),
            fk::TensorWrite<fk::float3>::build(m_out.ptr()));
        m_stream.sync();
    }

    std::vector<float> output() const override {
        const float* data = reinterpret_cast<const float*>(m_out.ptr().data);
        return std::vector<float>(data, data + N * OUT_FLOATS);
    }

private:
    const NV12Image& m_frame{frameImage()};
    // The rectangles of every iteration, converted before the loops: what FKL's Crop takes
    std::vector<std::array<fk::Rect, N>> m_crops;
    fk::Tensor<fk::float3> m_out{OUT_W, OUT_H, N, 1, fk::MemType::Host};
    fk::Stream_<PA> m_stream;
};

template <fk::ParArch PA>
std::unique_ptr<FklCpu> makePipeline(size_t n, const Workload& w) {
    std::unique_ptr<FklCpu> fkl;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((BATCHES[Is] == n ? void(fkl = std::make_unique<FklCpuPipeline<BATCHES[Is], PA>>(w)) : void()), ...);
    }(std::make_index_sequence<BATCHES.size()>{});
    if (!fkl) {
        throw std::runtime_error("Batch size " + std::to_string(n) + " is not one of the benchmarked ones");
    }
    return fkl;
}

} // namespace

const uint8_t* fklFrame(size_t* pitch) {
    const auto raw = frameImage().getData().ptr();
    *pitch = raw.dims.pitch;
    return raw.data;
}

std::unique_ptr<FklCpu> makeFklCpu(size_t n, bool multithread, const Workload& w) {
    return multithread ? makePipeline<fk::ParArch::CPU_MULTITHREAD>(n, w) : makePipeline<fk::ParArch::CPU>(n, w);
}

std::vector<float> referenceOutput(const Workload& w, size_t n, size_t t, Conventions conventions) {
    return reference::hostReference(frameImage(), *w.regions, n, t, conventions);
}

Conversion fklConversion() {
    return reference::conversion();
}

int poolThreads() {
    return static_cast<int>(fk::cpuThreadPool().size());
}

void runOnFklPool(size_t chunks, const std::function<void(size_t)>& function) {
    static fk::Stream_<fk::ParArch::CPU_MULTITHREAD> stream;
    const std::function<void(size_t)>* f = &function;
    stream.enqueue([chunks, f](const fk::cpu_multithread::Completion& done) {
        fk::cpu_multithread::enqueueChunks(chunks, [f](const std::size_t chunk) { (*f)(chunk); },
                                           [done](const std::exception_ptr error) { done(error); });
    });
    stream.sync();
}

} // namespace bench::cpu
