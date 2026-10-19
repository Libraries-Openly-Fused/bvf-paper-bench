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

// FKL's side of the GPU program (src/fkl_gpu.h): compiled as CUDA by nvcc, or as HIP on ROCm (CMakeLists.txt sets the
// language of this file), with FKL's GPU backend of that platform. FKL BVF is the chain of common/bench_gpu.h's fkl_bvf,
// on the stream of PyTorch's pipeline.

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include "bench_reference.h"
#include "fkl_gpu.h"

#include <array>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace bench::pytorch {

namespace {

using namespace geometry;
using namespace pipeline;
using reference::NV12Image;
using reference::YUVToRGB;

#if defined(__HIPCC__)
using GpuStream = hipStream_t;
#else
using GpuStream = cudaStream_t;
#endif

// The frame in device memory (FKL's input) and in host memory (the references'), generated and uploaded once
struct Frames {
    NV12Image device{FRAME_W, FRAME_H};
    NV12Image host{FRAME_W, FRAME_H, fk::MemType::Host};

    Frames() {
        auto deviceData = device.getData();
        auto hostData = host.getData();
        generateFrame([&](int x, int y, uint8_t value) {
            deviceData.at(x, y) = value;
            hostData.at(x, y) = value;
        });
        fk::Stream stream;
        device.upload(stream);
        stream.sync();
    }
};

Frames& frames() {
    static Frames f;
    return f;
}

template <size_t N>
class FklBvf final : public FklGpu {
public:
    FklBvf(GpuStream stream, const RegionSets& regions) : m_stream(stream), m_rects(regions.iterations()) {
        for (size_t t = 0; t < m_rects.size(); ++t) {
            for (size_t i = 0; i < N; ++i) {
                const Region& r = regions.at(t, i);
                m_rects[t][i] = fk::Rect(r.x, r.y, r.width, r.height);
            }
        }
        frames();
    }

    void run(size_t t) override {
        fk::executeOperations<fk::TransformDPP<>>(m_stream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(frames().device),
            YUVToRGB::build(),
            fk::Crop<>::build(m_rects[t]),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, fk::float3{0.f, 0.f, 0.f}),
            fk::Mul<fk::float3>::build(fk::float3{MUL[0], MUL[1], MUL[2]}),
            fk::Sub<fk::float3>::build(fk::float3{SUB[0], SUB[1], SUB[2]}),
            fk::Div<fk::float3>::build(fk::float3{DIV[0], DIV[1], DIV[2]}),
            fk::TensorWrite<fk::float3>::build(m_out.ptr()));
    }

    std::vector<float> download() override {
        std::vector<float> host(N * OUT_FLOATS);
#if defined(__HIPCC__)
        gpuErrchk(hipMemcpyAsync(host.data(), m_out.ptr().data, host.size() * sizeof(float), hipMemcpyDeviceToHost,
                                 m_stream));
#else
        gpuErrchk(cudaMemcpyAsync(host.data(), m_out.ptr().data, host.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                  m_stream));
#endif
        m_stream.sync();
        return host;
    }

    void fillNaN() override {
#if defined(__HIPCC__)
        gpuErrchk(hipMemsetAsync(m_out.ptr().data, 0xFF, N * OUT_FLOATS * sizeof(float), m_stream));
#else
        gpuErrchk(cudaMemsetAsync(m_out.ptr().data, 0xFF, N * OUT_FLOATS * sizeof(float), m_stream));
#endif
    }

private:
    // PyTorch's stream, not owned
    fk::Stream m_stream;
    // The rectangles of every iteration, converted before the loops: what FKL's Crop takes
    std::vector<std::array<fk::Rect, N>> m_rects;
    fk::Tensor<fk::float3> m_out{OUT_W, OUT_H, N, 1, fk::MemType::Device};
};

// Calls f.template operator()<N>() for the batch size N == n, or throws
template <typename F>
void withBatch(size_t n, F&& f) {
    bool found = false;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((BATCHES[Is] == n ? void((f.template operator()<BATCHES[Is]>(), found = true)) : void()), ...);
    }(std::make_index_sequence<BATCHES.size()>{});
    if (!found) {
        throw std::runtime_error("Batch size " + std::to_string(n) + " is not one of the benchmarked ones");
    }
}

} // namespace

std::unique_ptr<FklGpu> makeFklGpu(size_t n, void* stream, const RegionSets& regions) {
    std::unique_ptr<FklGpu> fkl;
    withBatch(n, [&]<size_t N>() { fkl = std::make_unique<FklBvf<N>>(static_cast<GpuStream>(stream), regions); });
    return fkl;
}

const uint8_t* hostFrame(size_t* pitch) {
    const auto raw = frames().host.getData().ptr();
    *pitch = raw.dims.pitch;
    return raw.data;
}

std::vector<float> fklCpuReference(size_t n, const RegionSets& regions, size_t t) {
    std::vector<float> result;
    withBatch(n, [&]<size_t N>() {
        std::array<fk::Rect, N> crops{};
        for (size_t i = 0; i < N; ++i) {
            const Region& r = regions.at(t, i);
            crops[i] = fk::Rect(r.x, r.y, r.width, r.height);
        }
        fk::Stream_<fk::ParArch::CPU> cpuStream;
        fk::Tensor<fk::float3> out(OUT_W, OUT_H, N, 1, fk::MemType::Host);
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(frames().host),
            YUVToRGB::build(),
            fk::Crop<>::build(crops),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, fk::float3{0.f, 0.f, 0.f}),
            fk::Mul<fk::float3>::build(fk::float3{MUL[0], MUL[1], MUL[2]}),
            fk::Sub<fk::float3>::build(fk::float3{SUB[0], SUB[1], SUB[2]}),
            fk::Div<fk::float3>::build(fk::float3{DIV[0], DIV[1], DIV[2]}),
            fk::TensorWrite<fk::float3>::build(out.ptr()));
        cpuStream.sync();
        const float* data = reinterpret_cast<const float*>(out.ptr().data);
        result.assign(data, data + N * OUT_FLOATS);
    });
    return result;
}

std::vector<float> pixelCenterReference(size_t n, const RegionSets& regions, size_t t) {
    return reference::hostReference(frames().host, regions, n, t, pipeline::Conventions::PixelCenters);
}

pipeline::Conversion fklConversion() {
    return reference::conversion();
}

std::string parArchName() {
    return std::string(fk::toStrView(fk::defaultParArch));
}

std::string describeGpuAndFkl() {
    std::ostringstream os;
    int device = 0, runtime = 0, driver = 0;
#if defined(__HIPCC__)
    gpuErrchk(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    gpuErrchk(hipGetDeviceProperties(&prop, device));
    gpuErrchk(hipRuntimeGetVersion(&runtime));
    gpuErrchk(hipDriverGetVersion(&driver));
    os << "GPU: " << prop.name << " (" << prop.gcnArchName << ", " << prop.multiProcessorCount << " CUs, "
       << (prop.totalGlobalMem >> 20) << " MiB)\n"
       << "HIP runtime: " << runtime << ", HIP driver: " << driver << "\n"
       << "FKL's code: HIP, " << BENCH_GPU_COMPILER << ", flags " << BENCH_GPU_FLAGS << "\n";
#else
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    gpuErrchk(cudaGetDeviceProperties(&prop, device));
    gpuErrchk(cudaRuntimeGetVersion(&runtime));
    gpuErrchk(cudaDriverGetVersion(&driver));
    os << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ", " << prop.multiProcessorCount << " SMs, "
       << (prop.totalGlobalMem >> 20) << " MiB)\n"
       << "CUDA runtime: " << runtime / 1000 << "." << (runtime % 1000) / 10
       << ", CUDA driver API: " << driver / 1000 << "." << (driver % 1000) / 10 << "\n"
       << "FKL's code: CUDA, " << BENCH_GPU_COMPILER << ", flags " << BENCH_GPU_FLAGS << "\n";
#endif
    os << "FKL: branch " << BENCH_FKL_BRANCH << ", commit " << BENCH_FKL_COMMIT << ", backend " << parArchName() << "\n";
    return os.str();
}

} // namespace bench::pytorch
