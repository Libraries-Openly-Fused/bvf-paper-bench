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

// Tests of ErodeBVF and DilateBVF (common/bench_neighborhood_ops.h) against the host reference of
// common/bench_neighborhood.h, on FKL's CPU backend and on the GPU, for every mask size and number of passes of the
// experiment:
//   - one PLANE_W x PLANE_H plane read directly (PerThreadRead): the result must be bit-identical to the reference,
//     since the minimum and the maximum pick one of the values they read;
//   - a batch of regions of different sizes, each one cropped and resized to PLANE_W x PLANE_H before the morphology
//     (the chain of the experiment, without the color conversion): the reference is the morphology of the resized
//     planes, which TransformDPP computes on the CPU. The resize multiplies and adds, so the backends may differ in the
//     last bits (fused multiply-add contraction): within 1e-5, relative.
// Exits with 0 when every case passes.

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include "bench_neighborhood.h"
#include "bench_neighborhood_ops.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace {

using bench::neighborhood::MorphOp;
using bench::neighborhood::PLANE_H;
using bench::neighborhood::PLANE_W;

constexpr size_t PLANE_PIXELS = static_cast<size_t>(PLANE_W) * PLANE_H;

// The source of the batched cases and its regions, of different sizes, all of them upscaled to PLANE_W x PLANE_H
constexpr int SOURCE_W = 200;
constexpr int SOURCE_H = 120;
constexpr size_t BATCH = 3;
constexpr std::array<fk::Rect, BATCH> REGIONS{fk::Rect(0, 0, 20, 10), fk::Rect(70, 40, 32, 16),
                                              fk::Rect(152, 96, 48, 24)};

std::vector<fk::float3> randomImage(const size_t pixels, const unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> level(0, 255);
    std::vector<fk::float3> image(pixels);
    for (auto& value : image) {
        value = {static_cast<float>(level(gen)), static_cast<float>(level(gen)), static_cast<float>(level(gen))};
    }
    return image;
}

template <int W, int H>
fk::RawPtr<fk::ND::_2D, fk::float3> hostPtr(std::vector<fk::float3>& image) {
    return {image.data(), fk::PtrDims<fk::ND::_2D>(W, H, W * sizeof(fk::float3))};
}

std::vector<fk::float3> tensorValues(const fk::Tensor<fk::float3>& tensor, const size_t planes) {
    std::vector<fk::float3> values;
    for (size_t z = 0; z < planes; ++z)
        for (int y = 0; y < PLANE_H; ++y)
            for (int x = 0; x < PLANE_W; ++x) values.push_back(tensor.at(x, y, static_cast<int>(z)));
    return values;
}

// relative = 0: bit-identical; otherwise within relative * max(1, |expected|)
bool matches(const char* what, const std::vector<fk::float3>& got, const std::vector<fk::float3>& expected,
             const float relative) {
    for (size_t i = 0; i < got.size(); ++i) {
        const float g[3] = {got[i].x, got[i].y, got[i].z};
        const float e[3] = {expected[i].x, expected[i].y, expected[i].z};
        for (int c = 0; c < 3; ++c) {
            const bool ok = relative == 0.f ? g[c] == e[c]
                                            : std::fabs(g[c] - e[c]) <= relative * std::max(1.f, std::fabs(e[c]));
            if (!ok) {
                std::printf("%s: plane %zu pixel %zu channel %d got %.9g expected %.9g\n", what, i / PLANE_PIXELS,
                            i % PLANE_PIXELS, c, g[c], e[c]);
                return false;
            }
        }
    }
    return true;
}

// The morphology of the experiment after the read IOp, as fused by executeOperations
template <int K, int PASSES, fk::ParArch PA, typename ReadIOp, typename WriteIOp>
void runChain(fk::Stream_<PA>& stream, const ReadIOp& read, const WriteIOp& write) {
    if constexpr (PASSES == 1) {
        fk::executeOperations<fk::TransformDPP<PA>>(stream, read, fk::ErodeBVF<K>::build(), write);
    } else {
        fk::executeOperations<fk::TransformDPP<PA>>(stream, read, fk::ErodeBVF<K>::build(), fk::DilateBVF<K>::build(),
                                                    write);
    }
}

template <int K, int PASSES>
bool runCase() {
    constexpr MorphOp OP = PASSES == 1 ? MorphOp::Erode : MorphOp::Open;
    char name[96];
    bool ok = true;

    // One plane, read directly
    std::vector<fk::float3> plane = randomImage(PLANE_PIXELS, 7u + K * 10u + PASSES);
    std::vector<fk::float3> expected(PLANE_PIXELS);
    bench::neighborhood::hostMorphPlane(plane.data(), expected.data(), K, OP);
    {
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, 1, 1, fk::MemType::Host);
        fk::Stream_<fk::ParArch::CPU> cpu;
        runChain<K, PASSES>(cpu, fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(hostPtr<PLANE_W, PLANE_H>(plane)),
                            fk::TensorWrite<fk::float3>::build(out.ptr()));
        std::snprintf(name, sizeof(name), "CPU %s %dx%d plane", bench::neighborhood::name(OP), K, K);
        ok = matches(name, tensorValues(out, 1), expected, 0.f) && ok;
    }
    {
        fk::Ptr2D<fk::float3> in(PLANE_W, PLANE_H);
        for (int y = 0; y < PLANE_H; ++y)
            for (int x = 0; x < PLANE_W; ++x) in.at(fk::Point{x, y, 0}) = plane[static_cast<size_t>(y) * PLANE_W + x];
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, 1);
        fk::Stream stream;
        in.upload(stream);
        runChain<K, PASSES>(stream, fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(in.ptr()),
                            fk::TensorWrite<fk::float3>::build(out.ptr()));
        out.download(stream);
        stream.sync();
        std::snprintf(name, sizeof(name), "GPU %s %dx%d plane", bench::neighborhood::name(OP), K, K);
        ok = matches(name, tensorValues(out, 1), expected, 0.f) && ok;
    }

    // A batch of regions, cropped and resized: the reference is the morphology of the resized planes
    std::vector<fk::float3> source = randomImage(static_cast<size_t>(SOURCE_W) * SOURCE_H, 11u + K * 10u + PASSES);
    const auto resizeOf = [](const auto& sourcePtr) {
        return fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(sourcePtr)
            .then(fk::Crop<>::build(REGIONS))
            .then(fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(
                fk::Size(PLANE_W, PLANE_H)));
    };
    std::vector<fk::float3> batchExpected(BATCH * PLANE_PIXELS);
    {
        fk::Tensor<fk::float3> resized(PLANE_W, PLANE_H, BATCH, 1, fk::MemType::Host);
        fk::Stream_<fk::ParArch::CPU> cpu;
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpu, resizeOf(hostPtr<SOURCE_W, SOURCE_H>(source)),
                                                                   fk::TensorWrite<fk::float3>::build(resized.ptr()));
        const std::vector<fk::float3> resizedValues = tensorValues(resized, BATCH);
        bench::neighborhood::hostMorph(resizedValues.data(), batchExpected.data(), BATCH, K, OP);
    }
    {
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, BATCH, 1, fk::MemType::Host);
        fk::Stream_<fk::ParArch::CPU> cpu;
        runChain<K, PASSES>(cpu, resizeOf(hostPtr<SOURCE_W, SOURCE_H>(source)),
                            fk::TensorWrite<fk::float3>::build(out.ptr()));
        std::snprintf(name, sizeof(name), "CPU %s %dx%d batch", bench::neighborhood::name(OP), K, K);
        ok = matches(name, tensorValues(out, BATCH), batchExpected, 1e-5f) && ok;
    }
    {
        fk::Ptr2D<fk::float3> in(SOURCE_W, SOURCE_H);
        for (int y = 0; y < SOURCE_H; ++y)
            for (int x = 0; x < SOURCE_W; ++x) in.at(fk::Point{x, y, 0}) = source[static_cast<size_t>(y) * SOURCE_W + x];
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, BATCH);
        fk::Stream stream;
        in.upload(stream);
        runChain<K, PASSES>(stream, resizeOf(in.ptr()), fk::TensorWrite<fk::float3>::build(out.ptr()));
        out.download(stream);
        stream.sync();
        std::snprintf(name, sizeof(name), "GPU %s %dx%d batch", bench::neighborhood::name(OP), K, K);
        ok = matches(name, tensorValues(out, BATCH), batchExpected, 1e-5f) && ok;
    }
    std::printf("%s %dx%d (%d pass%s): %s\n", bench::neighborhood::name(OP), K, K, PASSES, PASSES == 1 ? "" : "es",
                ok ? "PASS" : "FAIL");
    return ok;
}

template <size_t... I>
bool runAll(std::index_sequence<I...>) {
    constexpr auto KS = bench::neighborhood::PART_A_KS;
    constexpr auto PS = bench::neighborhood::PART_A_PASSES;
    constexpr size_t NP = PS.size();
    return (runCase<KS[I / NP], PS[I % NP]>() & ...);
}

} // namespace

int main() {
    using namespace bench::neighborhood;
    const bool ok = runAll(std::make_index_sequence<PART_A_KS.size() * PART_A_PASSES.size()>{});
    std::printf("%s\n", ok ? "All ErodeBVF/DilateBVF tests passed" : "ErodeBVF/DilateBVF tests FAILED");
    return ok ? 0 : 1;
}
