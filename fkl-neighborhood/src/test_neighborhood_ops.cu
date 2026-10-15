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

// Tests of BoxBVF (common/bench_neighborhood_ops.h) and of FKL's MorphologyDPP box blur (executeBoxBlur, and
// MorphologyDPP<ParArch::CPU> with its passes) against the host reference of common/bench_neighborhood.h
// (hostBoxPlane, sums in double), for every box size and number of passes of the experiment, and of GaussianBVF
// (common/bench_neighborhood_ops.h, nppdx-wins/) against a separable host reference in double: Gaussians of 5 and 21
// taps, asymmetric taps that do not sum to 1, and a batch with one set of taps per crop:
//   - one PLANE_W x PLANE_H plane read directly (PerThreadRead);
//   - a batch of CROP_W x CROP_H crops of a larger image, one of them at its corner (the chain of the experiment,
//     without the color conversion): each crop replicates its own edge, it does not read the image around it.
// The values are 8 bit levels; the sums are in float in another order than the reference: within 1e-5, relative.
// Exits with 0 when every case passes.

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/morphology.h>

#include "bench_neighborhood.h"
#include "bench_neighborhood_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace {

using bench::neighborhood::CROP_H;
using bench::neighborhood::CROP_W;
using bench::neighborhood::PLANE_FLOATS;
using bench::neighborhood::PLANE_H;
using bench::neighborhood::PLANE_PIXELS;
using bench::neighborhood::PLANE_W;

// The source of the batched cases and its crops, the first one at its corner and the last one at its far corner
constexpr int SOURCE_W = 200;
constexpr int SOURCE_H = 120;
constexpr size_t BATCH = 3;
constexpr std::array<fk::Rect, BATCH> CROPS{fk::Rect(0, 0, CROP_W, CROP_H), fk::Rect(70, 40, CROP_W, CROP_H),
                                            fk::Rect(SOURCE_W - CROP_W, SOURCE_H - CROP_H, CROP_W, CROP_H)};

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

// The host box (hostBoxPlane) of planes planes of fk::float3, one after the other
std::vector<fk::float3> hostBox(const std::vector<fk::float3>& planes, const size_t count, const int k,
                                const int passes) {
    std::vector<fk::float3> out(planes.size());
    std::vector<double> in(PLANE_FLOATS), blurred(PLANE_FLOATS);
    for (size_t p = 0; p < count; ++p) {
        for (size_t i = 0; i < PLANE_PIXELS; ++i) {
            const fk::float3 v = planes[p * PLANE_PIXELS + i];
            in[i * 3] = v.x;
            in[i * 3 + 1] = v.y;
            in[i * 3 + 2] = v.z;
        }
        bench::neighborhood::hostBoxPlane(in.data(), blurred.data(), k, passes);
        for (size_t i = 0; i < PLANE_PIXELS; ++i) {
            out[p * PLANE_PIXELS + i] = {static_cast<float>(blurred[i * 3]), static_cast<float>(blurred[i * 3 + 1]),
                                         static_cast<float>(blurred[i * 3 + 2])};
        }
    }
    return out;
}

bool matches(const char* what, const std::vector<fk::float3>& got, const std::vector<fk::float3>& expected) {
    for (size_t i = 0; i < got.size(); ++i) {
        const float g[3] = {got[i].x, got[i].y, got[i].z};
        const float e[3] = {expected[i].x, expected[i].y, expected[i].z};
        for (int c = 0; c < 3; ++c) {
            if (!(std::fabs(g[c] - e[c]) <= 1e-5f * std::max(1.f, std::fabs(e[c])))) {
                std::printf("%s: plane %zu pixel %zu channel %d got %.9g expected %.9g\n", what, i / PLANE_PIXELS,
                            i % PLANE_PIXELS, c, g[c], e[c]);
                return false;
            }
        }
    }
    return true;
}

// The box of the experiment as BVF operations after the read IOp, fused by executeOperations
template <int K, int PASSES, fk::ParArch PA, typename ReadIOp, typename WriteIOp>
void runChain(fk::Stream_<PA>& stream, const ReadIOp& read, const WriteIOp& write) {
    if constexpr (PASSES == 1) {
        fk::executeOperations<fk::TransformDPP<PA>>(stream, read, fk::BoxBVF<K>::build(), write);
    } else {
        fk::executeOperations<fk::TransformDPP<PA>>(stream, read, fk::BoxBVF<K>::build(), fk::BoxBVF<K>::build(),
                                                    write);
    }
}

template <int K>
using Details = fk::MorphologyDPPDetails<fk::float3, 16, 8, K, K>;

template <int K>
constexpr Details<K> details() {
    return Details<K>{PLANE_W, PLANE_H, K, K, K / 2, K / 2};
}

// The box of the experiment with MorphologyDPP on the CPU: the passes of executeBoxBlur
template <int K, int PASSES, typename ReadIOp, typename WriteIOp>
void runDppCpu(const ReadIOp& read, const WriteIOp& write) {
    fk::MorphologyDPP<fk::ParArch::CPU, Details<K>>::exec(
        details<K>(), read, write, fk::morphology_detail::boxBlurReducers<fk::float3, PASSES>(details<K>()));
}

template <int K, int PASSES>
bool runCase() {
    char name[96];
    bool ok = true;
    const auto check = [&](const char* what, const std::vector<fk::float3>& got,
                           const std::vector<fk::float3>& expected) {
        std::snprintf(name, sizeof(name), "%s %dx%d, %d pass%s", what, K, K, PASSES, PASSES == 1 ? "" : "es");
        ok = matches(name, got, expected) && ok;
    };

    // One plane, read directly
    std::vector<fk::float3> plane = randomImage(PLANE_PIXELS, 7u + K * 10u + PASSES);
    const std::vector<fk::float3> expected = hostBox(plane, 1, K, PASSES);
    const auto planeRead = [](const auto& ptr) { return fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(ptr); };
    {
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, 1, 1, fk::MemType::Host);
        fk::Stream_<fk::ParArch::CPU> cpu;
        runChain<K, PASSES>(cpu, planeRead(hostPtr<PLANE_W, PLANE_H>(plane)),
                            fk::TensorWrite<fk::float3>::build(out.ptr()));
        check("CPU BoxBVF plane", tensorValues(out, 1), expected);
        fk::Tensor<fk::float3> dppOut(PLANE_W, PLANE_H, 1, 1, fk::MemType::Host);
        runDppCpu<K, PASSES>(planeRead(hostPtr<PLANE_W, PLANE_H>(plane)),
                             fk::TensorWrite<fk::float3>::build(dppOut.ptr()));
        check("CPU MorphologyDPP box plane", tensorValues(dppOut, 1), expected);
    }
    {
        fk::Ptr2D<fk::float3> in(PLANE_W, PLANE_H);
        for (int y = 0; y < PLANE_H; ++y)
            for (int x = 0; x < PLANE_W; ++x) in.at(fk::Point{x, y, 0}) = plane[static_cast<size_t>(y) * PLANE_W + x];
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, 1);
        fk::Tensor<fk::float3> dppOut(PLANE_W, PLANE_H, 1);
        fk::Stream stream;
        in.upload(stream);
        runChain<K, PASSES>(stream, planeRead(in.ptr()), fk::TensorWrite<fk::float3>::build(out.ptr()));
        const bool launched = fk::executeBoxBlur<PASSES>(details<K>(), planeRead(in.ptr()),
                                                         fk::TensorWrite<fk::float3>::build(dppOut.ptr()), stream);
        out.download(stream);
        dppOut.download(stream);
        stream.sync();
        check("GPU BoxBVF plane", tensorValues(out, 1), expected);
        if (!launched) {
            std::printf("executeBoxBlur rejected valid details\n");
            ok = false;
        }
        check("GPU executeBoxBlur plane", tensorValues(dppOut, 1), expected);
    }

    // A batch of crops of a larger image: the reference is the box of each crop, replicating its own edge
    std::vector<fk::float3> source = randomImage(static_cast<size_t>(SOURCE_W) * SOURCE_H, 11u + K * 10u + PASSES);
    std::vector<fk::float3> crops(BATCH * PLANE_PIXELS);
    for (size_t i = 0; i < BATCH; ++i)
        for (int y = 0; y < CROP_H; ++y)
            for (int x = 0; x < CROP_W; ++x)
                crops[i * PLANE_PIXELS + static_cast<size_t>(y) * PLANE_W + x] =
                    source[static_cast<size_t>(CROPS[i].y + y) * SOURCE_W + CROPS[i].x + x];
    const std::vector<fk::float3> batchExpected = hostBox(crops, BATCH, K, PASSES);
    const auto cropOf = [](const auto& sourcePtr) {
        return fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(sourcePtr).then(fk::Crop<>::build(CROPS));
    };
    {
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, BATCH, 1, fk::MemType::Host);
        fk::Stream_<fk::ParArch::CPU> cpu;
        runChain<K, PASSES>(cpu, cropOf(hostPtr<SOURCE_W, SOURCE_H>(source)),
                            fk::TensorWrite<fk::float3>::build(out.ptr()));
        check("CPU BoxBVF batch", tensorValues(out, BATCH), batchExpected);
        fk::Tensor<fk::float3> dppOut(PLANE_W, PLANE_H, BATCH, 1, fk::MemType::Host);
        runDppCpu<K, PASSES>(cropOf(hostPtr<SOURCE_W, SOURCE_H>(source)),
                             fk::TensorWrite<fk::float3>::build(dppOut.ptr()));
        check("CPU MorphologyDPP box batch", tensorValues(dppOut, BATCH), batchExpected);
    }
    {
        fk::Ptr2D<fk::float3> in(SOURCE_W, SOURCE_H);
        for (int y = 0; y < SOURCE_H; ++y)
            for (int x = 0; x < SOURCE_W; ++x) in.at(fk::Point{x, y, 0}) = source[static_cast<size_t>(y) * SOURCE_W + x];
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, BATCH);
        fk::Tensor<fk::float3> dppOut(PLANE_W, PLANE_H, BATCH);
        fk::Stream stream;
        in.upload(stream);
        runChain<K, PASSES>(stream, cropOf(in.ptr()), fk::TensorWrite<fk::float3>::build(out.ptr()));
        const bool launched = fk::executeBoxBlur<PASSES>(details<K>(), cropOf(in.ptr()),
                                                         fk::TensorWrite<fk::float3>::build(dppOut.ptr()), stream);
        out.download(stream);
        dppOut.download(stream);
        stream.sync();
        check("GPU BoxBVF batch", tensorValues(out, BATCH), batchExpected);
        if (!launched) {
            std::printf("executeBoxBlur rejected valid details\n");
            ok = false;
        }
        check("GPU executeBoxBlur batch", tensorValues(dppOut, BATCH), batchExpected);
    }
    std::printf("box %dx%d (%d pass%s): %s\n", K, K, PASSES, PASSES == 1 ? "" : "es", ok ? "PASS" : "FAIL");
    return ok;
}

// ------------------------------------------------------------------------------------------------
// GaussianBVF
// ------------------------------------------------------------------------------------------------

// The K taps of a Gaussian of standard deviation sigma, centered, normalized to a sum of 1
template <int K>
fk::Array<float, K> gaussianTaps(const double sigma) {
    double taps[K];
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        const double d = i - K / 2;
        taps[i] = std::exp(-d * d / (2.0 * sigma * sigma));
        sum += taps[i];
    }
    fk::Array<float, K> out{};
    for (int i = 0; i < K; ++i) {
        out[i] = static_cast<float>(taps[i] / sum);
    }
    return out;
}

// The separable filter of the taps on planes planes of fk::float3, one after the other, each replicating its own
// edge: rows, then columns, in double (the taps as float, as GaussianBVF uses them)
template <int K>
std::vector<fk::float3> hostGaussian(const std::vector<fk::float3>& planes, const size_t count,
                                     const fk::Array<float, K>& taps) {
    std::vector<fk::float3> out(planes.size());
    std::vector<double> rows(PLANE_FLOATS);
    const auto channel = [](const fk::float3& v, const int c) { return c == 0 ? v.x : (c == 1 ? v.y : v.z); };
    for (size_t p = 0; p < count; ++p) {
        const fk::float3* plane = planes.data() + p * PLANE_PIXELS;
        for (int y = 0; y < PLANE_H; ++y)
            for (int x = 0; x < PLANE_W; ++x)
                for (int c = 0; c < 3; ++c) {
                    double sum = 0.0;
                    for (int k = 0; k < K; ++k) {
                        const int sx = std::clamp(x + k - K / 2, 0, PLANE_W - 1);
                        sum += static_cast<double>(taps[k]) * channel(plane[static_cast<size_t>(y) * PLANE_W + sx], c);
                    }
                    rows[(static_cast<size_t>(y) * PLANE_W + x) * 3 + c] = sum;
                }
        for (int y = 0; y < PLANE_H; ++y)
            for (int x = 0; x < PLANE_W; ++x) {
                double v[3];
                for (int c = 0; c < 3; ++c) {
                    double sum = 0.0;
                    for (int k = 0; k < K; ++k) {
                        const int sy = std::clamp(y + k - K / 2, 0, PLANE_H - 1);
                        sum += static_cast<double>(taps[k]) * rows[(static_cast<size_t>(sy) * PLANE_W + x) * 3 + c];
                    }
                    v[c] = sum;
                }
                out[p * PLANE_PIXELS + static_cast<size_t>(y) * PLANE_W + x] = {
                    static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])};
            }
    }
    return out;
}

// Asymmetric taps that do not sum to 1 (a ramp times the Gaussian, times 1.7): a reversed tap index or a normalization
// in the operation would not match the reference
template <int K>
fk::Array<float, K> asymmetricTaps() {
    const fk::Array<float, K> g = gaussianTaps<K>(1.0);
    fk::Array<float, K> out{};
    for (int i = 0; i < K; ++i) {
        out[i] = static_cast<float>(1.7 * g[i] * (0.5 + i / (K - 1.0)));
    }
    return out;
}

template <int K>
bool runGaussianCase(const char* what, const fk::Array<float, K>& taps) {
    char name[128];
    bool ok = true;
    const auto check = [&](const char* where, const std::vector<fk::float3>& got,
                           const std::vector<fk::float3>& expected) {
        std::snprintf(name, sizeof(name), "%s, %s", where, what);
        ok = matches(name, got, expected) && ok;
    };
    const auto planeRead = [](const auto& ptr) { return fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(ptr); };

    // One plane, read directly
    std::vector<fk::float3> plane = randomImage(PLANE_PIXELS, 101u + K);
    const std::vector<fk::float3> expected = hostGaussian<K>(plane, 1, taps);
    {
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, 1, 1, fk::MemType::Host);
        fk::Stream_<fk::ParArch::CPU> cpu;
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpu, planeRead(hostPtr<PLANE_W, PLANE_H>(plane)),
                                                                  fk::GaussianBVF<K>::build(taps),
                                                                  fk::TensorWrite<fk::float3>::build(out.ptr()));
        check("CPU GaussianBVF plane", tensorValues(out, 1), expected);
    }
    {
        fk::Ptr2D<fk::float3> in(PLANE_W, PLANE_H);
        for (int y = 0; y < PLANE_H; ++y)
            for (int x = 0; x < PLANE_W; ++x) in.at(fk::Point{x, y, 0}) = plane[static_cast<size_t>(y) * PLANE_W + x];
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, 1);
        fk::Stream stream;
        in.upload(stream);
        fk::executeOperations<fk::TransformDPP<>>(stream, planeRead(in.ptr()), fk::GaussianBVF<K>::build(taps),
                                                  fk::TensorWrite<fk::float3>::build(out.ptr()));
        out.download(stream);
        stream.sync();
        check("GPU GaussianBVF plane", tensorValues(out, 1), expected);
    }

    // A batch of crops of a larger image, each replicating its own edge
    std::vector<fk::float3> source = randomImage(static_cast<size_t>(SOURCE_W) * SOURCE_H, 103u + K);
    std::vector<fk::float3> crops(BATCH * PLANE_PIXELS);
    for (size_t i = 0; i < BATCH; ++i)
        for (int y = 0; y < CROP_H; ++y)
            for (int x = 0; x < CROP_W; ++x)
                crops[i * PLANE_PIXELS + static_cast<size_t>(y) * PLANE_W + x] =
                    source[static_cast<size_t>(CROPS[i].y + y) * SOURCE_W + CROPS[i].x + x];
    const std::vector<fk::float3> batchExpected = hostGaussian<K>(crops, BATCH, taps);
    const auto cropOf = [](const auto& sourcePtr) {
        return fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(sourcePtr).then(fk::Crop<>::build(CROPS));
    };
    {
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, BATCH, 1, fk::MemType::Host);
        fk::Stream_<fk::ParArch::CPU> cpu;
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpu, cropOf(hostPtr<SOURCE_W, SOURCE_H>(source)),
                                                                  fk::GaussianBVF<K>::build(taps),
                                                                  fk::TensorWrite<fk::float3>::build(out.ptr()));
        check("CPU GaussianBVF batch", tensorValues(out, BATCH), batchExpected);
    }
    {
        fk::Ptr2D<fk::float3> in(SOURCE_W, SOURCE_H);
        for (int y = 0; y < SOURCE_H; ++y)
            for (int x = 0; x < SOURCE_W; ++x)
                in.at(fk::Point{x, y, 0}) = source[static_cast<size_t>(y) * SOURCE_W + x];
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, BATCH);
        fk::Stream stream;
        in.upload(stream);
        fk::executeOperations<fk::TransformDPP<>>(stream, cropOf(in.ptr()), fk::GaussianBVF<K>::build(taps),
                                                  fk::TensorWrite<fk::float3>::build(out.ptr()));
        out.download(stream);
        stream.sync();
        check("GPU GaussianBVF batch", tensorValues(out, BATCH), batchExpected);
    }
    std::printf("GaussianBVF %s: %s\n", what, ok ? "PASS" : "FAIL");
    return ok;
}

// One set of taps per crop of a batch (the std::array builder of the incomplete GaussianBVF): every plane is filtered
// with its own taps
bool runPerPlaneTapsCase() {
    constexpr int K = 5;
    const std::array<fk::Array<float, K>, BATCH> taps{gaussianTaps<K>(1.0), asymmetricTaps<K>(),
                                                      fk::Array<float, K>{{0.f, 0.f, 1.f, 0.f, 0.f}}};
    std::vector<fk::float3> source = randomImage(static_cast<size_t>(SOURCE_W) * SOURCE_H, 107u);
    std::vector<fk::float3> expected(BATCH * PLANE_PIXELS);
    for (size_t i = 0; i < BATCH; ++i) {
        std::vector<fk::float3> crop(PLANE_PIXELS);
        for (int y = 0; y < CROP_H; ++y)
            for (int x = 0; x < CROP_W; ++x)
                crop[static_cast<size_t>(y) * PLANE_W + x] =
                    source[static_cast<size_t>(CROPS[i].y + y) * SOURCE_W + CROPS[i].x + x];
        const auto planeExpected = hostGaussian<K>(crop, 1, taps[i]);
        std::copy(planeExpected.begin(), planeExpected.end(), expected.begin() + i * PLANE_PIXELS);
    }
    const auto cropOf = [](const auto& sourcePtr) {
        return fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(sourcePtr).then(fk::Crop<>::build(CROPS));
    };
    bool ok = true;
    {
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, BATCH, 1, fk::MemType::Host);
        fk::Stream_<fk::ParArch::CPU> cpu;
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpu, cropOf(hostPtr<SOURCE_W, SOURCE_H>(source)),
                                                                  fk::GaussianBVF<K>::build(taps),
                                                                  fk::TensorWrite<fk::float3>::build(out.ptr()));
        ok = matches("CPU GaussianBVF batch with per-plane taps", tensorValues(out, BATCH), expected) && ok;
    }
    {
        fk::Ptr2D<fk::float3> in(SOURCE_W, SOURCE_H);
        for (int y = 0; y < SOURCE_H; ++y)
            for (int x = 0; x < SOURCE_W; ++x)
                in.at(fk::Point{x, y, 0}) = source[static_cast<size_t>(y) * SOURCE_W + x];
        fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, BATCH);
        fk::Stream stream;
        in.upload(stream);
        fk::executeOperations<fk::TransformDPP<>>(stream, cropOf(in.ptr()), fk::GaussianBVF<K>::build(taps),
                                                  fk::TensorWrite<fk::float3>::build(out.ptr()));
        out.download(stream);
        stream.sync();
        ok = matches("GPU GaussianBVF batch with per-plane taps", tensorValues(out, BATCH), expected) && ok;
    }
    std::printf("GaussianBVF batch with per-plane taps: %s\n", ok ? "PASS" : "FAIL");
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
    bool ok = runAll(std::make_index_sequence<PART_A_KS.size() * PART_A_PASSES.size()>{});
    ok = runGaussianCase<5>("Gaussian 5 taps, sigma 1", gaussianTaps<5>(1.0)) & ok;
    ok = runGaussianCase<21>("Gaussian 21 taps, sigma 5", gaussianTaps<21>(5.0)) & ok;
    ok = runGaussianCase<5>("asymmetric 5 taps, not normalized", asymmetricTaps<5>()) & ok;
    ok = runPerPlaneTapsCase() & ok;
    std::printf("%s\n", ok ? "All BoxBVF, GaussianBVF and MorphologyDPP box tests passed" : "Tests FAILED");
    return ok ? 0 : 1;
}
