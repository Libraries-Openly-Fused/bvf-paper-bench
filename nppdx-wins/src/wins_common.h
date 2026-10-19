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

// The nppdx-wins exploration (a smoke test, not a result of the paper): a 5x5 box blur BEFORE a resize with a
// compile-time ratio. NPPDx is built for this order (BoxBlur -> Resize in one kernel, the ratio a template). FKL
// cannot stage it in one kernel: a DPP's output is a write, so it cannot feed a resize.
//   W1, regions  ReadNV12 -> YUVtoRGB -> Box 5x5 over the frame -> Crop 32x16 -> Resize 1:2 to 64x32 -> Mul -> Sub ->
//                Div -> N float3 planes (benchmark_wins_regions), Experiment 1's positions
//   W2, frame    ReadNV12 -> YUVtoRGB -> Box 5x5 -> Resize 2:1 to 1920x1080 -> Mul -> Sub -> Div -> one float3 plane
//                (benchmark_wins_frame)
// Implementations, on the harness of common/bench_gpu.h:
//   nppdx        one kernel: NPPDx's NV12 ingest, FromLimitedLuma/Chroma, ColorConvert BT.601, BoxBlur<5, 5> and
//                Resize<J, K, bilinear> on shared memory tiles, then the normalization and float3 write written by
//                hand (NPPDx has no float output), as fkl-vs-nppdx's NPPDx kernel. W1 in frame mode (below).
//   fkl_inline   one executeOperations call: ReadNV12 -> YUVtoRGB -> BoxBVF<5> -> [Crop ->] Resize -> Mul -> Sub ->
//                Div -> TensorWrite. Every output value evaluates the NV12 read and conversion 4 x 25 = 100 times.
//   fkl_staged   two launches: MorphologyDPP's box (executeBoxBlur) into a float3 intermediate in memory (the
//                regions expanded by the box radius for W1, the frame for W2), then one TransformDPP kernel
//                [Crop ->] Resize -> Mul -> Sub -> Div -> TensorWrite reading it.
//
// Edges and sampling differ between the libraries; this is documented, not matched (exp3-exploration/
// box-followup.md):
// - FKL's Resize samples at dst * (src / dst), with no half-pixel offset, and clamps its reads to the image it reads
//   (the crop). Both FKL forms are "box over the frame, crop, resize clamped at the crop".
// - NPPDx samples at pixel centers, (dst + 0.5) * J / K - 0.5. In W1's frame mode its ingest also gives the resize
//   the boxed frame beyond the crop. NPPDx is "box over the frame, resize reading the boxed frame around the region".
// - W2: both replicate the frame edge for the box. FKL's 2:1 then takes the boxed pixel at (2x, 2y) (bilinear weights
//   1 and 0); NPPDx's takes the mean of the 2x2 boxed pixels around (2x + 0.5, 2y + 0.5).
// Each implementation is checked against a reference of its own semantics: the frame RGB of FKL's CPU backend (float),
// then the box and the resize in double, then the normalization in float.

#ifndef BVF_BENCH_WINS_COMMON_H
#define BVF_BENCH_WINS_COMMON_H

#include <fused_kernel/core/utils/utils.h> // gpuErrchk
#include <fused_kernel/algorithms/image_processing/linear_filter.h>
#include <fused_kernel/algorithms/image_processing/morphology.h>

#include <nppdx.hpp>
#include <nppdx/detail/backend/gaussian_blur_coeff_tables.hpp>

#include "bench_gpu.h"
#include "bench_neighborhood_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace bench::wins {

constexpr int BOX_K = 5;
constexpr int BOX_R = BOX_K / 2;

// W1: 32x16 regions (BENCH_REGIONS=32x16), resized 1:2 to 64x32
constexpr int REG_W = 32;
constexpr int REG_H = 16;
constexpr int W1_OUT_W = 64;
constexpr int W1_OUT_H = 32;
constexpr size_t W1_PLANE_FLOATS = static_cast<size_t>(W1_OUT_W) * W1_OUT_H * 3;
// The regions of fkl_staged's box: the region plus the box radius on every side
constexpr int EXP_W = REG_W + 2 * BOX_R;
constexpr int EXP_H = REG_H + 2 * BOX_R;

// W2: the frame resized 2:1
constexpr int W2_OUT_W = FRAME_W / 2;
constexpr int W2_OUT_H = FRAME_H / 2;
constexpr size_t W2_FLOATS = static_cast<size_t>(W2_OUT_W) * W2_OUT_H * 3;

// ------------------------------------------------------------------------------------------------
// References
// ------------------------------------------------------------------------------------------------

// The RGB of the whole frame on FKL's CPU backend (ReadNV12 -> YUVtoRGB, float), interleaved; computed once
inline const std::vector<float>& frameRgb(const Context& ctx) {
    static std::vector<float> rgb;
    if (rgb.empty()) {
        fk::Stream_<fk::ParArch::CPU> cpuStream;
        fk::Tensor<fk::float3> out(FRAME_W, FRAME_H, 1, 1, fk::MemType::Host);
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameHost), YUVToRGB::build(),
            fk::TensorWrite<fk::float3>::build(out.ptr()));
        cpuStream.sync();
        const float* data = reinterpret_cast<const float*>(out.ptr().data);
        rgb.assign(data, data + static_cast<size_t>(FRAME_W) * FRAME_H * 3);
    }
    return rgb;
}

// The 5x5 box of the frame RGB at (x, y), replicating the frame edge, in double
inline double boxedAt(const std::vector<float>& rgb, const int x, const int y, const int c) {
    double sum = 0.0;
    for (int dy = -BOX_R; dy <= BOX_R; ++dy) {
        const int sy = std::clamp(y + dy, 0, FRAME_H - 1);
        for (int dx = -BOX_R; dx <= BOX_R; ++dx) {
            const int sx = std::clamp(x + dx, 0, FRAME_W - 1);
            sum += rgb[(static_cast<size_t>(sy) * FRAME_W + sx) * 3 + c];
        }
    }
    return sum / (BOX_K * BOX_K);
}

// The normalization in float, as the pipelines
inline float normalized(const double v, const int c) {
    const float f = static_cast<float>(v);
    return (f * MUL[c] - SUB[c]) / DIV[c];
}

// FKL's bilinear (interpolation.h): at (srcX, srcY) of a w x h image g, the right and bottom neighbors clamped to it
template <typename G>
double fklBilinear(const G& g, const float srcX, const float srcY, const int w, const int h, const int c) {
    const int x1 = static_cast<int>(std::floor(srcX));
    const int y1 = static_cast<int>(std::floor(srcY));
    const int x2 = std::min(x1 + 1, w - 1);
    const int y2 = std::min(y1 + 1, h - 1);
    const double ax = static_cast<double>(srcX) - x1;
    const double ay = static_cast<double>(srcY) - y1;
    return (g(x1, y1, c) * (1.0 - ax) + g(x2, y1, c) * ax) * (1.0 - ay) +
           (g(x1, y2, c) * (1.0 - ax) + g(x2, y2, c) * ax) * ay;
}

// Bilinear at pixel centers without clamping (NPPDx: its tile and halo hold the values around the image)
template <typename G>
double centerBilinear(const G& g, const double srcX, const double srcY, const int c) {
    const int x1 = static_cast<int>(std::floor(srcX));
    const int y1 = static_cast<int>(std::floor(srcY));
    const double ax = srcX - x1;
    const double ay = srcY - y1;
    return (g(x1, y1, c) * (1.0 - ax) + g(x1 + 1, y1, c) * ax) * (1.0 - ay) +
           (g(x1, y1 + 1, c) * (1.0 - ax) + g(x1 + 1, y1 + 1, c) * ax) * ay;
}

// FKL's resize factor (resize.h: the float of 1 / (dst / src))
inline float fklFactor(const int src, const int dst) {
    return static_cast<float>(1.0 / (static_cast<double>(dst) / static_cast<double>(src)));
}

// W1: the N planes of iteration t, with FKL's semantics (FklFloat) or NPPDx's (NppdxPixelCenterFloat)
template <size_t N>
std::vector<float> regionsReference(const Context& ctx, const Reference r, const size_t t) {
    if (r != Reference::FklFloat && r != Reference::NppdxPixelCenterFloat) {
        throw std::runtime_error("nppdx-wins has float references only");
    }
    const auto& rgb = frameRgb(ctx);
    std::vector<float> out(N * W1_PLANE_FLOATS);
    const float fx = fklFactor(REG_W, W1_OUT_W);
    const float fy = fklFactor(REG_H, W1_OUT_H);
    for (size_t i = 0; i < N; ++i) {
        const Region& region = ctx.regions->at(t, i);
        if (region.width != REG_W || region.height != REG_H) {
            throw std::runtime_error("W1 needs 32x16 regions (BENCH_REGIONS=32x16)");
        }
        const int rx = static_cast<int>(region.x);
        const int ry = static_cast<int>(region.y);
        const auto g = [&](const int x, const int y, const int c) { return boxedAt(rgb, rx + x, ry + y, c); };
        for (int oy = 0; oy < W1_OUT_H; ++oy) {
            for (int ox = 0; ox < W1_OUT_W; ++ox) {
                for (int c = 0; c < 3; ++c) {
                    const double v =
                        r == Reference::FklFloat
                            ? fklBilinear(g, static_cast<float>(ox) * fx, static_cast<float>(oy) * fy, REG_W, REG_H, c)
                            : centerBilinear(g, (ox + 0.5) * REG_W / W1_OUT_W - 0.5,
                                             (oy + 0.5) * REG_H / W1_OUT_H - 0.5, c);
                    out[i * W1_PLANE_FLOATS + (static_cast<size_t>(oy) * W1_OUT_W + ox) * 3 + c] = normalized(v, c);
                }
            }
        }
    }
    return out;
}

// One separable pass (rows, then columns) of the 1D weights wts (centered) over a w x h image of 3 channels, in double.
// clamp: replicating the image edge, the output is w x h; otherwise only where the window is inside the image, the
// output is (w - n + 1) x (h - n + 1)
inline std::vector<double> separablePass(const std::vector<double>& in, const int w, const int h,
                                         const std::vector<double>& wts, const bool clamp, int& ow, int& oh) {
    const int n = static_cast<int>(wts.size());
    const int r = n / 2;
    ow = clamp ? w : w - n + 1;
    oh = clamp ? h : h - n + 1;
    std::vector<double> rows(static_cast<size_t>(ow) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < ow; ++x) {
            for (int c = 0; c < 3; ++c) {
                double sum = 0.0;
                for (int i = 0; i < n; ++i) {
                    const int sx = clamp ? std::clamp(x + i - r, 0, w - 1) : x + i;
                    sum += wts[i] * in[(static_cast<size_t>(y) * w + sx) * 3 + c];
                }
                rows[(static_cast<size_t>(y) * ow + x) * 3 + c] = sum;
            }
        }
    }
    std::vector<double> out(static_cast<size_t>(ow) * oh * 3);
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            for (int c = 0; c < 3; ++c) {
                double sum = 0.0;
                for (int i = 0; i < n; ++i) {
                    const int sy = clamp ? std::clamp(y + i - r, 0, h - 1) : y + i;
                    sum += wts[i] * rows[(static_cast<size_t>(sy) * ow + x) * 3 + c];
                }
                out[(static_cast<size_t>(y) * ow + x) * 3 + c] = sum;
            }
        }
    }
    return out;
}

// W2: the filtered frame (double), computed once per semantics.
// - FKL (both forms): every pass replicates the edge of the image it reads (BoxBVF clamps to the image before it;
//   MorphologyDPP replicates the border of every pass; LinearFilterDPP replicates the frame edge).
// - NPPDx: the ingest replicates the frame edge into the halo, and every step filters the whole tile with its halo, so
//   a second pass reads the first pass's values beyond the frame instead of replicating its edge: the passes run on the
//   frame extended by the total halo, keeping only the values whose window is inside it.
// With one pass the two are the same.
template <typename Spec>
const std::vector<double>& filteredFrame(const Context& ctx, const bool nppdxSemantics) {
    static std::map<bool, std::vector<double>> cache;
    if (cache.count(nppdxSemantics)) {
        return cache.at(nppdxSemantics);
    }
    const auto& rgb = frameRgb(ctx);
    std::vector<double> wts;
    if constexpr (Spec::GAUSSIAN) {
        wts = Spec::taps();
    } else {
        wts.assign(Spec::K, 1.0 / Spec::K);
    }
    std::vector<double> image;
    int w = FRAME_W, h = FRAME_H;
    if (nppdxSemantics) {
        const int pad = Spec::HALO;
        w = FRAME_W + 2 * pad;
        h = FRAME_H + 2 * pad;
        image.resize(static_cast<size_t>(w) * h * 3);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int sx = std::clamp(x - pad, 0, FRAME_W - 1);
                const int sy = std::clamp(y - pad, 0, FRAME_H - 1);
                for (int c = 0; c < 3; ++c) {
                    image[(static_cast<size_t>(y) * w + x) * 3 + c] = rgb[(static_cast<size_t>(sy) * FRAME_W + sx) * 3 + c];
                }
            }
        }
    } else {
        image.assign(rgb.begin(), rgb.end());
    }
    for (int pass = 0; pass < Spec::PASSES; ++pass) {
        int ow = 0, oh = 0;
        image = separablePass(image, w, h, wts, !nppdxSemantics, ow, oh);
        w = ow;
        h = oh;
    }
    if (w != FRAME_W || h != FRAME_H) {
        throw std::runtime_error("The filtered frame has the wrong size");
    }
    cache[nppdxSemantics] = std::move(image);
    return cache.at(nppdxSemantics);
}

// W2: the output plane with FKL's or NPPDx's semantics (the same in every iteration: no regions)
template <typename Spec>
std::vector<float> frameReference(const Context& ctx, const Reference r) {
    if (r != Reference::FklFloat && r != Reference::NppdxPixelCenterFloat) {
        throw std::runtime_error("nppdx-wins has float references only");
    }
    static std::map<Reference, std::vector<float>> cache;
    if (cache.count(r)) {
        return cache.at(r);
    }
    const auto& filtered = filteredFrame<Spec>(ctx, r == Reference::NppdxPixelCenterFloat);
    const auto g = [&](const int x, const int y, const int c) {
        return filtered[(static_cast<size_t>(std::clamp(y, 0, FRAME_H - 1)) * FRAME_W + std::clamp(x, 0, FRAME_W - 1)) *
                            3 + c];
    };
    std::vector<float> out(W2_FLOATS);
    const float fx = fklFactor(FRAME_W, W2_OUT_W);
    const float fy = fklFactor(FRAME_H, W2_OUT_H);
    for (int oy = 0; oy < W2_OUT_H; ++oy) {
        for (int ox = 0; ox < W2_OUT_W; ++ox) {
            for (int c = 0; c < 3; ++c) {
                const double v = r == Reference::FklFloat
                                     ? fklBilinear(g, static_cast<float>(ox) * fx, static_cast<float>(oy) * fy, FRAME_W,
                                                   FRAME_H, c)
                                     : centerBilinear(g, (ox + 0.5) * 2.0 - 0.5, (oy + 0.5) * 2.0 - 0.5, c);
                out[(static_cast<size_t>(oy) * W2_OUT_W + ox) * 3 + c] = normalized(v, c);
            }
        }
    }
    cache[r] = out;
    return out;
}

// ------------------------------------------------------------------------------------------------
// The fastest validated configuration (NPPDx tile and block, MorphologyDPP tile): every candidate is validated on the
// first timed iteration and timed alone (latency, every iteration); the fastest correct one is used. The main run
// writes the choices to choices.csv and every candidate to choice_sweep.csv; the other modes read choices.csv.
// A main run can also take some choices as given (pinned): WINS_PINNED_CHOICES=<file> with lines "key,index" (the
// header line is skipped); those keys are not swept, and choices.csv says "pinned".
// ------------------------------------------------------------------------------------------------
struct Choices {
    std::filesystem::path outDir;
    bool sweep{true};
    std::map<std::string, size_t> chosen;
    std::map<std::string, bool> pinned;
    std::vector<std::string> sweepRows;
};
inline Choices choices;

inline void writeChoices() {
    std::ofstream file(choices.outDir / "choices.csv");
    file << "key,index,how\n";
    for (const auto& [k, index] : choices.chosen) {
        file << k << "," << index << "," << (choices.pinned[k] ? "pinned" : "swept") << "\n";
    }
}

// Reads "key,index[,...]" lines after a header
inline std::map<std::string, size_t> readChoiceFile(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    std::map<std::string, size_t> read;
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::string key, index;
        if (std::getline(row, key, ',') && std::getline(row, index, ',')) {
            read[key] = std::stoul(index);
        }
    }
    return read;
}

inline void setupChoices(const std::filesystem::path& outDir, const std::string& mode) {
    choices.outDir = outDir;
    choices.sweep = mode.empty();
    if (choices.sweep) {
        if (const char* pinned = std::getenv("WINS_PINNED_CHOICES")) {
            for (const auto& [key, index] : readChoiceFile(pinned)) {
                choices.chosen[key] = index;
                choices.pinned[key] = true;
                std::cout << key << ": candidate " << index << " pinned (" << pinned << ")" << std::endl;
            }
            writeChoices();
        }
        return;
    }
    if (!std::filesystem::exists(outDir / "choices.csv")) {
        throw std::runtime_error("Run the main benchmark first: it writes " + (outDir / "choices.csv").string());
    }
    choices.chosen = readChoiceFile(outDir / "choices.csv");
}

template <typename Make, typename Label>
size_t chooseFastest(Context& ctx, const std::string& key, const size_t count, const Make& make, const Label& label,
                     const std::vector<float>& reference, const float tolerance) {
    if (choices.chosen.count(key)) {
        return choices.chosen.at(key);
    }
    if (!choices.sweep) {
        throw std::runtime_error("No choice for " + key + " in " + (choices.outDir / "choices.csv").string());
    }
    size_t best = count;
    float bestMedian = 0.f;
    std::vector<std::string> rows;
    for (size_t i = 0; i < count; ++i) {
        bool launched = false;
        ErrorStats errors;
        TimingStats timing;
        try {
            const Implementation impl = make(i);
            impl.fillNaN();
            gpuErrchk(cudaDeviceSynchronize());
            impl.run(WARMUP_ITERS);
            gpuErrchk(cudaStreamSynchronize(ctx.stream));
            errors = compare(impl.download(), reference);
            timing = computeStats(measureSequential(ctx, {impl})[0]);
            launched = true;
        } catch (const std::exception& e) {
            std::cout << key << ": " << label(i) << " cannot run: " << e.what() << std::endl;
            cudaGetLastError();
        }
        const bool correct = launched && validation::withinTolerance(errors, tolerance);
        std::ostringstream row;
        row << key << "," << label(i) << "," << (launched ? "yes" : "no") << "," << std::fixed << std::setprecision(2)
            << timing.median << "," << std::scientific << std::setprecision(3) << errors.max() << ","
            << (correct ? "yes" : "no");
        rows.push_back(row.str());
        std::cout << key << ": " << label(i) << " median " << std::fixed << std::setprecision(2) << timing.median
                  << " us, max abs error " << std::scientific << std::setprecision(2) << errors.max()
                  << std::defaultfloat << (correct ? "" : " (not within the tolerance)") << std::endl;
        if (correct && (best == count || timing.median < bestMedian)) {
            best = i;
            bestMedian = timing.median;
        }
    }
    if (best == count) {
        throw std::runtime_error("No configuration of " + key + " is within the tolerance");
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        choices.sweepRows.push_back(rows[i] + (i == best ? ",yes" : ",no"));
    }
    choices.chosen[key] = best;
    writeChoices();
    std::ofstream sweep(choices.outDir / "choice_sweep.csv");
    sweep << "key,candidate,launched,median_us,max_abs_err,within_tolerance,chosen\n";
    for (const auto& r : choices.sweepRows) {
        sweep << r << "\n";
    }
    std::cout << key << ": " << label(best) << " chosen" << std::endl;
    return best;
}

// Calls f.template operator()<CONFIGS[index]>()
template <const auto& CONFIGS, typename F>
void withConfig(const size_t index, F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((Is == index ? f.template operator()<CONFIGS[Is]>() : void()), ...);
    }(std::make_index_sequence<CONFIGS.size()>{});
}

// ------------------------------------------------------------------------------------------------
// NPPDx
// ------------------------------------------------------------------------------------------------

struct Normalization {
    float mul[3];
    float sub[3];
    float div[3];
};
inline Normalization normalization() {
    return Normalization{{MUL[0], MUL[1], MUL[2]}, {SUB[0], SUB[1], SUB[2]}, {DIV[0], DIV[1], DIV[2]}};
}

struct NppdxConfig {
    unsigned tileX, tileY, block;
};
inline std::string configLabel(const NppdxConfig& c) {
    return "tile " + std::to_string(c.tileX) + "x" + std::to_string(c.tileY) + " block " +
           (c.block == 0 ? std::string("recommended") : std::to_string(c.block));
}

// Explicit 1D block size (BlockDim<X>), or 0 for NPPDx's recommended one
template <unsigned BlockX>
using NppdxBlock = std::conditional_t<BlockX == 0, decltype(nppdx::Block()),
                                      decltype(nppdx::Block() + nppdx::BlockDim<BlockX>())>;

#define WINS_NPPDX_LAYOUT(TX, TY, MH, CH, BX)                                                                         \
    nppdx::TileSize<TX, TY>() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, MH)() +                                             \
        NPPDX_MAKE_HALO(nppdx::CumulativeHalo, CH)() + nppdx::SM<NPPDX_SM>() + NppdxBlock<BX>()

// The filter before the resize, for NPPDx and the references: a K x K box in PASSES passes (the second NPPDx BoxBlur
// step reads the first one's output with the halo that is left), or NPPDx's GaussianBlur<RT, TAIL> (separable FIR)
template <int K_, int PASSES_>
struct BoxSpec {
    static constexpr bool GAUSSIAN = false;
    static constexpr int K = K_;
    static constexpr int PASSES = PASSES_;
    static constexpr int STEP_HALO = K_ / 2;
    static constexpr int HALO = STEP_HALO * PASSES_;
    using Operator = nppdx::BoxBlur<K_, K_>;
    static constexpr nppdx::function FUNCTION = nppdx::function::box_blur;
};
template <int RT, nppdx::gaussian_tail_width TAIL>
struct GaussianSpec {
    static constexpr bool GAUSSIAN = true;
    // NPPDx's bands (operators/function.hpp, tap_configs): the standard tap count and the first RadiusTenths of the band
    // of RT, written out here: calling NPPDx's get_tap_config in this program made nvcc emit its device version, which
    // reads a host-only table
    static constexpr int BAND_TAPS = RT <= 9 ? 3 : RT <= 20 ? 5 : 7 + 2 * ((RT - 21) / 10);
    static constexpr int BAND_MIN = RT <= 9 ? 1 : RT <= 20 ? 10 : 21 + 10 * ((RT - 21) / 10);
    static constexpr int K = TAIL == nppdx::gaussian_tail_width::wide ? 2 * BAND_TAPS - 1 : BAND_TAPS;
    static constexpr int ROW = RT - BAND_MIN;
    static constexpr int PASSES = 1;
    static constexpr int STEP_HALO = K / 2;
    static constexpr int HALO = STEP_HALO;
    using Operator = nppdx::GaussianBlur<RT, TAIL>;
    static constexpr nppdx::function FUNCTION = nppdx::function::gaussian_blur;
    // NPPDx's taps (detail/backend/gaussian_blur_coeff_tables.hpp), for FKL's coefficients and the references
    static std::vector<double> taps() {
        const auto& row = nppdx::detail::backend::gaussian_blur::gaussian_fir_1d_weights<K, TAIL>[ROW];
        return std::vector<double>(row.begin(), row.end());
    }
};

// The NPPDx operations of one TX x TY tile: NV12 ingest with the halo of the filter and the resize, the conversion,
// the filter, Resize<J, K, bilinear> (as fused_resize.cu and fkl-vs-nppdx/benchmarks/nppdx_common.h)
template <unsigned TX_, unsigned TY_, unsigned BlockX, unsigned J, unsigned K, typename Spec>
struct FilterResizeOps {
    static constexpr unsigned TX = TX_;
    static constexpr unsigned TY = TY_;
    using ResizePre = decltype(nppdx::Function<nppdx::function::resize>() +
                               nppdx::Resize<J, K, nppdx::interpolation_method::bilinear>() +
                               nppdx::TileSize<TX, TY>() + nppdx::SM<NPPDX_SM>() + NppdxBlock<BlockX>());
    static constexpr nppdx::Halo4 resizeHalo = ResizePre::local_halo;
    static constexpr nppdx::Halo4 total = nppdx::Halo4::make_uniform(Spec::HALO) + resizeHalo;
    // The halo left in the input of the second pass (with one pass, Blur2 is Blur and is not run)
    static constexpr nppdx::Halo4 second =
        Spec::PASSES == 2 ? nppdx::Halo4::make_uniform(Spec::HALO - Spec::STEP_HALO) + resizeHalo : total;
    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::nv12>() +
                            WINS_NPPDX_LAYOUT(TX, TY, total, total, BlockX));
    using FromLimitedY = decltype(nppdx::Function<nppdx::function::affine_channel_map>() +
                                  nppdx::FromLimitedLuma<nppdx::bit_depth::bpp_8u>() +
                                  WINS_NPPDX_LAYOUT(TX, TY, total, total, BlockX));
    using FromLimitedUV = decltype(nppdx::Function<nppdx::function::affine_channel_map>() +
                                   nppdx::FromLimitedChroma<nppdx::bit_depth::bpp_8u>() +
                                   WINS_NPPDX_LAYOUT(TX, TY, total, total, BlockX));
    using Convert = decltype(nppdx::Function<nppdx::function::color_convert>() +
                             nppdx::ColorConvert<nppdx::color_space::yuv_bt601, nppdx::color_space::rgb,
                                                 nppdx::bit_depth::bpp_8u, nppdx::bit_depth::bpp_8u>() +
                             WINS_NPPDX_LAYOUT(TX, TY, total, total, BlockX));
    using Blur = decltype(nppdx::Function<Spec::FUNCTION>() + typename Spec::Operator() +
                          WINS_NPPDX_LAYOUT(TX, TY, total, total, BlockX));
    using Blur2 = decltype(nppdx::Function<Spec::FUNCTION>() + typename Spec::Operator() +
                           WINS_NPPDX_LAYOUT(TX, TY, total, second, BlockX));
    using Resize = decltype(ResizePre() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, total)() +
                            NPPDX_MAKE_HALO(nppdx::CumulativeHalo, resizeHalo)());
    using InTile = nppdx::input_storage_of_t<Ingest>;
    using BlurTile = nppdx::output_storage_of_t<Blur>;
    using TmpTile = nppdx::temp_storage_of_t<Resize>;
    using OutTile = nppdx::output_storage_of_t<Resize>;
    static constexpr unsigned OX = OutTile::ChannelSliceType::width;
    static constexpr unsigned OY = OutTile::ChannelSliceType::height;
    static constexpr size_t smem = nppdx::shared_memory::compute_total_tile_storage<InTile, BlurTile, TmpTile, OutTile>();
    static constexpr unsigned block = std::max({Ingest::block_dim.x, FromLimitedY::block_dim.x,
                                                FromLimitedUV::block_dim.x, Convert::block_dim.x, Blur::block_dim.x,
                                                Blur2::block_dim.x, Resize::block_dim.x});

    template <typename Image, typename Tiles>
    __device__ static void run(const Image& image, Tiles& tiles, const int w, const int h) {
        auto& in = std::get<0>(tiles);
        Ingest().execute(image, in, w, h);
        FromLimitedY().execute(in, w, h);
        FromLimitedUV().execute(in, w, h);
        Convert().execute(in, w, h);
        Blur().execute(in, std::get<1>(tiles), w, h);
        if constexpr (Spec::PASSES == 2) {
            // The second pass writes into the input tile, which is no longer needed
            Blur2().execute(std::get<1>(tiles), in, w, h);
            Resize().execute(in, std::get<2>(tiles), std::get<3>(tiles), w, h);
        } else {
            Resize().execute(std::get<1>(tiles), std::get<2>(tiles), std::get<3>(tiles), w, h);
        }
        __syncthreads();
    }
};
static_assert(BoxSpec<5, 2>::HALO == 4 && BoxSpec<9, 1>::HALO == 4, "Box halos");

// W1's operations: the 5x5 box, one pass
template <unsigned TX, unsigned TY, unsigned BlockX, unsigned J, unsigned K>
using BoxResizeOps = FilterResizeOps<TX, TY, BlockX, J, K, BoxSpec<BOX_K, 1>>;

template <typename Ops>
__device__ auto sliceTiles() {
    extern __shared__ unsigned char smem[];
    return nppdx::shared_memory::slice_into_tile_storage<typename Ops::InTile, typename Ops::BlurTile,
                                                         typename Ops::TmpTile, typename Ops::OutTile>(smem);
}

// The normalization and float3 write of the resized tile at (x0, y0) of a w x h plane, written by hand
template <typename Ops, typename Tile>
__device__ void writeTile(const Tile& tile, const Normalization& n, float* plane, const int x0, const int y0,
                          const int w, const int h) {
    const int threads = blockDim.x * blockDim.y * blockDim.z;
    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    for (int i = tid; i < static_cast<int>(Ops::OX * Ops::OY); i += threads) {
        const int tx = i % Ops::OX;
        const int ty = i / Ops::OX;
        const int x = x0 + tx;
        const int y = y0 + ty;
        if (x < w && y < h) {
            float* pixel = plane + (static_cast<size_t>(y) * w + x) * 3;
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                pixel[c] = ((tile.channel(c)(ty, tx) * n.mul[c]) - n.sub[c]) / n.div[c];
            }
        }
    }
}

template <size_t N>
struct Origins {
    int2 origin[N];
};

// W1 in frame mode (exp3-exploration/box-followup.md, section 1): NPPDx takes a tile's position from blockIdx and the
// image base, with no offset parameter, and clamps its reads to the image. So the image base is moved back by one tile
// and the image is the region plus one tile on every side; the grid is one tile larger and its first row and column
// of blocks return at once. Every other block processes the tile of the region it would without the shift, and its
// halo is read from the frame pixels around the region (no read is clamped). The moved base is outside the frame for
// regions near its top or left edge, but no block reads before the region's halo.
template <typename Ops, size_t N>
__global__ void nppdxRegionsKernel(const uint8_t* y, const uint8_t* uv, const int pitch, const Origins<N> origins,
                                   const Normalization n, float* out) {
    if (blockIdx.x == 0 || blockIdx.y == 0) {
        return;
    }
    auto tiles = sliceTiles<Ops>();
    const int2 o = origins.origin[blockIdx.z];
    const int baseX = o.x - static_cast<int>(Ops::TX);
    const int baseY = o.y - static_cast<int>(Ops::TY);
    const nppdx::ImageLayout<const uint8_t, 2> image{
        {{y + static_cast<ptrdiff_t>(baseY) * pitch + baseX, uv + static_cast<ptrdiff_t>(baseY / 2) * pitch + baseX}},
        {{pitch, pitch}}};
    Ops::run(image, tiles, REG_W + 2 * static_cast<int>(Ops::TX), REG_H + 2 * static_cast<int>(Ops::TY));
    writeTile<Ops>(std::get<3>(tiles), n, out + static_cast<size_t>(blockIdx.z) * W1_PLANE_FLOATS,
                   (static_cast<int>(blockIdx.x) - 1) * static_cast<int>(Ops::OX),
                   (static_cast<int>(blockIdx.y) - 1) * static_cast<int>(Ops::OY), W1_OUT_W, W1_OUT_H);
}

// W2: the frame is the image (its edge replicated into the halo), one block per tile
template <typename Ops>
__global__ void nppdxFrameKernel(const uint8_t* y, const uint8_t* uv, const int pitch, const Normalization n,
                                 float* out) {
    auto tiles = sliceTiles<Ops>();
    const nppdx::ImageLayout<const uint8_t, 2> image{{{y, uv}}, {{pitch, pitch}}};
    Ops::run(image, tiles, FRAME_W, FRAME_H);
    writeTile<Ops>(std::get<3>(tiles), n, out, static_cast<int>(blockIdx.x * Ops::OX),
                   static_cast<int>(blockIdx.y * Ops::OY), W2_OUT_W, W2_OUT_H);
}

template <auto Kernel, size_t SMEM>
void allowSharedMemory() {
    if constexpr (SMEM > 48 * 1024) {
        gpuErrchk(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    }
}

// ------------------------------------------------------------------------------------------------
// FKL
// ------------------------------------------------------------------------------------------------

using Bilinear = fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>;

inline auto normalizedWrite(const fk::Tensor<fk::float3>& out) {
    return fk::Mul<fk::float3>::build(mulValue)
        .then(fk::Sub<fk::float3>::build(subValue))
        .then(fk::Div<fk::float3>::build(divValue))
        .then(fk::TensorWrite<fk::float3>::build(out.ptr()));
}

inline auto rgbRead(const Context& ctx) {
    return fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice).then(YUVToRGB::build());
}

// The MorphologyDPP tiles fkl_staged is compiled for (TW x TH threads per block), as Experiment 3
struct DppTile {
    int width, height;
};
constexpr std::array<DppTile, 3> DPP_TILES{{{16, 8}, {32, 8}, {32, 16}}};
inline std::string dppLabel(const size_t i) {
    return "DPP tile " + std::to_string(DPP_TILES[i].width) + "x" + std::to_string(DPP_TILES[i].height);
}

template <size_t N>
void setFloatIO(Implementation& impl, const std::shared_ptr<fk::Tensor<fk::float3>>& out, const size_t floats) {
    impl.download = [out, floats]() { return downloadDevice(out->ptr().data, floats); };
    impl.fillNaN = [out, floats]() { gpuErrchk(cudaMemset(out->ptr().data, 0xFF, floats * sizeof(float))); };
}

} // namespace bench::wins

#endif // BVF_BENCH_WINS_COMMON_H
