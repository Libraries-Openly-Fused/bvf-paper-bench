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

// nppdx-wins, W2 (wins_common.h): ReadNV12 -> YUVtoRGB -> filter -> Resize 2:1 to 1920x1080 -> Mul -> Sub -> Div ->
// one float3 plane, the whole 3840x2160 frame (batch "1": one frame; the harness's regions are not used). nppdx,
// fkl_inline (and fkl_staged_separable for the Gaussian) are the baselines of fkl_staged in the harness's files; every
// median is in latency.csv, gpu_time.csv and gpu_time_l2_flushed.csv. The filter (CMake target):
//   WINS_GAUSSIAN 0   a WINS_BOX_K x WINS_BOX_K box in WINS_PASSES passes (benchmark_wins_frame: 5x5, one pass;
//                     _k9: 9x9; _k5_p2: 5x5 twice). fkl_inline: BoxBVF<K> per pass; fkl_staged: executeBoxBlur<PASSES>
//   WINS_GAUSSIAN 1   NPPDx's GaussianBlur<50, wide>, 21 taps, sigma 5 (benchmark_wins_frame_gauss21). fkl_inline:
//                     GaussianBVF<21> with NPPDx's taps (common/bench_neighborhood_ops.h); fkl_staged: LinearFilterDPP
//                     21x21 with the products of NPPDx's taps; and fkl_staged_separable, LinearFilterDPP 21x1 then 1x21
//                     (three kernels)

#include "wins_common.h"

#include <iostream>

#ifndef WINS_GAUSSIAN
#define WINS_GAUSSIAN 0
#endif
#ifndef WINS_BOX_K
#define WINS_BOX_K 5
#endif
#ifndef WINS_PASSES
#define WINS_PASSES 1
#endif

namespace bench::wins {

#if WINS_GAUSSIAN
using Spec = GaussianSpec<50, nppdx::gaussian_tail_width::wide>;
static_assert(Spec::K == 21 && Spec::K == nppdx::GaussianBlur<50, nppdx::gaussian_tail_width::wide>::value.taps(),
              "GaussianSpec's tap count is NPPDx's");
#else
using Spec = BoxSpec<WINS_BOX_K, WINS_PASSES>;
#endif
static_assert(Spec::PASSES == 1 || Spec::PASSES == 2, "One or two passes");

inline std::string filterName() {
    if constexpr (Spec::GAUSSIAN) {
        return "Gaussian, NPPDx's GaussianBlur<50, wide> (21 taps, sigma 5)";
    } else {
        return "box " + std::to_string(Spec::K) + "x" + std::to_string(Spec::K) +
               (Spec::PASSES == 2 ? " in two passes" : "");
    }
}

constexpr std::array<NppdxConfig, 20> CONFIGS{{{64, 32, 0}, {64, 32, 128}, {64, 32, 256}, {64, 32, 512},
                                               {32, 32, 0}, {32, 32, 128}, {32, 32, 256}, {32, 32, 512},
                                               {32, 16, 0}, {32, 16, 128}, {32, 16, 256}, {32, 16, 512},
                                               {16, 16, 0}, {16, 16, 128}, {16, 16, 256}, {16, 16, 512},
                                               {16, 8, 0}, {16, 8, 128}, {16, 8, 256}, {16, 8, 512}}};

std::shared_ptr<fk::Tensor<fk::float3>> outputPlane() {
    return std::make_shared<fk::Tensor<fk::float3>>(W2_OUT_W, W2_OUT_H, 1, 1, fk::MemType::Device);
}

// NPPDx: one kernel over the frame
template <NppdxConfig C>
Implementation makeNppdx(Context& ctx) {
    using Ops = FilterResizeOps<C.tileX, C.tileY, C.block, 2, 1, Spec>;
    constexpr auto kernel = nppdxFrameKernel<Ops>;
    allowSharedMemory<kernel, Ops::smem>();
    auto out = outputPlane();
    const Normalization n = normalization();
    Implementation impl{"nppdx", Reference::NppdxPixelCenterFloat};
    impl.stages.emplace_back(Spec::GAUSSIAN ? "1_nppdx_gaussian_resize_normalize" : "1_nppdx_box_resize_normalize",
                             [&ctx, out, n](size_t) {
        const dim3 grid{(FRAME_W + Ops::TX - 1) / Ops::TX, (FRAME_H + Ops::TY - 1) / Ops::TY, 1};
        beforeKernel();
        kernel<<<grid, dim3(Ops::block), Ops::smem, ctx.stream>>>(ctx.y, ctx.uv, ctx.pitch, n,
                                                                  reinterpret_cast<float*>(out->ptr().data));
        gpuErrchk(cudaGetLastError());
    });
    impl.run = runStages(impl.stages);
    setFloatIO<1>(impl, out, W2_FLOATS);
    return impl;
}

// FKL, the filter as a BVF operation before the resize (one BoxBVF per pass, or GaussianBVF with NPPDx's taps): one
// kernel, 4 x (K^2)^PASSES frame evaluations per output value
template <typename S = Spec>
Implementation makeFklInline(Context& ctx) {
    auto out = outputPlane();
    // The Gaussian's taps (NPPDx's), the parameter of GaussianBVF; unused by the box
    fk::Array<float, S::K> taps{};
    if constexpr (S::GAUSSIAN) {
        const auto values = S::taps();
        for (int i = 0; i < S::K; ++i) {
            taps[i] = static_cast<float>(values[i]);
        }
    }
    Implementation impl{"fkl_inline", Reference::FklFloat};
    impl.stages.emplace_back(S::GAUSSIAN ? "1_gaussian_resize_normalize" : "1_box_resize_normalize",
                             [&ctx, out, taps](size_t) {
        beforeKernel();
        if constexpr (S::GAUSSIAN) {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                                                      fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                                                      YUVToRGB::build(), fk::GaussianBVF<S::K>::build(taps),
                                                      Bilinear::build(fk::Size(W2_OUT_W, W2_OUT_H)),
                                                      normalizedWrite(*out));
        } else if constexpr (S::PASSES == 1) {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                                                      fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                                                      YUVToRGB::build(), fk::BoxBVF<S::K>::build(),
                                                      Bilinear::build(fk::Size(W2_OUT_W, W2_OUT_H)),
                                                      normalizedWrite(*out));
        } else {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                                                      fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                                                      YUVToRGB::build(), fk::BoxBVF<S::K>::build(),
                                                      fk::BoxBVF<S::K>::build(),
                                                      Bilinear::build(fk::Size(W2_OUT_W, W2_OUT_H)),
                                                      normalizedWrite(*out));
        }
    });
    impl.run = runStages(impl.stages);
    setFloatIO<1>(impl, out, W2_FLOATS);
    return impl;
}

// FKL's 21x21 coefficients: the products of NPPDx's taps (Gaussian only)
template <typename S>
std::shared_ptr<fk::Ptr2D<fk::float3>> gaussianCoefficients(Context& ctx) {
    static std::shared_ptr<fk::Ptr2D<fk::float3>> coefficients;
    if (!coefficients) {
        const auto taps = S::taps();
        const int n = static_cast<int>(taps.size());
        coefficients = std::make_shared<fk::Ptr2D<fk::float3>>(n, n);
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                coefficients->at(fk::Point{x, y, 0}) = fk::make_set<fk::float3>(static_cast<float>(taps[y] * taps[x]));
            }
        }
        coefficients->upload(*ctx.fkStream);
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
    }
    return coefficients;
}

// FKL staged: the DPP of the filter (MorphologyDPP's box, or LinearFilterDPP for the Gaussian) of the frame into a
// float3 frame in memory (3840x2160x12 bytes), then one TransformDPP kernel Resize -> normalize reading it
template <int TW, int TH>
Implementation makeFklStaged(Context& ctx) {
    auto inter = std::make_shared<fk::Tensor<fk::float3>>(FRAME_W, FRAME_H, 1, 1, fk::MemType::Device);
    auto out = outputPlane();
    const fk::RawPtr<fk::ND::_2D, fk::float3> frame{inter->ptr().data,
                                                     {FRAME_W, FRAME_H, FRAME_W * sizeof(fk::float3)}};
    Implementation impl{"fkl_staged", Reference::FklFloat};
    if constexpr (Spec::GAUSSIAN) {
        const auto coefficients = gaussianCoefficients<Spec>(ctx);
        impl.stages.emplace_back("1_linear_filter_dpp_of_the_frame", [&ctx, inter, frame, coefficients](size_t) {
            using Details = fk::LinearFilterDPPDetails<fk::float3, TW, TH, Spec::K, Spec::K>;
            const Details details{FRAME_W, FRAME_H, Spec::K, Spec::K, Spec::STEP_HALO, Spec::STEP_HALO};
            beforeKernel();
            if (!fk::executeLinearFilter(details,
                                         fk::make_tuple(rgbRead(ctx), fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(
                                                                          coefficients->ptr())),
                                         fk::Mul<fk::float3, fk::float3, fk::float3, fk::UnaryType>::build(),
                                         fk::Add<fk::float3, fk::float3, fk::float3, fk::UnaryType>::build(),
                                         fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(frame), *ctx.fkStream)) {
                throw std::runtime_error("executeLinearFilter rejected its details");
            }
        });
    } else {
        impl.stages.emplace_back("1_morphology_dpp_box_of_the_frame", [&ctx, inter, frame](size_t) {
            using Details = fk::MorphologyDPPDetails<fk::float3, TW, TH, Spec::K, Spec::K>;
            const Details details{FRAME_W, FRAME_H, Spec::K, Spec::K, Spec::STEP_HALO, Spec::STEP_HALO};
            beforeKernel();
            if (!fk::executeBoxBlur<Spec::PASSES>(details, rgbRead(ctx),
                                                  fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(frame),
                                                  *ctx.fkStream)) {
                throw std::runtime_error("executeBoxBlur rejected its details");
            }
        });
    }
    impl.stages.emplace_back("2_resize_normalize", [&ctx, out, frame](size_t) {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                                                  fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(frame),
                                                  Bilinear::build(fk::Size(W2_OUT_W, W2_OUT_H)),
                                                  normalizedWrite(*out));
    });
    impl.run = runStages(impl.stages);
    setFloatIO<1>(impl, out, W2_FLOATS);
    return impl;
}

inline Implementation nppdx(Context& ctx) {
    const auto reference = frameReference<Spec>(ctx, Reference::NppdxPixelCenterFloat);
    const size_t index = chooseFastest(
        ctx, "nppdx_frame", CONFIGS.size(),
        [&](size_t i) {
            Implementation impl;
            withConfig<CONFIGS>(i, [&]<NppdxConfig C>() { impl = makeNppdx<C>(ctx); });
            return impl;
        },
        [](size_t i) { return configLabel(CONFIGS[i]); }, reference, TOLERANCE);
    Implementation impl;
    withConfig<CONFIGS>(index, [&]<NppdxConfig C>() { impl = makeNppdx<C>(ctx); });
    return impl;
}

// The Gaussian with FKL's DPPs used separably (not one of the two forms asked for: LinearFilterDPP has no separable
// path, so this is the best FKL can do with existing components): LinearFilterDPP 21x1 of the frame into a float3
// frame, LinearFilterDPP 1x21 of it into a second one, then TransformDPP Resize -> normalize: three kernels, two
// intermediates of 3840x2160x12 bytes
template <int TW, int TH, typename S = Spec>
Implementation makeFklStagedSeparable(Context& ctx) {
    static_assert(S::GAUSSIAN, "The separable form is the Gaussian's");
    auto rowsPassed = std::make_shared<fk::Tensor<fk::float3>>(FRAME_W, FRAME_H, 1, 1, fk::MemType::Device);
    auto filtered = std::make_shared<fk::Tensor<fk::float3>>(FRAME_W, FRAME_H, 1, 1, fk::MemType::Device);
    auto out = outputPlane();
    const auto view = [](const std::shared_ptr<fk::Tensor<fk::float3>>& t) {
        return fk::RawPtr<fk::ND::_2D, fk::float3>{t->ptr().data, {FRAME_W, FRAME_H, FRAME_W * sizeof(fk::float3)}};
    };
    const auto rowsView = view(rowsPassed);
    const auto filteredView = view(filtered);
    // The 1D taps as a 21x1 and a 1x21 coefficient image
    const auto taps = S::taps();
    auto rowTaps = std::make_shared<fk::Ptr2D<fk::float3>>(Spec::K, 1);
    auto columnTaps = std::make_shared<fk::Ptr2D<fk::float3>>(1, Spec::K);
    for (int i = 0; i < Spec::K; ++i) {
        rowTaps->at(fk::Point{i, 0, 0}) = fk::make_set<fk::float3>(static_cast<float>(taps[i]));
        columnTaps->at(fk::Point{0, i, 0}) = fk::make_set<fk::float3>(static_cast<float>(taps[i]));
    }
    rowTaps->upload(*ctx.fkStream);
    columnTaps->upload(*ctx.fkStream);
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    using Mul3 = fk::Mul<fk::float3, fk::float3, fk::float3, fk::UnaryType>;
    using Add3 = fk::Add<fk::float3, fk::float3, fk::float3, fk::UnaryType>;
    Implementation impl{"fkl_staged_separable", Reference::FklFloat};
    impl.stages.emplace_back("1_linear_filter_dpp_rows", [&ctx, rowsPassed, rowsView, rowTaps](size_t) {
        using Details = fk::LinearFilterDPPDetails<fk::float3, TW, TH, Spec::K, 1>;
        const Details details{FRAME_W, FRAME_H, Spec::K, 1, Spec::STEP_HALO, 0};
        beforeKernel();
        if (!fk::executeLinearFilter(details,
                                     fk::make_tuple(rgbRead(ctx),
                                                    fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(rowTaps->ptr())),
                                     Mul3::build(), Add3::build(),
                                     fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(rowsView), *ctx.fkStream)) {
            throw std::runtime_error("executeLinearFilter rejected its details");
        }
    });
    impl.stages.emplace_back("2_linear_filter_dpp_columns",
                             [&ctx, rowsPassed, filtered, rowsView, filteredView, columnTaps](size_t) {
        using Details = fk::LinearFilterDPPDetails<fk::float3, TW, TH, 1, Spec::K>;
        const Details details{FRAME_W, FRAME_H, 1, Spec::K, 0, Spec::STEP_HALO};
        beforeKernel();
        if (!fk::executeLinearFilter(details,
                                     fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(rowsView),
                                                    fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(
                                                        columnTaps->ptr())),
                                     Mul3::build(), Add3::build(),
                                     fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(filteredView),
                                     *ctx.fkStream)) {
            throw std::runtime_error("executeLinearFilter rejected its details");
        }
    });
    impl.stages.emplace_back("3_resize_normalize", [&ctx, filtered, filteredView, out](size_t) {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                                                  fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(filteredView),
                                                  Bilinear::build(fk::Size(W2_OUT_W, W2_OUT_H)),
                                                  normalizedWrite(*out));
    });
    impl.run = runStages(impl.stages);
    setFloatIO<1>(impl, out, W2_FLOATS);
    return impl;
}

template <typename S = Spec>
Implementation fklStagedSeparable(Context& ctx) {
    const auto reference = frameReference<S>(ctx, Reference::FklFloat);
    const auto make = [&](size_t i) {
        Implementation impl;
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((Is == i ? (impl = makeFklStagedSeparable<DPP_TILES[Is].width, DPP_TILES[Is].height, S>(ctx), 0) : 0),
             ...);
        }(std::make_index_sequence<DPP_TILES.size()>{});
        return impl;
    };
    return make(chooseFastest(ctx, "fkl_staged_separable_frame", DPP_TILES.size(), make, dppLabel, reference,
                              TOLERANCE));
}

inline Implementation fklStaged(Context& ctx) {
    const auto reference = frameReference<Spec>(ctx, Reference::FklFloat);
    const auto make = [&](size_t i) {
        Implementation impl;
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((Is == i ? (impl = makeFklStaged<DPP_TILES[Is].width, DPP_TILES[Is].height>(ctx), 0) : 0), ...);
        }(std::make_index_sequence<DPP_TILES.size()>{});
        return impl;
    };
    return make(chooseFastest(ctx, "fkl_staged_frame", DPP_TILES.size(), make, dppLabel, reference, TOLERANCE));
}

struct WinsFrame {
    static constexpr const char* title = "nppdx-wins W2: a filter before a 2:1 resize of the 4K frame (smoke test)";
    static constexpr const char* fklName = "fkl_staged";
    static constexpr const char* fklLabel = "FKL staged";
    static constexpr const char* fklShortLabel = "staged";
    static constexpr std::array<size_t, 1> batches{1};

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        static_assert(N == 1, "One frame");
        std::vector<Implementation> impls;
        impls.push_back(nppdx(ctx));
        impls.push_back(makeFklInline(ctx));
        if constexpr (Spec::GAUSSIAN) {
            impls.push_back(fklStagedSeparable(ctx));
        }
        return impls;
    }

    template <size_t N>
    static Implementation fklImplementation(Context& ctx) {
        return fklStaged(ctx);
    }

    template <size_t N>
    static std::vector<float> reference(const Context& ctx, Reference r, size_t) {
        return frameReference<Spec>(ctx, r);
    }

    static void setup(const std::filesystem::path& outDir, const std::string& mode) { setupChoices(outDir, mode); }

    static void workload(std::ostream& os) {
        os << "bvf-paper-bench: commit " << BENCH_REPO_COMMIT << "\n"
           << "Workload (nppdx-wins W2, smoke test): the " << FRAME_W << "x" << FRAME_H << " NV12 frame (seed "
           << FRAME_SEED << "), NV12->RGB BT.601 limited range, " << filterName() << " (per channel, replicating the "
           << "frame edge; with two passes FKL replicates the edge of every pass and NPPDx filters the extended frame), "
           << "resized 2:1 (bilinear) to " << W2_OUT_W << "x" << W2_OUT_H << ", then Mul (1/255), Sub (0.485, "
           << "0.456, 0.406), Div (0.229, 0.224, 0.225), one float3 plane. Batch 1 is the frame (the harness's "
           << "regions are not used).\n"
           << "Semantics (documented, not matched): FKL's 2:1 bilinear samples at (2x, 2y), i.e. the boxed pixel; "
           << "NPPDx's at pixel centers, the mean of the 2x2 boxed pixels. Each is checked against a reference of its "
           << "own semantics.\n";
    }

    static void describe(std::ostream& os, const std::filesystem::path&) {
        os << "NPPDx: " << NPPDX_GIT_VERSION << ", operations for SM " << NPPDX_SM << "\n"
           << "Filter: " << filterName() << "\n"
           << "nppdx: one kernel over the frame (wins_common.h): NV12 ingest of the tile with the halo of the filter "
              "and the resize, FromLimitedLuma/Chroma, ColorConvert BT.601, the filter (BoxBlur steps or "
              "GaussianBlur), Resize<2, 1, bilinear>, the "
              "normalization and float3 write written by hand; the fastest validated of tiles 64x32, 32x32, 32x16, "
              "16x16, 16x8 with the recommended block and 128, 256, 512 (choices.csv, choice_sweep.csv).\n"
           << "fkl_inline: one kernel, ReadNV12 -> YUVtoRGB -> BoxBVF<K> per pass (GaussianBVF<21> with NPPDx's taps "
              "for the Gaussian) -> Resize -> Mul -> Sub -> Div -> TensorWrite (4 x (K^2)^passes NV12 reads and "
              "conversions per output value).\n"
           << "fkl_staged: two kernels, the DPP of the filter (MorphologyDPP, executeBoxBlur; LinearFilterDPP 21x21 "
              "for the Gaussian) of the frame into a float3 frame in memory "
              "(" << static_cast<size_t>(FRAME_W) * FRAME_H * 12 << " bytes written, then read by the resize), then "
              "TransformDPP Resize -> Mul -> Sub -> Div -> TensorWrite; the fastest validated DPP tile of 16x8, 32x8, "
              "32x16.\n";
        if constexpr (Spec::GAUSSIAN) {
            os << "fkl_staged_separable (an extra, not one of the two FKL forms): three kernels, LinearFilterDPP 21x1 "
                  "of the frame into a float3 frame, LinearFilterDPP 1x21 of it into a second one (two intermediates "
                  "of " << static_cast<size_t>(FRAME_W) * FRAME_H * 12 << " bytes), then TransformDPP Resize -> Mul -> "
                  "Sub -> Div -> TensorWrite; the fastest validated LinearFilterDPP tile of 16x8, 32x8, 32x16.\n";
        }
    }

    static std::string diagnostic(const Context&) { return std::string(); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench::wins

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::wins::WinsFrame>(argc, argv);
}
