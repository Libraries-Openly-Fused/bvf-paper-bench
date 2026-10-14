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

// nppdx-wins, W1 (wins_common.h): ReadNV12 -> YUVtoRGB -> Box 5x5 over the frame -> Crop 32x16 -> Resize 1:2 to 64x32
// -> Mul -> Sub -> Div -> N float3 planes, at Experiment 1's positions (BENCH_REGIONS=32x16). nppdx and fkl_inline are
// the baselines of fkl_staged in the harness's files; every median is in latency.csv, gpu_time.csv and
// gpu_time_l2_flushed.csv.

#include "wins_common.h"

#include <iostream>

namespace bench::wins {

static_assert(geometry::REGION_SIZE_COUNT == 1 && geometry::REGION_SIZES[0].width == REG_W &&
                  geometry::REGION_SIZES[0].height == REG_H,
              "W1 is built for 32x16 regions (BENCH_REGIONS=32x16)");

constexpr std::array<NppdxConfig, 9> CONFIGS{{{32, 16, 0}, {32, 16, 128}, {32, 16, 256},
                                              {16, 16, 0}, {16, 16, 128}, {16, 16, 256},
                                              {16, 8, 0}, {16, 8, 128}, {16, 8, 256}}};

// NPPDx in frame mode, one kernel for the N regions
template <size_t N, NppdxConfig C>
Implementation makeNppdx(Context& ctx) {
    using Ops = BoxResizeOps<C.tileX, C.tileY, C.block, 1, 2>;
    constexpr auto kernel = nppdxRegionsKernel<Ops, N>;
    allowSharedMemory<kernel, Ops::smem>();
    auto out = std::make_shared<fk::Tensor<fk::float3>>(W1_OUT_W, W1_OUT_H, N, 1, fk::MemType::Device);
    auto origins = std::make_shared<std::vector<Origins<N>>>(ctx.regions->iterations());
    for (size_t t = 0; t < origins->size(); ++t) {
        for (size_t i = 0; i < N; ++i) {
            const Region& r = ctx.regions->at(t, i);
            (*origins)[t].origin[i] = int2{static_cast<int>(r.x), static_cast<int>(r.y)};
        }
    }
    const Normalization n = normalization();
    Implementation impl{"nppdx", Reference::NppdxPixelCenterFloat};
    impl.stages.emplace_back("1_nppdx_box_resize_normalize", [&ctx, out, origins, n](size_t t) {
        const dim3 grid{REG_W / Ops::TX + 1, REG_H / Ops::TY + 1, static_cast<unsigned>(N)};
        beforeKernel();
        kernel<<<grid, dim3(Ops::block), Ops::smem, ctx.stream>>>(ctx.y, ctx.uv, ctx.pitch, (*origins)[t], n,
                                                                  reinterpret_cast<float*>(out->ptr().data));
        gpuErrchk(cudaGetLastError());
    });
    impl.run = runStages(impl.stages);
    setFloatIO<N>(impl, out, N * W1_PLANE_FLOATS);
    return impl;
}

// FKL, the box as a BVF operation before the crop: one kernel, 100 frame evaluations per output value
template <size_t N>
Implementation makeFklInline(Context& ctx) {
    auto out = std::make_shared<fk::Tensor<fk::float3>>(W1_OUT_W, W1_OUT_H, N, 1, fk::MemType::Device);
    const auto regions = fklRegionSets<N>(ctx);
    Implementation impl{"fkl_inline", Reference::FklFloat};
    impl.stages.emplace_back("1_box_crop_resize_normalize", [&ctx, out, regions](size_t t) {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream, fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                                                  YUVToRGB::build(), fk::BoxBVF<BOX_K>::build(),
                                                  fk::Crop<>::build((*regions)[t]),
                                                  Bilinear::build(fk::Size(W1_OUT_W, W1_OUT_H)),
                                                  normalizedWrite(*out));
    });
    impl.run = runStages(impl.stages);
    setFloatIO<N>(impl, out, N * W1_PLANE_FLOATS);
    return impl;
}

// FKL staged: MorphologyDPP's box of the regions expanded by the box radius (frame pixels around the region; the
// replicated edge of the expanded region is not read) into N float3 planes of EXP_W x EXP_H, then one TransformDPP
// kernel reads them as one EXP_W x (EXP_H * N) image: Crop (the interior of every plane) -> Resize -> normalize
template <size_t N, int TW, int TH>
Implementation makeFklStaged(Context& ctx) {
    auto inter = std::make_shared<fk::Tensor<fk::float3>>(EXP_W, EXP_H, N, 1, fk::MemType::Device);
    auto out = std::make_shared<fk::Tensor<fk::float3>>(W1_OUT_W, W1_OUT_H, N, 1, fk::MemType::Device);
    auto expanded = std::make_shared<std::vector<std::array<fk::Rect, N>>>(ctx.regions->iterations());
    for (size_t t = 0; t < expanded->size(); ++t) {
        for (size_t i = 0; i < N; ++i) {
            const Region& r = ctx.regions->at(t, i);
            (*expanded)[t][i] = fk::Rect(r.x - BOX_R, r.y - BOX_R, EXP_W, EXP_H);
        }
    }
    std::array<fk::Rect, N> interiors{};
    for (size_t i = 0; i < N; ++i) {
        interiors[i] = fk::Rect(BOX_R, BOX_R + static_cast<int>(i) * EXP_H, REG_W, REG_H);
    }
    const fk::RawPtr<fk::ND::_2D, fk::float3> stacked{
        inter->ptr().data, {EXP_W, static_cast<unsigned>(EXP_H * N), EXP_W * sizeof(fk::float3)}};
    Implementation impl{"fkl_staged", Reference::FklFloat};
    impl.stages.emplace_back("1_morphology_dpp_box_of_the_expanded_regions", [&ctx, inter, expanded](size_t t) {
        using Details = fk::MorphologyDPPDetails<fk::float3, TW, TH, BOX_K, BOX_K>;
        const Details details{EXP_W, EXP_H, BOX_K, BOX_K, BOX_R, BOX_R};
        beforeKernel();
        if (!fk::executeBoxBlur<1>(details, rgbRead(ctx).then(fk::Crop<>::build((*expanded)[t])),
                                   fk::TensorWrite<fk::float3>::build(inter->ptr()), *ctx.fkStream)) {
            throw std::runtime_error("executeBoxBlur rejected its details");
        }
    });
    impl.stages.emplace_back("2_crop_resize_normalize", [&ctx, out, interiors, stacked](size_t) {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                                                  fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(stacked),
                                                  fk::Crop<>::build(interiors),
                                                  Bilinear::build(fk::Size(W1_OUT_W, W1_OUT_H)),
                                                  normalizedWrite(*out));
    });
    impl.run = runStages(impl.stages);
    setFloatIO<N>(impl, out, N * W1_PLANE_FLOATS);
    return impl;
}

template <size_t N>
Implementation nppdx(Context& ctx) {
    const auto reference = regionsReference<N>(ctx, Reference::NppdxPixelCenterFloat, WARMUP_ITERS);
    const size_t index = chooseFastest(
        ctx, "nppdx_N" + std::to_string(N), CONFIGS.size(),
        [&](size_t i) {
            Implementation impl;
            withConfig<CONFIGS>(i, [&]<NppdxConfig C>() { impl = makeNppdx<N, C>(ctx); });
            return impl;
        },
        [](size_t i) { return configLabel(CONFIGS[i]); }, reference, TOLERANCE);
    Implementation impl;
    withConfig<CONFIGS>(index, [&]<NppdxConfig C>() { impl = makeNppdx<N, C>(ctx); });
    return impl;
}

template <size_t N>
Implementation fklStaged(Context& ctx) {
    const auto reference = regionsReference<N>(ctx, Reference::FklFloat, WARMUP_ITERS);
    const auto make = [&](size_t i) {
        Implementation impl;
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((Is == i ? (impl = makeFklStaged<N, DPP_TILES[Is].width, DPP_TILES[Is].height>(ctx), 0) : 0), ...);
        }(std::make_index_sequence<DPP_TILES.size()>{});
        return impl;
    };
    return make(chooseFastest(ctx, "fkl_staged_N" + std::to_string(N), DPP_TILES.size(), make, dppLabel, reference,
                              TOLERANCE));
}

struct WinsRegions {
    static constexpr const char* title = "nppdx-wins W1: box 5x5 before a 1:2 resize, 32x16 regions (smoke test)";
    static constexpr const char* fklName = "fkl_staged";
    static constexpr const char* fklLabel = "FKL staged";
    static constexpr const char* fklShortLabel = "staged";
    static constexpr std::array<size_t, 3> batches{1, 16, 100};

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        impls.push_back(nppdx<N>(ctx));
        impls.push_back(makeFklInline<N>(ctx));
        return impls;
    }

    template <size_t N>
    static Implementation fklImplementation(Context& ctx) {
        return fklStaged<N>(ctx);
    }

    template <size_t N>
    static std::vector<float> reference(const Context& ctx, Reference r, size_t t) {
        return regionsReference<N>(ctx, r, t);
    }

    static void setup(const std::filesystem::path& outDir, const std::string& mode) { setupChoices(outDir, mode); }

    static void workload(std::ostream& os) {
        os << "bvf-paper-bench: commit " << BENCH_REPO_COMMIT << "\n"
           << "Workload (nppdx-wins W1, smoke test): " << FRAME_W << "x" << FRAME_H << " NV12 frame (seed "
           << FRAME_SEED << "), NV12->RGB BT.601 limited range, box 5x5 over the frame (the mean, per channel), N in "
           << "{1, 16, 100} regions of 32x16 per iteration at Experiment 1's positions (BENCH_REGIONS=32x16), each "
           << "resized 1:2 (bilinear) to 64x32, then Mul (1/255), Sub (0.485, 0.456, 0.406), Div (0.229, 0.224, "
           << "0.225), float3 tensor of N planes 64x32.\n"
           << "Semantics (documented, not matched): FKL samples the resize at dst * 0.5 and clamps at the crop (box "
           << "over the frame, crop, resize); NPPDx samples at pixel centers and its frame mode gives the resize the "
           << "boxed frame around the region. Each is checked against a reference of its own semantics.\n";
    }

    static void describe(std::ostream& os, const std::filesystem::path&) {
        os << "NPPDx: " << NPPDX_GIT_VERSION << ", operations for SM " << NPPDX_SM << "\n"
           << "nppdx: one kernel in frame mode (wins_common.h): NV12 ingest of the region with the halo of the box and "
              "the resize from the frame pixels around it, FromLimitedLuma/Chroma, ColorConvert BT.601, BoxBlur<5, 5>, "
              "Resize<1, 2, bilinear>, the normalization and float3 write written by hand; the fastest validated of "
              "tiles 32x16, 16x16, 16x8 with the recommended block and 128 and 256 (choices.csv, "
              "choice_sweep.csv).\n"
           << "fkl_inline: one kernel, ReadNV12 -> YUVtoRGB -> BoxBVF<5> -> Crop -> Resize -> Mul -> Sub -> Div -> "
              "TensorWrite (100 NV12 reads and conversions per output value).\n"
           << "fkl_staged: two kernels, MorphologyDPP box (executeBoxBlur) of the regions expanded by 2 pixels "
              "(36x20) into N float3 planes in memory (" << EXP_W * EXP_H * 12 << " bytes per region written, then "
              "read), then TransformDPP Crop (the 32x16 interior) -> Resize -> Mul -> Sub -> Div -> TensorWrite; the "
              "fastest validated MorphologyDPP tile of 16x8, 32x8, 32x16.\n";
    }

    static std::string diagnostic(const Context&) { return std::string(); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench::wins

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::wins::WinsRegions>(argc, argv);
}
