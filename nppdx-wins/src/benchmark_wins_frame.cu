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

// nppdx-wins, W2 (wins_common.h): ReadNV12 -> YUVtoRGB -> Box 5x5 -> Resize 2:1 to 1920x1080 -> Mul -> Sub -> Div ->
// one float3 plane, the whole 3840x2160 frame (batch "1": one frame; the harness's regions are not used). nppdx and
// fkl_inline are the baselines of fkl_staged in the harness's files; every median is in latency.csv, gpu_time.csv and
// gpu_time_l2_flushed.csv.

#include "wins_common.h"

#include <iostream>

namespace bench::wins {

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
    using Ops = BoxResizeOps<C.tileX, C.tileY, C.block, 2, 1>;
    constexpr auto kernel = nppdxFrameKernel<Ops>;
    allowSharedMemory<kernel, Ops::smem>();
    auto out = outputPlane();
    const Normalization n = normalization();
    Implementation impl{"nppdx", Reference::NppdxPixelCenterFloat};
    impl.stages.emplace_back("1_nppdx_box_resize_normalize", [&ctx, out, n](size_t) {
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

// FKL, the box as a BVF operation before the resize: one kernel, 100 frame evaluations per output value
inline Implementation makeFklInline(Context& ctx) {
    auto out = outputPlane();
    Implementation impl{"fkl_inline", Reference::FklFloat};
    impl.stages.emplace_back("1_box_resize_normalize", [&ctx, out](size_t) {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream, fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                                                  YUVToRGB::build(), fk::BoxBVF<BOX_K>::build(),
                                                  Bilinear::build(fk::Size(W2_OUT_W, W2_OUT_H)),
                                                  normalizedWrite(*out));
    });
    impl.run = runStages(impl.stages);
    setFloatIO<1>(impl, out, W2_FLOATS);
    return impl;
}

// FKL staged: MorphologyDPP's box of the frame into a float3 frame in memory (3840x2160x12 bytes), then one
// TransformDPP kernel Resize -> normalize reading it
template <int TW, int TH>
Implementation makeFklStaged(Context& ctx) {
    auto inter = std::make_shared<fk::Tensor<fk::float3>>(FRAME_W, FRAME_H, 1, 1, fk::MemType::Device);
    auto out = outputPlane();
    const fk::RawPtr<fk::ND::_2D, fk::float3> frame{inter->ptr().data,
                                                     {FRAME_W, FRAME_H, FRAME_W * sizeof(fk::float3)}};
    Implementation impl{"fkl_staged", Reference::FklFloat};
    impl.stages.emplace_back("1_morphology_dpp_box_of_the_frame", [&ctx, inter, frame](size_t) {
        using Details = fk::MorphologyDPPDetails<fk::float3, TW, TH, BOX_K, BOX_K>;
        const Details details{FRAME_W, FRAME_H, BOX_K, BOX_K, BOX_R, BOX_R};
        beforeKernel();
        if (!fk::executeBoxBlur<1>(details, rgbRead(ctx), fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(frame),
                                   *ctx.fkStream)) {
            throw std::runtime_error("executeBoxBlur rejected its details");
        }
    });
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
    const auto reference = frameReference(ctx, Reference::NppdxPixelCenterFloat);
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

inline Implementation fklStaged(Context& ctx) {
    const auto reference = frameReference(ctx, Reference::FklFloat);
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
    static constexpr const char* title = "nppdx-wins W2: box 5x5 before a 2:1 resize of the 4K frame (smoke test)";
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
        return impls;
    }

    template <size_t N>
    static Implementation fklImplementation(Context& ctx) {
        return fklStaged(ctx);
    }

    template <size_t N>
    static std::vector<float> reference(const Context& ctx, Reference r, size_t) {
        return frameReference(ctx, r);
    }

    static void setup(const std::filesystem::path& outDir, const std::string& mode) { setupChoices(outDir, mode); }

    static void workload(std::ostream& os) {
        os << "bvf-paper-bench: commit " << BENCH_REPO_COMMIT << "\n"
           << "Workload (nppdx-wins W2, smoke test): the " << FRAME_W << "x" << FRAME_H << " NV12 frame (seed "
           << FRAME_SEED << "), NV12->RGB BT.601 limited range, box 5x5 (the mean, per channel, replicating the frame "
           << "edge), resized 2:1 (bilinear) to " << W2_OUT_W << "x" << W2_OUT_H << ", then Mul (1/255), Sub (0.485, "
           << "0.456, 0.406), Div (0.229, 0.224, 0.225), one float3 plane. Batch 1 is the frame (the harness's "
           << "regions are not used).\n"
           << "Semantics (documented, not matched): FKL's 2:1 bilinear samples at (2x, 2y), i.e. the boxed pixel; "
           << "NPPDx's at pixel centers, the mean of the 2x2 boxed pixels. Each is checked against a reference of its "
           << "own semantics.\n";
    }

    static void describe(std::ostream& os, const std::filesystem::path&) {
        os << "NPPDx: " << NPPDX_GIT_VERSION << ", operations for SM " << NPPDX_SM << "\n"
           << "nppdx: one kernel over the frame (wins_common.h): NV12 ingest of the tile with the halo of the box and "
              "the resize, FromLimitedLuma/Chroma, ColorConvert BT.601, BoxBlur<5, 5>, Resize<2, 1, bilinear>, the "
              "normalization and float3 write written by hand; the fastest validated of tiles 64x32, 32x32, 32x16, "
              "16x16, 16x8 with the recommended block and 128, 256, 512 (choices.csv, choice_sweep.csv).\n"
           << "fkl_inline: one kernel, ReadNV12 -> YUVtoRGB -> BoxBVF<5> -> Resize -> Mul -> Sub -> Div -> "
              "TensorWrite (100 NV12 reads and conversions per output value).\n"
           << "fkl_staged: two kernels, MorphologyDPP box (executeBoxBlur) of the frame into a float3 frame in memory "
              "(" << static_cast<size_t>(FRAME_W) * FRAME_H * 12 << " bytes written, then read by the resize), then "
              "TransformDPP Resize -> Mul -> Sub -> Div -> TensorWrite; the fastest validated MorphologyDPP tile of "
              "16x8, 32x8, 32x16.\n";
    }

    static std::string diagnostic(const Context&) { return std::string(); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench::wins

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::wins::WinsFrame>(argc, argv);
}
