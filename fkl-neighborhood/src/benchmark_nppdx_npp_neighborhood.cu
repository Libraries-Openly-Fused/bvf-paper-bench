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

// Experiment 3, part (b): FKL's MorphologyDPP against NPPDx+NPP on the pipeline of the neighborhood experiment
// (common/bench_neighborhood.h), with a MORPH_K x MORPH_K mask and the operation NEIGHBORHOOD_OP (0: erode, 1: open;
// one build each). NPPDx kernels are compiled for one region size, so, as fkl-vs-nppdx, this program is built for one
// of the sizes (BENCH_REGIONS 20x10, 32x16 or 48x24, with BENCH_NPPDX=ON: fkl-neighborhood/CMakeLists.txt). The harness
// is common/bench_gpu.h.
//
//   nppdx_npp  without hand written pixel processing code: the NPPDx kernel of fkl-vs-nppdx's NPPDx+NPP pipeline
//              (nppdx_npp_figure1.h: NV12 ingest of the region, limited to full range, ColorConvert BT.601, Resize,
//              8 bit RGB write; one launch for the N regions), then NPP with the function it offers for each step
//              (NPPDx 0.1.2 has no morphology): per region nppiErodeBorder_8u_C3R (erode), or nppiErodeBorder_8u_C3R into
//              a plane buffer and nppiDilateBorder_8u_C3R (open), with a K x K mask of ones, replicating the border of
//              the plane; then nppiConvert_8u32f_C3R, nppiMulC, nppiSubC and nppiDivC_32f_C3IR on the N planes as one
//              image: N + 5 kernels (erode), 2N + 5 (open). Every tile and block size NPPDx supports for these regions
//              is validated and its kernel timed alone; the fastest correct one is used.
//   fkl_dpp    FKL's MorphologyDPP, one kernel (neighborhood_common.h)

#include "benchmarks/nppdx_npp_figure1.h"

#include "neighborhood_common.h"

#ifndef NEIGHBORHOOD_OP
#error "Define NEIGHBORHOOD_OP, 0 (erode) or 1 (open) (fkl-neighborhood/CMakeLists.txt does)"
#endif

namespace bench::neighborhood {

constexpr MorphOp OP = NEIGHBORHOOD_OP == 0 ? MorphOp::Erode : MorphOp::Open;
static_assert(NEIGHBORHOOD_OP == 0 || NEIGHBORHOOD_OP == 1, "0 (erode) or 1 (open)");
constexpr int K = MORPH_K;
static_assert(PLANE_W == figure1::RES_W && PLANE_H == figure1::RES_H, "The NPPDx kernel writes RES_W x RES_H planes");

// The NPPDx configurations (tile and block) of fkl-vs-nppdx's NPPDx+NPP kernel for the region size of this build
inline constexpr auto NPPDX_CONFIGS = figure1::nppdxRgbConfigs<NPPDX_SM>();

constexpr size_t PLANE_BYTES = static_cast<size_t>(PLANE_W) * PLANE_H * 3;

struct NppdxNppState {
    // The resized 8 bit RGB planes the NPPDx kernel writes, the morphology's (and its first pass, for the opening)
    uint8_t* resized{nullptr};
    uint8_t* firstPass{nullptr};
    uint8_t* morphed{nullptr};
    float* out{nullptr};
    // The K x K mask of ones of nppiErodeBorder / nppiDilateBorder
    Npp8u* mask{nullptr};
    ~NppdxNppState() {
        cudaFree(resized);
        cudaFree(firstPass);
        cudaFree(morphed);
        cudaFree(out);
        cudaFree(mask);
    }
};

// The origins of the regions of every iteration, NPPDx's argument (a kernel parameter), converted before the loops
template <size_t N>
std::shared_ptr<const std::vector<figure1::CropOrigins<N>>> nppdxOrigins(const Context& ctx) {
    auto origins = std::make_shared<std::vector<figure1::CropOrigins<N>>>(ctx.regions->iterations());
    for (size_t t = 0; t < origins->size(); ++t) {
        for (size_t i = 0; i < N; ++i) {
            const Region& r = ctx.regions->at(t, i);
            (*origins)[t].origin[i] = int2{static_cast<int>(r.x), static_cast<int>(r.y)};
        }
    }
    return origins;
}

// NPPDx+NPP with the NPPDx configuration C; with KERNEL_ONLY, only its NPPDx kernel (timed alone to choose C)
template <size_t N, figure1::NppdxConfig C, bool KERNEL_ONLY = false>
Implementation makeNppdxNpp(Context& ctx) {
    using Kernel = figure1::NppdxRgb<NPPDX_SM, N, C.block, C.tileX, C.tileY>;
    Kernel::init();
    auto st = std::make_shared<NppdxNppState>();
    gpuErrchk(cudaMalloc(&st->resized, N * PLANE_BYTES));
    gpuErrchk(cudaMalloc(&st->firstPass, N * PLANE_BYTES));
    gpuErrchk(cudaMalloc(&st->morphed, N * PLANE_BYTES));
    gpuErrchk(cudaMalloc(&st->out, N * PLANE_FLOATS * sizeof(float)));
    const std::vector<Npp8u> ones(static_cast<size_t>(K) * K, 1);
    gpuErrchk(cudaMalloc(&st->mask, ones.size()));
    gpuErrchk(cudaMemcpy(st->mask, ones.data(), ones.size(), cudaMemcpyHostToDevice));
    const auto origins = nppdxOrigins<N>(ctx);

    Implementation impl{"nppdx_npp", Reference::NppdxPixelCenters8bit};
    impl.stages.emplace_back("1_nppdx_kernel", [&ctx, st, origins](size_t t) {
        beforeKernel();
        Kernel::run(ctx.stream, ctx.y, ctx.uv, ctx.pitch, (*origins)[t], st->resized);
    });
    if constexpr (!KERNEL_ONLY) {
        const std::string morphStage =
            OP == MorphOp::Erode ? "2_per_region_erode_border" : "2_per_region_erode_then_dilate_border";
        impl.stages.emplace_back(morphStage, [&ctx, st](size_t) {
            const NppiSize plane{PLANE_W, PLANE_H}, mask{K, K};
            const NppiPoint anchor{K / 2, K / 2}, origin{0, 0};
            const int step = PLANE_W * 3;
            for (size_t i = 0; i < N; ++i) {
                const Npp8u* src = st->resized + i * PLANE_BYTES;
                Npp8u* out = st->morphed + i * PLANE_BYTES;
                if constexpr (OP == MorphOp::Erode) {
                    beforeKernel();
                    NPP_CHECK(nppiErodeBorder_8u_C3R_Ctx(src, step, plane, origin, out, step, plane, st->mask, mask,
                                                         anchor, NPP_BORDER_REPLICATE, ctx.npp));
                } else {
                    Npp8u* firstPass = st->firstPass + i * PLANE_BYTES;
                    beforeKernel();
                    NPP_CHECK(nppiErodeBorder_8u_C3R_Ctx(src, step, plane, origin, firstPass, step, plane, st->mask,
                                                         mask, anchor, NPP_BORDER_REPLICATE, ctx.npp));
                    beforeKernel();
                    NPP_CHECK(nppiDilateBorder_8u_C3R_Ctx(firstPass, step, plane, origin, out, step, plane, st->mask,
                                                          mask, anchor, NPP_BORDER_REPLICATE, ctx.npp));
                }
            }
        });
        impl.stages.emplace_back("3_convert_mulc_subc_divc", [&ctx, st](size_t) {
            const NppiSize planes{PLANE_W, static_cast<int>(N) * PLANE_H};
            const int step = PLANE_W * 3 * static_cast<int>(sizeof(float));
            beforeKernel();
            NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(st->morphed, PLANE_W * 3, st->out, step, planes, ctx.npp));
            beforeKernel();
            NPP_CHECK(nppiMulC_32f_C3IR_Ctx(MUL, st->out, step, planes, ctx.npp));
            beforeKernel();
            NPP_CHECK(nppiSubC_32f_C3IR_Ctx(SUB, st->out, step, planes, ctx.npp));
            beforeKernel();
            NPP_CHECK(nppiDivC_32f_C3IR_Ctx(DIV, st->out, step, planes, ctx.npp));
        });
    }
    impl.run = runStages(impl.stages);
    impl.download = [st]() { return downloadDevice(st->out, N * PLANE_FLOATS); };
    impl.fillNaN = [st]() {
        // Every buffer the pipeline writes: values no correct execution leaves (NaN floats)
        gpuErrchk(cudaMemset(st->resized, 0xAB, N * PLANE_BYTES));
        gpuErrchk(cudaMemset(st->firstPass, 0xAB, N * PLANE_BYTES));
        gpuErrchk(cudaMemset(st->morphed, 0xAB, N * PLANE_BYTES));
        gpuErrchk(cudaMemset(st->out, 0xFF, N * PLANE_FLOATS * sizeof(float)));
    };
    return impl;
}

// ------------------------------------------------------------------------------------------------
// Choice of the NPPDx configuration, as fkl-vs-nppdx and as the MorphologyDPP tile (neighborhood_common.h): every
// configuration is validated (the whole pipeline, regions of the first timed iteration, against its reference) and
// its NPPDx kernel timed alone (on the regions of every iteration); the fastest correct one is used. The main run
// writes it to nppdx_npp_configs.csv, the other modes read it.
// ------------------------------------------------------------------------------------------------
struct ConfigSweepRow {
    size_t batch{0};
    figure1::NppdxConfig config{};
    TimingStats timing;
    ErrorStats errors;
    bool launched{false};
    bool chosen{false};
    bool correct() const { return launched && validation::withinTolerance(errors, TOLERANCE_8BIT); }
};

struct ConfigChoice {
    std::filesystem::path outDir;
    bool sweep{true};
    std::map<size_t, size_t> chosen;
    std::vector<ConfigSweepRow> rows;
};
inline ConfigChoice configChoice;

constexpr const char* CONFIG_FILE = "nppdx_npp_configs.csv";
constexpr const char* CONFIG_SWEEP_FILE = "nppdx_npp_config_sweep.csv";

inline size_t configIndex(unsigned tileX, unsigned tileY, unsigned block) {
    for (size_t i = 0; i < NPPDX_CONFIGS.size(); ++i) {
        const auto& c = NPPDX_CONFIGS[i];
        if (c.tileX == tileX && c.tileY == tileY && c.block == block) {
            return i;
        }
    }
    throw std::runtime_error("Unknown NPPDx configuration " + std::to_string(tileX) + "x" + std::to_string(tileY) +
                             " block " + std::to_string(block));
}

inline void setupConfigChoice(const std::filesystem::path& outDir, const std::string& mode) {
    configChoice.outDir = outDir;
    configChoice.sweep = mode.empty();
    if (configChoice.sweep) {
        return;
    }
    std::ifstream in(outDir / CONFIG_FILE);
    if (!in) {
        throw std::runtime_error(std::string("Run the main benchmark first: it chooses the NPPDx configurations and "
                                             "writes them to ") + (outDir / CONFIG_FILE).string());
    }
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        size_t batch = 0;
        unsigned tileX = 0, tileY = 0, block = 0;
        if (std::sscanf(line.c_str(), "%zu,%ux%u,%u", &batch, &tileX, &tileY, &block) == 4) {
            configChoice.chosen[batch] = configIndex(tileX, tileY, block);
        }
    }
}

inline void writeConfigFiles() {
    std::ofstream configs(configChoice.outDir / CONFIG_FILE);
    configs << "N,tile,requested_block\n";
    for (const auto& [batch, index] : configChoice.chosen) {
        const auto& c = NPPDX_CONFIGS[index];
        configs << batch << "," << c.tileX << "x" << c.tileY << "," << c.block << "\n";
    }
    std::ofstream sweep(configChoice.outDir / CONFIG_SWEEP_FILE);
    sweep << "N,tile,requested_block,launched,kernel_median_us,kernel_p10_us,kernel_p90_us,kernel_mean_us,"
             "max_abs_err,within_tolerance,chosen\n";
    for (const auto& r : configChoice.rows) {
        sweep << r.batch << "," << r.config.tileX << "x" << r.config.tileY << "," << r.config.block << ","
              << (r.launched ? "yes" : "no") << "," << std::fixed << std::setprecision(2) << r.timing.median << ","
              << r.timing.p10 << "," << r.timing.p90 << "," << r.timing.mean << "," << std::scientific
              << std::setprecision(3) << r.errors.max() << std::defaultfloat << "," << (r.correct() ? "yes" : "no")
              << "," << (r.chosen ? "yes" : "no") << "\n";
    }
}

template <size_t N>
ConfigSweepRow sweepConfig(Context& ctx, const size_t index, const std::vector<float>& reference) {
    ConfigSweepRow row{N, NPPDX_CONFIGS[index]};
    figure1::withConfig<NPPDX_CONFIGS>(index, [&]<figure1::NppdxConfig C>() {
        try {
            const Implementation pipeline = makeNppdxNpp<N, C>(ctx);
            pipeline.fillNaN();
            gpuErrchk(cudaDeviceSynchronize());
            pipeline.run(WARMUP_ITERS);
            gpuErrchk(cudaStreamSynchronize(ctx.stream));
            row.errors = compare(pipeline.download(), reference);
            row.launched = true;
        } catch (const std::exception& e) {
            // A configuration that cannot run here (too many threads, registers or shared memory) is never chosen
            std::cout << "N " << N << ": NPPDx tile " << C.tileX << "x" << C.tileY << " block " << C.block
                      << " cannot run: " << e.what() << std::endl;
            gpuErrchk(cudaGetLastError());
            return;
        }
        row.timing = computeStats(measureInterleaved(ctx, {makeNppdxNpp<N, C, true>(ctx)})[0]);
    });
    return row;
}

template <size_t N>
size_t chooseConfig(Context& ctx) {
    if (configChoice.chosen.count(N)) {
        return configChoice.chosen.at(N);
    }
    if (!configChoice.sweep) {
        throw std::runtime_error("No NPPDx configuration for N = " + std::to_string(N) + " in " +
                                 (configChoice.outDir / CONFIG_FILE).string());
    }
    const auto reference = morphReference<N>(ctx, Reference::NppdxPixelCenters8bit, WARMUP_ITERS, K, OP);
    const size_t first = configChoice.rows.size();
    size_t best = NPPDX_CONFIGS.size();
    for (size_t i = 0; i < NPPDX_CONFIGS.size(); ++i) {
        configChoice.rows.push_back(sweepConfig<N>(ctx, i, reference));
        const ConfigSweepRow& r = configChoice.rows.back();
        if (r.correct() &&
            (best == NPPDX_CONFIGS.size() || r.timing.median < configChoice.rows[first + best].timing.median)) {
            best = i;
        }
    }
    if (best == NPPDX_CONFIGS.size()) {
        throw std::runtime_error("No NPPDx configuration is within the tolerance for N = " + std::to_string(N));
    }
    configChoice.rows[first + best].chosen = true;
    configChoice.chosen[N] = best;
    const auto& c = NPPDX_CONFIGS[best];
    std::cout << "N " << N << ": NPPDx tile " << c.tileX << "x" << c.tileY << " block " << c.block
              << " (fastest correct of " << NPPDX_CONFIGS.size() << " configurations)" << std::endl;
    writeConfigFiles();
    return best;
}

struct NppdxNppVsFklDpp {
    static constexpr const char* title =
        "FKL MorphologyDPP vs NPPDx+NPP, neighborhood experiment (fkl-neighborhood)";
    static constexpr const char* fklName = "fkl_dpp";
    static constexpr const char* fklLabel = "FKL DPP";
    static constexpr const char* fklShortLabel = "DPP";

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        figure1::withConfig<NPPDX_CONFIGS>(chooseConfig<N>(ctx), [&]<figure1::NppdxConfig C>() {
            impls.push_back(makeNppdxNpp<N, C>(ctx));
        });
        return impls;
    }

    template <size_t N>
    static Implementation fklImplementation(Context& ctx) {
        return fklDpp<N, K, OP>(ctx);
    }

    template <size_t N>
    static std::vector<float> reference(const Context& ctx, Reference r, size_t t) {
        return morphReference<N>(ctx, r, t, K, OP);
    }

    static void setup(const std::filesystem::path& outDir, const std::string& mode) {
        setupTileChoice(outDir, mode);
        setupConfigChoice(outDir, mode);
    }

    static void workload(std::ostream& os) { describeWorkload(os, K, OP, BATCHES); }

    static void describe(std::ostream& os, const std::filesystem::path&) {
        const NppLibraryVersion* npp = nppGetLibVersion();
        os << "NPPDx: " << NPPDX_GIT_VERSION << ", operations for SM " << NPPDX_SM << "\n"
           << "NPP: " << npp->major << "." << npp->minor << "." << npp->build << "\n"
           << "NPPDx+NPP pipeline (nppdx_npp), without hand written pixel processing code: the NPPDx kernel of "
              "fkl-vs-nppdx's NPPDx+NPP pipeline (NV12 ingest of the region, FromLimitedLuma and FromLimitedChroma, "
              "ColorConvert BT.601, Resize " << figure1::RESIZE_J << ":" << figure1::RESIZE_K << " bilinear, 8 bit "
              "RGB write; one launch for the N regions, blockIdx.z the region), then NPP (NPPDx 0.1.2 has no "
              "morphology): per region "
           << (OP == MorphOp::Erode ? "nppiErodeBorder_8u_C3R"
                                    : "nppiErodeBorder_8u_C3R into a plane buffer, then nppiDilateBorder_8u_C3R")
           << " with a " << K << "x" << K << " mask of ones (oSrcSize the plane, oSrcOffset 0, NPP_BORDER_REPLICATE), "
              "then nppiConvert_8u32f_C3R, nppiMulC, nppiSubC and nppiDivC_32f_C3IR on the N planes as one image: "
           << (OP == MorphOp::Erode ? "N + 5" : "2N + 5") << " kernels. The resized RGB is rounded to 8 bits before "
              "the morphology and the normalization; checked against " << referenceName(Reference::NppdxPixelCenters8bit)
           << " (the reference of fkl-vs-nppdx without the letterbox, then the morphology).\n"
           << "NPPDx configurations: every tile NPPDx supports for these regions (as fkl-vs-nppdx) with NPPDx's "
              "recommended block and BlockDim 64, 128, 256 and 512, validated and its kernel timed alone for every "
              "batch size, the fastest correct one used (" << CONFIG_SWEEP_FILE << ", " << CONFIG_FILE << "):";
        for (const auto& [batch, index] : configChoice.chosen) {
            const auto& c = NPPDX_CONFIGS[index];
            os << " N = " << batch << ": tile " << c.tileX << "x" << c.tileY << ", block " << c.block << ";";
        }
        os << "\n";
    }

    static std::string diagnostic(const Context&) { return std::string(); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench::neighborhood

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::neighborhood::NppdxNppVsFklDpp>(argc, argv);
}
