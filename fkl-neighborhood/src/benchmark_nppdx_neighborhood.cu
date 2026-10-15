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
// Experiment 3, part (b): FKL's MorphologyDPP box blur against NPPDx on the pipeline of the neighborhood experiment
// (common/bench_neighborhood.h): BOX_K x BOX_K box, one pass. NPPDx has the box (BoxBlur) between its color conversion
// and its write, so its kernel does the whole pipeline (nppdx_box.h):
//
//   nppdx       one kernel, the NPPDx operations (NV12 ingest of the crop, limited to full range, ColorConvert BT.601,
//               BoxBlur) and the normalization and float3 write written by hand, as fkl-vs-nppdx's NPPDx kernel (NPPDx
//               has no float output): 1 kernel
//   fkl_dpp     FKL's MorphologyDPP box blur, one kernel (neighborhood_common.h)
//
// Every tile and block size NPPDx supports for the crops is validated and its NPPDx kernel timed alone, for every batch
// size; the fastest correct one is used. The harness is common/bench_gpu.h. (The kernel of nppdx_box.h also has NPPDx's
// 8 bit write, which an NPPDx+NPP program used until it left the experiment; this program instantiates the float
// write only.)

#include "nppdx_box.h"

#include "neighborhood_common.h"

#include <cstdio>

namespace bench::neighborhood {

using namespace nppdx_box;

// The kernel with the hand written float write (nppdx_box.h)
constexpr bool FLOAT_WRITE = true;
constexpr int K = BOX_K;
constexpr const char* NPPDX_NAME = "nppdx";
// Float from the conversion to the write
constexpr Reference NPPDX_REFERENCE = Reference::FklFloat;

// The NPPDx configurations (tile and block) of the kernel for the crops
inline constexpr auto NPPDX_CONFIGS = nppdxConfigs<NPPDX_SM, FLOAT_WRITE>();

struct NppdxState {
    // The float output
    float* out{nullptr};
    ~NppdxState() { cudaFree(out); }
};

// The origins of the crops of every iteration, NPPDx's argument (a kernel parameter), converted before the loops
template <size_t N>
std::shared_ptr<const std::vector<CropOrigins<N>>> nppdxOrigins(const Context& ctx) {
    auto origins = std::make_shared<std::vector<CropOrigins<N>>>(ctx.regions->iterations());
    for (size_t t = 0; t < origins->size(); ++t) {
        for (size_t i = 0; i < N; ++i) {
            const Region& r = ctx.regions->at(t, i);
            (*origins)[t].origin[i] = int2{static_cast<int>(r.x), static_cast<int>(r.y)};
        }
    }
    return origins;
}

inline Normalization normalization() {
    return Normalization{{MUL[0], MUL[1], MUL[2]}, {SUB[0], SUB[1], SUB[2]}, {DIV[0], DIV[1], DIV[2]}};
}

// The NPPDx pipeline with the configuration C (one kernel)
template <size_t N, NppdxConfig C>
Implementation makeNppdx(Context& ctx) {
    using Kernel = NppdxBox<NPPDX_SM, N, FLOAT_WRITE, C.block, C.tileX, C.tileY>;
    Kernel::init();
    auto st = std::make_shared<NppdxState>();
    gpuErrchk(cudaMalloc(&st->out, N * PLANE_FLOATS * sizeof(float)));
    const auto origins = nppdxOrigins<N>(ctx);
    const Normalization n = normalization();

    Implementation impl{NPPDX_NAME, NPPDX_REFERENCE};
    impl.stages.emplace_back("1_nppdx_kernel_with_float_write", [&ctx, st, origins, n](size_t t) {
        beforeKernel();
        Kernel::run(ctx.stream, ctx.y, ctx.uv, ctx.pitch, (*origins)[t], n, st->out, nullptr);
    });
    impl.run = runStages(impl.stages);
    impl.download = [st]() { return downloadDevice(st->out, N * PLANE_FLOATS); };
    impl.fillNaN = [st]() { gpuErrchk(cudaMemset(st->out, 0xFF, N * PLANE_FLOATS * sizeof(float))); };
    return impl;
}

// ------------------------------------------------------------------------------------------------
// Choice of the NPPDx configuration, as fkl-vs-nppdx and as the MorphologyDPP tile (neighborhood_common.h): every
// configuration is validated (the whole pipeline, regions of the first timed iteration, against its reference) and
// its NPPDx kernel timed alone (on the regions of every iteration); the fastest correct one is used. The main run
// writes it to <name>_configs.csv, the other modes read it.
// ------------------------------------------------------------------------------------------------
struct ConfigSweepRow {
    size_t batch{0};
    NppdxConfig config{};
    TimingStats timing;
    ErrorStats errors;
    bool launched{false};
    bool chosen{false};
    bool correct() const { return launched && validation::withinTolerance(errors, toleranceOf(NPPDX_REFERENCE)); }
};

struct ConfigChoice {
    std::filesystem::path outDir;
    bool sweep{true};
    std::map<size_t, size_t> chosen;
    std::vector<ConfigSweepRow> rows;
};
inline ConfigChoice configChoice;

inline std::string configFile() { return std::string(NPPDX_NAME) + "_configs.csv"; }
inline std::string configSweepFile() { return std::string(NPPDX_NAME) + "_config_sweep.csv"; }

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
    std::ifstream in(outDir / configFile());
    if (!in) {
        throw std::runtime_error("Run the main benchmark first: it chooses the NPPDx configurations and writes them "
                                 "to " + (outDir / configFile()).string());
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
    std::ofstream configs(configChoice.outDir / configFile());
    configs << "N,tile,requested_block\n";
    for (const auto& [batch, index] : configChoice.chosen) {
        const auto& c = NPPDX_CONFIGS[index];
        configs << batch << "," << c.tileX << "x" << c.tileY << "," << c.block << "\n";
    }
    std::ofstream sweep(configChoice.outDir / configSweepFile());
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
    withConfig<NPPDX_CONFIGS>(index, [&]<NppdxConfig C>() {
        try {
            const Implementation pipeline = makeNppdx<N, C>(ctx);
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
        // The pipeline is the NPPDx kernel alone
        row.timing = computeStats(measureInterleaved(ctx, {makeNppdx<N, C>(ctx)})[0]);
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
                                 (configChoice.outDir / configFile()).string());
    }
    const auto reference = boxReferenceOf<N>(ctx, NPPDX_REFERENCE, WARMUP_ITERS, K, 1);
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

struct NppdxVsFklDpp {
    static constexpr const char* title = "FKL MorphologyDPP box blur vs NPPDx, neighborhood experiment (fkl-neighborhood)";
    static constexpr const char* fklName = "fkl_dpp";
    static constexpr const char* fklLabel = "FKL DPP";
    static constexpr const char* fklShortLabel = "DPP";
    static constexpr geometry::RegionSize regionSize = CROP_SIZE;

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        withConfig<NPPDX_CONFIGS>(chooseConfig<N>(ctx), [&]<NppdxConfig C>() { impls.push_back(makeNppdx<N, C>(ctx)); });
        return impls;
    }

    template <size_t N>
    static Implementation fklImplementation(Context& ctx) {
        return fklDpp<N, K, 1>(ctx);
    }

    template <size_t N>
    static std::vector<float> reference(const Context& ctx, Reference r, size_t t) {
        return boxReferenceOf<N>(ctx, r, t, K, 1);
    }

    static void setup(const std::filesystem::path& outDir, const std::string& mode) {
        setupTileChoice(outDir, mode);
        setupConfigChoice(outDir, mode);
    }

    static void workload(std::ostream& os) { describeWorkload(os, K, 1, BATCHES); }

    static void describe(std::ostream& os, const std::filesystem::path&) {
        os << "NPPDx: " << NPPDX_GIT_VERSION << ", operations for SM " << NPPDX_SM << "\n"
           << "NPPDx pipeline (nppdx): one kernel (nppdx_box.h): NPPDx's NV12 ingest of the crop (the crop is the "
              "image, so the halo of " << HALO << " pixels replicates its edge), FromLimitedLuma and "
              "FromLimitedChroma, ColorConvert BT.601 and BoxBlur<" << K << ", " << K << "> on a shared memory "
              "tile with the halo, then the normalization (Mul, Sub, Div) and the float3 write written by hand, "
              "as fkl-vs-nppdx's NPPDx kernel (NPPDx has no float output); one launch for the N crops, blockIdx.z "
              "the crop: 1 kernel. Float from the conversion to the write; checked against "
           << referenceName(NPPDX_REFERENCE) << ".\n";
        os << "NPPDx configurations: every tile of the crop that NPPDx supports (the whole crop, then halvings of the "
              "longer side down to 64 pixels, in 4x2 pixel cells) with NPPDx's recommended block and BlockDim 64, "
              "128, 256 and 512, validated and its kernel timed alone for every batch size, the fastest correct one "
              "used (" << configSweepFile() << ", " << configFile() << "):";
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
    return bench::gpuBenchMain<bench::neighborhood::NppdxVsFklDpp>(argc, argv);
}
