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

// What the programs of the neighborhood experiment share (common/bench_neighborhood.h): its references, FKL's
// MorphologyDPP pipeline with the choice of its tile, FKL's pipeline with the BVF operations ErodeBVF and DilateBVF,
// and the description of the workload for the platform file. The programs run on the harness of common/bench_gpu.h
// through its optional Program members (fklImplementation, reference, workload...).

#ifndef BVF_BENCH_NEIGHBORHOOD_COMMON_H
#define BVF_BENCH_NEIGHBORHOOD_COMMON_H

#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/image_processing/morphology.h>

#include "bench_gpu.h"
#include "bench_neighborhood.h"
#include "bench_neighborhood_ops.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace bench::neighborhood {

// ------------------------------------------------------------------------------------------------
// References
// ------------------------------------------------------------------------------------------------

// The reference of conventions r for the regions of iteration t: the one of Figure 1 without the letterbox
// (cpuReference / nppConventionReference8bit of common/bench_gpu.h, normalized), then the morphology on every plane
// (hostMorph, exact after the normalization: see bench_neighborhood.h)
template <size_t N>
std::vector<float> morphReference(const Context& ctx, const Reference r, const size_t t, const int k, const MorphOp op) {
    std::vector<float> base;
    switch (r) {
        case Reference::FklFloat: base = cpuReference<N, false, false>(ctx, t); break;
        case Reference::Fkl8bit: base = cpuReference<N, true, false>(ctx, t); break;
        default: base = nppConventionReference8bit<N, false>(ctx, t); break;
    }
    if (base.size() != N * PLANE_FLOATS) {
        throw std::runtime_error("Unexpected size of the reference without the letterbox");
    }
    std::vector<float> out(base.size());
    hostMorph(reinterpret_cast<const fk::float3*>(base.data()), reinterpret_cast<fk::float3*>(out.data()), N, k, op);
    return out;
}

// ------------------------------------------------------------------------------------------------
// FKL
// ------------------------------------------------------------------------------------------------

template <size_t N>
std::shared_ptr<fk::Tensor<fk::float3>> planesTensor() {
    return std::make_shared<fk::Tensor<fk::float3>>(PLANE_W, PLANE_H, N, 1, fk::MemType::Device);
}

template <size_t N>
void setPlanesIO(Implementation& impl, const std::shared_ptr<fk::Tensor<fk::float3>>& out) {
    impl.download = [out]() { return downloadDevice(out->ptr().data, N * PLANE_FLOATS); };
    impl.fillNaN = [out]() { gpuErrchk(cudaMemset(out->ptr().data, 0xFF, N * PLANE_FLOATS * sizeof(float))); };
}

// The Figure 1 read chain up to the resize, as one fused read IOp (Backwards Vertical Fusion), and the pointwise tail
// with the write, as one fused write IOp (Vertical Fusion)
template <size_t N>
auto resizedRead(const Context& ctx, const std::array<fk::Rect, N>& regions) {
    return fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice)
        .then(YUVToRGB::build())
        .then(fk::Crop<>::build(regions))
        .then(fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(
            fk::Size(PLANE_W, PLANE_H)));
}

inline auto normalizedWrite(const fk::Tensor<fk::float3>& out) {
    return fk::Mul<fk::float3>::build(mulValue)
        .then(fk::Sub<fk::float3>::build(subValue))
        .then(fk::Div<fk::float3>::build(divValue))
        .then(fk::TensorWrite<fk::float3>::build(out.ptr()));
}

// FKL's MorphologyDPP with a TW x TH tile: one executeErode / executeOpen call, one kernel for the whole pipeline. The
// DPP reads the resized regions through the BVF chain and writes through the VF tail, so nothing is in memory between
// the NV12 frame and the output; its grid has one z plane per region (FusedKernelLibrary PR #393). Its shared memory
// stages the tile, plus the halo of every pass, of the K x K mask.
template <size_t N, int K, MorphOp OP, int TW, int TH>
Implementation makeFklDpp(Context& ctx) {
    auto out = planesTensor<N>();
    const auto regions = fklRegionSets<N>(ctx);
    Implementation impl{"fkl_dpp", Reference::FklFloat};
    impl.run = [&ctx, out, regions](size_t t) {
        using Details = fk::MorphologyDPPDetails<fk::float3, TW, TH, K, K>;
        const Details details{PLANE_W, PLANE_H, K, K, K / 2, K / 2};
        beforeKernel();
        bool launched = false;
        if constexpr (OP == MorphOp::Erode) {
            launched = fk::executeErode(details, resizedRead<N>(ctx, (*regions)[t]), normalizedWrite(*out), *ctx.fkStream);
        } else {
            launched = fk::executeOpen(details, resizedRead<N>(ctx, (*regions)[t]), normalizedWrite(*out), *ctx.fkStream);
        }
        if (!launched) {
            throw std::runtime_error("MorphologyDPP rejected its details");
        }
    };
    setPlanesIO<N>(impl, out);
    return impl;
}

// FKL with the morphology as BVF operations (common/bench_neighborhood_ops.h): one executeOperations call, one kernel,
// in which every output value evaluates the read chain K x K times per pass (K^4 times for the opening), and every
// evaluation of the resize reads four values of the RGB conversion
template <size_t N, int K, MorphOp OP>
Implementation makeFklNaive(Context& ctx) {
    auto out = planesTensor<N>();
    const auto regions = fklRegionSets<N>(ctx);
    Implementation impl{"fkl_bvf_ops", Reference::FklFloat};
    impl.run = [&ctx, out, regions](size_t t) {
        const auto read = fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice);
        const auto resize = fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(
            fk::Size(PLANE_W, PLANE_H));
        beforeKernel();
        if constexpr (OP == MorphOp::Erode) {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream, read, YUVToRGB::build(),
                fk::Crop<>::build((*regions)[t]), resize, fk::ErodeBVF<K>::build(),
                fk::Mul<fk::float3>::build(mulValue), fk::Sub<fk::float3>::build(subValue),
                fk::Div<fk::float3>::build(divValue), fk::TensorWrite<fk::float3>::build(out->ptr()));
        } else {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream, read, YUVToRGB::build(),
                fk::Crop<>::build((*regions)[t]), resize, fk::ErodeBVF<K>::build(), fk::DilateBVF<K>::build(),
                fk::Mul<fk::float3>::build(mulValue), fk::Sub<fk::float3>::build(subValue),
                fk::Div<fk::float3>::build(divValue), fk::TensorWrite<fk::float3>::build(out->ptr()));
        }
    };
    setPlanesIO<N>(impl, out);
    return impl;
}

// ------------------------------------------------------------------------------------------------
// Choice of the MorphologyDPP tile
// ------------------------------------------------------------------------------------------------

// Every tile of DPP_TILES is validated (regions of the first timed iteration, against the float reference) and timed
// alone (latency, CPU + GPU, as in the main benchmark, on the regions of every iteration); the fastest correct one is
// the DPP of the comparison for that batch size. The main run chooses and writes the tile of every batch size to
// fkl_dpp_tiles.csv in its output folder; the other modes (--gpu-only-l2-flushed, --profile-kernels, --ncu-target)
// read it from there, so that every mode measures the same DPP.
struct TileSweepRow {
    size_t batch{0};
    Tile tile{};
    TimingStats timing;
    ErrorStats errors;
    bool chosen{false};
    bool correct() const { return validation::withinTolerance(errors, TOLERANCE); }
};

struct TileChoice {
    std::filesystem::path outDir;
    bool sweep{true};
    std::map<size_t, size_t> chosen;
    std::vector<TileSweepRow> rows;
};
inline TileChoice tileChoice;

constexpr const char* TILE_FILE = "fkl_dpp_tiles.csv";
constexpr const char* TILE_SWEEP_FILE = "fkl_dpp_tile_sweep.csv";

inline size_t tileIndex(const int width, const int height) {
    for (size_t i = 0; i < DPP_TILES.size(); ++i) {
        if (DPP_TILES[i].width == width && DPP_TILES[i].height == height) {
            return i;
        }
    }
    throw std::runtime_error("Unknown MorphologyDPP tile " + std::to_string(width) + "x" + std::to_string(height));
}

// Called by the program's setup(): the main run (mode "") chooses the tiles, the other modes read them
inline void setupTileChoice(const std::filesystem::path& outDir, const std::string& mode) {
    tileChoice.outDir = outDir;
    tileChoice.sweep = mode.empty();
    if (tileChoice.sweep) {
        return;
    }
    std::ifstream in(outDir / TILE_FILE);
    if (!in) {
        throw std::runtime_error(std::string("Run the main benchmark first: it chooses the MorphologyDPP tiles and "
                                             "writes them to ") + (outDir / TILE_FILE).string());
    }
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        size_t batch = 0;
        int width = 0, height = 0;
        char comma = 0, x = 0;
        std::istringstream row(line);
        if (row >> batch >> comma >> width >> x >> height) {
            tileChoice.chosen[batch] = tileIndex(width, height);
        }
    }
}

inline void writeTileFiles() {
    std::ofstream tiles(tileChoice.outDir / TILE_FILE);
    tiles << "N,tile\n";
    for (const auto& [batch, index] : tileChoice.chosen) {
        tiles << batch << "," << DPP_TILES[index].width << "x" << DPP_TILES[index].height << "\n";
    }
    std::ofstream sweep(tileChoice.outDir / TILE_SWEEP_FILE);
    sweep << "N,tile,median_us,p10_us,p90_us,mean_us,max_abs_err,within_tolerance,chosen\n";
    for (const auto& r : tileChoice.rows) {
        sweep << r.batch << "," << r.tile.width << "x" << r.tile.height << "," << std::fixed << std::setprecision(2)
              << r.timing.median << "," << r.timing.p10 << "," << r.timing.p90 << "," << r.timing.mean << ","
              << std::scientific << std::setprecision(3) << r.errors.max() << std::defaultfloat << ","
              << (r.correct() ? "yes" : "no") << "," << (r.chosen ? "yes" : "no") << "\n";
    }
}

// Calls f.template operator()<I>() for the tile DPP_TILES[index]
template <typename F>
void withTile(const size_t index, F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((Is == index ? (f.template operator()<Is>(), 0) : 0), ...);
    }(std::make_index_sequence<DPP_TILES.size()>{});
}

template <size_t N, int K, MorphOp OP>
TileSweepRow sweepTile(Context& ctx, const size_t index, const std::vector<float>& reference) {
    TileSweepRow row{N, DPP_TILES[index]};
    withTile(index, [&]<size_t I>() {
        const Implementation impl = makeFklDpp<N, K, OP, DPP_TILES[I].width, DPP_TILES[I].height>(ctx);
        impl.fillNaN();
        gpuErrchk(cudaDeviceSynchronize());
        impl.run(WARMUP_ITERS);
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
        row.errors = compare(impl.download(), reference);
        row.timing = computeStats(measureInterleaved(ctx, {impl})[0]);
    });
    return row;
}

// The tile of batch N: chosen the first time (main run) or read (other modes)
template <size_t N, int K, MorphOp OP>
size_t chooseTile(Context& ctx) {
    if (tileChoice.chosen.count(N)) {
        return tileChoice.chosen.at(N);
    }
    if (!tileChoice.sweep) {
        throw std::runtime_error("No MorphologyDPP tile for N = " + std::to_string(N) + " in " +
                                 (tileChoice.outDir / TILE_FILE).string());
    }
    const auto reference = morphReference<N>(ctx, Reference::FklFloat, WARMUP_ITERS, K, OP);
    const size_t first = tileChoice.rows.size();
    size_t best = DPP_TILES.size();
    for (size_t i = 0; i < DPP_TILES.size(); ++i) {
        tileChoice.rows.push_back(sweepTile<N, K, OP>(ctx, i, reference));
        const TileSweepRow& r = tileChoice.rows.back();
        std::cout << "N " << N << ": MorphologyDPP tile " << r.tile.width << "x" << r.tile.height << " median "
                  << std::fixed << std::setprecision(2) << r.timing.median << " us, max abs error " << std::scientific
                  << std::setprecision(2) << r.errors.max() << std::defaultfloat << std::endl;
        if (r.correct() &&
            (best == DPP_TILES.size() || r.timing.median < tileChoice.rows[first + best].timing.median)) {
            best = i;
        }
    }
    if (best == DPP_TILES.size()) {
        throw std::runtime_error("No MorphologyDPP tile is within the tolerance for N = " + std::to_string(N));
    }
    tileChoice.rows[first + best].chosen = true;
    tileChoice.chosen[N] = best;
    writeTileFiles();
    return best;
}

// FKL's MorphologyDPP with the tile of batch N
template <size_t N, int K, MorphOp OP>
Implementation fklDpp(Context& ctx) {
    Implementation impl;
    withTile(chooseTile<N, K, OP>(ctx), [&]<size_t I>() {
        impl = makeFklDpp<N, K, OP, DPP_TILES[I].width, DPP_TILES[I].height>(ctx);
    });
    return impl;
}

// The tiles of the run, for the platform file
inline std::string tileSummary() {
    std::ostringstream os;
    os << "MorphologyDPP tiles (TW x TH threads per block, one output value per thread): every tile of {";
    for (size_t i = 0; i < DPP_TILES.size(); ++i) {
        os << (i ? ", " : "") << DPP_TILES[i].width << "x" << DPP_TILES[i].height;
    }
    os << "} validated and timed alone for every batch size, the fastest correct one used (" << TILE_SWEEP_FILE
       << ", " << TILE_FILE << "):";
    for (const auto& [batch, index] : tileChoice.chosen) {
        os << " N = " << batch << ": " << DPP_TILES[index].width << "x" << DPP_TILES[index].height << ";";
    }
    os << "\n";
    return os.str();
}

// ------------------------------------------------------------------------------------------------
// Platform file
// ------------------------------------------------------------------------------------------------

// The workload of a program with a K x K mask, operation op and the given batch sizes
template <size_t B>
void describeWorkload(std::ostream& os, const int k, const MorphOp op, const std::array<size_t, B>& batches) {
    std::ostringstream list;
    for (size_t i = 0; i < B; ++i) {
        list << (i ? ", " : "") << batches[i];
    }
    os << "Workload (Experiment 3, neighborhood operations): " << geometry::describe(false, list.str())
       << ". Unlike Figure 1 there is no AddBorder: the resized regions go to the morphology, " << name(op) << " with a "
       << k << "x" << k
       << " rectangular mask anchored at its center ("
       << (op == MorphOp::Erode ? "one pass of minimum" : "a pass of minimum, then a pass of maximum")
       << ", per channel), replicating the border of the " << PLANE_W << "x" << PLANE_H << " image in every pass; then "
       << "Mul (1/255), Sub (0.485, 0.456, 0.406), Div (0.229, 0.224, 0.225), float3 tensor of N planes " << PLANE_W
       << "x" << PLANE_H << "\n"
       << "FKL DPP: one executeErode / executeOpen call, one kernel: FKL's MorphologyDPP<float3, TW, TH, " << k << ", "
       << k << "> over the N regions (one grid z plane each), reading them through the BVF chain ReadNV12 -> YUVtoRGB "
       << "-> Crop -> Resize fused into its reads, staging them per tile with the halo of its passes in shared memory, "
       << "and writing through Mul -> Sub -> Div -> TensorWrite fused into its write: no intermediates in memory, "
       << "float3 from the color conversion to the write.\n"
       << tileSummary()
       << "References: the Figure 1 references without the letterbox, normalized, then the morphology on the host "
       << "(exact after the normalization, which is non-decreasing in every channel).\n";
}

} // namespace bench::neighborhood

#endif // BVF_BENCH_NEIGHBORHOOD_COMMON_H
