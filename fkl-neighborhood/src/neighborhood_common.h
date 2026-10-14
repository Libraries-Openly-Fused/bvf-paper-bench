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
// MorphologyDPP box blur with the choice of its tile, FKL's pipeline with the BVF operation BoxBVF, and the
// description of the workload for the platform file. The programs run on the harness of common/bench_gpu.h through
// its optional Program members (regionSize, fklImplementation, reference, workload...).

#ifndef BVF_BENCH_NEIGHBORHOOD_COMMON_H
#define BVF_BENCH_NEIGHBORHOOD_COMMON_H

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

// The first N crops of iteration t on FKL's CPU backend: ReadNV12 -> YUVtoRGB -> Crop, in float (FKL's conversion),
// or with the RGB rounded to 8 bits right after the conversion as in cpuReference (common/bench_gpu.h), like NPP's 8
// bit RGB frame. N planes of PLANE_W x PLANE_H interleaved RGB, one after the other.
template <size_t N, bool EIGHT_BIT>
std::vector<double> cropRgb(const Context& ctx, const size_t t) {
    const auto crops = fklRects<N>(ctx, t);
    fk::Stream_<fk::ParArch::CPU> cpuStream;
    fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, N, 1, fk::MemType::Host);
    const auto rgbRead = fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameHost).then(YUVToRGB::build());
    const auto write = fk::TensorWrite<fk::float3>::build(out.ptr());
    if constexpr (EIGHT_BIT) {
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
            rgbRead.then(fk::NormalizePixel<fk::float3, fk::ColorDepth::p8bit>::build())
                .then(fk::Add<fk::float3>::build(fk::make_set<fk::float3>(QUANTIZATION_OFFSET)))
                .then(fk::SaturateDenormalizePixel<fk::float3, fk::uchar3, fk::ColorDepth::p8bit>::build())
                .then(fk::Cast<fk::uchar3, fk::float3>::build()),
            fk::Crop<>::build(crops), write);
    } else {
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream, rgbRead, fk::Crop<>::build(crops), write);
    }
    cpuStream.sync();
    const float* data = reinterpret_cast<const float*>(out.ptr().data);
    return std::vector<double>(data, data + N * PLANE_FLOATS);
}

// The reference of conventions r for the regions of iteration t, a k x k box in passes passes: the crops (float or 8
// bit RGB), the box in double, rounded as the implementation rounds it, then normalized (boxReference,
// common/bench_neighborhood.h)
//   FklFloat         FKL's float RGB, float box (FKL, NPPDx with its float write)
//   FklFloatRounded  FKL's float RGB, box rounded half up to 8 bits (NPPDx's 8 bit write: NPPDx+NPP)
//   Fkl8bit          8 bit RGB, float box (NPP's 32f box, OpenCV's float filters)
//   Fkl8bitFloorBox  8 bit RGB, box rounded down, floor(sum / k^2) (NPP's 8 bit box)
template <size_t N>
std::vector<float> boxReferenceOf(const Context& ctx, const Reference r, const size_t t, const int k,
                                  const int passes) {
    switch (r) {
        case Reference::FklFloat:
            return boxReference(cropRgb<N, false>(ctx, t), N, k, passes, Rounding::None, MUL, SUB, DIV);
        case Reference::FklFloatRounded:
            return boxReference(cropRgb<N, false>(ctx, t), N, k, passes, Rounding::HalfUp, MUL, SUB, DIV);
        case Reference::Fkl8bit:
            return boxReference(cropRgb<N, true>(ctx, t), N, k, passes, Rounding::None, MUL, SUB, DIV);
        case Reference::Fkl8bitFloorBox:
            return boxReference(cropRgb<N, true>(ctx, t), N, k, passes, Rounding::Floor, MUL, SUB, DIV);
        default: throw std::runtime_error("No box reference with NPP's resize sampling: there is no resize");
    }
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

// The read chain up to the crop, as one fused read IOp (Backwards Vertical Fusion), and the pointwise tail with the
// write, as one fused write IOp (Vertical Fusion)
template <size_t N>
auto cropRead(const Context& ctx, const std::array<fk::Rect, N>& regions) {
    return fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice)
        .then(YUVToRGB::build())
        .then(fk::Crop<>::build(regions));
}

inline auto normalizedWrite(const fk::Tensor<fk::float3>& out) {
    return fk::Mul<fk::float3>::build(mulValue)
        .then(fk::Sub<fk::float3>::build(subValue))
        .then(fk::Div<fk::float3>::build(divValue))
        .then(fk::TensorWrite<fk::float3>::build(out.ptr()));
}

// FKL's MorphologyDPP box blur with a TW x TH tile: one executeBoxBlur call (FusedKernelLibrary PR #393), one kernel
// for the whole pipeline. The DPP reads the crops through the BVF chain and writes through the VF tail, so nothing is
// in memory between the NV12 frame and the output; its grid has one z plane per crop. Every pass sums its window (an
// Add reducer), and the Mul by 1 / K^(2 PASSES) runs in its write, in front of the normalization. Its shared memory
// stages the tile, plus the halo of every pass, of the K x K box.
template <size_t N, int K, int PASSES, int TW, int TH>
Implementation makeFklDpp(Context& ctx) {
    auto out = planesTensor<N>();
    const auto regions = fklRegionSets<N>(ctx);
    Implementation impl{"fkl_dpp", Reference::FklFloat};
    impl.run = [&ctx, out, regions](size_t t) {
        using Details = fk::MorphologyDPPDetails<fk::float3, TW, TH, K, K>;
        const Details details{PLANE_W, PLANE_H, K, K, K / 2, K / 2};
        beforeKernel();
        if (!fk::executeBoxBlur<PASSES>(details, cropRead<N>(ctx, (*regions)[t]), normalizedWrite(*out),
                                        *ctx.fkStream)) {
            throw std::runtime_error("executeBoxBlur rejected its details");
        }
    };
    setPlanesIO<N>(impl, out);
    return impl;
}

// FKL with the box as a BVF operation (common/bench_neighborhood_ops.h): one executeOperations call, one kernel, in
// which every output value evaluates the read chain (an NV12 read and a color conversion) K x K times per pass (K^4
// times for two passes)
template <size_t N, int K, int PASSES>
Implementation makeFklNaive(Context& ctx) {
    static_assert(PASSES == 1 || PASSES == 2, "One or two passes");
    auto out = planesTensor<N>();
    const auto regions = fklRegionSets<N>(ctx);
    Implementation impl{"fkl_bvf_ops", Reference::FklFloat};
    impl.run = [&ctx, out, regions](size_t t) {
        const auto read = fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice);
        beforeKernel();
        if constexpr (PASSES == 1) {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream, read, YUVToRGB::build(),
                fk::Crop<>::build((*regions)[t]), fk::BoxBVF<K>::build(), fk::Mul<fk::float3>::build(mulValue),
                fk::Sub<fk::float3>::build(subValue), fk::Div<fk::float3>::build(divValue),
                fk::TensorWrite<fk::float3>::build(out->ptr()));
        } else {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream, read, YUVToRGB::build(),
                fk::Crop<>::build((*regions)[t]), fk::BoxBVF<K>::build(), fk::BoxBVF<K>::build(),
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

template <size_t N, int K, int PASSES>
TileSweepRow sweepTile(Context& ctx, const size_t index, const std::vector<float>& reference) {
    TileSweepRow row{N, DPP_TILES[index]};
    withTile(index, [&]<size_t I>() {
        const Implementation impl = makeFklDpp<N, K, PASSES, DPP_TILES[I].width, DPP_TILES[I].height>(ctx);
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
template <size_t N, int K, int PASSES>
size_t chooseTile(Context& ctx) {
    if (tileChoice.chosen.count(N)) {
        return tileChoice.chosen.at(N);
    }
    if (!tileChoice.sweep) {
        throw std::runtime_error("No MorphologyDPP tile for N = " + std::to_string(N) + " in " +
                                 (tileChoice.outDir / TILE_FILE).string());
    }
    const auto reference = boxReferenceOf<N>(ctx, Reference::FklFloat, WARMUP_ITERS, K, PASSES);
    const size_t first = tileChoice.rows.size();
    size_t best = DPP_TILES.size();
    for (size_t i = 0; i < DPP_TILES.size(); ++i) {
        tileChoice.rows.push_back(sweepTile<N, K, PASSES>(ctx, i, reference));
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

// FKL's MorphologyDPP box blur with the tile of batch N
template <size_t N, int K, int PASSES>
Implementation fklDpp(Context& ctx) {
    Implementation impl;
    withTile(chooseTile<N, K, PASSES>(ctx), [&]<size_t I>() {
        impl = makeFklDpp<N, K, PASSES, DPP_TILES[I].width, DPP_TILES[I].height>(ctx);
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

// The workload of a program with a k x k box in passes passes and the given batch sizes
template <size_t B>
void describeWorkload(std::ostream& os, const int k, const int passes, const std::array<size_t, B>& batches) {
    std::ostringstream list;
    for (size_t i = 0; i < B; ++i) {
        list << (i ? ", " : "") << batches[i];
    }
    os << "bvf-paper-bench: commit " << BENCH_REPO_COMMIT << " (FKL: the commit above)\n"
       << "Workload (Experiment 3, neighborhood operations): " << FRAME_W << "x" << FRAME_H << " NV12 frame (uniform "
       << "random limited-range values, Y in [" << LUMA_MIN << ", " << LUMA_MAX << "], U and V in [" << CHROMA_MIN
       << ", " << CHROMA_MAX << "], seed " << FRAME_SEED << "), NV12->RGB BT.601 limited range, N in {" << list.str()
       << "} crops of " << CROP_W << "x" << CROP_H << " per iteration, new every iteration (warm-up and timed) and "
       << "precomputed before the loop (seed " << REGION_SEED << ", batch N uses the first N of every iteration), at "
       << "uniform random even positions at least " << REGION_MARGIN << " pixels from the frame edges "
       << "(cropRegionSets, common/bench_neighborhood.h). No resize and no added rows: each crop is blurred with a "
       << k << "x" << k << " box (the mean of the window centered on the value, per channel, replicating the edge of "
       << "the crop" << (passes > 1 ? "; " + std::to_string(passes) + " passes, each replicating the edge of the crop "
                                          "it reads" : std::string(", one pass"))
       << "), then Mul (1/255), Sub (0.485, 0.456, 0.406), Div (0.229, 0.224, 0.225), float3 tensor of N planes "
       << PLANE_W << "x" << PLANE_H << "\n"
       << "FKL DPP: one executeBoxBlur<" << passes << "> call (FusedKernelLibrary PR #393), one kernel: FKL's "
       << "MorphologyDPP<float3, TW, TH, " << k << ", " << k << "> with an Add reducer per pass over the N crops (one "
       << "grid z plane each), reading them through the BVF chain ReadNV12 -> YUVtoRGB -> Crop fused into its reads, "
       << "staging them per tile with the halo of its passes in shared memory, and writing through Mul (1/" << k * k
       << (passes > 1 ? "^" + std::to_string(passes) : std::string()) << ") -> Mul -> Sub -> Div -> TensorWrite fused "
       << "into its write: no intermediates in memory, float3 from the color conversion to the write.\n"
       << tileSummary()
       << "References: the crops of FKL's CPU backend (ReadNV12 -> YUVtoRGB -> Crop), in float or with the RGB "
       << "rounded to 8 bits after the conversion (the 8 bit RGB frame of the libraries that convert with NPP), the box "
       << "in double, rounded as the implementation rounds it (none, half up for NPPDx's 8 bit write, down for NPP's 8 "
       << "bit box), then the normalization in float (boxReference, common/bench_neighborhood.h): the box before the "
       << "normalization, as every pipeline.\n";
}

} // namespace bench::neighborhood

#endif // BVF_BENCH_NEIGHBORHOOD_COMMON_H
