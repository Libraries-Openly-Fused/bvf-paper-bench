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

// Experiment 3, part (a): the morphology written as BVF operations (ErodeBVF, DilateBVF) against FKL's MorphologyDPP,
// on the pipeline of the neighborhood experiment (common/bench_neighborhood.h), for one mask size and number of passes
// (NEIGHBORHOOD_K, NEIGHBORHOOD_PASSES: one build per combination, fkl-neighborhood/CMakeLists.txt) and the batch
// sizes PART_A_BATCHES. The harness is common/bench_gpu.h.
//
//   fkl_bvf_ops  one executeOperations call, one kernel: ReadNV12 -> YUVtoRGB -> Crop -> Resize -> ErodeBVF<K>
//                [-> DilateBVF<K>] -> Mul -> Sub -> Div -> Write. Every output value evaluates the chain before the
//                morphology K^2 times per pass (K^4 for the opening), and each evaluation of the resize reads four
//                values of the color conversion: 4 K^2 (erode) or 4 K^4 (open) NV12 reads and conversions per output
//                value
//   fkl_dpp      one MorphologyDPP call, one kernel, the same chain fused into its reads and writes, every resized
//                value of a tile and its halo computed once (see neighborhood_common.h)

#include "neighborhood_common.h"

#ifndef NEIGHBORHOOD_K
#error "Define NEIGHBORHOOD_K, the mask size (fkl-neighborhood/CMakeLists.txt does)"
#endif
#ifndef NEIGHBORHOOD_PASSES
#error "Define NEIGHBORHOOD_PASSES, 1 (erode) or 2 (open) (fkl-neighborhood/CMakeLists.txt does)"
#endif

namespace bench::neighborhood {

constexpr int K = NEIGHBORHOOD_K;
constexpr MorphOp OP = NEIGHBORHOOD_PASSES == 1 ? MorphOp::Erode : MorphOp::Open;
static_assert(NEIGHBORHOOD_PASSES == 1 || NEIGHBORHOOD_PASSES == 2, "1 pass (erode) or 2 passes (open)");

struct BvfOpsVsDpp {
    static constexpr const char* title = "FKL BVF morphology operations vs FKL MorphologyDPP (fkl-neighborhood)";
    static constexpr const char* fklName = "fkl_dpp";
    static constexpr const char* fklLabel = "FKL DPP";
    static constexpr const char* fklShortLabel = "DPP";
    static constexpr auto batches = PART_A_BATCHES;

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        impls.push_back(makeFklNaive<N, K, OP>(ctx));
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

    static void setup(const std::filesystem::path& outDir, const std::string& mode) { setupTileChoice(outDir, mode); }

    static void workload(std::ostream& os) { describeWorkload(os, K, OP, batches); }

    static void describe(std::ostream& os, const std::filesystem::path&) {
        const long long k2 = static_cast<long long>(K) * K;
        os << "Morphology: " << name(OP) << ", " << K << "x" << K << " (" << NEIGHBORHOOD_PASSES << " pass"
           << (NEIGHBORHOOD_PASSES == 1 ? "" : "es") << "), batch sizes {";
        for (size_t i = 0; i < batches.size(); ++i) {
            os << (i ? ", " : "") << batches[i];
        }
        os << "}\n"
           << "FKL BVF operations pipeline (fkl_bvf_ops): one executeOperations call, one kernel: ReadNV12 -> YUVtoRGB "
              "-> Crop -> Resize -> ErodeBVF<" << K << ">"
           << (OP == MorphOp::Open ? " -> DilateBVF<" + std::to_string(K) + ">" : std::string()) << " -> Mul -> Sub "
           << "-> Div -> Write (common/bench_neighborhood_ops.h). Every output value evaluates the chain before the "
              "morphology " << (OP == MorphOp::Erode ? k2 : k2 * k2) << " times (K^" << 2 * NEIGHBORHOOD_PASSES
           << "), and every evaluation of the bilinear resize reads 4 values of the color conversion: "
           << 4 * (OP == MorphOp::Erode ? k2 : k2 * k2) << " NV12 reads and conversions per output value.\n";
    }

    static std::string diagnostic(const Context&) { return std::string(); }

    static bool extraMode(Context&, const std::string&, const std::filesystem::path&) { return false; }
};

} // namespace bench::neighborhood

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::neighborhood::BvfOpsVsDpp>(argc, argv);
}
