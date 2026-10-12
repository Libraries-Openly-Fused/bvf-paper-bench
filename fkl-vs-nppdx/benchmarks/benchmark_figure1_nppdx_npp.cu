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

// Figure 1 pipeline of the Backwards Vertical Fusion paper with NPPDx and NPP, without hand written pixel processing
// code, and with FKL:
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// The workload of benchmark_figure1_bvf.cu (common/bench_geometry.h; NV12->RGB BT.601 limited range, bilinear resize,
// black rows added on top and bottom up to a square, then (x * mul - sub) / div, written as float3 into a tensor of N
// planes).
//
//   NPPDx+NPP: only what NPPDx and NPP provide (nppdx_npp_figure1.h): a kernel that only calls NPPDx operations
//              (NV12 ingest of the crop -> limited to full range -> ColorConvert BT.601 -> Resize J:K bilinear -> 8 bit
//              RGB write), one or more thread blocks per crop (one per tile, blockIdx.z is the crop), then NPP, since
//              NPPDx has no border and no float output: CopyConstBorder per crop (NPP has no batched border), then
//              Convert 8u -> 32f, MulC, SubC and DivC on all crops at once (N + 5 kernels). Every tile and block size NPPDx supports for these crops is validated and timed (the NPPDx
//              kernel alone), and the fastest correct one is compared with FKL.
//   FKL:       one executeOperations call (Backwards Vertical Fusion), the paper's Listing 2.
//
// The shared parts (FKL, references, validation, choice of the NPPDx configuration, timing, modes) are in
// figure1_common.h.

#include "benchmarks/figure1_common.h"
#include "benchmarks/nppdx_npp_figure1.h"

namespace figure1 {

// The NPPDx pipeline of this benchmark for figure1_common.h: the NPPDx kernel and the NPP calls of nppdx_npp_figure1.h
struct NppdxNppPipeline {
    static constexpr const char* implementation = "nppdx_npp";
    static constexpr const char* label = "NPPDx+NPP";
    static constexpr const char* name = "benchmark_figure1_nppdx_npp";
    // Every configuration tried: the candidate tiles NPPDx supports for these crops (with the 8 bit RGB write), each
    // with NPPDx's recommended block (block 0) and with BlockDim 64, 128, 256 and 512
    static constexpr auto configs = nppdxRgbConfigs<NPPDX_SM>();
    // The resized RGB is rounded to 8 bits, and a value on a half level can round differently than on the host
    static constexpr float tolerance = TOLERANCE_8BIT;
    static constexpr const char* referenceName =
        "the same host chain with the resize sampled at pixel centers ((dst + 0.5) * scale - 0.5, NPPDx's convention) "
        "and the resized RGB rounded to 8 bits as NPPDx writes it (+ 0.5, clamped to [0, 255])";
    static std::string description() {
        return "no hand written pixel processing code: a kernel that only calls NPPDx operations (NV12 ingest of the "
               "crop, FromLimitedLuma and FromLimitedChroma, ColorConvert BT.601, Resize " + std::to_string(RESIZE_J) +
               ":" + std::to_string(RESIZE_K) + " bilinear, 8 bit RGB write), then NPP with the function it offers for "
               "each step, as in the GPU benchmark's NPP pipeline: nppiCopyConstBorder_8u_C3R once per crop (NPP has "
               "no batched border), then nppiConvert_8u32f_C3R, nppiMulC_32f_C3IR, nppiSubC_32f_C3IR and "
               "nppiDivC_32f_C3IR on the N planes as one image; N + 5 kernels. The resized RGB is rounded to 8 bits "
               "before the normalization. The times include the NPP calls; the configuration sweep times the NPPDx "
               "kernel alone.";
    }

    template <size_t N, NppdxConfig C>
    using kernel = NppdxRgb<NPPDX_SM, N, C.block, C.tileX, C.tileY>;

    // The resized 8 bit RGB crops the NPPDx kernel writes, NPP's bordered planes (in the NppTail) and the float3 output
    template <size_t N>
    struct Buffers {
        uint8_t* resized{nullptr};
        float* out{nullptr};
        NppTail tail{N};
        Buffers() {
            gpuErrchk(cudaMalloc(&resized, N * RES_W * RES_H * 3));
            gpuErrchk(cudaMalloc(&out, N * OUT_FLOATS * sizeof(float)));
        }
        ~Buffers() {
            cudaFree(resized);
            cudaFree(out);
        }
        Buffers(const Buffers&) = delete;
        Buffers& operator=(const Buffers&) = delete;
    };

    // Fills every buffer the pipeline writes with values no correct execution leaves (NaN floats), before a validation
    template <size_t N>
    static void poison(const Buffers<N>& b, cudaStream_t stream) {
        gpuErrchk(cudaMemsetAsync(b.resized, 0xAB, N * RES_W * RES_H * 3, stream));
        b.tail.poison(stream);
        gpuErrchk(cudaMemsetAsync(b.out, 0xFF, N * OUT_FLOATS * sizeof(float), stream));
    }

    template <size_t N>
    static std::vector<float> reference(const Workload& w, const std::array<fk::Rect, N>& crops) {
        return pixelCenterReference<N>(w, crops, true);
    }

    // One execution: the NPPDx kernel, then NPP's border, 8u -> 32f and normalization; before() runs before every NPP
    // call (the benchmark flushes before an execution itself)
    template <size_t N, NppdxConfig C, typename Before>
    static void run(Workload& w, const CropOrigins<N>& origins, const Buffers<N>& b, Before&& before) {
        kernel<N, C>::run(w.stream, w.y, w.uv, w.pitch, origins, b.resized);
        b.tail.run(b.resized, b.out, norm, w.npp, before);
    }

    template <size_t N, NppdxConfig C>
    static void runKernel(Workload& w, const CropOrigins<N>& origins, const Buffers<N>& b) {
        kernel<N, C>::run(w.stream, w.y, w.uv, w.pitch, origins, b.resized);
    }

    static void analyze(Workload& w) {
        std::cout << "### NPPDx recommended block configuration" << std::endl;
        analyzeBlock<0>(w);
        std::cout << "### BlockDim<256>" << std::endl;
        analyzeBlock<256>(w);
    }

    // What NPPDx computes: the NPPDx kernel alone (N = 1, the first crop, the first and largest tile of configs, block
    // size BlockX) against the host reference with every sampling convention, with and without clamping RGB before the
    // resize, always rounded to 8 bits as NPPDx writes it
    template <unsigned BlockX>
    static void analyzeBlock(Workload& w) {
        using Nppdx = NppdxRgb<NPPDX_SM, 1, BlockX, configs[0].tileX, configs[0].tileY>;
        Nppdx::init();
        std::cout << "Recommended block sizes: Ingest " << Nppdx::Ingest::block_dim.x << ", Convert "
                  << Nppdx::Convert::block_dim.x << ", Resize " << Nppdx::Resize::block_dim.x << ", Exgest "
                  << Nppdx::Exgest::block_dim.x << std::endl;
        std::cout << "NPPDx kernel: tile " << configs[0].tileX << "x" << configs[0].tileY << ", block "
                  << Nppdx::block.x << "x" << Nppdx::block.y << "x" << Nppdx::block.z << ", dynamic shared memory " << Nppdx::smem
                  << " bytes, resize halo (l,r,t,b) " << Nppdx::halo.left_top.x << "," << Nppdx::halo.right_bottom.x
                  << "," << Nppdx::halo.left_top.y << "," << Nppdx::halo.right_bottom.y << std::endl;
        const CropOrigins<1> origins{{int2{static_cast<int>(w.crops[0].x), static_cast<int>(w.crops[0].y)}}};
        uint8_t* resized = nullptr;
        gpuErrchk(cudaMalloc(&resized, RES_W * RES_H * 3));
        Nppdx::run(w.stream, w.y, w.uv, w.pitch, origins, resized);
        gpuErrchk(cudaStreamSynchronize(w.stream));
        const auto nppdx = download(resized, RES_W * RES_H * 3);
        gpuErrchk(cudaFree(resized));
        const char* names[] = {"dst*scale (FKL)", "pixel centers", "dst*scale-0.25 (NPP)"};
        for (bool clampRGB : {false, true}) {
            for (Sampling s : {Sampling::DstTimesScale, Sampling::PixelCenters, Sampling::NppMinusQuarter}) {
                const auto ref = hostVariant(w, w.crops[0], clampRGB, s, true).resized;
                int maxLevels = 0;
                size_t differing = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    const int d = std::abs(static_cast<int>(nppdx[i]) - static_cast<int>(ref[i]));
                    maxLevels = std::max(maxLevels, d);
                    differing += d != 0;
                }
                std::cout << "NPPDx 8 bit resized RGB vs host reference, "
                          << (clampRGB ? "RGB clamped to [0,255], " : "RGB not clamped,      ")
                          << names[static_cast<int>(s)] << ": " << differing << " of " << ref.size()
                          << " values differ, max " << maxLevels << " levels" << std::endl;
            }
        }
    }
};

} // namespace figure1

int main(int argc, char** argv) {
    return figure1::figure1Main<figure1::NppdxNppPipeline>(argc, argv);
}
