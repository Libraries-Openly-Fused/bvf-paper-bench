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

// Figure 1 pipeline of the Backwards Vertical Fusion paper, as one kernel in each library:
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// One 3840x2160 NV12 frame in device memory and N crops, with the sizes and positions of common/bench_geometry.h
// (by default 32x16 crops resized to 64x32), NV12->RGB BT.601 limited range, bilinear resize, black rows added on top
// and bottom up to a square, then (x * mul - sub) / div, written as float3 into a tensor of N planes.
//
//   NPPDx: one CUDA kernel built from NPPDx operations (nppdx_figure1.h), one or more thread blocks per crop (one per
//          tile of the crop, blockIdx.z is the crop): NV12 ingest of the tile -> limited to full range -> ColorConvert
//          BT.601 -> Resize J:K bilinear in shared memory, then AddBorder, Mul, Sub, Div and the float3 write written
//          by hand in the same kernel, because NPPDx has no border operation, its affine map only takes integer
//          constants, and it has no float output format. Every tile and block size NPPDx supports for these crops is
//          validated and timed, and the fastest correct one is compared with FKL.
//   FKL:   one executeOperations call (Backwards Vertical Fusion), the paper's Listing 2.
//
// benchmark_figure1_nppdx_npp.cu compares FKL with the same pipeline in NPPDx and NPP without hand written pixel
// processing code.
// The shared parts (FKL, references, validation, choice of the NPPDx configuration, timing, modes) are in
// figure1_common.h.

#include "benchmarks/figure1_common.h"
#include "benchmarks/nppdx_figure1.h"

namespace figure1 {

// Ingest + limited to full range + ColorConvert only: writes the whole input tile (with its halo) so that the
// conversion can be checked alone
template <typename Ingest, typename FromLimitedY, typename FromLimitedUV, typename Convert, typename InputTile>
__global__ void nppdxConvertOnlyKernel(const uint8_t* y, const uint8_t* uv, const int pitch, const int2 o, float* out) {
    extern __shared__ unsigned char smem[];
    auto tiles = nppdx::shared_memory::slice_into_tile_storage<InputTile>(smem);
    auto inputTile = std::get<0>(tiles);
    const nppdx::ImageLayout<const uint8_t, 2> crop{{{y + static_cast<size_t>(o.y) * pitch + o.x,
                                                      uv + static_cast<size_t>(o.y / 2) * pitch + o.x}},
                                                    {{pitch, pitch}}};
    Ingest().execute(crop, inputTile, CROP_W, CROP_H);
    FromLimitedY().execute(inputTile, CROP_W, CROP_H);
    FromLimitedUV().execute(inputTile, CROP_W, CROP_H);
    Convert().execute(inputTile, CROP_W, CROP_H);
    __syncthreads();
    constexpr int TW = InputTile::ChannelSliceType::width, TH = InputTile::ChannelSliceType::height;
    for (int idx = threadIdx.x; idx < TW * TH; idx += blockDim.x) {
        for (int c = 0; c < 3; ++c) {
            out[idx * 3 + c] = inputTile.channel(c)(idx / TW, idx % TW);
        }
    }
}

// The NPPDx pipeline of this benchmark for figure1_common.h: the single kernel of nppdx_figure1.h
struct NppdxPipeline {
    static constexpr const char* implementation = "nppdx";
    static constexpr const char* label = "NPPDx";
    static constexpr const char* name = "benchmark_figure1_bvf";
    // Every configuration tried: the candidate tiles NPPDx supports for these crops, each with NPPDx's recommended
    // block (block 0) and with BlockDim 64, 128, 256 and 512
    static constexpr auto configs = nppdxFigure1Configs<NPPDX_SM>();
    static constexpr float tolerance = TOLERANCE;
    static constexpr const char* referenceName =
        "the same host chain with the resize sampled at pixel centers ((dst + 0.5) * scale - 0.5, NPPDx's convention)";
    static std::string description() {
        return "one kernel; NPPDx operations (NV12 ingest of the crop, FromLimitedLuma and FromLimitedChroma, "
               "ColorConvert BT.601, Resize " + std::to_string(RESIZE_J) + ":" + std::to_string(RESIZE_K) +
               " bilinear) followed by hand written CUDA code for the border, the normalization and the float3 write "
               "(NPPDx has no border operation and no floating point output).";
    }

    template <size_t N, NppdxConfig C>
    using kernel = NppdxFigure1<NPPDX_SM, N, C.block, C.tileX, C.tileY>;

    template <size_t N>
    struct Buffers {
        float* out{nullptr};
        Buffers() { gpuErrchk(cudaMalloc(&out, N * OUT_FLOATS * sizeof(float))); }
        ~Buffers() { cudaFree(out); }
        Buffers(const Buffers&) = delete;
        Buffers& operator=(const Buffers&) = delete;
    };

    // Fills the output with NaN before a validation, so that a value the kernel does not write is an error
    template <size_t N>
    static void poison(const Buffers<N>& b, cudaStream_t stream) {
        gpuErrchk(cudaMemsetAsync(b.out, 0xFF, N * OUT_FLOATS * sizeof(float), stream));
    }

    // NPPDx samples the bilinear resize at pixel centers, FKL at dst * scale (see --analyze)
    template <size_t N>
    static std::vector<float> reference(const Workload& w, const std::array<fk::Rect, N>& crops) {
        return pixelCenterReference<N>(w, crops, false);
    }

    template <size_t N, NppdxConfig C, typename Before>
    static void run(Workload& w, const CropOrigins<N>& origins, const Buffers<N>& b, Before&&) {
        kernel<N, C>::run(w.stream, w.y, w.uv, w.pitch, origins, norm, b.out);
    }

    template <size_t N, NppdxConfig C>
    static void runKernel(Workload& w, const CropOrigins<N>& origins, const Buffers<N>& b) {
        run<N, C>(w, origins, b, noFlush);
    }

    static void analyze(Workload& w) {
        std::cout << "### NPPDx recommended block configuration" << std::endl;
        analyzeBlock<0>(w);
        std::cout << "### BlockDim<256>" << std::endl;
        analyzeBlock<256>(w);
    }

    // What NPPDx computes, with the first (largest) tile of configs and block size BlockX
    template <unsigned BlockX>
    static void analyzeBlock(Workload& w) {
        using Nppdx = NppdxFigure1<NPPDX_SM, 1, BlockX, configs[0].tileX, configs[0].tileY>;
        Nppdx::init();
        std::cout << "Recommended block sizes: Ingest " << Nppdx::Ingest::block_dim.x << ", Convert "
                  << Nppdx::Convert::block_dim.x << ", Resize " << Nppdx::Resize::block_dim.x << std::endl;
        std::cout << "NPPDx kernel: tile " << configs[0].tileX << "x" << configs[0].tileY << ", block "
                  << Nppdx::block.x << "x" << Nppdx::block.y << "x" << Nppdx::block.z << ", dynamic shared memory " << Nppdx::smem
                  << " bytes, resize halo (l,r,t,b) " << Nppdx::halo.left_top.x << "," << Nppdx::halo.right_bottom.x
                  << "," << Nppdx::halo.left_top.y << "," << Nppdx::halo.right_bottom.y << std::endl;
        const CropOrigins<1> origins{{int2{static_cast<int>(w.crops[0].x), static_cast<int>(w.crops[0].y)}}};
        float* out = nullptr;
        gpuErrchk(cudaMalloc(&out, OUT_FLOATS * sizeof(float)));
        Nppdx::run(w.stream, w.y, w.uv, w.pitch, origins, norm, out);
        gpuErrchk(cudaStreamSynchronize(w.stream));
        const auto nppdx = download(out, OUT_FLOATS);
        gpuErrchk(cudaFree(out));
        const char* names[] = {"dst*scale (FKL)", "pixel centers", "dst*scale-0.25 (NPP)"};
        for (bool clampRGB : {false, true}) {
            for (Sampling s : {Sampling::DstTimesScale, Sampling::PixelCenters, Sampling::NppMinusQuarter}) {
                const Error e = compare(nppdx, hostVariant(w, w.crops[0], clampRGB, s, false).plane);
                std::cout << "NPPDx vs host reference, "
                          << (clampRGB ? "RGB clamped to [0,255], " : "RGB not clamped,      ")
                          << names[static_cast<int>(s)] << ": max " << e.max << ", mean " << e.mean << std::endl;
            }
        }
        {
            using In = typename Nppdx::InputTile;
            constexpr int TW = In::ChannelSliceType::width, TH = In::ChannelSliceType::height;
            std::cout << "Input tile storage " << TW << "x" << TH << ", output tile storage "
                      << Nppdx::OutputTile::ChannelSliceType::width << "x"
                      << Nppdx::OutputTile::ChannelSliceType::height << ", temp tile storage "
                      << Nppdx::TempTile::ChannelSliceType::width << "x" << Nppdx::TempTile::ChannelSliceType::height
                      << std::endl;
            float* tile = nullptr;
            gpuErrchk(cudaMalloc(&tile, TW * TH * 3 * sizeof(float)));
            constexpr size_t smemIn = nppdx::shared_memory::compute_total_tile_storage<In>();
            constexpr auto convertOnly =
                nppdxConvertOnlyKernel<typename Nppdx::Ingest, typename Nppdx::FromLimitedY,
                                       typename Nppdx::FromLimitedUV, typename Nppdx::Convert, In>;
            if constexpr (smemIn > 48 * 1024) {
                gpuErrchk(cudaFuncSetAttribute(convertOnly, cudaFuncAttributeMaxDynamicSharedMemorySize, smemIn));
            }
            convertOnly<<<1, Nppdx::block, smemIn, w.stream>>>(w.y, w.uv, w.pitch, origins.origin[0], tile);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaStreamSynchronize(w.stream));
            const auto t = download(tile, TW * TH * 3);
            gpuErrchk(cudaFree(tile));
            // Compare tile pixel (tx, ty) with crop pixel (tx - hl, ty - ht), clamped to the crop
            const int hl = Nppdx::halo.left_top.x, ht = Nppdx::halo.left_top.y;
            float maxRaw = 0.f, maxClamped = 0.f;
            const fk::Rect& r = w.crops[0];
            for (int ty = 0; ty < TH; ++ty) {
                for (int tx = 0; tx < TW; ++tx) {
                    const int cx = std::min(CROP_W - 1, std::max(0, tx - hl));
                    const int cy = std::min(CROP_H - 1, std::max(0, ty - ht));
                    const fk::float3 rgb = YUVToRGB::exec(
                        w.frameHost->readAt(fk::Point{static_cast<int>(r.x) + cx, static_cast<int>(r.y) + cy, 0}));
                    const float c3[3] = {rgb.x, rgb.y, rgb.z};
                    for (int c = 0; c < 3; ++c) {
                        const float v = t[(ty * TW + tx) * 3 + c];
                        maxRaw = std::max(maxRaw, std::abs(v - c3[c]));
                        maxClamped = std::max(maxClamped, std::abs(v - std::min(255.f, std::max(0.f, c3[c]))));
                    }
                }
            }
            std::cout << "Ingest + FromLimited + ColorConvert tile vs FKL conversion (8 bit levels): max diff not "
                         "clamped " << maxRaw << ", clamped to [0,255] " << maxClamped << std::endl;
        }
        const auto ref = hostVariant(w, w.crops[0], false, Sampling::DstTimesScale, false).plane;
        // The first rows of the top border and of the resized crop, the middle, the last resized row and the first row
        // of the bottom border (when there is one)
        for (const int y : {0, BORDER_TOP, BORDER_TOP + 1, OUT_H / 2 - 2, BORDER_TOP + RES_H - 1, BORDER_TOP + RES_H}) {
            if (y >= OUT_H) {
                continue;
            }
            for (const int x : {0, 1, OUT_W / 2 - 1, OUT_W - 1}) {
                const size_t i = (static_cast<size_t>(y) * OUT_W + x) * 3;
                std::cout << "  (" << x << "," << y << ") NPPDx " << nppdx[i] << " " << nppdx[i + 1] << " "
                          << nppdx[i + 2] << " | FKL reference " << ref[i] << " " << ref[i + 1] << " " << ref[i + 2]
                          << std::endl;
            }
        }
    }
};

} // namespace figure1

int main(int argc, char** argv) {
    return figure1::figure1Main<figure1::NppdxPipeline>(argc, argv);
}
