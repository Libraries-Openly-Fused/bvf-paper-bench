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

// The Figure 1 pipeline with NPPDx and NPP, without hand written pixel processing code: only what the two libraries
// provide. NPPDx has no host API, so its operations run inside a CUDA kernel written for them, but the kernel only
// calls NPPDx operations, in the order of the pipeline (NV12 ingest of the crop, limited to full range, color
// conversion BT.601, resize, 8 bit RGB write), one thread block per tile of a crop. NPPDx has no border operation and
// no floating point output, so the rest is NPP: nppiSet makes the N 8 bit RGB output planes black (one call for all
// of them, as one image), the NPPDx kernel then writes every resized crop between its black rows, and Convert 8u ->
// 32f, MulC, SubC and DivC run on the N planes at once. Six kernels for every N, with the intermediates in memory, and
// the resized RGB rounded to 8 bits before the normalization. The sizes are those of common/bench_geometry.h. Used by
// benchmark_figure1_nppdx_npp.cu. nppdx_figure1.h has the single kernel version, with hand written CUDA code.

#ifndef FIGURE1_NPPDX_NPP_FIGURE1_H
#define FIGURE1_NPPDX_NPP_FIGURE1_H

#include "benchmarks/nppdx_common.h"
#include "bench_npp.h"

namespace figure1 {

// The NPPDx kernel: NPPDx operations only. Each thread block processes one tile of one crop: NPPDx takes the position
// of the tile from blockIdx.x and blockIdx.y, and blockIdx.z selects the crop, which is given to NPPDx as an image of
// CROP_W x CROP_H inside the frame. NPPDx writes the resized crop as 8 bit RGB into its output plane, from row
// BORDER_TOP: a plane has the width of the resized crop, so its rows are those of NPPDx's 8 bit RGB image.
static_assert(OUT_W == RES_W, "The border adds rows only, so a resized crop is an image inside its output plane");
template <typename Ingest, typename FromLimitedY, typename FromLimitedUV, typename Convert, typename Resize,
          typename Exgest, typename InputTile, typename OutputTile, typename TempTile, size_t N>
__global__ void nppdxRgbKernel(const uint8_t* y, const uint8_t* uv, const int pitch, const CropOrigins<N> crops,
                               uint8_t* planes) {
    extern __shared__ unsigned char smem[];
    auto tiles = nppdx::shared_memory::slice_into_tile_storage<InputTile, OutputTile, TempTile>(smem);
    auto inputTile = std::get<0>(tiles);
    auto resizedTile = std::get<1>(tiles);
    auto tempTile = std::get<2>(tiles);

    const int2 o = crops.origin[blockIdx.z];
    const nppdx::ImageLayout<const uint8_t, 2> crop{{{y + static_cast<size_t>(o.y) * pitch + o.x,
                                                      uv + static_cast<size_t>(o.y / 2) * pitch + o.x}},
                                                    {{pitch, pitch}}};
    uint8_t* resizedCrop = planes + (static_cast<size_t>(blockIdx.z) * OUT_H + BORDER_TOP) * OUT_W * 3;

    Ingest().execute(crop, inputTile, CROP_W, CROP_H);
    FromLimitedY().execute(inputTile, CROP_W, CROP_H);
    FromLimitedUV().execute(inputTile, CROP_W, CROP_H);
    Convert().execute(inputTile, CROP_W, CROP_H);
    Resize().execute(inputTile, tempTile, resizedTile, CROP_W, CROP_H);
    Exgest().execute(resizedTile, resizedCrop, RES_W, RES_H);
}

// The NPPDx kernel in one configuration: TileX x TileY input tiles, block size BlockX (0: NPPDx's recommended one)
template <int Arch, size_t N, unsigned BlockX, unsigned TileX, unsigned TileY>
struct NppdxRgb : NppdxOps<Arch, BlockX, TileX, TileY> {
    using Ops = NppdxOps<Arch, BlockX, TileX, TileY>;
    using typename Ops::Ingest, typename Ops::FromLimitedY, typename Ops::FromLimitedUV, typename Ops::Convert,
        typename Ops::Resize, typename Ops::InputTile, typename Ops::OutputTile, typename Ops::TempTile;
    // 8 bit RGB write of the resized tile, as in NPPDx's resize example
    using Exgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                            nppdx::OutputFormat<nppdx::packing_format::rgb24>() +
                            nppdx::TileSize<Ops::outTileX, Ops::outTileY>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, nppdx::Halo4::make_empty())() +
                            NPPDX_MAKE_HALO(nppdx::CumulativeHalo, nppdx::Halo4::make_empty())() + nppdx::SM<Arch>() +
                            NppdxBlock<BlockX>());
    static_assert(nppdx::is_supported_v<Exgest, Arch>);
    static constexpr size_t smem = Ops::smem;
    // The write of the larger output tile is part of the kernel, so its block counts too, as in NPPDx's resize example
    static constexpr dim3 block{std::max(Ops::blockBeforeWrite, Exgest::block_dim.x), 1, 1};
    static constexpr dim3 grid = Ops::gridOf(N);

    static constexpr auto kernel = nppdxRgbKernel<Ingest, FromLimitedY, FromLimitedUV, Convert, Resize, Exgest,
                                                  InputTile, OutputTile, TempTile, N>;

    static void init() {
        if constexpr (smem > 48 * 1024) {
            gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        }
    }

    // Writes the N resized crops as 8 bit RGB into the N output planes (OUT_W x OUT_H each, one after the other), from
    // row BORDER_TOP of each; the other rows are not written
    static void run(cudaStream_t stream, const uint8_t* y, const uint8_t* uv, int pitch, const CropOrigins<N>& crops,
                    uint8_t* planes) {
        kernel<<<grid, block, smem, stream>>>(y, uv, pitch, crops, planes);
        gpuErrchk(cudaGetLastError());
    }
};

// The configurations of this kernel (with its 8 bit RGB write)
template <int Arch>
constexpr auto nppdxRgbConfigs() {
    return nppdxConfigs<Arch, true>();
}

// The NPP part of the pipeline and its 8 bit RGB planes: before the NPPDx kernel, border() makes the N planes black
// (AddBorder: the kernel then writes the resized crops between the black rows; NPP has no batched border, but the N
// planes are one image for nppiSet); after it, normalize() converts them to float and normalizes them, on the N planes
// as one image of OUT_W x N * OUT_H. normalize() runs before() before every NPP call (the GPU only benchmark with the
// L2 flushed flushes there).
class NppParts {
public:
    explicit NppParts(size_t n) : m_n(n) { gpuErrchk(cudaMalloc(&m_planes, n * OUT_W * OUT_H * 3)); }
    ~NppParts() { cudaFree(m_planes); }
    NppParts(const NppParts&) = delete;
    NppParts& operator=(const NppParts&) = delete;

    uint8_t* planes() const { return m_planes; }

    void border(const NppStreamContext& ctx) const {
        static const Npp8u black[3] = {0, 0, 0};
        NPP_CHECK(nppiSet_8u_C3R_Ctx(black, m_planes, OUT_W * 3, size(), ctx));
    }

    template <typename Before>
    void normalize(float* out, const Normalization& n, const NppStreamContext& ctx, Before&& before) const {
        const int step = OUT_W * 3 * static_cast<int>(sizeof(float));
        before();
        NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(m_planes, OUT_W * 3, out, step, size(), ctx));
        before();
        NPP_CHECK(nppiMulC_32f_C3IR_Ctx(n.mul, out, step, size(), ctx));
        before();
        NPP_CHECK(nppiSubC_32f_C3IR_Ctx(n.sub, out, step, size(), ctx));
        before();
        NPP_CHECK(nppiDivC_32f_C3IR_Ctx(n.div, out, step, size(), ctx));
    }

private:
    NppiSize size() const { return NppiSize{OUT_W, static_cast<int>(m_n) * OUT_H}; }

    size_t m_n{0};
    uint8_t* m_planes{nullptr};
};

} // namespace figure1

#endif // FIGURE1_NPPDX_NPP_FIGURE1_H
