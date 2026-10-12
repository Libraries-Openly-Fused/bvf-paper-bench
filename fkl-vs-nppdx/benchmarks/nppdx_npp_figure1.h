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
// no floating point output, so the rest is NPP, with the function NPP offers for each step: CopyConstBorder for the
// black rows, once per crop (NPP has no batched border), then Convert 8u -> 32f, MulC, SubC and DivC on the N planes
// at once. N + 5 kernels, with the intermediates in memory, and the resized RGB rounded to 8 bits before the
// normalization. The sizes are those of common/bench_geometry.h. Used by benchmark_nppdx_npp.cu.
// nppdx_figure1.h has the single kernel version, with hand written CUDA code.

#ifndef FIGURE1_NPPDX_NPP_FIGURE1_H
#define FIGURE1_NPPDX_NPP_FIGURE1_H

#include "benchmarks/nppdx_common.h"
#include "bench_npp.h"

namespace figure1 {

// The NPPDx kernel: NPPDx operations only. Each thread block processes one tile of one crop: NPPDx takes the position
// of the tile from blockIdx.x and blockIdx.y, and blockIdx.z selects the crop, which is given to NPPDx as an image of
// CROP_W x CROP_H inside the frame. The resized crops are written one after the other as 8 bit RGB.
template <typename Ingest, typename FromLimitedY, typename FromLimitedUV, typename Convert, typename Resize,
          typename Exgest, typename InputTile, typename OutputTile, typename TempTile, size_t N>
__global__ void nppdxRgbKernel(const uint8_t* y, const uint8_t* uv, const int pitch, const CropOrigins<N> crops,
                               uint8_t* resized) {
    extern __shared__ unsigned char smem[];
    auto tiles = nppdx::shared_memory::slice_into_tile_storage<InputTile, OutputTile, TempTile>(smem);
    auto inputTile = std::get<0>(tiles);
    auto resizedTile = std::get<1>(tiles);
    auto tempTile = std::get<2>(tiles);

    const int2 o = crops.origin[blockIdx.z];
    const nppdx::ImageLayout<const uint8_t, 2> crop{{{y + static_cast<size_t>(o.y) * pitch + o.x,
                                                      uv + static_cast<size_t>(o.y / 2) * pitch + o.x}},
                                                    {{pitch, pitch}}};
    uint8_t* resizedCrop = resized + static_cast<size_t>(blockIdx.z) * RES_W * RES_H * 3;

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

    // Writes the N resized crops, RES_W x RES_H 8 bit RGB each, one after the other in resized
    static void run(cudaStream_t stream, const uint8_t* y, const uint8_t* uv, int pitch, const CropOrigins<N>& crops,
                    uint8_t* resized) {
        kernel<<<grid, block, smem, stream>>>(y, uv, pitch, crops, resized);
        gpuErrchk(cudaGetLastError());
    }
};

// The configurations of this kernel (with its 8 bit RGB write)
template <int Arch>
constexpr auto nppdxRgbConfigs() {
    return nppdxConfigs<Arch, true>();
}

// The NPP part of the pipeline, after the NPPDx kernel, and its 8 bit RGB output planes: per crop the black rows
// (CopyConstBorder; NPP has no batched border), then 8 bit -> float and the normalization on the N planes at once, as
// one image of OUT_W x N * OUT_H. before() runs before every NPP call (the GPU only benchmark with the L2 flushed
// flushes there).
class NppTail {
public:
    explicit NppTail(size_t n) : m_n(n) { gpuErrchk(cudaMalloc(&m_bordered, n * OUT_W * OUT_H * 3)); }
    ~NppTail() { cudaFree(m_bordered); }
    NppTail(const NppTail&) = delete;
    NppTail& operator=(const NppTail&) = delete;

    // Fills the bordered planes with values the border never writes, before a validation
    void poison(cudaStream_t stream) const {
        gpuErrchk(cudaMemsetAsync(m_bordered, 0xAB, m_n * OUT_W * OUT_H * 3, stream));
    }

    template <typename Before>
    void run(const uint8_t* resized, float* out, const Normalization& n, const NppStreamContext& ctx,
             Before&& before) const {
        static const Npp8u black[3] = {0, 0, 0};
        for (size_t i = 0; i < m_n; ++i) {
            before();
            NPP_CHECK(nppiCopyConstBorder_8u_C3R_Ctx(resized + i * RES_W * RES_H * 3, RES_W * 3, NppiSize{RES_W, RES_H},
                                                     m_bordered + i * OUT_W * OUT_H * 3, OUT_W * 3,
                                                     NppiSize{OUT_W, OUT_H}, BORDER_TOP, 0, black, ctx));
        }
        const NppiSize planes{OUT_W, static_cast<int>(m_n) * OUT_H};
        const int step = OUT_W * 3 * static_cast<int>(sizeof(float));
        before();
        NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(m_bordered, OUT_W * 3, out, step, planes, ctx));
        before();
        NPP_CHECK(nppiMulC_32f_C3IR_Ctx(n.mul, out, step, planes, ctx));
        before();
        NPP_CHECK(nppiSubC_32f_C3IR_Ctx(n.sub, out, step, planes, ctx));
        before();
        NPP_CHECK(nppiDivC_32f_C3IR_Ctx(n.div, out, step, planes, ctx));
    }

private:
    size_t m_n{0};
    uint8_t* m_bordered{nullptr};
};

} // namespace figure1

#endif // FIGURE1_NPPDX_NPP_FIGURE1_H
