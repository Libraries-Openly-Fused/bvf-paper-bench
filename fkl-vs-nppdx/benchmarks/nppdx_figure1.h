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

// The NPPDx implementation of the Figure 1 pipeline as one CUDA kernel: one or more thread blocks per crop (one per
// tile of the crop), built from NPPDx operations (NV12 ingest, limited to full range, color conversion BT.601,
// resize), with the border, the normalization and the float3 write written by hand, since NPPDx has no border
// operation and no floating point output (and its affine channel map takes integer parameters only). The sizes are
// those of common/bench_geometry.h. Used by benchmark_figure1_bvf.cu and compile_time_pipeline.cu.
// nppdx_npp_figure1.h has the version without hand written pixel processing code.

#ifndef FIGURE1_NPPDX_FIGURE1_H
#define FIGURE1_NPPDX_FIGURE1_H

#include "benchmarks/nppdx_common.h"

namespace figure1 {

template <typename Ingest, typename FromLimitedY, typename FromLimitedUV, typename Convert, typename Resize,
          typename InputTile, typename OutputTile, typename TempTile, size_t N>
__global__ void nppdxFigure1Kernel(const uint8_t* y, const uint8_t* uv, const int pitch,
                                   const CropOrigins<N> crops, const Normalization n, float* out) {
    extern __shared__ unsigned char smem[];
    auto tiles = nppdx::shared_memory::slice_into_tile_storage<InputTile, OutputTile, TempTile>(smem);
    auto inputTile = std::get<0>(tiles);
    auto resizedTile = std::get<1>(tiles);
    auto tempTile = std::get<2>(tiles);

    // ReadNV12 + Crop: the crop is ingested as a CROP_W x CROP_H NV12 image inside the frame; NPPDx takes the position
    // of this block's tile in the crop from blockIdx.x and blockIdx.y, and blockIdx.z is the crop
    const int2 o = crops.origin[blockIdx.z];
    const nppdx::ImageLayout<const uint8_t, 2> crop{{{y + static_cast<size_t>(o.y) * pitch + o.x,
                                                      uv + static_cast<size_t>(o.y / 2) * pitch + o.x}},
                                                    {{pitch, pitch}}};
    Ingest().execute(crop, inputTile, CROP_W, CROP_H);
    // YUVtoRGB, BT.601 limited range, in place on the shared memory tile
    FromLimitedY().execute(inputTile, CROP_W, CROP_H);
    FromLimitedUV().execute(inputTile, CROP_W, CROP_H);
    Convert().execute(inputTile, CROP_W, CROP_H);
    // Resize, J input pixels to K output pixels in both dimensions
    Resize().execute(inputTile, tempTile, resizedTile, CROP_W, CROP_H);
    __syncthreads();

    // AddBorder + Mul + Sub + Div + Write float3 (no NPPDx operations for these). Each block writes its
    // resized tile, plus the top border rows if it is in the first tile row and the bottom ones if in the last
    constexpr int OTW = OutputTile::ChannelSliceType::width;
    constexpr int OTH = OutputTile::ChannelSliceType::height;
    const int ox = blockIdx.x * OTW;
    const int oy = blockIdx.y * OTH;
    const int rowBegin = blockIdx.y == 0 ? 0 : BORDER_TOP + oy;
    const int rowEnd = blockIdx.y == gridDim.y - 1 ? OUT_H : BORDER_TOP + oy + OTH;
    float* plane = out + static_cast<size_t>(blockIdx.z) * OUT_W * OUT_H * 3;
    const int numThreads = blockDim.x * blockDim.y * blockDim.z;
    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    for (int idx = tid; idx < (rowEnd - rowBegin) * OTW; idx += numThreads) {
        const int tx = idx % OTW;
        const int outY = rowBegin + idx / OTW;
        const bool inside = outY >= BORDER_TOP && outY < BORDER_TOP + RES_H;
        float* pixel = plane + (static_cast<size_t>(outY) * OUT_W + ox + tx) * 3;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const float value = inside ? resizedTile.channel(c)(outY - BORDER_TOP - oy, tx) : 0.f;
            pixel[c] = ((value * n.mul[c]) - n.sub[c]) / n.div[c];
        }
    }
}

// The kernel in one configuration: TileX x TileY input tiles, block size BlockX (0: NPPDx's recommended one)
template <int Arch, size_t N, unsigned BlockX, unsigned TileX, unsigned TileY>
struct NppdxFigure1 : NppdxOps<Arch, BlockX, TileX, TileY> {
    using Ops = NppdxOps<Arch, BlockX, TileX, TileY>;
    using typename Ops::Ingest, typename Ops::FromLimitedY, typename Ops::FromLimitedUV, typename Ops::Convert,
        typename Ops::Resize, typename Ops::InputTile, typename Ops::OutputTile, typename Ops::TempTile;
    static constexpr size_t smem = Ops::smem;
    static constexpr dim3 block{Ops::blockBeforeWrite, 1, 1};
    static constexpr dim3 grid = Ops::gridOf(N);

    static constexpr auto kernel = nppdxFigure1Kernel<Ingest, FromLimitedY, FromLimitedUV, Convert, Resize, InputTile,
                                                      OutputTile, TempTile, N>;

    static void init() {
        if constexpr (smem > 48 * 1024) {
            gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        }
    }

    static void run(cudaStream_t stream, const uint8_t* y, const uint8_t* uv, int pitch, const CropOrigins<N>& crops,
                    const Normalization& n, float* out) {
        kernel<<<grid, block, smem, stream>>>(y, uv, pitch, crops, n, out);
        gpuErrchk(cudaGetLastError());
    }
};

// The configurations of this kernel (it has no 8 bit RGB write)
template <int Arch>
constexpr auto nppdxFigure1Configs() {
    return nppdxConfigs<Arch, false>();
}

} // namespace figure1

#endif // FIGURE1_NPPDX_FIGURE1_H
