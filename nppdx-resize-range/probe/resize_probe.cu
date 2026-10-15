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

// One point of the NPPDx resize-range sweep: the pipeline of NPPDx's example 04_resize/fused_resize.cu without its
// color conversion, ingest (RGB24) -> BoxBlur<5, 5> -> Resize<J, K, Method> -> exgest (RGB24), in one kernel, with the
// input tile PROBE_TX x PROBE_TY, for the NPPDx SM PROBE_SM. Every type is built as in the example.
//
// Compiled once per point by sweep.py with -DPROBE_J, -DPROBE_K, -DPROBE_METHOD (nearest, bilinear, bicubic or
// lanczos3), -DPROBE_TX, -DPROBE_TY and -DPROBE_SM. If NPPDx rejects the point, the compiler prints NPPDx's
// static_assert. If it accepts it, the program prints the tiles, the shared memory and the block of the point and,
// with --launch, runs the kernel once on a random RGB24 image and compares it with CPU references.

#include <nppdx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "cpu_resize_reference.hpp" // NPPDx's example reference (example/nppdx/common)

#if !defined(PROBE_J) || !defined(PROBE_K) || !defined(PROBE_METHOD) || !defined(PROBE_TX) || !defined(PROBE_TY) || \
    !defined(PROBE_SM)
#error "Define PROBE_J, PROBE_K, PROBE_METHOD, PROBE_TX, PROBE_TY and PROBE_SM"
#endif

#define PROBE_STR2(x) #x
#define PROBE_STR(x) PROBE_STR2(x)

namespace probe {

constexpr unsigned int J = PROBE_J;
constexpr unsigned int K = PROBE_K;
constexpr unsigned int TX = PROBE_TX;
constexpr unsigned int TY = PROBE_TY;
constexpr int SM = PROBE_SM;
constexpr nppdx::interpolation_method METHOD = nppdx::interpolation_method::PROBE_METHOD;
constexpr unsigned int BOX = 5;
constexpr unsigned int CHANNELS = 3;

// As run_nppdx_fused of fused_resize.cu
using ResizePre = decltype(nppdx::Function<nppdx::function::resize>() + nppdx::Resize<J, K, METHOD>() +
                           nppdx::TileSize<TX, TY>() + nppdx::SM<SM>() + nppdx::Block());
constexpr auto resize_halo = ResizePre::local_halo;

using BoxBlurPre = decltype(nppdx::Function<nppdx::function::box_blur>() + nppdx::BoxBlur<BOX, BOX>() +
                            nppdx::TileSize<TX, TY>() + nppdx::SM<SM>() + nppdx::Block());
constexpr auto box_halo = BoxBlurPre::local_halo;

constexpr auto total_halo = box_halo + resize_halo;

using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                        nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<TX, TY>() +
                        NPPDX_MAKE_HALO(nppdx::MemoryHalo, total_halo)() +
                        NPPDX_MAKE_HALO(nppdx::CumulativeHalo, total_halo)() + nppdx::SM<SM>() + nppdx::Block());

using BoxBlur = decltype(BoxBlurPre() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, total_halo)() +
                         NPPDX_MAKE_HALO(nppdx::CumulativeHalo, total_halo)());

using Resize = decltype(ResizePre() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, total_halo)() +
                        NPPDX_MAKE_HALO(nppdx::CumulativeHalo, resize_halo)());

using InputTile = nppdx::input_storage_of_t<Ingest>;
using BlurredTile = nppdx::output_storage_of_t<BoxBlur>;
using IntermediateTile = nppdx::temp_storage_of_t<Resize>;
using ResizedTile = nppdx::output_storage_of_t<Resize>;

constexpr unsigned int OX = ResizedTile::ChannelSliceType::width;
constexpr unsigned int OY = ResizedTile::ChannelSliceType::height;

constexpr auto empty_halo = nppdx::Halo4::make_empty();

using Exgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                        nppdx::OutputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<OX, OY>() +
                        NPPDX_MAKE_HALO(nppdx::MemoryHalo, empty_halo)() +
                        NPPDX_MAKE_HALO(nppdx::CumulativeHalo, empty_halo)() + nppdx::SM<SM>() + nppdx::Block());

constexpr unsigned int SMEM =
    nppdx::shared_memory::compute_total_tile_storage<InputTile, BlurredTile, IntermediateTile, ResizedTile>();

// The example's launch: the block of the ingest for a downscale, of the exgest otherwise, and the grid of that step
constexpr bool DOWNSCALE = J > K;
constexpr dim3 BLOCK = DOWNSCALE ? Ingest::block_dim : Exgest::block_dim;

} // namespace probe

// fused_blur_resize_convert_kernel of fused_resize.cu without the color conversion
template <typename Ingest, typename BoxBlur, typename Resize, typename Exgest, typename InputTile, typename BlurredTile,
          typename IntermediateTile, typename ResizedTile>
__global__ void fusedBlurResizeKernel(const uint8_t* input, uint8_t* output, const unsigned int inputWidth,
                                      const unsigned int inputHeight, const unsigned int outputWidth,
                                      const unsigned int outputHeight) {
    extern __shared__ unsigned char smem[];
    auto tiles = nppdx::shared_memory::slice_into_tile_storage<InputTile, BlurredTile, IntermediateTile, ResizedTile>(
        smem);
    auto inputTile = std::get<0>(tiles);
    auto blurredTile = std::get<1>(tiles);
    auto intermediateTile = std::get<2>(tiles);
    auto resizedTile = std::get<3>(tiles);

    Ingest().execute(input, inputTile, inputWidth, inputHeight);
    BoxBlur().execute(inputTile, blurredTile, inputWidth, inputHeight);
    Resize().execute(blurredTile, intermediateTile, resizedTile, inputWidth, inputHeight);
    Exgest().execute(resizedTile, output, outputWidth, outputHeight);
}

namespace probe {

constexpr auto kernel = fusedBlurResizeKernel<Ingest, BoxBlur, Resize, Exgest, InputTile, BlurredTile,
                                              IntermediateTile, ResizedTile>;

template <typename Tile>
void printTile(const char* name) {
    std::printf(",%s=%ux%u", name, Tile::ChannelSliceType::width, Tile::ChannelSliceType::height);
}

void printPoint() {
    std::printf("point J=%u,K=%u,method=%s,tile=%ux%u,sm=%d,box_halo=%d,resize_halo=%d", J, K,
                PROBE_STR(PROBE_METHOD), TX, TY, SM, box_halo.left_top.x, resize_halo.left_top.x);
    printTile<InputTile>("input_storage");
    printTile<BlurredTile>("blurred_storage");
    printTile<IntermediateTile>("intermediate_storage");
    printTile<ResizedTile>("output_storage");
    std::printf(",smem_bytes=%u,block_ingest=%u,block_boxblur=%u,block_resize=%u,block_exgest=%u,block=%u\n", SMEM,
                Ingest::block_dim.x * Ingest::block_dim.y * Ingest::block_dim.z,
                BoxBlur::block_dim.x * BoxBlur::block_dim.y * BoxBlur::block_dim.z,
                Resize::block_dim.x * Resize::block_dim.y * Resize::block_dim.z,
                Exgest::block_dim.x * Exgest::block_dim.y * Exgest::block_dim.z, BLOCK.x * BLOCK.y * BLOCK.z);
}

// 5x5 mean replicating the image edge, in float, as fused_resize.cu's cpu_box_blur_5x5_float
std::vector<float> hostBox(const std::vector<uint8_t>& in, int w, int h) {
    constexpr int r = BOX / 2;
    std::vector<float> out(in.size());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (unsigned int c = 0; c < CHANNELS; ++c) {
                float sum = 0.f;
                for (int dy = -r; dy <= r; ++dy) {
                    for (int dx = -r; dx <= r; ++dx) {
                        const int sx = std::clamp(x + dx, 0, w - 1);
                        const int sy = std::clamp(y + dy, 0, h - 1);
                        sum += static_cast<float>(in[(static_cast<size_t>(sy) * w + sx) * CHANNELS + c]);
                    }
                }
                out[(static_cast<size_t>(y) * w + x) * CHANNELS + c] = sum / float(BOX * BOX);
            }
        }
    }
    return out;
}

// NPPDx's example reference (cpu_reference::resize) computes the phase of an output pixel in float,
// theta = frac((o + 0.5f) * (J / K) - 0.5f). Where theta is exactly 0, the float can come out just under 1, which moves
// the reference's taps by one pixel (at 7:3, 13:11 or 3:19, for every method). The exact reference is the same resample
// with the phase and the center index from integers, as in NPPDx's kernel (compute_phase_index and
// compute_center_input_index of detail/backend/resize_operations.hpp), and the taps and weights of the example
// reference in double. Nearest has ties where theta is exactly 1/2 (J even): there both source pixels are accepted.
constexpr int exactSpan() {
    const unsigned int muNum = METHOD == nppdx::interpolation_method::bicubic    ? 2u
                               : METHOD == nppdx::interpolation_method::lanczos3 ? 3u
                                                                                 : 1u;
    const unsigned int muDen = METHOD == nppdx::interpolation_method::nearest ? 2u : 1u;
    const unsigned int radiusNum = J > K ? muNum * J : muNum;
    const unsigned int radiusDen = J > K ? muDen * K : muDen;
    const unsigned int raw = (2 * radiusNum + radiusDen - 1) / radiusDen;
    return METHOD == nppdx::interpolation_method::nearest ? 1 : int(raw + raw % 2);
}

double exactKernel(double x) {
    constexpr double pi = 3.14159265358979323846;
    const double a = std::abs(x);
    switch (METHOD) {
    case nppdx::interpolation_method::nearest: return a <= 0.5 ? 1.0 : 0.0;
    case nppdx::interpolation_method::bilinear: return a < 1.0 ? 1.0 - a : 0.0;
    case nppdx::interpolation_method::bicubic:
        if (a < 1.0) return (1.5 * a - 2.5) * a * a + 1.0;
        if (a < 2.0) return ((-0.5 * a + 2.5) * a - 4.0) * a + 2.0;
        return 0.0;
    case nppdx::interpolation_method::lanczos3:
        if (x == 0.0) return 1.0;
        if (a >= 3.0) return 0.0;
        return (std::sin(pi * x) / (pi * x)) * (std::sin(pi * x / 3.0) / (pi * x / 3.0));
    default: return 0.0;
    }
}

struct AxisTaps {
    int start = 0;
    std::vector<double> weights;
    bool tie = false; // nearest: the source pixels start - 1 and start are at the same distance
};

AxisTaps exactTaps(long long o) {
    constexpr int span = exactSpan();
    const long long twoK = 2LL * K;
    const long long num = (2 * o + 1) * static_cast<long long>(J) - static_cast<long long>(K);
    const long long center = num >= 0 ? num / twoK : (num - twoK + 1) / twoK;
    const long long phase = num - center * twoK; // theta = phase / 2K, in [0, 1)
    const int delta = span % 2 == 0 ? 2 : (2 * phase < twoK ? 1 : 3);
    AxisTaps taps;
    taps.start = int(center) - (span - delta) / 2;
    const double theta = double(phase) / double(twoK);
    const double l = double(span - delta) / 2.0;
    for (int j = 0; j < span; ++j) taps.weights.push_back(exactKernel(theta + l - double(j)));
    taps.tie = METHOD == nppdx::interpolation_method::nearest && 2 * phase == twoK;
    return taps;
}

// The smallest difference between an output value and the exact reference (any tie of nearest)
int exactDiff(const std::vector<float>& blurred, int w, int h, const AxisTaps& tx, const AxisTaps& ty, unsigned int c,
              int value) {
    auto at = [&](int x, int y) {
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        return double(blurred[(static_cast<size_t>(y) * w + x) * CHANNELS + c]);
    };
    auto diff = [&](double v) { return std::abs(value - int(std::floor(std::clamp(v, 0.0, 255.0) + 0.5))); };
    if (METHOD == nppdx::interpolation_method::nearest) {
        int best = diff(at(tx.start, ty.start));
        if (tx.tie) best = std::min(best, diff(at(tx.start - 1, ty.start)));
        if (ty.tie) best = std::min(best, diff(at(tx.start, ty.start - 1)));
        if (tx.tie && ty.tie) best = std::min(best, diff(at(tx.start - 1, ty.start - 1)));
        return best;
    }
    double sum = 0.0, weights = 0.0;
    for (size_t jy = 0; jy < ty.weights.size(); ++jy) {
        for (size_t jx = 0; jx < tx.weights.size(); ++jx) {
            const double wgt = ty.weights[jy] * tx.weights[jx];
            if (wgt == 0.0) continue;
            sum += wgt * at(tx.start + int(jx), ty.start + int(jy));
            weights += wgt;
        }
    }
    return diff(weights > 1e-6 ? sum / weights : sum);
}

#define PROBE_CUDA(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            std::printf("launch result=cuda_error,call=%s,error=%s\n", #call, cudaGetErrorName(err_));                 \
            return 0;                                                                                                  \
        }                                                                                                              \
    } while (0)

// Runs the kernel once on a random RGB24 image of at least 3 x 3 input tiles and compares it with the CPU references:
// the 5x5 box above, then the exact resample above (max_diff...) and NPPDx's example resize (example_max_diff...),
// both replicating the edge of the blurred image, rounded to 8 bits. The kernel blurs the replicated input outside the
// image instead, so the comparison is also given for the interior: the output pixels whose resize footprint and box
// window are inside the image.
int launch(unsigned int blockThreads) {
    int device = 0;
    PROBE_CUDA(cudaGetDevice(&device));
    int optin = 0;
    PROBE_CUDA(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    int maxThreads = 0;
    PROBE_CUDA(cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, device));
    cudaDeviceProp prop{};
    PROBE_CUDA(cudaGetDeviceProperties(&prop, device));
    std::printf("device name=%s,cc=%d.%d,max_shared_memory_per_block_optin=%d,max_threads_per_block=%d\n", prop.name,
                prop.major, prop.minor, optin, maxThreads);

    // Interior: the input position of the output pixel's center is at least the resize radius (NPPDx's halo of the
    // resize) plus the box radius plus one pixel away from every image edge
    const double margin = double(resize_halo.left_top.x) + double(BOX / 2) + 1.0;
    // At least 3 x 3 tiles, 2 x 2 output tiles and an interior of 2 x 2 tiles
    unsigned int tilesX = 3, tilesY = 3;
    while ((tilesX * TX) * K / J < 2 * OX) ++tilesX;
    while ((tilesY * TY) * K / J < 2 * OY) ++tilesY;
    while (tilesX * TX < 2 * margin + 2 * TX) ++tilesX;
    while (tilesY * TY < 2 * margin + 2 * TY) ++tilesY;
    const unsigned int w = tilesX * TX, h = tilesY * TY;
    const unsigned int ow = w * K / J, oh = h * K / J; // the size of NPPDx's reference

    std::vector<uint8_t> hostIn(static_cast<size_t>(w) * h * CHANNELS);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : hostIn) v = static_cast<uint8_t>(dist(gen));

    uint8_t* dIn = nullptr;
    uint8_t* dOut = nullptr;
    const size_t outBytes = static_cast<size_t>(ow) * oh * CHANNELS;
    PROBE_CUDA(cudaMalloc(&dIn, hostIn.size()));
    PROBE_CUDA(cudaMalloc(&dOut, outBytes));
    PROBE_CUDA(cudaMemcpy(dIn, hostIn.data(), hostIn.size(), cudaMemcpyHostToDevice));
    PROBE_CUDA(cudaMemset(dOut, 0, outBytes));

    const cudaError_t attr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (attr != cudaSuccess) {
        std::printf("launch result=attribute_error,smem_bytes=%u,error=%s\n", SMEM, cudaGetErrorName(attr));
        return 0;
    }
    const dim3 grid = DOWNSCALE ? Ingest::calculate_grid_dim(w, h) : Exgest::calculate_grid_dim(ow, oh);
    kernel<<<grid, dim3(blockThreads), SMEM>>>(dIn, dOut, w, h, ow, oh);
    const cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::printf("launch result=launch_error,block=%u,error=%s\n", blockThreads, cudaGetErrorName(launchErr));
        return 0;
    }
    const cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::printf("launch result=execution_error,block=%u,error=%s\n", blockThreads, cudaGetErrorName(syncErr));
        return 0;
    }
    std::vector<uint8_t> hostOut(outBytes);
    PROBE_CUDA(cudaMemcpy(hostOut.data(), dOut, outBytes, cudaMemcpyDeviceToHost));

    const auto blurred = hostBox(hostIn, int(w), int(h));
    const auto reference =
        nppdx::example::cpu_reference::resize<J, K, METHOD, CHANNELS>(blurred, int(w), int(h));

    std::vector<AxisTaps> tapsX, tapsY;
    for (unsigned int x = 0; x < ow; ++x) tapsX.push_back(exactTaps(x));
    for (unsigned int y = 0; y < oh; ++y) tapsY.push_back(exactTaps(y));

    int maxDiff = 0, maxDiffInterior = 0, exampleMaxDiff = 0, exampleMaxDiffInterior = 0;
    size_t over1 = 0, over1Interior = 0, interior = 0;
    for (unsigned int y = 0; y < oh; ++y) {
        const double cy = (y + 0.5) * double(J) / double(K) - 0.5;
        for (unsigned int x = 0; x < ow; ++x) {
            const double cx = (x + 0.5) * double(J) / double(K) - 0.5;
            const bool inside = cx - margin >= 0.0 && cx + margin <= double(w) - 1.0 && cy - margin >= 0.0 &&
                                cy + margin <= double(h) - 1.0;
            interior += inside ? 1 : 0;
            for (unsigned int c = 0; c < CHANNELS; ++c) {
                const size_t i = (static_cast<size_t>(y) * ow + x) * CHANNELS + c;
                const int diff = exactDiff(blurred, int(w), int(h), tapsX[x], tapsY[y], c, int(hostOut[i]));
                const float ref = std::clamp(reference[i], 0.0f, 255.0f);
                const int exampleDiff = std::abs(int(hostOut[i]) - int(std::floor(ref + 0.5f)));
                maxDiff = std::max(maxDiff, diff);
                exampleMaxDiff = std::max(exampleMaxDiff, exampleDiff);
                over1 += diff > 1 ? 1 : 0;
                if (inside) {
                    maxDiffInterior = std::max(maxDiffInterior, diff);
                    exampleMaxDiffInterior = std::max(exampleMaxDiffInterior, exampleDiff);
                    over1Interior += diff > 1 ? 1 : 0;
                }
            }
        }
    }
    std::printf("launch result=ran,image=%ux%u,output=%ux%u,grid=%ux%u,block=%u,smem_bytes=%u,max_diff=%d,"
                "values_over_1=%zu,interior_pixels=%zu,max_diff_interior=%d,interior_values_over_1=%zu,"
                "example_max_diff=%d,example_max_diff_interior=%d\n",
                w, h, ow, oh, grid.x, grid.y, blockThreads, SMEM, maxDiff, over1, interior, maxDiffInterior,
                over1Interior, exampleMaxDiff, exampleMaxDiffInterior);
    cudaFree(dIn);
    cudaFree(dOut);
    return 0;
}

} // namespace probe

int main(int argc, char** argv) {
    probe::printPoint();
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--launch") == 0) {
            // --launch [threads]: the example's block by default
            const unsigned int threads =
                (i + 1 < argc) ? static_cast<unsigned int>(std::stoul(argv[i + 1])) : probe::BLOCK.x;
            return probe::launch(threads);
        }
    }
    return 0;
}
