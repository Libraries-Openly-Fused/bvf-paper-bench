// ROCm port of src/bvf_bench.cu: latency and correctness benchmark for the Backwards Vertical Fusion paper, Figure 1
// pipeline on an AMD GPU:
// ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// with the frame, sizes, batch sizes and crop positions of common/bench_geometry.h.
//
// Implementations: rpp (the fastest of the RPP variants below, chosen by a selection pass; the only RPP pipeline in the
// measurements), fkl_vf_hf and fkl_bvf. RPP variants (ROCm Performance Primitives, HIP backend):
//   rpp_per_op         one RPP call per operation, batched where RPP can:
//                      rppt_yuv_to_rgb on the full frame (8 bit RGB), one rppt_crop_mirror_normalize for all the
//                      crops (8u -> 32f, every batch image aliasing the frame), hipMemsetAsync of the output tensor
//                      (RPP has no constant border), one rppt_resize writing each resized crop into the rows of its
//                      output plane below the top border, then rppt_multiply_scalar, rppt_tensor_subtract_tensor and
//                      rppt_tensor_divide_tensor on the whole output tensor
//   rpp_npp_like       the same calls, but with exactly the intermediates of the paper's NPP pipeline: the resized
//                      crops go to their own float3 buffer, and one batched rppt_crop F32 -> F32 copies them into
//                      those rows of the output planes (the counterpart of NPP's CopyConstBorder)
//   rpp_fused          rppt_yuv_to_rgb, memset, rppt_resize_mirror_normalize (crop + 8u -> 32f + resize + Mul, RPP's
//                      fused primitive) into those rows, then Sub and Div
//   rpp_normalize      rpp_per_op with Mul + Sub + Div in one rppt_normalize
//   rpp_fused_normalize rppt_yuv_to_rgb, memset, rppt_resize_mirror_normalize (crop + 8u -> 32f + resize), rppt_normalize
//   fkl_vf_hf          four FKL kernels with the intermediates in memory (VF + HF, no BVF): 8 bit RGB frame, float3
//                      crops, float3 resized crops, float3 output (fkl_vf_hf_float3_frame only in the breakdown)
//   fkl_bvf            one FKL kernel for the whole chain (BVF, Listing 2 of the paper)
//
// No HIP graphs anywhere. GPU time only (rocprofiler-sdk kernel dispatch tracing) runs only where the ROCm stack
// delivers kernel dispatch records; on WSL2 it does not, and the benchmark says so and skips those passes.

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <hip/hip_runtime.h>
#include <rpp/rpp.h>
#include <rpp/rpp_version.h>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "bench_geometry.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define RPP_CHECK(expr)                                                                                     \
    do {                                                                                                    \
        const RppStatus status_ = (expr);                                                                   \
        if (status_ != RPP_SUCCESS) {                                                                       \
            throw std::runtime_error(std::string("RPP error ") + std::to_string(status_) + " in " + #expr); \
        }                                                                                                   \
    } while (0)

#define ROCP_CHECK(expr)                                                                                      \
    do {                                                                                                      \
        const rocprofiler_status_t status_ = (expr);                                                          \
        if (status_ != ROCPROFILER_STATUS_SUCCESS) {                                                          \
            throw std::runtime_error(std::string("rocprofiler error ") + rocprofiler_get_status_string(status_) + \
                                     " in " + #expr);                                                         \
        }                                                                                                     \
    } while (0)

namespace bench {

// Frame, crop, resize, border and output sizes, batch sizes, seeds and crop positions, shared by every benchmark
using namespace geometry;
constexpr int WARMUP_ITERS = 100;
constexpr int TIMED_ITERS = 1000;
constexpr float TOLERANCE = 1e-3f;
// RPP variants within this max abs error (normalized units) of the RPP-convention reference are numerically
// equivalent; the fastest of them is the benchmark's RPP pipeline (selection pass). One 8 bit level is
// (1/255) / 0.229 = 0.017.
constexpr float RPP_EQUIVALENCE = 1e-5f;

// Pointwise tail: ImageNet normalization of 8 bit RGB
constexpr float MUL[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
constexpr float SUB[3] = {0.485f, 0.456f, 0.406f};
constexpr float DIV[3] = {0.229f, 0.224f, 0.225f};
// 8 bit quantization of the RGB values for the reference of the 8 bit pipelines:
// normalize to [0, 1], add half a level so that the truncating denormalization rounds to nearest,
// saturate and denormalize to uchar3
constexpr float QUANTIZATION_OFFSET = 0.5f / 255.f;

// RPP's HIP kernels load and store 8 pixels per thread and only check that the first one is inside the ROI, so the rows
// of RPP's intermediates are padded to a multiple of 8 pixels (a 20 pixel crop row would be written as 24, over the start
// of the next row; 32 pixel crops need no padding), and every RPP buffer gets some slack at its end. The output tensor
// is contiguous, like FKL's, so its rows must be a multiple of 8 pixels themselves.
constexpr int RPP_PIXEL_ALIGN = 8;
constexpr size_t RPP_SLACK_BYTES = 1 << 16;
constexpr int alignUp(int v, int a) { return (v + a - 1) / a * a; }
static_assert(OUT_W % RPP_PIXEL_ALIGN == 0, "RPP's kernels write the output rows 8 pixels at a time");

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Full, fk::ColorPrimitives::bt2020>;

const fk::float3 mulValue{MUL[0], MUL[1], MUL[2]};
const fk::float3 subValue{SUB[0], SUB[1], SUB[2]};
const fk::float3 divValue{DIV[0], DIV[1], DIV[2]};
const fk::float3 black{0.f, 0.f, 0.f};

struct Context {
    hipStream_t stream{};
    std::unique_ptr<fk::Stream> fkStream;
    // rppt_yuv_to_rgb's 8 bit RGB frame (host copy), the input of the RPP-convention reference
    std::vector<uint8_t> rppRgb8;
    std::unique_ptr<NV12Image> frameDevice;
    std::unique_ptr<NV12Image> frameHost;
    uint8_t* y{nullptr};
    uint8_t* uv{nullptr};
    int pitch{0};
    std::array<fk::Rect, MAX_BATCH> crops{};
};

void initContext(Context& ctx) {
    // A blocking stream: rppt_yuv_to_rgb's hipMemcpyToSymbol calls go to the null stream (RPP takes no stream for them),
    // and the null stream serializes with blocking streams, so every operation of every pipeline is ordered
    gpuErrchk(hipStreamCreate(&ctx.stream));
    ctx.fkStream = std::make_unique<fk::Stream>(ctx.stream);

    // Synthetic NV12 frame with uniform random bytes, identical for every implementation
    ctx.frameDevice = std::make_unique<NV12Image>(FRAME_W, FRAME_H);
    ctx.frameHost = std::make_unique<NV12Image>(FRAME_W, FRAME_H, fk::MemType::Host);
    auto deviceData = ctx.frameDevice->getData();
    auto hostData = ctx.frameHost->getData();
    generateFrame([&](int x, int y, uint8_t value) {
        deviceData.at(x, y) = value;
        hostData.at(x, y) = value;
    });
    ctx.frameDevice->upload(*ctx.fkStream);
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    const auto raw = deviceData.ptr();
    ctx.pitch = static_cast<int>(raw.dims.pitch);
    ctx.y = raw.data;
    ctx.uv = raw.data + static_cast<size_t>(FRAME_H) * ctx.pitch;

    // Crop rectangles at random positions, the first N are used for batch N
    const auto crops = makeCrops();
    for (size_t i = 0; i < MAX_BATCH; ++i) {
        ctx.crops[i] = fk::Rect(crops[i].x, crops[i].y, crops[i].width, crops[i].height);
    }
}

// Cache flush before every GPU operation, for the GPU only benchmark with flushed caches: simulates a GPU that runs
// other kernels in between, so that no intermediate result (nor the input frame) can be assumed to still be in L2 or
// in the Infinity Cache (MALL). The flush kernel reads a buffer of twice the sum of the L2 and MALL sizes.
// beforeKernel() is called right before every library call / kernel / memset of the pipelines; it does nothing
// unless the flushed benchmark is running. Only used where kernel dispatch tracing works (not on WSL2).
__global__ void flushCachesKernel(const int4* buffer, size_t count, int* sink) {
    int acc = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int4 v = buffer[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x7fffffff) {
        *sink = acc; // never true for the zeroed buffer, keeps the loads
    }
}

struct CacheFlusher {
    int4* buffer{nullptr};
    size_t count{0};
    int* sink{nullptr};
    hipStream_t stream{};
    int blocks{0};
};
CacheFlusher* activeFlusher = nullptr;

inline void beforeKernel() {
    if (activeFlusher) {
        flushCachesKernel<<<activeFlusher->blocks, 256, 0, activeFlusher->stream>>>(activeFlusher->buffer,
                                                                                  activeFlusher->count, activeFlusher->sink);
    }
}

// Each implementation enqueues one complete pipeline on ctx.stream with run(), and returns the
// N x outH x outW x 3 float output with download(). fillNaN() poisons the output before validation.
struct Implementation {
    std::string name;
    std::string reference; // fkl_cpu_float, fkl_cpu_8bit_rgb or rpp_sampling_8bit
    std::function<void()> run;
    std::function<std::vector<float>()> download;
    std::function<void()> fillNaN;
    // The GPU operations (library calls, kernels, memsets) of the pipeline, in order, for the per operation breakdown
    std::vector<std::pair<std::string, std::function<void()>>> stages;
    std::string variant; // for "rpp": the RPP variant it is
};

// NaN poisoning and readback of the outputs are ordered on the pipeline stream, like the pipelines themselves
void poisonDevice(hipStream_t stream, void* data, size_t floats) {
    gpuErrchk(hipMemsetAsync(data, 0xFF, floats * sizeof(float), stream));
}

std::vector<float> downloadDevice(hipStream_t stream, const void* data, size_t floats) {
    std::vector<float> host(floats);
    gpuErrchk(hipMemcpyAsync(host.data(), data, floats * sizeof(float), hipMemcpyDeviceToHost, stream));
    gpuErrchk(hipStreamSynchronize(stream));
    return host;
}

template <size_t N>
std::array<fk::Rect, N> firstCrops(const Context& ctx) {
    std::array<fk::Rect, N> crops{};
    std::copy_n(ctx.crops.begin(), N, crops.begin());
    return crops;
}

template <size_t N>
Implementation makeFklBvf(Context& ctx) {
    auto out = std::make_shared<fk::Tensor<fk::float3>>(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const auto crops = firstCrops<N>(ctx);
    Implementation impl{"fkl_bvf", "fkl_cpu_float"};
    impl.run = [&ctx, out, crops]() {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
            YUVToRGB::build(),
            fk::Crop<>::build(crops),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
            fk::Mul<fk::float3>::build(mulValue),
            fk::Sub<fk::float3>::build(subValue),
            fk::Div<fk::float3>::build(divValue),
            fk::TensorWrite<fk::float3>::build(out->ptr()));
    };
    impl.download = [&ctx, out]() { return downloadDevice(ctx.stream, out->ptr().data, N * OUT_FLOATS); };
    impl.fillNaN = [&ctx, out]() { poisonDevice(ctx.stream, out->ptr().data, N * OUT_FLOATS); };
    return impl;
}

template <size_t N, typename T>
std::array<fk::RawPtr<fk::ND::_2D, T>, N> planeViews(const fk::Tensor<T>& tensor, int width, int height) {
    const auto raw = tensor.ptr();
    if (raw.dims.pitch != width * sizeof(T)) {
        throw std::runtime_error("Expected a contiguous Tensor");
    }
    std::array<fk::RawPtr<fk::ND::_2D, T>, N> views{};
    for (size_t i = 0; i < N; ++i) {
        views[i].data = raw.data + i * width * height;
        views[i].dims = fk::PtrDims<fk::ND::_2D>(width, height, width * sizeof(T));
    }
    return views;
}

// VF + HF without BVF: four kernels with the intermediates in memory, the same intermediates as the paper's OpenCV
// and NPP pipelines: an 8 bit RGB frame, quantized as in the 8 bit reference (normalize, add half a level, saturate,
// denormalize), then float3 crops, float3 resized crops and the float3 output. Unchanged from bvf_bench.cu.
// With EIGHT_BIT_FRAME = false the frame is float3 instead; that variant is only used for the per kernel breakdown.
template <size_t N, bool EIGHT_BIT_FRAME>
Implementation makeFklVfHf(Context& ctx) {
    constexpr bool EIGHT_BIT = EIGHT_BIT_FRAME;
    using RGB = std::conditional_t<EIGHT_BIT, fk::uchar3, fk::float3>;
    auto rgb = std::make_shared<fk::Ptr2D<RGB>>(FRAME_W, FRAME_H, 0, fk::MemType::Device);
    auto cropped = std::make_shared<fk::Tensor<fk::float3>>(CROP_W, CROP_H, N, 1, fk::MemType::Device);
    auto resized = std::make_shared<fk::Tensor<fk::float3>>(RES_W, RES_H, N, 1, fk::MemType::Device);
    auto out = std::make_shared<fk::Tensor<fk::float3>>(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const auto crops = firstCrops<N>(ctx);
    const auto croppedViews = planeViews<N>(*cropped, CROP_W, CROP_H);
    const auto resizedViews = planeViews<N>(*resized, RES_W, RES_H);
    Implementation impl{EIGHT_BIT ? "fkl_vf_hf" : "fkl_vf_hf_float3_frame", EIGHT_BIT ? "fkl_cpu_8bit_rgb" : "fkl_cpu_float"};
    impl.stages.emplace_back(EIGHT_BIT ? "1_nv12_to_rgb8_full_frame" : "1_nv12_to_rgb_float3_full_frame", [&ctx, rgb]() {
        if constexpr (EIGHT_BIT) {
            beforeKernel();
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                YUVToRGB::build(),
                fk::NormalizePixel<fk::float3, fk::ColorDepth::p8bit>::build(),
                fk::Add<fk::float3>::build(fk::make_set<fk::float3>(QUANTIZATION_OFFSET)),
                fk::SaturateDenormalizePixel<fk::float3, fk::uchar3, fk::ColorDepth::p8bit>::build(),
                fk::PerThreadWrite<fk::ND::_2D, fk::uchar3>::build(rgb->ptr()));
        } else {
            beforeKernel();
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                YUVToRGB::build(),
                fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(rgb->ptr()));
        }
    });
    impl.stages.emplace_back("2_crop_to_float3", [&ctx, rgb, cropped, crops]() {
        if constexpr (EIGHT_BIT) {
            beforeKernel();
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                fk::PerThreadRead<fk::ND::_2D, RGB>::build(rgb->ptr()),
                fk::Crop<>::build(crops),
                fk::Cast<fk::uchar3, fk::float3>::build(),
                fk::TensorWrite<fk::float3>::build(cropped->ptr()));
        } else {
            beforeKernel();
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                fk::PerThreadRead<fk::ND::_2D, RGB>::build(rgb->ptr()),
                fk::Crop<>::build(crops),
                fk::TensorWrite<fk::float3>::build(cropped->ptr()));
        }
    });
    impl.stages.emplace_back("3_resize", [&ctx, resized, croppedViews]() {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(croppedViews),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::TensorWrite<fk::float3>::build(resized->ptr()));
    });
    impl.stages.emplace_back("4_addborder_mul_sub_div_write", [&ctx, out, resizedViews]() {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(resizedViews),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
            fk::Mul<fk::float3>::build(mulValue),
            fk::Sub<fk::float3>::build(subValue),
            fk::Div<fk::float3>::build(divValue),
            fk::TensorWrite<fk::float3>::build(out->ptr()));
    });
    impl.run = [stages = impl.stages]() {
        for (const auto& stage : stages) {
            stage.second();
        }
    };
    impl.download = [&ctx, out]() { return downloadDevice(ctx.stream, out->ptr().data, N * OUT_FLOATS); };
    impl.fillNaN = [&ctx, out]() { poisonDevice(ctx.stream, out->ptr().data, N * OUT_FLOATS); };
    return impl;
}

// ------------------------------------------------------------------------------------------------
// RPP pipelines. Every buffer and parameter tensor is allocated and filled once, outside of the timing. Each pipeline
// has its own RPP handle created with its batch size N: RPP's HIP executors size their grids with the handle's batch
// size, not with the descriptors' n. Each call gets its ROIs in the format it works in (XYWH for rppt_crop and
// rppt_crop_mirror_normalize, LTRB for rppt_resize and rppt_resize_mirror_normalize): given the other format, RPP
// launches a conversion kernel that rewrites the caller's ROI tensor in place, so a second call would read converted
// ROIs. RpptROI tensors are in device memory, the other parameter tensors in pinned host memory (RPP reads some of
// them on the host). The descriptors follow RPP's element strides; the output tensor is N planes of outW x outH
// float3, contiguous.
// ------------------------------------------------------------------------------------------------
template <typename T>
T* hostAlloc(size_t count) {
    T* p = nullptr;
    gpuErrchk(hipHostMalloc(reinterpret_cast<void**>(&p), count * sizeof(T), hipHostMallocDefault));
    return p;
}
template <typename T>
T* deviceAlloc(size_t bytes) {
    T* p = nullptr;
    gpuErrchk(hipMalloc(reinterpret_cast<void**>(&p), bytes));
    return p;
}

struct RppState {
    size_t n{0};
    rppHandle_t handle{};
    int cropPitch{0};    // pixels per row of the float3 crop buffer
    int resizedPitch{0}; // pixels per row of the float3 resized crop buffer (rpp_npp_like only)
    uint8_t* rgb8{nullptr};
    float* cropped{nullptr};
    float* resized{nullptr};
    float* tensor{nullptr};
    RpptROI* cropRoisXywh{nullptr}; // crops on the RGB frame, for rppt_crop_mirror_normalize (device)
    RpptROI* cropRoisLtrb{nullptr}; // crops on the RGB frame, for rppt_resize_mirror_normalize (device)
    RpptROI* resizeRois{nullptr};   // whole crop, LTRB for rppt_resize (device)
    RpptROI* resizedRois{nullptr};  // whole resized crop, XYWH for rppt_crop (device, rpp_npp_like only)
    RpptImagePatch* dstSizes{nullptr};
    float* cmnOffset{nullptr};
    float* cmnMultiplier{nullptr};
    Rpp32u* mirror{nullptr};
    float* rmnMean{nullptr};      // 0
    float* rmnStdDevMul{nullptr}; // 255: (x - 0) / 255 = x * Mul
    float* rmnStdDevOne{nullptr}; // 1: the resized 0..255 values
    float* mulScalar{nullptr};
    RpptROI3D* volumeRoi{nullptr};
    float* subRow{nullptr}; // per channel constants tiled over one output row (device)
    float* divRow{nullptr};
    float* normMean{nullptr};   // 255 * sub tiled over one output row, for rppt_normalize (device)
    float* normStdDev{nullptr}; // 255 * div tiled over one output row (device)
    Rpp32u* rowsRoi{nullptr};
    Rpp32u* rowRoi{nullptr};
    RpptDesc nv12Desc, rgbDesc, frameBatchDesc, cropDesc, resizedDesc, outInteriorDesc;
    RpptGenericDesc volumeDesc{}, rowsDesc{}, rowDesc{};
    ~RppState() {
        if (handle) {
            rppDestroy(handle, RppBackend::RPP_HIP_BACKEND);
        }
        (void)hipFree(rgb8);
        (void)hipFree(cropped);
        (void)hipFree(resized);
        (void)hipFree(tensor);
        (void)hipFree(cropRoisXywh);
        (void)hipFree(cropRoisLtrb);
        (void)hipFree(resizeRois);
        (void)hipFree(resizedRois);
        (void)hipFree(subRow);
        (void)hipFree(divRow);
        (void)hipFree(normMean);
        (void)hipFree(normStdDev);
        for (void* p : {static_cast<void*>(dstSizes), static_cast<void*>(cmnOffset), static_cast<void*>(cmnMultiplier),
                        static_cast<void*>(mirror), static_cast<void*>(rmnMean), static_cast<void*>(rmnStdDevMul),
                        static_cast<void*>(rmnStdDevOne),
                        static_cast<void*>(mulScalar), static_cast<void*>(volumeRoi), static_cast<void*>(rowsRoi),
                        static_cast<void*>(rowRoi)}) {
            (void)hipHostFree(p);
        }
    }
};

std::shared_ptr<RppState> makeRppState(const Context& ctx, size_t n, bool resizedBuffer = false) {
    auto st = std::make_shared<RppState>();
    st->n = n;
    st->cropPitch = alignUp(CROP_W, RPP_PIXEL_ALIGN);
    st->resizedPitch = alignUp(RES_W, RPP_PIXEL_ALIGN);
    const int outW = OUT_W, outH = OUT_H;
    const unsigned N = static_cast<unsigned>(n);
    RPP_CHECK(static_cast<RppStatus>(rppCreate(&st->handle, n, 0, ctx.stream, RppBackend::RPP_HIP_BACKEND)));
    st->rgb8 = deviceAlloc<uint8_t>(static_cast<size_t>(FRAME_W) * FRAME_H * 3 + RPP_SLACK_BYTES);
    st->cropped = deviceAlloc<float>(n * st->cropPitch * CROP_H * 3 * sizeof(float) + RPP_SLACK_BYTES);
    if (resizedBuffer) {
        st->resized = deviceAlloc<float>(n * st->resizedPitch * RES_H * 3 * sizeof(float) + RPP_SLACK_BYTES);
        std::vector<RpptROI> resizedRois(n);
        for (auto& r : resizedRois) {
            r.xywhROI = {{0, 0}, RES_W, RES_H};
        }
        st->resizedRois = deviceAlloc<RpptROI>(n * sizeof(RpptROI));
        gpuErrchk(hipMemcpyAsync(st->resizedRois, resizedRois.data(), n * sizeof(RpptROI), hipMemcpyHostToDevice, ctx.stream));
    }
    // rppt_resize_mirror_normalize zeroes n * nStride floats from its destination pointer, which starts BORDER_TOP rows
    // into the tensor: the slack covers those rows after the last plane
    st->tensor = deviceAlloc<float>(n * OUT_FLOATS * sizeof(float) + BORDER_TOP * outW * 3 * sizeof(float) + RPP_SLACK_BYTES);

    std::vector<RpptROI> cropRoisXywh(n), cropRoisLtrb(n), resizeRois(n);
    st->dstSizes = hostAlloc<RpptImagePatch>(n);
    st->cmnOffset = hostAlloc<float>(n * 3);
    st->cmnMultiplier = hostAlloc<float>(n * 3);
    st->mirror = hostAlloc<Rpp32u>(n);
    st->rmnMean = hostAlloc<float>(n * 3);
    st->rmnStdDevMul = hostAlloc<float>(n * 3);
    st->rmnStdDevOne = hostAlloc<float>(n * 3);
    for (size_t i = 0; i < n; ++i) {
        const fk::Rect& r = ctx.crops[i];
        cropRoisXywh[i].xywhROI = {{static_cast<int>(r.x), static_cast<int>(r.y)}, CROP_W, CROP_H};
        cropRoisLtrb[i].ltrbROI = {{static_cast<int>(r.x), static_cast<int>(r.y)},
                                   {static_cast<int>(r.x) + CROP_W - 1, static_cast<int>(r.y) + CROP_H - 1}};
        resizeRois[i].ltrbROI = {{0, 0}, {CROP_W - 1, CROP_H - 1}};
        st->dstSizes[i] = {static_cast<Rpp32u>(RES_W), static_cast<Rpp32u>(RES_H)};
        st->mirror[i] = 0;
        for (int c = 0; c < 3; ++c) {
            st->cmnOffset[i * 3 + c] = 0.f;
            st->cmnMultiplier[i * 3 + c] = 1.f;
            // RPP's 8u -> 32f resize_mirror_normalize computes (x - mean) / stdDev on the 0..255 values: see
            // resizeMirrorNormalizeStage
            st->rmnMean[i * 3 + c] = 0.f;
            st->rmnStdDevMul[i * 3 + c] = 1.f / MUL[c];
            st->rmnStdDevOne[i * 3 + c] = 1.f;
        }
    }
    st->cropRoisXywh = deviceAlloc<RpptROI>(n * sizeof(RpptROI));
    st->cropRoisLtrb = deviceAlloc<RpptROI>(n * sizeof(RpptROI));
    st->resizeRois = deviceAlloc<RpptROI>(n * sizeof(RpptROI));
    gpuErrchk(hipMemcpyAsync(st->cropRoisXywh, cropRoisXywh.data(), n * sizeof(RpptROI), hipMemcpyHostToDevice, ctx.stream));
    gpuErrchk(hipMemcpyAsync(st->cropRoisLtrb, cropRoisLtrb.data(), n * sizeof(RpptROI), hipMemcpyHostToDevice, ctx.stream));
    gpuErrchk(hipMemcpyAsync(st->resizeRois, resizeRois.data(), n * sizeof(RpptROI), hipMemcpyHostToDevice, ctx.stream));

    st->mulScalar = hostAlloc<float>(1);
    st->mulScalar[0] = MUL[0];
    st->volumeRoi = hostAlloc<RpptROI3D>(1);
    st->volumeRoi->xyzwhdROI = {{0, 0, 0}, outW, outH, static_cast<int>(n)};
    std::vector<float> subRow(outW * 3), divRow(outW * 3), normMean(outW * 3), normStdDev(outW * 3);
    for (int i = 0; i < outW * 3; ++i) {
        subRow[i] = SUB[i % 3];
        divRow[i] = DIV[i % 3];
        normMean[i] = SUB[i % 3] / MUL[i % 3];
        normStdDev[i] = DIV[i % 3] / MUL[i % 3];
    }
    const auto uploadRow = [&](float*& dst, const std::vector<float>& src) {
        dst = deviceAlloc<float>(src.size() * sizeof(float) + RPP_SLACK_BYTES);
        gpuErrchk(hipMemcpyAsync(dst, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice, ctx.stream));
    };
    uploadRow(st->subRow, subRow);
    uploadRow(st->divRow, divRow);
    uploadRow(st->normMean, normMean);
    uploadRow(st->normStdDev, normStdDev);
    st->rowsRoi = hostAlloc<Rpp32u>(4);
    st->rowRoi = hostAlloc<Rpp32u>(4);
    const Rpp32u rows = N * outH, cols = outW * 3;
    st->rowsRoi[0] = 0; st->rowsRoi[1] = 0; st->rowsRoi[2] = rows; st->rowsRoi[3] = cols;
    st->rowRoi[0] = 0; st->rowRoi[1] = 0; st->rowRoi[2] = 1; st->rowRoi[3] = cols;

    st->nv12Desc = RpptDesc(1, 1, FRAME_H, FRAME_W, RpptDataType::U8, RpptLayout::NHWC);
    st->rgbDesc = RpptDesc(1, 3, FRAME_H, FRAME_W, RpptDataType::U8, RpptLayout::NHWC);
    // All the batch images alias the RGB frame (zero batch stride), each with its own crop ROI
    st->frameBatchDesc = RpptDesc(N, 3, FRAME_H, FRAME_W, RpptDataType::U8, RpptLayout::NHWC, 0,
                                  RpptStrides{0, 1, static_cast<Rpp32u>(FRAME_W * 3), 3});
    st->cropDesc = RpptDesc(N, 3, CROP_H, CROP_W, RpptDataType::F32, RpptLayout::NHWC, 0,
                            RpptStrides{static_cast<Rpp32u>(st->cropPitch * CROP_H * 3), 1,
                                        static_cast<Rpp32u>(st->cropPitch * 3), 3});
    st->resizedDesc = RpptDesc(N, 3, RES_H, RES_W, RpptDataType::F32, RpptLayout::NHWC, 0,
                               RpptStrides{static_cast<Rpp32u>(st->resizedPitch * RES_H * 3), 1,
                                           static_cast<Rpp32u>(st->resizedPitch * 3), 3});
    // Rows BORDER_TOP to BORDER_TOP + resH - 1 of every output plane
    st->outInteriorDesc = RpptDesc(N, 3, RES_H, RES_W, RpptDataType::F32, RpptLayout::NHWC,
                                   static_cast<Rpp32u>(BORDER_TOP * outW * 3 * sizeof(float)),
                                   RpptStrides{static_cast<Rpp32u>(outW * outH * 3), 1, static_cast<Rpp32u>(outW * 3), 3});
    // The scalar primitives launch one kernel per batch image: the whole tensor is one NDHWC volume of N planes
    RpptGenericDesc& v = st->volumeDesc;
    v.numDims = 5; v.offsetInBytes = 0; v.dataType = RpptDataType::F32; v.layout = RpptLayout::NDHWC;
    v.dims[0] = 1; v.dims[1] = N; v.dims[2] = outH; v.dims[3] = outW; v.dims[4] = 3;
    v.strides[4] = 1; v.strides[3] = 3; v.strides[2] = outW * 3; v.strides[1] = outH * outW * 3; v.strides[0] = N * outH * outW * 3;
    // Per channel Sub and Div: the tensor as rows of outW float3, minus / divided by one row of the constants
    // (broadcast over the rows)
    RpptGenericDesc& a = st->rowsDesc;
    a.numDims = 3; a.offsetInBytes = 0; a.dataType = RpptDataType::F32; a.layout = RpptLayout::NDHWC;
    a.dims[0] = 1; a.dims[1] = rows; a.dims[2] = cols; a.strides[2] = 1; a.strides[1] = cols; a.strides[0] = rows * cols;
    RpptGenericDesc& b = st->rowDesc;
    b = a;
    b.dims[1] = 1; b.strides[0] = cols;
    // The uploads above were enqueued on the benchmark stream; their host vectors must outlive them
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    return st;
}

// One GPU operation of an RPP pipeline: its name in the breakdown and the call
using Stage = std::pair<std::string, std::function<void()>>;
using RppStatePtr = std::shared_ptr<RppState>;

void rppYuvToRgb(const Context& ctx, const RppState& st) {
    beforeKernel();
    RPP_CHECK(rppt_yuv_to_rgb(ctx.y, ctx.uv, const_cast<RpptDesc*>(&st.nv12Desc), st.rgb8,
                              const_cast<RpptDesc*>(&st.rgbDesc), ctx.pitch, ctx.pitch, FRAME_W * 3, FRAME_W, FRAME_H,
                              RpptColorStandard_BT2020_NCL, RpptColorRange_FULL, st.handle, RppBackend::RPP_HIP_BACKEND));
}

Stage yuvToRgbStage(Context& ctx, RppStatePtr st) {
    return {"rppt_yuv_to_rgb_full_frame", [&ctx, st]() { rppYuvToRgb(ctx, *st); }};
}

// Crop + 8u -> 32f of all the crops, every batch image aliasing the RGB frame
Stage cropConvertStage(RppStatePtr st) {
    return {"rppt_crop_mirror_normalize_8u32f_batch", [st]() {
                beforeKernel();
                RPP_CHECK(rppt_crop_mirror_normalize(st->rgb8, &st->frameBatchDesc, st->cropped, &st->cropDesc,
                                                     st->cmnOffset, st->cmnMultiplier, st->mirror, st->cropRoisXywh,
                                                     RpptRoiType::XYWH, st->handle, RppBackend::RPP_HIP_BACKEND));
            }};
}

// The whole output tensor to zero (RPP has no constant border)
Stage zeroTensorStage(Context& ctx, RppStatePtr st) {
    return {"hipMemsetAsync_output_tensor", [&ctx, st]() {
                beforeKernel();
                gpuErrchk(hipMemsetAsync(st->tensor, 0, st->n * OUT_FLOATS * sizeof(float), ctx.stream));
            }};
}

// Only the top border rows of the first plane: rppt_resize_mirror_normalize zeroes everything after them itself
Stage zeroFirstBorderStage(Context& ctx, RppStatePtr st) {
    return {"hipMemsetAsync_first_plane_top_border", [&ctx, st]() {
                beforeKernel();
                gpuErrchk(hipMemsetAsync(st->tensor, 0, static_cast<size_t>(BORDER_TOP) * OUT_W * 3 * sizeof(float),
                                         ctx.stream));
            }};
}

// Resize of all the crops, written into rows BORDER_TOP.. of the output planes
Stage resizeIntoOutputStage(RppStatePtr st) {
    return {"rppt_resize_batch_into_output_rows", [st]() {
                beforeKernel();
                RPP_CHECK(rppt_resize(st->cropped, &st->cropDesc, st->tensor, &st->outInteriorDesc, st->dstSizes,
                                      RpptInterpolationType::BILINEAR, st->resizeRois, RpptRoiType::LTRB, st->handle,
                                      RppBackend::RPP_HIP_BACKEND));
            }};
}

// Resize of all the crops into their own float3 buffer (rpp_npp_like)
Stage resizeStage(RppStatePtr st) {
    return {"rppt_resize_batch", [st]() {
                beforeKernel();
                RPP_CHECK(rppt_resize(st->cropped, &st->cropDesc, st->resized, &st->resizedDesc, st->dstSizes,
                                      RpptInterpolationType::BILINEAR, st->resizeRois, RpptRoiType::LTRB, st->handle,
                                      RppBackend::RPP_HIP_BACKEND));
            }};
}

// Copy of the resized crops into rows BORDER_TOP.. of the output planes (rpp_npp_like, NPP's CopyConstBorder)
Stage copyIntoOutputStage(RppStatePtr st) {
    return {"rppt_crop_f32_batch_into_output_rows", [st]() {
                beforeKernel();
                RPP_CHECK(rppt_crop(st->resized, &st->resizedDesc, st->tensor, &st->outInteriorDesc, st->resizedRois,
                                    RpptRoiType::XYWH, st->handle, RppBackend::RPP_HIP_BACKEND));
            }};
}

// Crop + 8u -> 32f + resize of all the crops from the RGB frame into the output rows, and, with mul, the 1/255 scale:
// RPP's 8u -> 32f resize_mirror_normalize computes (x - mean) / stdDev on the 0..255 values, so the whole tail would
// be mean = 255 * sub, stdDev = 255 * div; but it only writes the resized pixels, while the border rows must hold the
// normalized black (0 - sub) / div, so the rest of the tail runs on the whole tensor afterwards
Stage resizeMirrorNormalizeStage(RppStatePtr st, bool mul) {
    return {mul ? "rppt_resize_mirror_normalize_8u32f_batch_mul_into_output_rows"
                : "rppt_resize_mirror_normalize_8u32f_batch_into_output_rows",
            [st, mul]() {
                beforeKernel();
                RPP_CHECK(rppt_resize_mirror_normalize(st->rgb8, &st->frameBatchDesc, st->tensor, &st->outInteriorDesc,
                                                       st->dstSizes, RpptInterpolationType::BILINEAR, st->rmnMean,
                                                       mul ? st->rmnStdDevMul : st->rmnStdDevOne, st->mirror,
                                                       st->cropRoisLtrb, RpptRoiType::LTRB, st->handle,
                                                       RppBackend::RPP_HIP_BACKEND));
            }};
}

Stage multiplyStage(RppStatePtr st) {
    return {"rppt_multiply_scalar", [st]() {
                beforeKernel();
                RPP_CHECK(rppt_multiply_scalar(st->tensor, &st->volumeDesc, st->tensor, &st->volumeDesc, st->mulScalar,
                                               st->volumeRoi, RpptRoi3DType::XYZWHD, st->handle,
                                               RppBackend::RPP_HIP_BACKEND));
            }};
}

Stage subtractStage(RppStatePtr st) {
    return {"rppt_tensor_subtract_tensor", [st]() {
                beforeKernel();
                RPP_CHECK(rppt_tensor_subtract_tensor(st->tensor, st->subRow, &st->rowsDesc, &st->rowDesc, st->tensor,
                                                      &st->rowsDesc, RPP_BROADCAST_ENABLE, st->rowsRoi, st->rowRoi,
                                                      st->handle, RppBackend::RPP_HIP_BACKEND));
            }};
}

Stage divideStage(RppStatePtr st) {
    return {"rppt_tensor_divide_tensor", [st]() {
                beforeKernel();
                RPP_CHECK(rppt_tensor_divide_tensor(st->tensor, st->divRow, &st->rowsDesc, &st->rowDesc, st->tensor,
                                                    &st->rowsDesc, RPP_BROADCAST_ENABLE, st->rowsRoi, st->rowRoi,
                                                    st->handle, RppBackend::RPP_HIP_BACKEND));
            }};
}

// Mul, Sub and Div in one call on the whole tensor (borders included): rppt_normalize on the rows of outW float3 with
// per column mean 255 * sub and stdDev 255 * div, fmaf(x - mean, 1 / stdDev, 0) = (x - 255 sub) / (255 div)
Stage normalizeStage(RppStatePtr st) {
    return {"rppt_normalize_mul_sub_div", [st]() {
                beforeKernel();
                RPP_CHECK(rppt_normalize(st->tensor, &st->rowsDesc, st->tensor, &st->rowsDesc, 1, st->normMean,
                                         st->normStdDev, 0, 1.f, 0.f, st->rowsRoi, st->handle,
                                         RppBackend::RPP_HIP_BACKEND));
            }};
}

Implementation makeRppImplementation(const std::string& name, Context& ctx, RppStatePtr st, std::vector<Stage> stages) {
    Implementation impl{name, "rpp_sampling_8bit"};
    for (size_t i = 0; i < stages.size(); ++i) {
        stages[i].first = std::to_string(i + 1) + "_" + stages[i].first;
    }
    impl.stages = std::move(stages);
    impl.run = [stages = impl.stages]() {
        for (const auto& stage : stages) {
            stage.second();
        }
    };
    impl.download = [&ctx, st]() { return downloadDevice(ctx.stream, st->tensor, st->n * OUT_FLOATS); };
    impl.fillNaN = [&ctx, st]() { poisonDevice(ctx.stream, st->tensor, st->n * OUT_FLOATS); };
    return impl;
}

// One RPP call per operation of Figure 1
template <size_t N>
Implementation makeRpp(Context& ctx) {
    auto st = makeRppState(ctx, N);
    return makeRppImplementation("rpp_per_op", ctx, st, {yuvToRgbStage(ctx, st), cropConvertStage(st), zeroTensorStage(ctx, st),
                                             resizeIntoOutputStage(st), multiplyStage(st), subtractStage(st),
                                             divideStage(st)});
}

// The same calls, with exactly the intermediates of the paper's NPP pipeline (8 bit frame, float3 crops, float3
// resized crops, output): the resize writes its own buffer, and one batched rppt_crop F32 -> F32 is the strided copy
// into the output rows, after the memset of the tensor
template <size_t N>
Implementation makeRppNppLike(Context& ctx) {
    auto st = makeRppState(ctx, N, true);
    return makeRppImplementation("rpp_npp_like", ctx, st, {yuvToRgbStage(ctx, st), cropConvertStage(st), resizeStage(st),
                                                      zeroTensorStage(ctx, st), copyIntoOutputStage(st),
                                                      multiplyStage(st), subtractStage(st), divideStage(st)});
}

// RPP's fused primitive: rppt_resize_mirror_normalize does crop + 8u -> 32f + resize + Mul, then Sub and Div
template <size_t N>
Implementation makeRppFused(Context& ctx) {
    auto st = makeRppState(ctx, N);
    return makeRppImplementation("rpp_fused", ctx, st, {yuvToRgbStage(ctx, st), zeroFirstBorderStage(ctx, st),
                                                   resizeMirrorNormalizeStage(st, true), subtractStage(st),
                                                   divideStage(st)});
}

// One RPP call per operation up to the resize, then Mul + Sub + Div in one rppt_normalize
template <size_t N>
Implementation makeRppNormalize(Context& ctx) {
    auto st = makeRppState(ctx, N);
    return makeRppImplementation("rpp_normalize", ctx, st, {yuvToRgbStage(ctx, st), cropConvertStage(st),
                                                       zeroTensorStage(ctx, st), resizeIntoOutputStage(st),
                                                       normalizeStage(st)});
}

// The fewest RPP calls: rppt_resize_mirror_normalize (crop + 8u -> 32f + resize) and rppt_normalize (Mul + Sub + Div)
template <size_t N>
Implementation makeRppFusedNormalize(Context& ctx) {
    auto st = makeRppState(ctx, N);
    return makeRppImplementation("rpp_fused_normalize", ctx, st, {yuvToRgbStage(ctx, st), zeroFirstBorderStage(ctx, st),
                                                             resizeMirrorNormalizeStage(st, false), normalizeStage(st)});
}

// References on FKL's CPU backend. The float reference is the Figure 1 chain itself. The 8 bit reference
// quantizes the RGB values to 8 bit right after the color conversion, like FKL VF+HF does.
template <size_t N, bool EIGHT_BIT>
std::vector<float> cpuReference(const Context& ctx) {
    const auto crops = firstCrops<N>(ctx);
    fk::Stream_<fk::ParArch::CPU> cpuStream;
    fk::Tensor<fk::float3> out(OUT_W, OUT_H, N, 1, fk::MemType::Host);
    const auto tail = [&](const auto& rgbRead) {
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
            rgbRead,
            fk::Crop<>::build(crops),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
            fk::Mul<fk::float3>::build(mulValue),
            fk::Sub<fk::float3>::build(subValue),
            fk::Div<fk::float3>::build(divValue),
            fk::TensorWrite<fk::float3>::build(out.ptr()));
    };
    const auto rgbRead = fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameHost).then(YUVToRGB::build());
    if constexpr (EIGHT_BIT) {
        tail(rgbRead.then(fk::NormalizePixel<fk::float3, fk::ColorDepth::p8bit>::build())
                    .then(fk::Add<fk::float3>::build(fk::make_set<fk::float3>(QUANTIZATION_OFFSET)))
                    .then(fk::SaturateDenormalizePixel<fk::float3, fk::uchar3, fk::ColorDepth::p8bit>::build())
                    .then(fk::Cast<fk::uchar3, fk::float3>::build()));
    } else {
        tail(rgbRead);
    }
    cpuStream.sync();
    const float* data = reinterpret_cast<const float*>(out.ptr().data);
    return std::vector<float>(data, data + N * OUT_FLOATS);
}

// Plain C++ reference with RPP's bilinear sampling convention, measured with linear ramps (--resize-convention):
// rppt_resize and rppt_resize_mirror_normalize sample at src = (dst + 0.5) * scale - 0.5 (pixel centers), with the
// sample position clamped to [0, size - 2] in both axes (the last source column and row never get any weight).
// The RGB values are RPP's own 8 bit frame (rppt_yuv_to_rgb's output, ctx.rppRgb8): its conversion differs from
// FKL's rounded float conversion in a few values (conversionDiagnostic), which this reference does not mix in. Same
// border and pointwise tail as the other references.
template <size_t N>
std::vector<float> rppConventionReference8bit(const Context& ctx) {
    if (ctx.rppRgb8.size() != static_cast<size_t>(FRAME_W) * FRAME_H * 3) {
        throw std::runtime_error("RPP's RGB frame (ctx.rppRgb8) is not available");
    }
    const int outW = OUT_W, outH = OUT_H;
    std::vector<float> out(N * OUT_FLOATS);
    const float scaleX = static_cast<float>(CROP_W) / RES_W;
    const float scaleY = static_cast<float>(CROP_H) / RES_H;
    std::vector<float> crop(static_cast<size_t>(CROP_W) * CROP_H * 3);
    const auto at = [&](int y, int x, int c) -> float& { return crop[(static_cast<size_t>(y) * CROP_W + x) * 3 + c]; };
    for (size_t i = 0; i < N; ++i) {
        const fk::Rect& r = ctx.crops[i];
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                const size_t idx = (static_cast<size_t>(r.y + y) * FRAME_W + (r.x + x)) * 3;
                for (int c = 0; c < 3; ++c) {
                    at(y, x, c) = ctx.rppRgb8[idx + c];
                }
            }
        }
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                float* dst = &out[((i * outH + y) * outW + x) * 3];
                const int ry = y - BORDER_TOP;
                float value[3] = {0.f, 0.f, 0.f};
                if (ry >= 0 && ry < RES_H) {
                    const float sx = std::min(static_cast<float>(CROP_W - 2), std::max(0.f, (x + 0.5f) * scaleX - 0.5f));
                    const float sy = std::min(static_cast<float>(CROP_H - 2), std::max(0.f, (ry + 0.5f) * scaleY - 0.5f));
                    const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
                    const int x1 = std::min(x0 + 1, CROP_W - 1), y1 = std::min(y0 + 1, CROP_H - 1);
                    const float fx = sx - x0, fy = sy - y0;
                    for (int c = 0; c < 3; ++c) {
                        value[c] = (at(y0, x0, c) * (1.f - fx) + at(y0, x1, c) * fx) * (1.f - fy) +
                                   (at(y1, x0, c) * (1.f - fx) + at(y1, x1, c) * fx) * fy;
                    }
                }
                for (int c = 0; c < 3; ++c) {
                    dst[c] = (value[c] * MUL[c] - SUB[c]) / DIV[c];
                }
            }
        }
    }
    return out;
}

struct ErrorStats {
    std::array<float, 3> maxPerChannel{0.f, 0.f, 0.f};
    double mean{0.};
    size_t nonFinite{0};
    float max() const { return std::max({maxPerChannel[0], maxPerChannel[1], maxPerChannel[2]}); }
};

ErrorStats compare(const std::vector<float>& output, const std::vector<float>& reference) {
    ErrorStats stats;
    double sum = 0.;
    for (size_t i = 0; i < output.size(); ++i) {
        const float err = std::abs(output[i] - reference[i]);
        if (!std::isfinite(err)) {
            stats.nonFinite++;
            stats.maxPerChannel[i % 3] = INFINITY;
            continue;
        }
        stats.maxPerChannel[i % 3] = std::max(stats.maxPerChannel[i % 3], err);
        sum += err;
    }
    stats.mean = sum / static_cast<double>(output.size());
    return stats;
}

// Statistics of a set of timed executions, in microseconds
struct TimingStats {
    double mean{0.}, variance{0.}, max{0.}, min{0.}, median{0.}, p10{0.}, p90{0.};
};

double percentile(const std::vector<float>& sorted, double p) {
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

TimingStats computeStats(std::vector<float> ms) {
    TimingStats t;
    double sum = 0.;
    for (const float v : ms) {
        sum += v;
    }
    t.mean = sum / ms.size() * 1000.;
    double sumOfDiff = 0.;
    for (const float v : ms) {
        sumOfDiff += (v * 1000. - t.mean) * (v * 1000. - t.mean);
    }
    t.variance = sumOfDiff / (ms.size() - 1);
    std::sort(ms.begin(), ms.end());
    t.min = ms.front() * 1000.;
    t.max = ms.back() * 1000.;
    t.median = percentile(ms, 0.5) * 1000.;
    t.p10 = percentile(ms, 0.1) * 1000.;
    t.p90 = percentile(ms, 0.9) * 1000.;
    return t;
}

// Latency of each complete pipeline, the way FKL's two-execution benchmarks measure: in every iteration all the
// implementations run one after the other, each one between two HIP events on the same stream (everything the
// host does to enqueue it is included) and followed by a synchronization. Inputs are already in device memory.
// Returns the TIMED_ITERS execution times of each implementation, in ms, in iteration order.
std::vector<std::vector<float>> measureInterleaved(const Context& ctx, const std::vector<Implementation>& impls,
                                                   int warmup = WARMUP_ITERS, int iters = TIMED_ITERS) {
    hipEvent_t start, stop;
    gpuErrchk(hipEventCreate(&start));
    gpuErrchk(hipEventCreate(&stop));
    std::vector<std::vector<float>> ms(impls.size(), std::vector<float>(iters));
    for (int i = 0; i < warmup + iters; ++i) {
        for (size_t k = 0; k < impls.size(); ++k) {
            gpuErrchk(hipEventRecord(start, ctx.stream));
            impls[k].run();
            gpuErrchk(hipEventRecord(stop, ctx.stream));
            gpuErrchk(hipEventSynchronize(stop));
            if (i >= warmup) {
                gpuErrchk(hipEventElapsedTime(&ms[k][i - warmup], start, stop));
            }
        }
    }
    gpuErrchk(hipEventDestroy(start));
    gpuErrchk(hipEventDestroy(stop));
    return ms;
}

// Time of each GPU operation of a multi-operation pipeline: HIP events between its operations on the same stream
struct StageTiming {
    std::string name;
    int batch{0};
    std::string stage;
    TimingStats timing;
};
std::vector<StageTiming> breakdown;

void measureStages(const Context& ctx, const Implementation& impl, int batch) {
    const size_t numStages = impl.stages.size();
    std::vector<hipEvent_t> events(numStages + 1);
    for (auto& e : events) {
        gpuErrchk(hipEventCreate(&e));
    }
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        impl.run();
        gpuErrchk(hipStreamSynchronize(ctx.stream));
    }
    std::vector<std::vector<float>> ms(numStages, std::vector<float>(TIMED_ITERS));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        gpuErrchk(hipEventRecord(events[0], ctx.stream));
        for (size_t s = 0; s < numStages; ++s) {
            impl.stages[s].second();
            gpuErrchk(hipEventRecord(events[s + 1], ctx.stream));
        }
        gpuErrchk(hipEventSynchronize(events[numStages]));
        for (size_t s = 0; s < numStages; ++s) {
            gpuErrchk(hipEventElapsedTime(&ms[s][i], events[s], events[s + 1]));
        }
    }
    for (auto& e : events) {
        gpuErrchk(hipEventDestroy(e));
    }
    std::vector<float> total(TIMED_ITERS, 0.f);
    for (size_t s = 0; s < numStages; ++s) {
        for (int i = 0; i < TIMED_ITERS; ++i) {
            total[i] += ms[s][i];
        }
        breakdown.push_back({impl.name, batch, impl.stages[s].first, computeStats(ms[s])});
    }
    breakdown.push_back({impl.name, batch, "total", computeStats(total)});
}

// ------------------------------------------------------------------------------------------------
// rocprofiler-sdk, in process (the counterpart of CUPTI in bvf_bench.cu):
//  - HIP runtime API callback tracing: kernel launches, memsets and memcpys of one pipeline (works on WSL2)
//  - kernel dispatch buffer tracing: GPU start and end timestamps of every kernel, for the GPU time only benchmark
//    (on WSL2 the SDK delivers no kernel dispatch records; kernelTracingAvailable() detects it)
// ------------------------------------------------------------------------------------------------
struct LaunchCounts {
    size_t kernels{0}, memsets{0}, memcpys{0};
};

struct KernelRecord {
    uint64_t start{0}, end{0}, kernelId{0};
};

namespace rocprof {
rocprofiler_context_id_t apiContext{};
rocprofiler_context_id_t kernelContext{};
rocprofiler_context_id_t codeObjectContext{};
rocprofiler_buffer_id_t kernelBuffer{};
bool configured = false;
LaunchCounts counts;
std::mutex mutex;
std::vector<KernelRecord> kernels;
std::map<uint64_t, std::string> kernelNames;

void apiCallback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*) {
    if (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) {
        return;
    }
    const char* name = nullptr;
    uint64_t length = 0;
    if (rocprofiler_query_callback_tracing_kind_operation_name(record.kind, record.operation, &name, &length) !=
            ROCPROFILER_STATUS_SUCCESS || name == nullptr) {
        return;
    }
    const std::string op(name, length);
    if (op.find("LaunchKernel") != std::string::npos || op.find("LaunchCooperativeKernel") != std::string::npos) {
        counts.kernels++;
    } else if (op.rfind("hipMemset", 0) == 0) {
        counts.memsets++;
    } else if (op.rfind("hipMemcpy", 0) == 0) {
        counts.memcpys++;
    }
}

void codeObjectCallback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*) {
    if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
        record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER &&
        record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
        const auto* data = static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(record.payload);
        std::lock_guard<std::mutex> lock(mutex);
        kernelNames[data->kernel_id] = data->kernel_name ? data->kernel_name : "";
    }
}

void kernelBufferCallback(rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_record_header_t** headers,
                          size_t numHeaders, void*, uint64_t) {
    std::lock_guard<std::mutex> lock(mutex);
    for (size_t i = 0; i < numHeaders; ++i) {
        const auto* h = headers[i];
        if (h->category == ROCPROFILER_BUFFER_CATEGORY_TRACING && h->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
            const auto* r = static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(h->payload);
            kernels.push_back({r->start_timestamp, r->end_timestamp, r->dispatch_info.kernel_id});
        }
    }
}

int toolInit(rocprofiler_client_finalize_t, void*) {
    try {
        ROCP_CHECK(rocprofiler_create_context(&apiContext));
        ROCP_CHECK(rocprofiler_configure_callback_tracing_service(apiContext, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                                  nullptr, 0, apiCallback, nullptr));
        ROCP_CHECK(rocprofiler_create_context(&codeObjectContext));
        ROCP_CHECK(rocprofiler_configure_callback_tracing_service(codeObjectContext, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                                  nullptr, 0, codeObjectCallback, nullptr));
        ROCP_CHECK(rocprofiler_create_context(&kernelContext));
        ROCP_CHECK(rocprofiler_create_buffer(kernelContext, 32 << 20, (32 << 20) - (1 << 20),
                                             ROCPROFILER_BUFFER_POLICY_LOSSLESS, kernelBufferCallback, nullptr,
                                             &kernelBuffer));
        ROCP_CHECK(rocprofiler_configure_buffer_tracing_service(kernelContext, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                                nullptr, 0, kernelBuffer));
        ROCP_CHECK(rocprofiler_start_context(codeObjectContext));
        configured = true;
    } catch (const std::exception& e) {
        std::cerr << "rocprofiler-sdk tool initialization failed: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

void toolFini(void*) {}

void requireConfigured() {
    if (!configured) {
        throw std::runtime_error("rocprofiler-sdk tool not initialized (is librocprofiler-sdk on the library path?)");
    }
}

void flushKernels() { ROCP_CHECK(rocprofiler_flush_buffer(kernelBuffer)); }

bool isFlushKernel(uint64_t kernelId) {
    const auto it = kernelNames.find(kernelId);
    return it != kernelNames.end() && it->second.find("flushCachesKernel") != std::string::npos;
}

// The HIP runtime executes hipMemset* (and some hipMemcpy*) with its own blit kernels; like CUPTI's kernel records in
// bvf_bench.cu, the GPU time only benchmark counts the kernels the pipelines launch, not memsets and copies
bool isRuntimeBlitKernel(uint64_t kernelId) {
    const auto it = kernelNames.find(kernelId);
    return it != kernelNames.end() && it->second.rfind("__amd_rocclr_", 0) == 0;
}
} // namespace rocprof

} // namespace bench

// The tool is only registered in the profiling child process (BENCH_ROCPROFILER=1), so that no profiler wraps the HIP
// API during the timed runs
extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t, const char*, uint32_t,
                                                                      rocprofiler_client_id_t* id) {
    if (std::getenv("BENCH_ROCPROFILER") == nullptr) {
        return nullptr;
    }
    id->name = "rocm_bench";
    static rocprofiler_tool_configure_result_t config{sizeof(rocprofiler_tool_configure_result_t),
                                                      bench::rocprof::toolInit, bench::rocprof::toolFini, nullptr};
    return &config;
}

namespace bench {

// Kernel launches, memsets and memcpys of one pipeline, from HIP runtime API tracing, in a separate pass after all
// the timings
LaunchCounts countLaunches(const Context& ctx, const Implementation& impl) {
    rocprof::requireConfigured();
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    rocprof::counts = LaunchCounts{};
    ROCP_CHECK(rocprofiler_start_context(rocprof::apiContext));
    impl.run();
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    ROCP_CHECK(rocprofiler_stop_context(rocprof::apiContext));
    // The stream synchronization itself is not a launch, memset or memcpy
    return rocprof::counts;
}

// Whether the ROCm stack delivers kernel dispatch records (not on WSL2, where rocprofiler-sdk uses the dxcore agents)
bool kernelTracingAvailable(Context& ctx) {
    rocprof::requireConfigured();
    const auto probe = makeFklBvf<1>(ctx);
    probe.run();
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    rocprof::flushKernels();
    {
        std::lock_guard<std::mutex> lock(rocprof::mutex);
        rocprof::kernels.clear();
    }
    ROCP_CHECK(rocprofiler_start_context(rocprof::kernelContext));
    probe.run();
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    ROCP_CHECK(rocprofiler_stop_context(rocprof::kernelContext));
    rocprof::flushKernels();
    std::lock_guard<std::mutex> lock(rocprof::mutex);
    const bool available = !rocprof::kernels.empty();
    rocprof::kernels.clear();
    return available;
}

struct Result {
    std::string name;
    int batch{0};
    TimingStats timing;
    // Mean over the timed iterations of (this implementation's time / FKL BVF's time) in the same iteration
    double meanSpeedupOfBvf{NAN};
    // GPU time only: sum of the execution times of the pipeline's kernels (kernel dispatch tracing), per execution
    TimingStats gpuTiming;
    double meanGpuSpeedupOfBvf{NAN};
    bool gpuMeasured{false};
    // GPU time of the HIP runtime's blit kernels (memsets and copies, __amd_rocclr_*) of each execution, not part of
    // the GPU time only sums, and their dispatches per execution (median)
    TimingStats gpuBlitTiming, gpuFlushedBlitTiming;
    double gpuBlitDispatches{0.}, gpuFlushedBlitDispatches{0.};
    // GPU time only with the caches flushed before every GPU operation
    TimingStats gpuFlushedTiming;
    double meanGpuFlushedSpeedupOfBvf{NAN};
    bool gpuFlushedMeasured{false};
    ErrorStats errorVsOwnReference;
    ErrorStats errorVsFloatReference;
    ErrorStats errorVsRppConventionReference;
    // The paper's 8 bit reference: FKL's conversion rounded to 8 bit, FKL's sampling (src = dst * scale)
    ErrorStats errorVs8bitReference;
    std::string reference;
    LaunchCounts launches;
};

// The RPP variants, all numerically equivalent on the HIP backend. The benchmark's RPP pipeline ("rpp") is the fastest
// of them, chosen by the selection pass (selectRpp) or given with BENCH_RPP_VARIANT (profiling child, counter targets);
// with allRppVariants (--validate) every variant is validated
const std::array<std::string, 5> RPP_VARIANTS{"rpp_per_op", "rpp_npp_like", "rpp_fused", "rpp_normalize",
                                              "rpp_fused_normalize"};
std::string selectedRpp;
bool allRppVariants = false;

template <size_t N>
Implementation makeRppVariant(Context& ctx, const std::string& variant) {
    if (variant == "rpp_per_op") return makeRpp<N>(ctx);
    if (variant == "rpp_npp_like") return makeRppNppLike<N>(ctx);
    if (variant == "rpp_fused") return makeRppFused<N>(ctx);
    if (variant == "rpp_normalize") return makeRppNormalize<N>(ctx);
    if (variant == "rpp_fused_normalize") return makeRppFusedNormalize<N>(ctx);
    throw std::runtime_error("Unknown RPP variant " + variant);
}

template <size_t N>
std::vector<Implementation> makeRppVariants(Context& ctx) {
    std::vector<Implementation> impls;
    for (const auto& v : RPP_VARIANTS) {
        impls.push_back(makeRppVariant<N>(ctx, v));
    }
    return impls;
}

template <size_t N>
Implementation makeSelectedRpp(Context& ctx) {
    if (selectedRpp.empty()) {
        throw std::runtime_error("No RPP variant selected (selection pass or BENCH_RPP_VARIANT)");
    }
    Implementation impl = makeRppVariant<N>(ctx, selectedRpp);
    impl.variant = selectedRpp;
    impl.name = "rpp";
    return impl;
}

// The implementations in the comparison, and FKL BVF last, which has no intermediates in memory and keeps float3 from
// the color conversion to the write
template <size_t N>
std::vector<Implementation> makeImplementations(Context& ctx) {
    std::vector<Implementation> impls = allRppVariants ? makeRppVariants<N>(ctx) : std::vector<Implementation>{};
    if (!allRppVariants) {
        impls.push_back(makeSelectedRpp<N>(ctx));
    }
    impls.push_back(makeFklVfHf<N, true>(ctx));
    impls.push_back(makeFklBvf<N>(ctx));
    return impls;
}

// Validation, interleaved timing with per-execution speedups, and per operation breakdown, for batch N. With timing
// false, validation only.
// Progress messages on stderr with BENCH_VERBOSE=1
void progress(const std::string& what) {
    static const bool verbose = std::getenv("BENCH_VERBOSE") != nullptr;
    if (verbose) {
        std::cerr << "  [" << what << "]" << std::endl;
    }
}

template <size_t N>
void runBatch(Context& ctx, std::vector<Result>& results, bool timing = true) {
    std::cout << "=== N = " << N << std::endl;
    progress("FKL CPU float reference");
    const auto refFloat = cpuReference<N, false>(ctx);
    progress("FKL CPU 8 bit reference");
    const auto ref8bit = cpuReference<N, true>(ctx);
    progress("RPP convention reference");
    const auto refRppConvention = rppConventionReference8bit<N>(ctx);
    progress("implementations");
    const auto impls = makeImplementations<N>(ctx);
    const auto ownReference = [&](const Implementation& impl) -> const std::vector<float>& {
        return impl.reference == "fkl_cpu_float" ? refFloat : impl.reference == "fkl_cpu_8bit_rgb" ? ref8bit : refRppConvention;
    };
    const size_t bvf = impls.size() - 1;
    std::vector<Result> batchResults;
    for (const auto& impl : impls) {
        Result r;
        r.name = impl.name;
        r.batch = static_cast<int>(N);
        progress("validating " + impl.name);
        impl.fillNaN();
        if (std::getenv("BENCH_VERBOSE") != nullptr && !impl.stages.empty()) {
            // Operation by operation, synchronizing after each one, to locate failures
            for (const auto& stage : impl.stages) {
                progress("  " + stage.first);
                stage.second();
                gpuErrchk(hipStreamSynchronize(ctx.stream));
            }
        } else {
            impl.run();
        }
        const hipError_t launchError = hipGetLastError();
        if (launchError != hipSuccess) {
            throw std::runtime_error(impl.name + ": HIP error after the pipeline: " + hipGetErrorString(launchError));
        }
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        const auto output = impl.download();
        r.reference = impl.reference;
        r.errorVsOwnReference = compare(output, ownReference(impl));
        r.errorVsFloatReference = compare(output, refFloat);
        r.errorVsRppConventionReference = compare(output, refRppConvention);
        r.errorVs8bitReference = compare(output, ref8bit);
        if (r.errorVsOwnReference.nonFinite != 0) {
            size_t first = 0;
            while (first < output.size() && std::isfinite(output[first])) {
                ++first;
            }
            std::cerr << "  " << impl.name << ": " << r.errorVsOwnReference.nonFinite << " of " << output.size()
                      << " output values are not finite, the first at float index " << first << " (plane "
                      << first / OUT_FLOATS << ")" << std::endl;
        }
        batchResults.push_back(r);
    }
    const auto checkFloat3Frame = [&](const Implementation& float3Frame) {
        float3Frame.fillNaN();
        float3Frame.run();
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        const ErrorStats float3FrameError = compare(float3Frame.download(), refFloat);
        if (float3FrameError.max() > TOLERANCE) {
            throw std::runtime_error("fkl_vf_hf_float3_frame does not match the float reference");
        }
    };
    if (!timing) {
        for (const Result& r : batchResults) {
            std::cout << std::left << std::setw(12) << r.name << std::right << "  max err " << std::scientific
                      << std::setprecision(2) << r.errorVsOwnReference.max() << " (" << r.reference << "), vs float "
                      << r.errorVsFloatReference.max() << ", vs RPP sampling " << r.errorVsRppConventionReference.max()
                      << ", vs 8 bit (FKL sampling) " << r.errorVs8bitReference.max() << std::defaultfloat << std::endl;
            results.push_back(r);
        }
        checkFloat3Frame(makeFklVfHf<N, false>(ctx));
        return;
    }
    progress("CPU + GPU timing");
    const auto ms = measureInterleaved(ctx, impls);
    // The outputs after the timed iterations must still be the validated ones (no library state, such as ROI tensors,
    // may change from one call to the next)
    for (size_t k = 0; k < impls.size(); ++k) {
        impls[k].fillNaN();
        impls[k].run();
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        const ErrorStats after = compare(impls[k].download(), ownReference(impls[k]));
        if (after.nonFinite != 0 || std::abs(after.max() - batchResults[k].errorVsOwnReference.max()) > 1e-6f) {
            throw std::runtime_error(impls[k].name + ": the output after the timed iterations differs from the validated "
                                     "one (max error " + std::to_string(after.max()) + ")");
        }
    }
    for (size_t k = 0; k < impls.size(); ++k) {
        Result& r = batchResults[k];
        r.timing = computeStats(ms[k]);
        if (k != bvf) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += ms[k][i] / ms[bvf][i];
            }
            r.meanSpeedupOfBvf = sum / TIMED_ITERS;
        }
        std::cout << std::left << std::setw(12) << r.name << std::right << std::fixed << std::setprecision(1)
                  << " median " << std::setw(7) << r.timing.median << " us  mean " << std::setw(7) << r.timing.mean
                  << " us  p10 " << std::setw(7) << r.timing.p10 << "  p90 " << std::setw(7) << r.timing.p90;
        if (k != bvf) {
            std::cout << "  FKL BVF mean speedup " << std::setw(6) << std::setprecision(2) << r.meanSpeedupOfBvf << "x";
        }
        std::cout << "  max err " << std::scientific << std::setprecision(2) << r.errorVsOwnReference.max() << " ("
                  << r.reference << ")" << std::defaultfloat << std::endl;
        results.push_back(r);
    }
    // Per operation breakdown of the multi-operation pipelines, and of VF + HF with a float3 frame for reference
    for (const auto& impl : impls) {
        if (!impl.stages.empty()) {
            measureStages(ctx, impl, static_cast<int>(N));
        }
    }
    const auto float3Frame = makeFklVfHf<N, false>(ctx);
    checkFloat3Frame(float3Frame);
    measureStages(ctx, float3Frame, static_cast<int>(N));
}

// ------------------------------------------------------------------------------------------------
// Selection of the RPP pipeline: every RPP variant is validated against the RPP-convention reference and timed (CPU +
// GPU, interleaved, SELECTION_WARMUP + SELECTION_ITERS iterations per N); the numerically equivalent variant with the
// lowest geometric mean of its medians over all N is the benchmark's RPP pipeline, the only one in the measurements
// ------------------------------------------------------------------------------------------------
constexpr int SELECTION_WARMUP = 20;
constexpr int SELECTION_ITERS = 200;

struct SelectionRow {
    int batch{0};
    std::string variant;
    ErrorStats error;
    TimingStats timing;
};
std::vector<SelectionRow> selectionRows;

template <size_t N>
void selectionBatch(Context& ctx) {
    const auto ref = rppConventionReference8bit<N>(ctx);
    const auto variants = makeRppVariants<N>(ctx);
    std::vector<ErrorStats> errors;
    for (const auto& v : variants) {
        v.fillNaN();
        v.run();
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        errors.push_back(compare(v.download(), ref));
    }
    const auto ms = measureInterleaved(ctx, variants, SELECTION_WARMUP, SELECTION_ITERS);
    std::cout << "RPP selection N = " << N << ":";
    for (size_t k = 0; k < variants.size(); ++k) {
        selectionRows.push_back({static_cast<int>(N), variants[k].name, errors[k], computeStats(ms[k])});
        std::cout << "  " << variants[k].name << " " << std::fixed << std::setprecision(1)
                  << selectionRows.back().timing.median << " us" << std::defaultfloat;
    }
    std::cout << std::endl;
}

template <size_t... Is>
void selectionAllBatches(Context& ctx, std::index_sequence<Is...>) {
    (selectionBatch<BATCHES[Is]>(ctx), ...);
}

std::string chooseRpp() {
    std::string best;
    double bestScore = INFINITY;
    for (const auto& v : RPP_VARIANTS) {
        double score = 0.;
        int rows = 0;
        bool equivalent = true;
        for (const auto& r : selectionRows) {
            if (r.variant != v) {
                continue;
            }
            equivalent = equivalent && r.error.nonFinite == 0 && r.error.max() <= RPP_EQUIVALENCE;
            score += std::log(r.timing.median);
            ++rows;
        }
        if (rows > 0 && equivalent && score < bestScore) {
            bestScore = score;
            best = v;
        }
    }
    if (best.empty()) {
        throw std::runtime_error("No RPP variant is numerically equivalent to the RPP-convention reference");
    }
    std::cout << "=== RPP pipeline of the benchmark: " << best << " (fastest numerically equivalent RPP variant)" << std::endl;
    return best;
}

void writeSelection(const std::filesystem::path& path) {
    std::ofstream os(path);
    os << "N,variant,selection_median_us,selection_p10_us,selection_p90_us,selection_mean_us,"
          "max_abs_err_vs_rpp_sampling_8bit_reference,numerically_equivalent,chosen,selection_iterations\n";
    for (const auto& r : selectionRows) {
        os << r.batch << "," << r.variant << "," << std::fixed
           << std::setprecision(2) << r.timing.median << "," << r.timing.p10 << "," << r.timing.p90 << ","
           << r.timing.mean << "," << std::scientific << std::setprecision(3) << r.error.max() << std::defaultfloat
           << "," << (r.error.nonFinite == 0 && r.error.max() <= RPP_EQUIVALENCE ? "yes" : "no") << ","
           << (r.variant == selectedRpp ? "yes" : "no") << "," << SELECTION_ITERS << "\n";
    }
}

// Kernel launches, memsets and memcpys of every GPU operation of the multi-operation pipelines (profiling child only)
struct OperationCount {
    std::string implementation;
    int batch{0};
    std::string operation;
    LaunchCounts counts;
};
std::vector<OperationCount> operationCounts;

void writeOperationCounts(const std::filesystem::path& path) {
    std::ofstream os(path);
    os << "implementation,N,operation,kernels,memsets,memcpys\n";
    for (const auto& o : operationCounts) {
        os << o.implementation << "," << o.batch << "," << o.operation
           << "," << o.counts.kernels << "," << o.counts.memsets << "," << o.counts.memcpys << "\n";
    }
}

// Kernel counts of every implementation for batch N, after all the timings
template <size_t N>
void countBatch(Context& ctx, std::vector<Result>& results) {
    for (const auto& impl : makeImplementations<N>(ctx)) {
        impl.run(); // warm-up, so that one-time work (code object loading) is not counted
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        const LaunchCounts launches = countLaunches(ctx, impl);
        auto r = std::find_if(results.begin(), results.end(), [&](const Result& x) {
            return x.name == impl.name && x.batch == static_cast<int>(N);
        });
        if (r == results.end()) {
            Result fresh;
            fresh.name = impl.name;
            fresh.batch = static_cast<int>(N);
            results.push_back(fresh);
            r = results.end() - 1;
        }
        r->launches = launches;
        // Per operation: which RPP call (or FKL kernel) launches what, including RPP's own copies inside its calls
        for (const auto& stage : impl.stages) {
            rocprof::counts = LaunchCounts{};
            ROCP_CHECK(rocprofiler_start_context(rocprof::apiContext));
            stage.second();
            gpuErrchk(hipStreamSynchronize(ctx.stream));
            ROCP_CHECK(rocprofiler_stop_context(rocprof::apiContext));
            operationCounts.push_back({impl.name, static_cast<int>(N), stage.first, rocprof::counts});
        }
        std::cout << std::left << std::setw(10) << impl.name << " N " << std::right << std::setw(3) << N
                  << "  kernels " << std::setw(4) << launches.kernels << "  memsets " << launches.memsets
                  << "  memcpys " << launches.memcpys << std::endl;
    }
}

// GPU time only benchmark: the same interleaved iterations as the CPU + GPU one, with kernel dispatch tracing on.
// The GPU time of one execution is the sum of the execution times of its kernels: no host work and no launch gaps.
// Only where kernelTracingAvailable(); never on WSL2.
template <size_t N>
void gpuOnlyBatch(Context& ctx, std::vector<Result>& results, CacheFlusher* flusher = nullptr) {
    const auto impls = makeImplementations<N>(ctx);
    const size_t bvf = impls.size() - 1;
    std::vector<size_t> kernelsPerRun(impls.size());
    for (size_t k = 0; k < impls.size(); ++k) {
        impls[k].run();
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        kernelsPerRun[k] = countLaunches(ctx, impls[k]).kernels;
    }
    activeFlusher = flusher;
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        for (const auto& impl : impls) {
            impl.run();
            gpuErrchk(hipStreamSynchronize(ctx.stream));
        }
    }
    rocprof::flushKernels();
    {
        std::lock_guard<std::mutex> lock(rocprof::mutex);
        rocprof::kernels.clear();
    }
    // Every execution runs inside a window of rocprofiler timestamps (the clock of the dispatch records); a record
    // belongs to the execution whose window contains its start. The executions are sequential (synchronized), so the
    // windows do not overlap.
    struct Window {
        rocprofiler_timestamp_t begin, end;
        size_t k;
        int i;
    };
    std::vector<Window> windows;
    windows.reserve(static_cast<size_t>(TIMED_ITERS) * impls.size());
    ROCP_CHECK(rocprofiler_start_context(rocprof::kernelContext));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        for (size_t k = 0; k < impls.size(); ++k) {
            Window w{0, 0, k, i};
            ROCP_CHECK(rocprofiler_get_timestamp(&w.begin));
            impls[k].run();
            gpuErrchk(hipStreamSynchronize(ctx.stream));
            ROCP_CHECK(rocprofiler_get_timestamp(&w.end));
            windows.push_back(w);
        }
        if (i % 100 == 99) {
            rocprof::flushKernels();
        }
    }
    ROCP_CHECK(rocprofiler_stop_context(rocprof::kernelContext));
    rocprof::flushKernels();
    activeFlusher = nullptr;
    std::vector<KernelRecord> records;
    std::vector<bool> blit;
    {
        std::lock_guard<std::mutex> lock(rocprof::mutex);
        for (const auto& r : rocprof::kernels) {
            if (!rocprof::isFlushKernel(r.kernelId)) {
                records.push_back(r);
            }
        }
        rocprof::kernels.clear();
        for (const auto& r : records) {
            blit.push_back(rocprof::isRuntimeBlitKernel(r.kernelId));
        }
    }
    // The GPU time of an execution is the sum of the execution times of its kernels; the HIP runtime's blit kernels
    // (memsets, copies) are summed separately, like CUPTI's kernel records in bvf_bench.cu, which have no memsets
    std::vector<std::vector<float>> ms(impls.size(), std::vector<float>(TIMED_ITERS, 0.f));
    std::vector<std::vector<float>> blitMs(impls.size(), std::vector<float>(TIMED_ITERS, 0.f));
    std::vector<std::vector<float>> blitCount(impls.size(), std::vector<float>(TIMED_ITERS, 0.f));
    std::vector<std::vector<size_t>> kernelCount(impls.size(), std::vector<size_t>(TIMED_ITERS, 0));
    size_t unassigned = 0;
    for (size_t j = 0; j < records.size(); ++j) {
        const auto& r = records[j];
        const auto w = std::upper_bound(windows.begin(), windows.end(), r.start,
                                        [](rocprofiler_timestamp_t t, const Window& x) { return t < x.begin; });
        if (w == windows.begin() || r.start > std::prev(w)->end) {
            unassigned += blit[j] ? 0 : 1;
            continue;
        }
        const Window& x = *std::prev(w);
        const float dur = static_cast<float>((r.end - r.start) * 1e-6);
        if (blit[j]) {
            blitMs[x.k][x.i] += dur;
            blitCount[x.k][x.i] += 1.f;
        } else {
            ms[x.k][x.i] += dur;
            kernelCount[x.k][x.i]++;
        }
    }
    for (size_t k = 0; k < impls.size(); ++k) {
        for (int i = 0; i < TIMED_ITERS; ++i) {
            if (kernelCount[k][i] != kernelsPerRun[k]) {
                throw std::runtime_error("GPU only benchmark: " + impls[k].name + " execution " + std::to_string(i) +
                                         " has " + std::to_string(kernelCount[k][i]) + " kernel records, expected " +
                                         std::to_string(kernelsPerRun[k]) + " (" + std::to_string(unassigned) +
                                         " kernel records outside every execution)");
            }
        }
    }
    for (size_t k = 0; k < impls.size(); ++k) {
        auto r = std::find_if(results.begin(), results.end(), [&](const Result& x) {
            return x.name == impls[k].name && x.batch == static_cast<int>(N);
        });
        if (r == results.end()) {
            Result fresh;
            fresh.name = impls[k].name;
            fresh.batch = static_cast<int>(N);
            fresh.launches.kernels = kernelsPerRun[k];
            results.push_back(fresh);
            r = results.end() - 1;
        }
        TimingStats& timing = flusher ? r->gpuFlushedTiming : r->gpuTiming;
        double& speedup = flusher ? r->meanGpuFlushedSpeedupOfBvf : r->meanGpuSpeedupOfBvf;
        (flusher ? r->gpuFlushedMeasured : r->gpuMeasured) = true;
        timing = computeStats(ms[k]);
        (flusher ? r->gpuFlushedBlitTiming : r->gpuBlitTiming) = computeStats(blitMs[k]);
        std::vector<float> counts = blitCount[k];
        std::sort(counts.begin(), counts.end());
        (flusher ? r->gpuFlushedBlitDispatches : r->gpuBlitDispatches) = percentile(counts, 0.5);
        if (k != bvf) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += ms[k][i] / ms[bvf][i];
            }
            speedup = sum / TIMED_ITERS;
        }
        std::cout << (flusher ? "GPU only, L2 + MALL flushed " : "GPU only ") << std::left << std::setw(12) << r->name
                  << std::right << " N " << std::setw(3) << N << std::fixed << std::setprecision(2) << "  median "
                  << std::setw(7) << timing.median << " us  p10 " << std::setw(7) << timing.p10 << "  p90 "
                  << std::setw(7) << timing.p90;
        if (k != bvf) {
            std::cout << "  FKL BVF mean speedup " << std::setw(6) << speedup << "x";
        }
        std::cout << std::defaultfloat << std::endl;
    }
}

std::string shell(const std::string& command) {
    std::string output;
    if (FILE* pipe = popen(command.c_str(), "r")) {
        char buffer[512];
        while (fgets(buffer, sizeof(buffer), pipe)) {
            output += buffer;
        }
        pclose(pipe);
    }
    while (!output.empty() && (output.back() == '\n' || output.back() == ' ')) {
        output.pop_back();
    }
    return output;
}

bool onWsl() {
    static const bool wsl = shell("grep -ci microsoft /proc/version 2>/dev/null") != "0";
    return wsl;
}

// L3 (MALL, Infinity Cache) size of the GPU in KiB: hipDeviceProp_t has no field for it. From BENCH_MALL_KIB if set,
// else from rocminfo, else from amd-smi
struct MallSize {
    size_t kib{0};
    std::string source;
};

MallSize mallSize() {
    static const MallSize mall = []() {
        if (const char* overrideKib = std::getenv("BENCH_MALL_KIB")) {
            std::cerr << "NOTE: MALL size taken from BENCH_MALL_KIB = " << overrideKib << " KiB" << std::endl;
            return MallSize{std::stoul(overrideKib), "BENCH_MALL_KIB override"};
        }
        std::string kib = shell("rocminfo 2>/dev/null | awk '/Device Type: *GPU/{g=1} g && /L3:/{print $2; exit}'");
        if (!kib.empty() && std::isdigit(static_cast<unsigned char>(kib[0]))) {
            return MallSize{std::stoul(kib), "rocminfo"};
        }
        kib = shell("amd-smi static --gpu 0 --cache 2>/dev/null | awk '/CACHE_ACRONYM: *L3/{f=1} f && /CACHE_SIZE:/{print $2; exit}'");
        if (!kib.empty() && std::isdigit(static_cast<unsigned char>(kib[0]))) {
            return MallSize{std::stoul(kib), "amd-smi"};
        }
        std::cerr << "\n*** WARNING: could not detect the MALL (Infinity Cache) size with rocminfo or amd-smi. Set "
                     "BENCH_MALL_KIB=<size in KiB> (for example 32768 for 32 MiB). The flushed GPU only benchmark "
                     "refuses to run without it. ***\n" << std::endl;
        return MallSize{0, "NOT DETECTED"};
    }();
    return mall;
}

// Floor of the CPU + GPU metric: the same hipEvent bracket and synchronization around nothing, around one empty
// kernel and around four, measured at the start of every run
__global__ void emptyKernel() {}

constexpr int FLOOR_WARMUP = 100;
constexpr int FLOOR_ITERS = 1000;

struct EventFloor {
    TimingStats empty, oneKernel, fourKernels;
};
EventFloor eventFloor;

void measureEventFloor(const Context& ctx) {
    hipEvent_t start, stop;
    gpuErrchk(hipEventCreate(&start));
    gpuErrchk(hipEventCreate(&stop));
    const auto measure = [&](int kernels) {
        std::vector<float> ms(FLOOR_ITERS);
        for (int i = 0; i < FLOOR_WARMUP + FLOOR_ITERS; ++i) {
            gpuErrchk(hipEventRecord(start, ctx.stream));
            for (int k = 0; k < kernels; ++k) {
                emptyKernel<<<1, 64, 0, ctx.stream>>>();
            }
            gpuErrchk(hipEventRecord(stop, ctx.stream));
            gpuErrchk(hipEventSynchronize(stop));
            if (i >= FLOOR_WARMUP) {
                gpuErrchk(hipEventElapsedTime(&ms[i - FLOOR_WARMUP], start, stop));
            }
        }
        return computeStats(ms);
    };
    eventFloor = {measure(0), measure(1), measure(4)};
    gpuErrchk(hipEventDestroy(start));
    gpuErrchk(hipEventDestroy(stop));
    std::cout << "Floor of the CPU + GPU metric: empty hipEvent bracket " << std::fixed << std::setprecision(2)
              << eventFloor.empty.median << " us, one empty kernel " << eventFloor.oneKernel.median
              << " us, four empty kernels " << eventFloor.fourKernels.median << " us (medians)" << std::defaultfloat
              << std::endl;
}

CacheFlusher makeCacheFlusher(const Context& ctx) {
    int device = 0;
    gpuErrchk(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    gpuErrchk(hipGetDeviceProperties(&prop, device));
    CacheFlusher flusher;
    const size_t l2 = static_cast<size_t>(prop.l2CacheSize);
    const MallSize mallInfo = mallSize();
    if (mallInfo.kib == 0) {
        throw std::runtime_error("MALL (Infinity Cache) size not detected: set BENCH_MALL_KIB=<KiB> to run the flushed "
                                 "GPU only benchmark");
    }
    const size_t mall = mallInfo.kib * 1024;
    const size_t bytes = 2 * (l2 + mall);
    flusher.count = bytes / sizeof(int4);
    gpuErrchk(hipMalloc(reinterpret_cast<void**>(&flusher.buffer), bytes));
    gpuErrchk(hipMemsetAsync(flusher.buffer, 0, bytes, ctx.stream));
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    gpuErrchk(hipMalloc(reinterpret_cast<void**>(&flusher.sink), sizeof(int)));
    flusher.stream = ctx.stream;
    flusher.blocks = prop.multiProcessorCount * 8;
    std::cout << "Flushing " << bytes / (1 << 20) << " MiB (2 x (L2 " << l2 / 1024 << " KiB + MALL " << mall / 1024
              << " KiB)) before every GPU operation" << std::endl;
    return flusher;
}

void freeCacheFlusher(CacheFlusher& flusher) {
    gpuErrchk(hipFree(flusher.buffer));
    gpuErrchk(hipFree(flusher.sink));
}

// Compares RPP's 8 bit NV12->RGB (rppt_yuv_to_rgb, BT.2020 full range) with FKL's float conversion, to show how RPP
// rounds and samples chroma
// rppt_yuv_to_rgb's 8 bit RGB frame, downloaded once (the input of the RPP-convention reference)
void computeRppRgb8(Context& ctx) {
    auto st = makeRppState(ctx, 1);
    rppYuvToRgb(ctx, *st);
    ctx.rppRgb8.resize(static_cast<size_t>(FRAME_W) * FRAME_H * 3);
    gpuErrchk(hipMemcpyAsync(ctx.rppRgb8.data(), st->rgb8, ctx.rppRgb8.size(), hipMemcpyDeviceToHost, ctx.stream));
    gpuErrchk(hipStreamSynchronize(ctx.stream));
}

void conversionDiagnostic(const Context& ctx, std::ostream& os) {
    const std::vector<uint8_t>& host = ctx.rppRgb8;
    size_t diffRound = 0, diffTrunc = 0, total = 0;
    int maxRound = 0, maxTrunc = 0;
    for (int y = 0; y < FRAME_H; ++y) {
        const uint8_t* row = host.data() + static_cast<size_t>(y) * FRAME_W * 3;
        for (int x = 0; x < FRAME_W; ++x) {
            const fk::uchar3 yuv = ctx.frameHost->readAt(fk::Point{x, y, 0});
            const fk::float3 rgb = YUVToRGB::exec(yuv);
            const float channels[3] = {rgb.x, rgb.y, rgb.z};
            for (int c = 0; c < 3; ++c) {
                const float clamped = std::min(255.f, std::max(0.f, channels[c]));
                const int rounded = static_cast<int>(std::floor(clamped + 0.5f));
                const int truncated = static_cast<int>(clamped);
                const int rpp = row[3 * x + c];
                diffRound += rpp != rounded;
                diffTrunc += rpp != truncated;
                maxRound = std::max(maxRound, std::abs(rpp - rounded));
                maxTrunc = std::max(maxTrunc, std::abs(rpp - truncated));
                total++;
            }
        }
    }
    os << "RPP NV12->RGB (rppt_yuv_to_rgb, RpptColorStandard_BT2020_NCL, RpptColorRange_FULL) vs FKL float conversion:\n"
       << "  vs round-to-nearest of FKL: " << diffRound << " of " << total << " channel values differ, max "
       << maxRound << " levels\n"
       << "  vs truncation of FKL:       " << diffTrunc << " of " << total << " channel values differ, max "
       << maxTrunc << " levels\n";
}

// RPP's bilinear sampling convention, measured as the paper did for NPP: resize linear ramps (value = position, exact)
// with rppt_resize F32 and read back the sample position of every output pixel
void resizeConvention(Context& ctx, const std::filesystem::path& outDir) {
    std::ofstream csv(outDir / "rpp_resize_convention.csv");
    csv << "axis,src,dst,scale,dst_index,measured_src_position,s_times_p,s_times_p_minus_0.25,"
           "pixel_center_s_times_p_plus_half_minus_half,pixel_center_clamped_to_0_size_minus_2\n";
    rppHandle_t handle{};
    RPP_CHECK(static_cast<RppStatus>(rppCreate(&handle, 1, 0, ctx.stream, RppBackend::RPP_HIP_BACKEND)));
    struct Case { bool vertical; int src, dst; };
    // The benchmark's crop -> resize in both axes, then other ratios
    const std::array<Case, 6> cases{{{false, CROP_W, RES_W}, {true, CROP_H, RES_H}, {false, 20, 64}, {false, 10, 32},
                                     {false, 8, 40}, {true, 20, 64}}};
    std::cout << "=== RPP rppt_resize (F32, BILINEAR) sampling convention from linear ramps" << std::endl;
    for (const auto& c : cases) {
        const int sw = c.vertical ? 8 : c.src, sh = c.vertical ? c.src : 2;
        const int dw = c.vertical ? 8 : c.dst, dh = c.vertical ? c.dst : 2;
        std::vector<float> in(static_cast<size_t>(sw) * sh * 3);
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < sw; ++x)
                for (int ch = 0; ch < 3; ++ch) in[(y * sw + x) * 3 + ch] = static_cast<float>(c.vertical ? y : x);
        float* dIn = deviceAlloc<float>(in.size() * sizeof(float) + RPP_SLACK_BYTES);
        float* dOut = deviceAlloc<float>(static_cast<size_t>(dw) * dh * 3 * sizeof(float) + RPP_SLACK_BYTES);
        gpuErrchk(hipMemcpyAsync(dIn, in.data(), in.size() * sizeof(float), hipMemcpyHostToDevice, ctx.stream));
        RpptDesc s(1, 3, sh, sw, RpptDataType::F32, RpptLayout::NHWC), d(1, 3, dh, dw, RpptDataType::F32, RpptLayout::NHWC);
        RpptROI* roi = deviceAlloc<RpptROI>(sizeof(RpptROI));
        RpptROI hostRoi;
        hostRoi.ltrbROI = {{0, 0}, {sw - 1, sh - 1}};
        gpuErrchk(hipMemcpyAsync(roi, &hostRoi, sizeof(RpptROI), hipMemcpyHostToDevice, ctx.stream));
        RpptImagePatch* sizes = hostAlloc<RpptImagePatch>(1);
        sizes->width = dw;
        sizes->height = dh;
        RPP_CHECK(rppt_resize(dIn, &s, dOut, &d, sizes, RpptInterpolationType::BILINEAR, roi, RpptRoiType::LTRB, handle,
                              RppBackend::RPP_HIP_BACKEND));
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        std::vector<float> out(static_cast<size_t>(dw) * dh * 3);
        gpuErrchk(hipMemcpyAsync(out.data(), dOut, out.size() * sizeof(float), hipMemcpyDeviceToHost, ctx.stream));
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        const float scale = static_cast<float>(c.src) / c.dst;
        double maxDevCenterClamped = 0.;
        for (int p = 0; p < c.dst; ++p) {
            const float measured = c.vertical ? out[(p * dw + 1) * 3] : out[(1 * dw + p) * 3];
            const float center = scale * (p + 0.5f) - 0.5f;
            const float centerClamped = std::min(static_cast<float>(c.src - 2), std::max(0.f, center));
            maxDevCenterClamped = std::max(maxDevCenterClamped, static_cast<double>(std::abs(measured - centerClamped)));
            csv << (c.vertical ? "y" : "x") << "," << c.src << "," << c.dst << "," << scale << "," << p << "," << measured
                << "," << scale * p << "," << scale * p - 0.25f << "," << center << "," << centerClamped << "\n";
        }
        std::cout << (c.vertical ? "vertical  " : "horizontal") << " " << std::setw(2) << c.src << " -> " << std::setw(2)
                  << c.dst << ": max |measured - clamp(s*(p+0.5)-0.5, 0, size-2)| = " << std::scientific
                  << std::setprecision(2) << maxDevCenterClamped << std::defaultfloat << std::endl;
        gpuErrchk(hipFree(dIn));
        gpuErrchk(hipFree(dOut));
        gpuErrchk(hipFree(roi));
        gpuErrchk(hipHostFree(sizes));
    }
    rppDestroy(handle, RppBackend::RPP_HIP_BACKEND);
}

// System, versions, workload and protocol of a run
void writePlatform(const std::filesystem::path& path, const std::string& diagnostic, bool kernelTracing,
                   const std::string& profilingError = "") {
    int device = 0, runtime = 0, driver = 0;
    gpuErrchk(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    gpuErrchk(hipGetDeviceProperties(&prop, device));
    gpuErrchk(hipRuntimeGetVersion(&runtime));
    gpuErrchk(hipDriverGetVersion(&driver));
    std::ofstream os(path);
    const MallSize mall = mallSize();
    const size_t l2Bytes = static_cast<size_t>(prop.l2CacheSize), mallBytes = mall.kib * 1024;
    const std::string driverVersion =
        shell("amd-smi static --gpu 0 --driver 2>/dev/null | sed -n 's/^ *VERSION: *//p' | head -1");
    const std::string osName = shell(". /etc/os-release && echo $PRETTY_NAME");
    const std::string rocmVersion = shell(std::string("cat ") + BENCH_ROCM_PATH + "/.info/version 2>/dev/null");
    // HIP reports the work-group processors of RDNA GPUs as multiProcessorCount; rocminfo reports the compute units
    const std::string rocminfoCUs =
        shell("rocminfo 2>/dev/null | awk '/Device Type: *GPU/{g=1} g && /Compute Unit:/{print $3; exit}'");
    const std::string cus = rocminfoCUs.empty() ? std::to_string(prop.multiProcessorCount) : rocminfoCUs;
    const std::string perfLevel =
        shell("(rocm-smi --showperflevel 2>/dev/null | grep -i 'performance level' | head -1 | sed 's/^.*: *//') || true");
    os << "Platform: " << prop.name << ", " << cus << " CUs, "
       << (prop.totalGlobalMem + (1ull << 29)) / (1ull << 30) << " GB, " << prop.gcnArchName << ", driver "
       << (driverVersion.empty() ? "unknown" : driverVersion) << " under " << (onWsl() ? "WSL2 " : "bare metal ")
       << osName << ", ROCm " << rocmVersion << ", RPP " << RPP_VERSION_MAJOR << "." << RPP_VERSION_MINOR << "."
       << RPP_VERSION_PATCH << "\n"
       << "GPU: " << prop.name << " (" << prop.gcnArchName << ", " << cus << " compute units (rocminfo; "
       << "hipDeviceProp_t::multiProcessorCount " << prop.multiProcessorCount << "), "
       << (prop.totalGlobalMem >> 20) << " MiB VRAM, max clock " << prop.clockRate / 1000 << " MHz)\n"
       << "Caches: L2 " << prop.l2CacheSize / 1024 << " KiB (hipDeviceProp_t::l2CacheSize), L3 / MALL (Infinity Cache) "
       << mall.kib << " KiB (" << mall.source << ")\n"
       << "Environment: " << (onWsl() ? "WSL2 (GPU through /dev/dxg)" : "bare metal") << "\n"
       << "Kernel driver: " << (driverVersion.empty() ? "unknown (amd-smi not available)" : driverVersion) << " ("
       << shell("uname -r") << ")\n"
       << "ROCm: " << rocmVersion << " (" << BENCH_ROCM_PATH << "), HIP runtime " << runtime << ", HIP driver " << driver
       << "\n"
       << "Compiler: " << __clang_version__ << "\n"
       << "HIP flags: " << BENCH_HIP_FLAGS << "\n"
       << "RPP: " << RPP_VERSION_MAJOR << "." << RPP_VERSION_MINOR << "." << RPP_VERSION_PATCH
       << " (ROCm 10.1 package amdrocm-rpp10.1, prebuilt; HIP backend; " << BENCH_RPP_LIB << ")\n"
       << "rocprofiler-sdk: " << BENCH_ROCPROFILER_LIB << "\n"
       << "FKL: " << BENCH_FKL_COMMIT
       << (std::string(BENCH_FKL_BRANCH) == "HEAD" ? std::string() : std::string(", branch ") + BENCH_FKL_BRANCH) << "\n"
       << "CPU: " << shell("lscpu | sed -n 's/^Model name: *//p'") << "\n"
       << "OS: " << osName << ", kernel " << shell("uname -r") << "\n"
       << "GPU clocks: not changed by the benchmark. rocm-smi performance level: "
       << (perfLevel.empty() ? std::string("not reported") : perfLevel) << "; amd-smi: " << shell("amd-smi metric --gpu 0 --clock 2>/dev/null | grep -m1 CLK_LOCKED | sed 's/^ *//'")
       << ", " << shell("amd-smi metric --gpu 0 --power 2>/dev/null | grep -m1 POWER_MANAGEMENT | sed 's/^ *//'")
       << " (empty fields: not reported on this system)\n"
       << "\nWorkload: " << describe() << ". NV12->RGB BT.2020 full range, AddBorder (black), then Mul "
       << "(1/255), Sub (0.485, 0.456, 0.406), Div (0.229, 0.224, 0.225), float3 tensor of N planes " << OUT_W << "x"
       << OUT_H << "\n";
    os << "Timing: " << WARMUP_ITERS << " warm-up then " << TIMED_ITERS << " timed iterations. In every iteration "
       << "the RPP pipeline (rpp), FKL VF+HF and FKL BVF run one after the other, each timed with HIP events on one "
       << "stream "
       << "around the complete pipeline (all host enqueue work included) and followed by a synchronization. "
       << "FKL BVF's speedup over each baseline is computed per iteration (baseline time / BVF time) and averaged "
       << "(mean speedup, FKL's two-execution convention). Median, p10, p90 with linear interpolation.\n"
       << "RPP pipeline: rpp = " << selectedRpp << ", the fastest of the five RPP variants below, which are "
       << "numerically equivalent (within " << RPP_EQUIVALENCE << " of the RPP-convention reference): a selection pass "
       << "validates and times every variant (CPU + GPU, interleaved, " << SELECTION_WARMUP << " warm-up + "
       << SELECTION_ITERS << " timed iterations per N) and picks the one with the lowest geometric mean "
       << "of its medians (rpp_selection.csv); only that one is measured afterwards.\n"
       << "RPP variants (HIP backend): rpp_per_op = rppt_yuv_to_rgb (BT2020_NCL, FULL; 1 kernel plus 8 hipMemcpyToSymbol per "
       << "call: its matrix and luma bias, and 6 integer constants used only by its cubic_v variant, uploaded on every "
       << "call), rppt_crop_mirror_normalize 8u->32f (one batched call, all batch images "
       << "aliasing the RGB frame with a zero batch stride, XYWH ROIs, offset 0, multiplier 1, no mirror), "
       << "hipMemsetAsync of the output tensor (RPP has no constant border), rppt_resize F32 bilinear (one batched call "
       << "writing each resized crop into rows " << BORDER_TOP << " to " << BORDER_TOP << " + resize height - 1 of its "
       << "output plane), rppt_multiply_scalar (the tensor as one NDHWC volume: RPP's scalar primitives launch one "
       << "kernel per batch image), rppt_tensor_subtract_tensor and rppt_tensor_divide_tensor (broadcast of one row of "
       << "the per channel constants over the rows of the tensor; each of these two calls also does 6 small "
       << "hipMemcpyAsync host-to-device copies of its broadcast metadata). rpp_npp_like = the same calls, but rppt_resize "
       << "writes the resized crops into their own float3 buffer and one batched rppt_crop F32->F32 (whole resized crop "
       << "ROIs) copies them into the output rows after the memset (the counterpart of NPP's CopyConstBorder). "
       << "rpp_fused = rppt_yuv_to_rgb, hipMemsetAsync of the top border rows of the first output plane, "
       << "rppt_resize_mirror_normalize 8u->32f (one batched call on the RGB frame with the crop ROIs: crop + convert + "
       << "resize + Mul, mean 0, stdDev 255, into the output rows; RPP's 8u->32f path computes (x - mean)/stdDev on the "
       << "0..255 values, so the whole tail would be mean 255*sub, stdDev 255*div, but the border rows, which "
       << "rppt_resize_mirror_normalize does not write, must hold the normalized black (0 - sub)/div), then "
       << "rppt_tensor_subtract_tensor and rppt_tensor_divide_tensor on the whole tensor. rpp_normalize = rpp_per_op with "
       << "Mul + Sub + Div in one rppt_normalize on the whole tensor (rows of outW float3, per column mean 255*sub and "
       << "stdDev 255*div: fmaf(x - mean, 1/stdDev, 0)). rpp_fused_normalize = rppt_yuv_to_rgb, the first-plane border "
       << "memset, rppt_resize_mirror_normalize (crop + convert + resize, mean 0, stdDev 1) and rppt_normalize (Mul + "
       << "Sub + Div): the fewest RPP calls. rppt_resize_mirror_normalize itself zeroes n*nStride floats "
       << "from its destination pointer (which starts " << BORDER_TOP << " rows into the tensor, so the tensor is "
       << "allocated with " << BORDER_TOP << " rows of slack after its last plane, and only the first plane's top "
       << "border rows are left to the memset) and then calls hipStreamSynchronize inside the call (library behaviour, "
       << "kept).\n"
       << "Outside the timed region (once, like NPP's batch descriptors in bvf_bench.cu): rppCreate (one handle per "
       << "pipeline with its batch size N, on the timing stream: RPP sizes its grids with the handle's batch size), all "
       << "buffers, the RPP descriptors, the ROI tensors (device memory; XYWH for rppt_crop and "
       << "rppt_crop_mirror_normalize, LTRB for rppt_resize and rppt_resize_mirror_normalize, the formats those calls "
       << "work in: given the other one, RPP converts the caller's ROI tensor in place with an extra kernel), "
       << "the destination sizes, normalization and scalar parameter tensors (pinned host memory) and the rows of "
       << "per channel constants (device memory). Inside the timed region, because a user cannot hoist them: RPP's own "
       << "copies inside its calls (the 8 hipMemcpyToSymbol of every rppt_yuv_to_rgb and the 6 host-to-device "
       << "hipMemcpyAsync of every rppt_tensor_subtract_tensor / rppt_tensor_divide_tensor, the zeroing memset and the "
       << "stream synchronization inside rppt_resize_mirror_normalize) and the hipMemsetAsync of the output tensor. RPP's "
       << "HIP kernels load and store 8 pixels per thread and check only the first one against the ROI width, so RPP "
       << "buffer rows are padded to a multiple of " << RPP_PIXEL_ALIGN << " pixels: the " << CROP_W << " pixel crop "
       << "rows have a pitch of " << alignUp(CROP_W, RPP_PIXEL_ALIGN) << " pixels"
       << (CROP_W % RPP_PIXEL_ALIGN == 0 ? " (no padding)" : "") << ", and the output rows (" << OUT_W << " pixels, "
       << "contiguous like FKL's) are a multiple of 8 themselves.\n"
       << "Intermediates: rpp_npp_like and FKL VF+HF write an 8 bit RGB frame, float3 crops, float3 resized crops and "
       << "the float3 output (the paper's NPP set). rpp_per_op and rpp_normalize write the 8 bit RGB frame and the float3 "
       << "crops, then the resized crops directly into the output tensor (after zeroing it), and then multiply, "
       << "subtract and divide it in place (rpp_normalize in one pass). rpp_fused and rpp_fused_normalize write the 8 "
       << "bit RGB frame and the output. FKL BVF has no intermediates in memory: float3 from the color conversion to "
       << "the write.\n"
       << "RPP-convention reference: RPP's own 8 bit RGB frame (rppt_yuv_to_rgb's output, downloaded once), RPP's "
       << "bilinear sampling (pixel centers clamped to [0, size - 2], measured with linear ramps: --resize-convention), "
       << "then the border and Mul, Sub, Div. Every RPP pipeline is also compared with the paper's 8 bit reference "
       << "(FKL's conversion rounded to 8 bit, FKL's sampling: max_abs_err_vs_8bit_reference_fkl_sampling)\n"
       << "Kernel counts: rocprofiler-sdk HIP runtime API tracing of one pipeline (kernel launch calls, hipMemset* "
       << "calls and hipMemcpy* calls, in separate columns) of one execution after a warm-up execution, in a child "
       << "process started after all timings: the rocprofiler-sdk tool is only registered in that child, so no profiler "
       << "wraps the HIP API during the timed runs\n"
       << "Stream: one blocking HIP stream for everything (FKL, every RPP handle, memsets, copies, events, flush "
       << "kernels); RPP's hipMemcpyToSymbol calls, which go to the null stream, serialize with it\n";
    if (!profilingError.empty()) {
        os << "PROFILING PASS FAILED (" << profilingError << "): kernel counts and GPU time only are not measured in "
           << "this run (the kernel counts in the CSV files are 0)\n";
    }
    os << "Floor of the CPU + GPU metric on this system (measured in this run, before the benchmark, "
       << FLOOR_WARMUP << " warm-up + " << FLOOR_ITERS << " timed iterations of hipEventRecord(start) -> work -> "
       << "hipEventRecord(stop) -> hipEventSynchronize(stop) on the benchmark stream): empty bracket "
       << std::fixed << std::setprecision(2) << eventFloor.empty.median << " us median (p10 " << eventFloor.empty.p10
       << ", p90 " << eventFloor.empty.p90 << "); one empty kernel in the bracket " << eventFloor.oneKernel.median
       << " us median (p10 " << eventFloor.oneKernel.p10 << ", p90 " << eventFloor.oneKernel.p90 << "); four empty "
       << "kernels " << eventFloor.fourKernels.median << " us median" << std::defaultfloat << "\n"
       << "GPU time only: "
       << (kernelTracing ? "a second pass with the same interleaved iterations and rocprofiler-sdk kernel dispatch "
                           "tracing on (in the profiling child process); the GPU time of one execution is the sum of "
                           "the execution times of its kernels."
                         : onWsl() ? "NOT MEASURED: rocprofiler-sdk delivers no kernel dispatch records on WSL2 "
                                     "(wsl-dxcore agents, no /dev/kfd), and hipEvents attached to each kernel distort the "
                                     "execution (an empty kernel measured about 31 us instead of about 2 us)."
                                   : !profilingError.empty() ? "NOT MEASURED (the profiling pass failed)."
                                                             : "NOT MEASURED: rocprofiler-sdk delivered no kernel dispatch records.")
       << " Cache flush buffer for the flushed GPU only benchmark: 2 x (L2 " << l2Bytes / 1024 << " KiB + MALL "
       << mallBytes / 1024 << " KiB) = " << 2 * (l2Bytes + mallBytes) / (1 << 20) << " MiB"
       << (kernelTracing ? ", read by a kernel before every GPU operation of every pipeline (memsets and RPP calls "
                           "included; RPP's internal copies run inside its calls, after the flush of the call). The "
                           "output files keep the CUDA benchmark's names (l2_flushed), here meaning L2 and MALL. "
                           "Dispatches of the HIP runtime's own blit kernels (__amd_rocclr_*, which execute memsets and "
                           "some copies) are not in the GPU time only sums, like CUPTI's kernel records in the CUDA "
                           "benchmark; their GPU time and dispatches per execution are in the runtime_blit_* columns of "
                           "gpu_time.csv and gpu_time_l2_flushed.csv. Records are attributed to executions by windows of "
                           "rocprofiler timestamps around each execution."
                         : " (not run).")
       << "\n"
       << "Memory traffic (Experiment 2): rocm/scripts/rocprof_counters.sh (rocprofv3 --pmc), not part of this program"
       << (onWsl() ? "; on WSL2 rocprofv3 --pmc reports \"Agent HW architecture is not supported, no counter metrics "
                     "found\" and aborts with \"Cannot open /dev/kfd\"."
                   : ". What its counters measure on gfx12: GL2C_EA (FETCH_SIZE, GL2C_EA_WRREQ) counts the requests "
                     "between L2 and the data fabric, with the Infinity Cache (MALL) behind the fabric, so L2 misses and "
                     "write-backs, part of which the MALL may serve: not DRAM traffic; rocprofiler-sdk lists no MALL or "
                     "DRAM counter for gfx1200; GL2C_EA_WRREQ counts the write-backs while a kernel runs, so the lines "
                     "still dirty in L2 when each kernel ends (up to the L2 size per kernel) are not counted in any "
                     "dispatch, and multi-kernel pipelines lose more of their writes than fused ones. The counters only "
                     "count at a profiling performance level "
                     "(at \"auto\" the GL2C counters read 0 on gfx1200); details in the platform.txt of its output "
                     "directory.")
       << "\n\n" << diagnostic;
}

// FKL's two-execution CSV format (times in ms): one row per value of the first column
void writeTwoExecutionHeader(std::ostream& two, const std::string& firstColumn, const std::string& baseline) {
    two << firstColumn;
    for (const std::string label : {baseline, std::string("fkl_bvf")}) {
        two << ", " << label << " MeanTime, " << label << " TimeVariance, " << label << " MaxTime, " << label
            << " MinTime";
    }
    two << ", Mean Speedup";
    for (const std::string label : {baseline, std::string("fkl_bvf")}) {
        two << ", " << label << " MedianTime, " << label << " P10Time, " << label << " P90Time";
    }
    two << std::endl;
}

void writeTwoExecutionRow(std::ostream& two, long long key, const TimingStats& tb, const TimingStats& tf,
                          double speedup) {
    two << key;
    for (const TimingStats* t : std::array<const TimingStats*, 2>{&tb, &tf}) {
        two << ", " << t->mean / 1000. << ", " << t->variance / 1e6 << ", " << t->max / 1000. << ", " << t->min / 1000.;
    }
    two << ", " << speedup;
    for (const TimingStats* t : std::array<const TimingStats*, 2>{&tb, &tf}) {
        two << ", " << t->median / 1000. << ", " << t->p10 / 1000. << ", " << t->p90 / 1000.;
    }
    two << std::endl;
}

const std::array<std::string, 2> BASELINES{"rpp", "fkl_vf_hf"};

template <size_t... Is>
void runAllBatches(Context& ctx, std::vector<Result>& results, std::index_sequence<Is...>) {
    (runBatch<BATCHES[Is]>(ctx, results), ...);
}

template <size_t... Is>
void validateAllBatches(Context& ctx, std::vector<Result>& results, std::index_sequence<Is...>) {
    (runBatch<BATCHES[Is]>(ctx, results, false), ...);
}

template <size_t... Is>
void gpuOnlyAllBatches(Context& ctx, std::vector<Result>& results, std::index_sequence<Is...>,
                       CacheFlusher* flusher = nullptr) {
    (gpuOnlyBatch<BATCHES[Is]>(ctx, results, flusher), ...);
}

template <size_t... Is>
void countAllBatches(Context& ctx, std::vector<Result>& results, std::index_sequence<Is...>) {
    (countBatch<BATCHES[Is]>(ctx, results), ...);
}

// ------------------------------------------------------------------------------------------------
// Profiling pass in a child process (BENCH_ROCPROFILER=1, the only process where the rocprofiler-sdk tool is
// registered): kernel counts, then GPU time only where kernel dispatch tracing works. The child writes one row per
// pipeline, which the parent merges into its results.
// ------------------------------------------------------------------------------------------------
void writeStats(std::ostream& os, const TimingStats& t) {
    os << "," << t.mean << "," << t.variance << "," << t.max << "," << t.min << "," << t.median << "," << t.p10 << ","
       << t.p90;
}

void writeProfileRows(const std::filesystem::path& file, const std::vector<Result>& results, bool tracing) {
    std::ofstream os(file);
    os << std::setprecision(17) << "tracing," << (tracing ? 1 : 0) << "\n";
    for (const auto& r : results) {
        os << r.name << "," << r.batch << "," << r.launches.kernels << ","
           << r.launches.memsets << "," << r.launches.memcpys << "," << (r.gpuMeasured ? 1 : 0);
        writeStats(os, r.gpuTiming);
        os << "," << r.meanGpuSpeedupOfBvf << "," << (r.gpuFlushedMeasured ? 1 : 0);
        writeStats(os, r.gpuFlushedTiming);
        os << "," << r.meanGpuFlushedSpeedupOfBvf;
        writeStats(os, r.gpuBlitTiming);
        os << "," << r.gpuBlitDispatches;
        writeStats(os, r.gpuFlushedBlitTiming);
        os << "," << r.gpuFlushedBlitDispatches << "\n";
    }
}

bool mergeProfileRows(const std::filesystem::path& file, std::vector<Result>& results) {
    std::ifstream is(file);
    std::string line;
    bool tracing = false;
    while (std::getline(is, line)) {
        std::vector<std::string> f;
        std::stringstream ss(line);
        for (std::string field; std::getline(ss, field, ',');) {
            f.push_back(field);
        }
        if (f.size() == 2 && f[0] == "tracing") {
            tracing = f[1] == "1";
            continue;
        }
        if (f.size() != 39) {
            throw std::runtime_error("Malformed profiling pass row: " + line);
        }
        const auto r = std::find_if(results.begin(), results.end(), [&](const Result& x) {
            return x.name == f[0] && x.batch == std::stoi(f[1]);
        });
        if (r == results.end()) {
            throw std::runtime_error("Profiling pass row without a timed result: " + line);
        }
        const auto stats = [&](size_t at) {
            return TimingStats{std::stod(f[at]),     std::stod(f[at + 1]), std::stod(f[at + 2]), std::stod(f[at + 3]),
                               std::stod(f[at + 4]), std::stod(f[at + 5]), std::stod(f[at + 6])};
        };
        r->launches = {std::stoul(f[2]), std::stoul(f[3]), std::stoul(f[4])};
        r->gpuMeasured = f[5] == "1";
        r->gpuTiming = stats(6);
        r->meanGpuSpeedupOfBvf = std::stod(f[13]);
        r->gpuFlushedMeasured = f[14] == "1";
        r->gpuFlushedTiming = stats(15);
        r->meanGpuFlushedSpeedupOfBvf = std::stod(f[22]);
        r->gpuBlitTiming = stats(23);
        r->gpuBlitDispatches = std::stod(f[30]);
        r->gpuFlushedBlitTiming = stats(31);
        r->gpuFlushedBlitDispatches = std::stod(f[38]);
    }
    return tracing;
}

// Runs this executable again with the profiler registered, and merges its rows; returns whether kernel dispatch
// tracing (GPU time only) was available
bool runProfilingChild(const std::filesystem::path& outDir, std::vector<Result>& results) {
    const auto file = std::filesystem::absolute(outDir / ".profiling_pass.csv");
    const std::string exe = std::filesystem::read_symlink("/proc/self/exe").string();
    const std::string command = "BENCH_ROCPROFILER=1 BENCH_RPP_VARIANT='" + selectedRpp + "' '" + exe + "' '" +
                                outDir.string() + "' --profile-pass '" + file.string() + "'";
    std::cout << "=== Profiling pass in a child process: " << command << std::endl;
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("The profiling pass failed");
    }
    const bool tracing = mergeProfileRows(file, results);
    std::filesystem::remove(file);
    return tracing;
}

void profilePass(Context& ctx, const std::filesystem::path& outDir, const std::filesystem::path& file) {
    std::vector<Result> results;
    std::cout << "=== Kernel counts (HIP runtime API tracing)" << std::endl;
    countAllBatches(ctx, results, std::make_index_sequence<BATCHES.size()>{});
    writeOperationCounts(outDir / "kernel_counts_per_operation.csv");
    const bool tracing = kernelTracingAvailable(ctx);
    if (tracing) {
        std::cout << "=== GPU time only (kernel dispatch tracing)" << std::endl;
        gpuOnlyAllBatches(ctx, results, std::make_index_sequence<BATCHES.size()>{});
        CacheFlusher flusher = makeCacheFlusher(ctx);
        gpuOnlyAllBatches(ctx, results, std::make_index_sequence<BATCHES.size()>{}, &flusher);
        freeCacheFlusher(flusher);
    } else {
        std::cout << "=== GPU time only: NOT MEASURED (no kernel dispatch records from rocprofiler-sdk on this system)"
                  << std::endl;
    }
    writeProfileRows(file, results, tracing);
}

void writeValidationRow(std::ostream& validation, const Result& r) {
    const auto& e = r.errorVsOwnReference;
    validation << r.name << "," << r.batch << "," << r.reference << "," << std::scientific << std::setprecision(3)
               << e.maxPerChannel[0] << "," << e.maxPerChannel[1] << "," << e.maxPerChannel[2] << "," << e.mean << ","
               << e.nonFinite << "," << r.errorVsFloatReference.max() << "," << r.errorVsRppConventionReference.max()
               << "," << (e.max() <= TOLERANCE ? "yes" : "no") << "," << r.errorVs8bitReference.max() << std::defaultfloat
               << "\n";
}

// Target for hardware counter collection (rocm/scripts/rocprof_counters.sh, rocprofv3 --pmc): three warm-up executions
// of one pipeline, then measureMarkerKernel, then one execution with the caches flushed before every GPU operation;
// the script sums the counters of the kernels dispatched after the marker, without the flush kernels
__global__ void measureMarkerKernel() {}

template <size_t N>
void rocprofTarget(Context& ctx, const std::string& name) {
    std::vector<Implementation> impls;
    if (std::find(RPP_VARIANTS.begin(), RPP_VARIANTS.end(), name) != RPP_VARIANTS.end()) {
        impls.push_back(makeRppVariant<N>(ctx, name));
    } else if (name == "rpp") {
        impls.push_back(makeSelectedRpp<N>(ctx));
    } else {
        impls.push_back(makeFklVfHf<N, true>(ctx));
        impls.push_back(makeFklVfHf<N, false>(ctx));
        impls.push_back(makeFklBvf<N>(ctx));
    }
    const auto impl = std::find_if(impls.begin(), impls.end(), [&](const Implementation& i) { return i.name == name; });
    if (impl == impls.end()) {
        throw std::runtime_error("Unknown implementation " + name);
    }
    CacheFlusher flusher = makeCacheFlusher(ctx);
    activeFlusher = &flusher;
    for (int i = 0; i < 3; ++i) {
        impl->run();
    }
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    measureMarkerKernel<<<1, 1, 0, ctx.stream>>>();
    impl->run();
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    activeFlusher = nullptr;
    freeCacheFlusher(flusher);
}

template <size_t... Is>
void rocprofTargetDispatch(Context& ctx, const std::string& name, size_t batch, std::index_sequence<Is...>) {
    bool found = false;
    ((BATCHES[Is] == batch ? (rocprofTarget<BATCHES[Is]>(ctx, name), found = true) : false), ...);
    if (!found) {
        throw std::runtime_error("Batch size " + std::to_string(batch) + " is not one of the benchmarked ones");
    }
}

} // namespace bench

int main(int argc, char** argv) {
    using namespace bench;
    try {
        if (argc == 2 && std::string(argv[1]) == "--batches") {
            // The batch sizes, for rocm/scripts/rocprof_counters.sh
            for (size_t i = 0; i < BATCHES.size(); ++i) {
                std::cout << (i ? " " : "") << BATCHES[i];
            }
            std::cout << std::endl;
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--workload") {
            // The workload (sizes, batch sizes, seeds), for the consistency check of rocm/scripts/run_all.sh
            std::cout << describe() << std::endl;
            return 0;
        }
        std::filesystem::path outDir = argc > 1 ? argv[1] : "results";
        std::filesystem::create_directories(outDir);

        Context ctx;
        initContext(ctx);
        if (const char* variant = std::getenv("BENCH_RPP_VARIANT")) {
            selectedRpp = variant; // the profiling child and the counter targets use the parent's RPP pipeline
        }

        if (argc > 3 && std::string(argv[2]) == "--profile-pass") {
            // Internal: the profiling child process started by the main benchmark
            profilePass(ctx, outDir, argv[3]);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--check-kernel-tracing") {
            // Exit status 0 if rocprofiler-sdk delivers kernel dispatch records here (run with BENCH_ROCPROFILER=1)
            const bool tracing = kernelTracingAvailable(ctx);
            std::cout << "Kernel dispatch tracing: " << (tracing ? "available" : "NOT available") << std::endl;
            return tracing ? 0 : 3;
        }
        if (argc > 4 && std::string(argv[2]) == "--rocprof-target") {
            rocprofTargetDispatch(ctx, argv[3], std::stoul(argv[4]), std::make_index_sequence<BATCHES.size()>{});
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--resize-convention") {
            resizeConvention(ctx, outDir);
            return 0;
        }
        const bool validateOnly = argc > 2 && std::string(argv[2]) == "--validate";
        if (argc > 2 && !validateOnly) {
            throw std::runtime_error(std::string("Unknown mode or missing arguments: ") + argv[2] +
                                     " (see the modes in README.md)");
        }

        computeRppRgb8(ctx);
        std::ostringstream diagnostic;
        conversionDiagnostic(ctx, diagnostic);
        std::cout << diagnostic.str();

        std::vector<Result> results;
        if (validateOnly) {
            allRppVariants = true;
            validateAllBatches(ctx, results, std::make_index_sequence<BATCHES.size()>{});
            return 0;
        }
        measureEventFloor(ctx);
        selectionAllBatches(ctx, std::make_index_sequence<BATCHES.size()>{});
        selectedRpp = chooseRpp();
        runAllBatches(ctx, results, std::make_index_sequence<BATCHES.size()>{});
        // Kernel counts, then GPU time only where kernel dispatch tracing works, in the profiling child process; a
        // failure there does not discard the CPU + GPU results
        bool tracing = false;
        std::string profilingError;
        try {
            tracing = runProfilingChild(outDir, results);
        } catch (const std::exception& e) {
            profilingError = e.what();
            std::cerr << "ERROR in the profiling pass: " << profilingError << std::endl;
        }

        std::ofstream latency(outDir / "latency.csv");
        latency << "implementation,N,median_us,p10_us,p90_us,iterations,max_abs_err,kernels_launched,mean_us,"
                   "fkl_bvf_mean_speedup\n";
        std::ofstream validation(outDir / "validation.csv");
        validation << "implementation,N,reference,max_abs_err_r,max_abs_err_g,max_abs_err_b,mean_abs_err,"
                      "non_finite_values,max_abs_err_vs_float_reference,max_abs_err_vs_rpp_sampling_8bit_reference,"
                      "within_tolerance_1e-3,max_abs_err_vs_8bit_reference_fkl_sampling\n";
        writeSelection(outDir / "rpp_selection.csv");
        std::ofstream kernels(outDir / "kernels.csv");
        kernels << "implementation,N,kernels,memsets,memcpys\n";
        for (const auto& r : results) {
            latency << r.name << "," << r.batch << "," << std::fixed << std::setprecision(2) << r.timing.median << ","
                    << r.timing.p10 << "," << r.timing.p90 << "," << TIMED_ITERS << "," << std::scientific
                    << std::setprecision(3) << r.errorVsOwnReference.max() << "," << r.launches.kernels << ","
                    << std::fixed << std::setprecision(2) << r.timing.mean << ",";
            if (!std::isnan(r.meanSpeedupOfBvf)) {
                latency << std::setprecision(3) << r.meanSpeedupOfBvf;
            }
            latency << "\n";
            writeValidationRow(validation, r);
            kernels << r.name << "," << r.batch << "," << r.launches.kernels << "," << r.launches.memsets << ","
                    << r.launches.memcpys << "\n";
        }
        // One CSV per baseline in FKL's two-execution format (times in ms):
        // <baseline>_vs_fkl_bvf_GPU_AMD.csv (CPU + GPU) and, where measured, <baseline>_vs_fkl_bvf_gpu_only_GPU_AMD.csv
        // and <baseline>_vs_fkl_bvf_gpu_only_l2_flushed_GPU_AMD.csv (the CUDA benchmark's names; on AMD the flush
        // buffer covers L2 and MALL)
        const std::string parArch(fk::toStrView(fk::defaultParArch));
        for (const int mode : {0, 1, 2}) {
            if (mode > 0 && !tracing) {
                continue;
            }
            const std::string suffix = mode == 0 ? "" : mode == 1 ? "gpu_only_" : "gpu_only_l2_flushed_";
            for (const std::string& baseline : BASELINES) {
                std::ofstream two(outDir / (baseline + "_vs_fkl_bvf_" + suffix + parArch + ".csv"));
                writeTwoExecutionHeader(two, "Batch size", baseline);
                for (const auto& b : results) {
                    if (b.name != baseline) {
                        continue;
                    }
                    const auto f = std::find_if(results.begin(), results.end(), [&](const Result& r) {
                        return r.name == "fkl_bvf" && r.batch == b.batch;
                    });
                    if (mode == 0) {
                        writeTwoExecutionRow(two, b.batch, b.timing, f->timing, b.meanSpeedupOfBvf);
                    } else if (mode == 1) {
                        writeTwoExecutionRow(two, b.batch, b.gpuTiming, f->gpuTiming, b.meanGpuSpeedupOfBvf);
                    } else {
                        writeTwoExecutionRow(two, b.batch, b.gpuFlushedTiming, f->gpuFlushedTiming,
                                             b.meanGpuFlushedSpeedupOfBvf);
                    }
                }
            }
        }
        if (tracing) {
            for (const bool flushed : {false, true}) {
                std::ofstream gpuTime(outDir / (flushed ? "gpu_time_l2_flushed.csv" : "gpu_time.csv"));
                gpuTime << "implementation,N,gpu_median_us,gpu_p10_us,gpu_p90_us,gpu_mean_us,iterations,kernels_launched,"
                           "fkl_bvf_mean_gpu_speedup,runtime_blit_median_us,runtime_blit_p10_us,runtime_blit_p90_us,"
                           "runtime_blit_mean_us,runtime_blit_dispatches_per_execution\n";
                for (const auto& r : results) {
                    const TimingStats& t = flushed ? r.gpuFlushedTiming : r.gpuTiming;
                    const double s = flushed ? r.meanGpuFlushedSpeedupOfBvf : r.meanGpuSpeedupOfBvf;
                    gpuTime << r.name << "," << r.batch << "," << std::fixed << std::setprecision(2) << t.median << ","
                            << t.p10 << "," << t.p90 << "," << t.mean << "," << TIMED_ITERS << "," << r.launches.kernels
                            << ",";
                    if (!std::isnan(s)) {
                        gpuTime << std::setprecision(3) << s;
                    }
                    const TimingStats& b = flushed ? r.gpuFlushedBlitTiming : r.gpuBlitTiming;
                    gpuTime << std::setprecision(2) << "," << b.median << "," << b.p10 << "," << b.p90 << "," << b.mean
                            << "," << std::setprecision(1) << (flushed ? r.gpuFlushedBlitDispatches : r.gpuBlitDispatches)
                            << "\n";
                }
            }
        }
        std::ofstream stages(outDir / "kernel_breakdown.csv");
        stages << "implementation,N,kernel,median_us,p10_us,p90_us\n";
        for (const auto& b : breakdown) {
            stages << b.name << "," << b.batch << "," << b.stage << "," << std::fixed << std::setprecision(2)
                   << b.timing.median << "," << b.timing.p10 << "," << b.timing.p90 << "\n";
        }
        writePlatform(outDir / "platform.txt", diagnostic.str(), tracing, profilingError);
        std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
        if (!profilingError.empty()) {
            std::cerr << "ERROR: the profiling pass failed; the CPU + GPU results were written" << std::endl;
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
