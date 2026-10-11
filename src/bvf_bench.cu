// Latency and correctness benchmark for the Backwards Vertical Fusion paper, Figure 1 pipeline:
// ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
//
// Implementations:
//   opencv_cuda        NPP NV12->RGB, then per crop cv::cuda convertTo, resize, copyMakeBorder,
//                      then multiply, subtract, divide on the whole output tensor
//   npp                NV12->RGB, per crop 8u->32f, one batched resize (nppiResize for N = 1, since
//                      nppiResizeBatch rejects a batch of one), per crop CopyConstBorder,
//                      then MulC, SubC, DivC on the whole output tensor
//   fkl_vf_hf          four FKL kernels with the intermediates in DRAM (VF + HF, no BVF), with the same
//                      intermediates as OpenCV and NPP: 8 bit RGB frame, then float3 crops, resized crops and output
//                      (fkl_vf_hf_float3_frame, with a float3 frame, is only in the per kernel breakdown)
//   fkl_bvf            one FKL kernel for the whole chain (BVF)

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <npp.h>

#include <cupti.h>
#include <nvtx3/nvToolsExt.h>

#include "bench_platform.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define NPP_CHECK(expr)                                                                                     \
    do {                                                                                                    \
        const NppStatus status_ = (expr);                                                                   \
        if (status_ != NPP_SUCCESS) {                                                                       \
            throw std::runtime_error(std::string("NPP error ") + std::to_string(status_) + " in " + #expr); \
        }                                                                                                   \
    } while (0)

namespace bench {

constexpr int FRAME_W = 3840;
constexpr int FRAME_H = 2160;
constexpr int CROP_W = 20;
constexpr int CROP_H = 10;
constexpr int RES_W = 64;
constexpr int RES_H = 32;
constexpr int BORDER_TOP = 16;
constexpr int BORDER_BOTTOM = 16;
constexpr int OUT_W = RES_W;
constexpr int OUT_H = RES_H + BORDER_TOP + BORDER_BOTTOM;
constexpr int MAX_BATCH = 100;
constexpr std::array<int, 5> BATCHES{1, 4, 16, 50, 100};
constexpr int WARMUP_ITERS = 100;
constexpr int TIMED_ITERS = 1000;
constexpr unsigned FRAME_SEED = 1;
constexpr unsigned CROP_SEED = 42;
constexpr float TOLERANCE = 1e-3f;

// Pointwise tail: ImageNet normalization of 8 bit RGB
constexpr float MUL[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
constexpr float SUB[3] = {0.485f, 0.456f, 0.406f};
constexpr float DIV[3] = {0.229f, 0.224f, 0.225f};
// 8 bit quantization of the RGB values for the reference of the 8 bit pipelines:
// normalize to [0, 1], add half a level so that the truncating denormalization rounds to nearest,
// saturate with __saturatef and denormalize to uchar3
constexpr float QUANTIZATION_OFFSET = 0.5f / 255.f;

// Crop, resize and output sizes of one benchmark: the main one (CROP_W x CROP_H resized to RES_W x RES_H), or one
// case of the region-size sweep. The pipelines take them from the Context, so that the same code runs both.
struct Geometry {
    int cropW{CROP_W}, cropH{CROP_H}, resW{RES_W}, resH{RES_H};
    int outW() const { return resW; }
    int outH() const { return resH + BORDER_TOP + BORDER_BOTTOM; }
    // Floats of one output plane
    size_t outFloats() const { return static_cast<size_t>(outW()) * outH() * 3; }
};

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Full, fk::ColorPrimitives::bt2020>;

const fk::float3 mulValue{MUL[0], MUL[1], MUL[2]};
const fk::float3 subValue{SUB[0], SUB[1], SUB[2]};
const fk::float3 divValue{DIV[0], DIV[1], DIV[2]};
const fk::float3 black{0.f, 0.f, 0.f};

struct Context {
    cudaStream_t stream{};
    std::unique_ptr<fk::Stream> fkStream;
    cv::cuda::Stream cvStream;
    NppStreamContext npp{};
    std::unique_ptr<NV12Image> frameDevice;
    std::unique_ptr<NV12Image> frameHost;
    const Npp8u* y{nullptr};
    const Npp8u* uv{nullptr};
    int pitch{0};
    Npp32f twist[3][4]{};
    Geometry geom;
    std::array<fk::Rect, MAX_BATCH> crops{};
};

NppStreamContext makeNppStreamContext(cudaStream_t stream) {
    NppStreamContext ctx{};
    int device = 0;
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    gpuErrchk(cudaGetDeviceProperties(&prop, device));
    ctx.hStream = stream;
    ctx.nCudaDeviceId = device;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    gpuErrchk(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device));
    gpuErrchk(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device));
    gpuErrchk(cudaStreamGetFlags(stream, &ctx.nStreamFlags));
    return ctx;
}

// The NPP color twist is dst = M * (Y, U, V) + offset. FKL's conversion is affine in (Y, U, V),
// so the exact same matrix and offsets are recovered by probing it.
void fillTwistFromFKL(Npp32f twist[3][4]) {
    const auto f = [](int y, int u, int v) {
        return YUVToRGB::exec(fk::uchar3{static_cast<uchar>(y), static_cast<uchar>(u), static_cast<uchar>(v)});
    };
    const fk::float3 offset = f(0, 0, 0);
    const fk::float3 colY = f(1, 0, 0) - offset;
    const fk::float3 colU = f(0, 1, 0) - offset;
    const fk::float3 colV = f(0, 0, 1) - offset;
    const float m[3][4] = {{colY.x, colU.x, colV.x, offset.x},
                           {colY.y, colU.y, colV.y, offset.y},
                           {colY.z, colU.z, colV.z, offset.z}};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            twist[r][c] = m[r][c];
        }
    }
}

void initContext(Context& ctx) {
    gpuErrchk(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
    ctx.fkStream = std::make_unique<fk::Stream>(ctx.stream);
    ctx.cvStream = cv::cuda::StreamAccessor::wrapStream(ctx.stream);
    ctx.npp = makeNppStreamContext(ctx.stream);
    fillTwistFromFKL(ctx.twist);

    // Synthetic NV12 frame with uniform random bytes, identical for every implementation
    ctx.frameDevice = std::make_unique<NV12Image>(FRAME_W, FRAME_H);
    ctx.frameHost = std::make_unique<NV12Image>(FRAME_W, FRAME_H, fk::MemType::Host);
    auto deviceData = ctx.frameDevice->getData();
    auto hostData = ctx.frameHost->getData();
    std::mt19937 gen(FRAME_SEED);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int y = 0; y < FRAME_H * 3 / 2; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const uchar value = static_cast<uchar>(byte(gen));
            deviceData.at(x, y) = value;
            hostData.at(x, y) = value;
        }
    }
    ctx.frameDevice->upload(*ctx.fkStream);
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    const auto raw = deviceData.ptr();
    cudaPointerAttributes attributes{};
    gpuErrchk(cudaPointerGetAttributes(&attributes, raw.data));
    if (attributes.type != cudaMemoryTypeDevice) {
        throw std::runtime_error("The NV12 frame is expected in device memory");
    }
    ctx.pitch = static_cast<int>(raw.dims.pitch);
    ctx.y = raw.data;
    ctx.uv = raw.data + static_cast<size_t>(FRAME_H) * ctx.pitch;

    // Crop rectangles at random positions, the first N are used for batch N
    std::mt19937 cropGen(CROP_SEED);
    std::uniform_int_distribution<int> distX(0, FRAME_W - CROP_W);
    std::uniform_int_distribution<int> distY(0, FRAME_H - CROP_H);
    for (auto& crop : ctx.crops) {
        const int x = distX(cropGen);
        const int y = distY(cropGen);
        crop = fk::Rect(x, y, CROP_W, CROP_H);
    }
}

// L2 flush before every kernel, for the GPU only benchmark with a flushed L2: simulates a GPU that runs other kernels
// in between, so that no intermediate result (nor the input frame) can be assumed to still be in L2. The flush kernel
// reads a buffer of twice the L2 size, which evicts every other line (writing back the dirty ones during the flush)
// and leaves L2 clean. Every library call of the pipelines launches exactly one kernel, and beforeKernel() is called
// right before each of them; it does nothing unless the flushed benchmark is running.
__global__ void flushL2Kernel(const int4* buffer, size_t count, int* sink) {
    int acc = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int4 v = __ldcg(buffer + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x7fffffff) {
        *sink = acc; // never true for the zeroed buffer, keeps the loads
    }
}

struct L2Flusher {
    int4* buffer{nullptr};
    size_t count{0};
    int* sink{nullptr};
    cudaStream_t stream{};
    int blocks{0};
};
L2Flusher* activeFlusher = nullptr;

inline void beforeKernel() {
    if (activeFlusher) {
        flushL2Kernel<<<activeFlusher->blocks, 256, 0, activeFlusher->stream>>>(activeFlusher->buffer,
                                                                              activeFlusher->count, activeFlusher->sink);
    }
}

void nv12ToRgb8(const Context& ctx, Npp8u* dst, int dstStep) {
    beforeKernel();
    const Npp8u* src[2] = {ctx.y, ctx.uv};
    int srcSteps[2] = {ctx.pitch, ctx.pitch};
    NPP_CHECK(nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(src, srcSteps, dst, dstStep, NppiSize{FRAME_W, FRAME_H},
                                                       ctx.twist, ctx.npp));
}

// Each implementation enqueues one complete pipeline on ctx.stream with run(), and returns the
// N x 64 x 64 x 3 float output with download(). fillNaN() poisons the output before validation.
struct Implementation {
    std::string name;
    bool eightBitRGB{false};
    std::function<void()> run;
    std::function<std::vector<float>()> download;
    std::function<void()> fillNaN;
    // Optional: the kernels of a multi-kernel pipeline, in order, for the per-kernel breakdown
    std::vector<std::pair<std::string, std::function<void()>>> stages;
};

std::vector<float> downloadDevice(const void* data, size_t floats) {
    std::vector<float> host(floats);
    gpuErrchk(cudaMemcpy(host.data(), data, floats * sizeof(float), cudaMemcpyDeviceToHost));
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
    const Geometry g = ctx.geom;
    auto out = std::make_shared<fk::Tensor<fk::float3>>(g.outW(), g.outH(), N, 1, fk::MemType::Device);
    const auto crops = firstCrops<N>(ctx);
    Implementation impl{"fkl_bvf", false};
    impl.run = [&ctx, out, crops, g]() {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
            YUVToRGB::build(),
            fk::Crop<>::build(crops),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(g.resW, g.resH)),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
            fk::Mul<fk::float3>::build(mulValue),
            fk::Sub<fk::float3>::build(subValue),
            fk::Div<fk::float3>::build(divValue),
            fk::TensorWrite<fk::float3>::build(out->ptr()));
    };
    impl.download = [out, g]() { return downloadDevice(out->ptr().data, N * g.outFloats()); };
    impl.fillNaN = [out, g]() { gpuErrchk(cudaMemset(out->ptr().data, 0xFF, N * g.outFloats() * sizeof(float))); };
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

// VF + HF without BVF: four kernels with the intermediates in DRAM, the same intermediates as the OpenCV and
// NPP pipelines: an 8 bit RGB frame (NPP's NV12 -> RGB can only write 8 or 16 bit integers, and OpenCV CUDA has no
// NV12 conversion), quantized as in the 8 bit reference (normalize, add half a level, __saturatef, denormalize),
// then float3 crops (like their per crop convertTo / Convert_8u32f), float3 resized crops and the float3 output.
// With EIGHT_BIT_FRAME = false the frame is float3 instead; that variant is only used for the per kernel breakdown.
template <size_t N, bool EIGHT_BIT_FRAME>
Implementation makeFklVfHf(Context& ctx) {
    constexpr bool EIGHT_BIT = EIGHT_BIT_FRAME;
    using RGB = std::conditional_t<EIGHT_BIT, fk::uchar3, fk::float3>;
    const Geometry g = ctx.geom;
    auto rgb = std::make_shared<fk::Ptr2D<RGB>>(FRAME_W, FRAME_H, 0, fk::MemType::Device);
    auto cropped = std::make_shared<fk::Tensor<fk::float3>>(g.cropW, g.cropH, N, 1, fk::MemType::Device);
    auto resized = std::make_shared<fk::Tensor<fk::float3>>(g.resW, g.resH, N, 1, fk::MemType::Device);
    auto out = std::make_shared<fk::Tensor<fk::float3>>(g.outW(), g.outH(), N, 1, fk::MemType::Device);
    const auto crops = firstCrops<N>(ctx);
    const auto croppedViews = planeViews<N>(*cropped, g.cropW, g.cropH);
    const auto resizedViews = planeViews<N>(*resized, g.resW, g.resH);
    Implementation impl{EIGHT_BIT ? "fkl_vf_hf" : "fkl_vf_hf_float3_frame", EIGHT_BIT};
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
    impl.stages.emplace_back("3_resize", [&ctx, resized, croppedViews, g]() {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(croppedViews),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(g.resW, g.resH)),
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
    impl.download = [out, g]() { return downloadDevice(out->ptr().data, N * g.outFloats()); };
    impl.fillNaN = [out, g]() { gpuErrchk(cudaMemset(out->ptr().data, 0xFF, N * g.outFloats() * sizeof(float))); };
    return impl;
}

struct OpenCVState {
    cv::cuda::GpuMat rgb8;
    std::vector<cv::Rect> rects;
    std::vector<cv::cuda::GpuMat> cropped;
    std::vector<cv::cuda::GpuMat> resized;
    cv::cuda::GpuMat tensor;
};

template <size_t N>
Implementation makeOpenCV(Context& ctx, std::shared_ptr<OpenCVState>& stateOut) {
    const Geometry g = ctx.geom;
    auto st = std::make_shared<OpenCVState>();
    st->rgb8.create(FRAME_H, FRAME_W, CV_8UC3);
    for (size_t i = 0; i < N; ++i) {
        const fk::Rect& r = ctx.crops[i];
        st->rects.emplace_back(r.x, r.y, r.width, r.height);
        st->cropped.emplace_back(g.cropH, g.cropW, CV_32FC3);
        st->resized.emplace_back(g.resH, g.resW, CV_32FC3);
    }
    cv::cuda::createContinuous(static_cast<int>(N) * g.outH(), g.outW(), CV_32FC3, st->tensor);
    stateOut = st;
    Implementation impl{"opencv_cuda", true};
    impl.stages.emplace_back("1_nv12_to_rgb8_full_frame_npp", [&ctx, st]() {
        nv12ToRgb8(ctx, st->rgb8.data, static_cast<int>(st->rgb8.step));
    });
    impl.stages.emplace_back("2_per_crop_convertTo_resize_copyMakeBorder", [&ctx, st, g]() {
        for (size_t i = 0; i < N; ++i) {
            beforeKernel();
            st->rgb8(st->rects[i]).convertTo(st->cropped[i], CV_32F, ctx.cvStream);
            beforeKernel();
            cv::cuda::resize(st->cropped[i], st->resized[i], cv::Size(g.resW, g.resH), 0., 0., cv::INTER_LINEAR,
                             ctx.cvStream);
            cv::cuda::GpuMat plane = st->tensor.rowRange(static_cast<int>(i) * g.outH(), static_cast<int>(i + 1) * g.outH());
            beforeKernel();
            cv::cuda::copyMakeBorder(st->resized[i], plane, BORDER_TOP, BORDER_BOTTOM, 0, 0, cv::BORDER_CONSTANT,
                                     cv::Scalar(0., 0., 0.), ctx.cvStream);
        }
    });
    impl.stages.emplace_back("3_multiply_subtract_divide", [&ctx, st]() {
        beforeKernel();
        cv::cuda::multiply(st->tensor, cv::Scalar(MUL[0], MUL[1], MUL[2]), st->tensor, 1., -1, ctx.cvStream);
        beforeKernel();
        cv::cuda::subtract(st->tensor, cv::Scalar(SUB[0], SUB[1], SUB[2]), st->tensor, cv::noArray(), -1, ctx.cvStream);
        beforeKernel();
        cv::cuda::divide(st->tensor, cv::Scalar(DIV[0], DIV[1], DIV[2]), st->tensor, 1., -1, ctx.cvStream);
    });
    impl.run = [stages = impl.stages]() {
        for (const auto& stage : stages) {
            stage.second();
        }
    };
    impl.download = [st, g]() { return downloadDevice(st->tensor.data, N * g.outFloats()); };
    impl.fillNaN = [st, g]() { gpuErrchk(cudaMemset(st->tensor.data, 0xFF, N * g.outFloats() * sizeof(float))); };
    return impl;
}

struct NPPState {
    Npp8u* rgb8{nullptr};
    int rgbStep{0};
    Npp32f* cropped{nullptr};
    Npp32f* resized{nullptr};
    Npp32f* tensor{nullptr};
    NppiImageDescriptor* srcDescriptors{nullptr};
    NppiImageDescriptor* dstDescriptors{nullptr};
    NppiResizeBatchROI_Advanced* rois{nullptr};
    std::vector<fk::Rect> rects;
    ~NPPState() {
        cudaFree(rgb8);
        cudaFree(cropped);
        cudaFree(resized);
        cudaFree(tensor);
        cudaFree(srcDescriptors);
        cudaFree(dstDescriptors);
        cudaFree(rois);
    }
};

template <size_t N>
Implementation makeNPP(Context& ctx) {
    const Geometry g = ctx.geom;
    const int cropStep = g.cropW * 3 * sizeof(Npp32f);
    const int outStep = g.outW() * 3 * sizeof(Npp32f);
    const size_t cropFloats = static_cast<size_t>(g.cropW) * g.cropH * 3;
    const size_t resizedFloats = static_cast<size_t>(g.resW) * g.resH * 3;
    const size_t outFloats = g.outFloats();
    auto st = std::make_shared<NPPState>();
    size_t rgbPitch = 0;
    gpuErrchk(cudaMallocPitch(reinterpret_cast<void**>(&st->rgb8), &rgbPitch, FRAME_W * 3, FRAME_H));
    st->rgbStep = static_cast<int>(rgbPitch);
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->cropped), N * cropFloats * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->resized), N * resizedFloats * sizeof(Npp32f)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->tensor), N * outFloats * sizeof(Npp32f)));
    // The batched resize descriptors live in device memory and are uploaded once, outside of the timing
    std::vector<NppiImageDescriptor> src(N), dst(N);
    std::vector<NppiResizeBatchROI_Advanced> rois(N);
    for (size_t i = 0; i < N; ++i) {
        src[i] = NppiImageDescriptor{st->cropped + i * cropFloats, cropStep, NppiSize{g.cropW, g.cropH}};
        dst[i] = NppiImageDescriptor{st->resized + i * resizedFloats, outStep, NppiSize{g.resW, g.resH}};
        rois[i] = NppiResizeBatchROI_Advanced{NppiRect{0, 0, g.cropW, g.cropH}, NppiRect{0, 0, g.resW, g.resH}};
        st->rects.push_back(ctx.crops[i]);
    }
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->srcDescriptors), N * sizeof(NppiImageDescriptor)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->dstDescriptors), N * sizeof(NppiImageDescriptor)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->rois), N * sizeof(NppiResizeBatchROI_Advanced)));
    gpuErrchk(cudaMemcpy(st->srcDescriptors, src.data(), N * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(st->dstDescriptors, dst.data(), N * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(st->rois, rois.data(), N * sizeof(NppiResizeBatchROI_Advanced), cudaMemcpyHostToDevice));

    Implementation impl{"npp", true};
    impl.stages.emplace_back("1_nv12_to_rgb8_full_frame", [&ctx, st]() { nv12ToRgb8(ctx, st->rgb8, st->rgbStep); });
    impl.stages.emplace_back("2_per_crop_convert_8u32f", [&ctx, st, g, cropFloats, cropStep]() {
        for (size_t i = 0; i < N; ++i) {
            const fk::Rect& r = st->rects[i];
            const Npp8u* cropSrc = st->rgb8 + static_cast<size_t>(r.y) * st->rgbStep + r.x * 3;
            beforeKernel();
            NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(cropSrc, st->rgbStep, st->cropped + i * cropFloats, cropStep,
                                                NppiSize{g.cropW, g.cropH}, ctx.npp));
        }
    });
    impl.stages.emplace_back("3_resize_batch", [&ctx, st, g, cropStep, outStep]() {
        if constexpr (N == 1) {
            // nppiResizeBatch rejects a batch of one image with NPP_BAD_ARGUMENT_ERROR
            beforeKernel();
            NPP_CHECK(nppiResize_32f_C3R_Ctx(st->cropped, cropStep, NppiSize{g.cropW, g.cropH},
                                             NppiRect{0, 0, g.cropW, g.cropH}, st->resized, outStep,
                                             NppiSize{g.resW, g.resH}, NppiRect{0, 0, g.resW, g.resH},
                                             NPPI_INTER_LINEAR, ctx.npp));
        } else {
            beforeKernel();
            NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(g.resW, g.resH, st->srcDescriptors, st->dstDescriptors, st->rois,
                                                           static_cast<unsigned int>(N), NPPI_INTER_LINEAR, ctx.npp));
        }
    });
    impl.stages.emplace_back("4_per_crop_copy_const_border", [&ctx, st, g, resizedFloats, outFloats, outStep]() {
        static const Npp32f zero[3] = {0.f, 0.f, 0.f};
        for (size_t i = 0; i < N; ++i) {
            beforeKernel();
            NPP_CHECK(nppiCopyConstBorder_32f_C3R_Ctx(st->resized + i * resizedFloats, outStep, NppiSize{g.resW, g.resH},
                                                      st->tensor + i * outFloats, outStep, NppiSize{g.outW(), g.outH()},
                                                      BORDER_TOP, 0, zero, ctx.npp));
        }
    });
    impl.stages.emplace_back("5_mulc_subc_divc", [&ctx, st, g, outStep]() {
        const NppiSize tensorSize{g.outW(), static_cast<int>(N) * g.outH()};
        beforeKernel();
        NPP_CHECK(nppiMulC_32f_C3IR_Ctx(MUL, st->tensor, outStep, tensorSize, ctx.npp));
        beforeKernel();
        NPP_CHECK(nppiSubC_32f_C3IR_Ctx(SUB, st->tensor, outStep, tensorSize, ctx.npp));
        beforeKernel();
        NPP_CHECK(nppiDivC_32f_C3IR_Ctx(DIV, st->tensor, outStep, tensorSize, ctx.npp));
    });
    impl.run = [stages = impl.stages]() {
        for (const auto& stage : stages) {
            stage.second();
        }
    };
    impl.download = [st, outFloats]() { return downloadDevice(st->tensor, N * outFloats); };
    impl.fillNaN = [st, outFloats]() { gpuErrchk(cudaMemset(st->tensor, 0xFF, N * outFloats * sizeof(Npp32f))); };
    return impl;
}

// References on FKL's CPU backend. The float reference is the Figure 1 chain itself. The 8 bit reference
// quantizes the RGB values to 8 bit right after the color conversion, like the OpenCV and NPP pipelines do.
template <size_t N, bool EIGHT_BIT>
std::vector<float> cpuReference(const Context& ctx) {
    const Geometry g = ctx.geom;
    const auto crops = firstCrops<N>(ctx);
    fk::Stream_<fk::ParArch::CPU> cpuStream;
    fk::Tensor<fk::float3> out(g.outW(), g.outH(), N, 1, fk::MemType::Host);
    const auto tail = [&](const auto& rgbRead) {
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
            rgbRead,
            fk::Crop<>::build(crops),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(g.resW, g.resH)),
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
    return std::vector<float>(data, data + N * g.outFloats());
}

// Plain C++ reference with NPP's bilinear sampling convention. FKL and OpenCV CUDA sample at src = dst * scale.
// NPP's nppiResize / nppiResizeBatch 32f linear sample at src = dst * scale - 0.25, clamped to the image
// (measured with linear ramps at upscale factors 2, 3.2 and 5; it is not the pixel-center convention,
// which would be dst * scale - 0.34375 here). Same 8 bit RGB, border and pointwise tail as the 8 bit reference.
constexpr float NPP_SAMPLE_OFFSET = -0.25f;

template <size_t N>
std::vector<float> nppConventionReference8bit(const Context& ctx) {
    const Geometry g = ctx.geom;
    const int outW = g.outW(), outH = g.outH();
    std::vector<float> out(N * g.outFloats());
    const float scaleX = static_cast<float>(g.cropW) / g.resW;
    const float scaleY = static_cast<float>(g.cropH) / g.resH;
    std::vector<float> crop(static_cast<size_t>(g.cropW) * g.cropH * 3);
    const auto at = [&](int y, int x, int c) -> float& { return crop[(static_cast<size_t>(y) * g.cropW + x) * 3 + c]; };
    for (size_t i = 0; i < N; ++i) {
        const fk::Rect& r = ctx.crops[i];
        for (int y = 0; y < g.cropH; ++y) {
            for (int x = 0; x < g.cropW; ++x) {
                const fk::float3 rgb = YUVToRGB::exec(
                    ctx.frameHost->readAt(fk::Point{static_cast<int>(r.x) + x, static_cast<int>(r.y) + y, 0}));
                const float channels[3] = {rgb.x, rgb.y, rgb.z};
                for (int c = 0; c < 3; ++c) {
                    at(y, x, c) = std::floor(std::min(255.f, std::max(0.f, channels[c])) + 0.5f);
                }
            }
        }
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                float* dst = &out[((i * outH + y) * outW + x) * 3];
                const int ry = y - BORDER_TOP;
                float value[3] = {0.f, 0.f, 0.f};
                if (ry >= 0 && ry < g.resH) {
                    const float sx = std::min(static_cast<float>(g.cropW - 1), std::max(0.f, x * scaleX + NPP_SAMPLE_OFFSET));
                    const float sy = std::min(static_cast<float>(g.cropH - 1), std::max(0.f, ry * scaleY + NPP_SAMPLE_OFFSET));
                    const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
                    const int x1 = std::min(x0 + 1, g.cropW - 1), y1 = std::min(y0 + 1, g.cropH - 1);
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
// implementations run one after the other, each one between two CUDA events on the same stream (everything the
// host does to enqueue it is included) and followed by a synchronization. Inputs are already in device memory.
// Returns the TIMED_ITERS execution times of each implementation, in ms, in iteration order.
std::vector<std::vector<float>> measureInterleaved(const Context& ctx, const std::vector<Implementation>& impls) {
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    std::vector<std::vector<float>> ms(impls.size(), std::vector<float>(TIMED_ITERS));
    for (int i = 0; i < WARMUP_ITERS + TIMED_ITERS; ++i) {
        for (size_t k = 0; k < impls.size(); ++k) {
            gpuErrchk(cudaEventRecord(start, ctx.stream));
            impls[k].run();
            gpuErrchk(cudaEventRecord(stop, ctx.stream));
            gpuErrchk(cudaEventSynchronize(stop));
            if (i >= WARMUP_ITERS) {
                gpuErrchk(cudaEventElapsedTime(&ms[k][i - WARMUP_ITERS], start, stop));
            }
        }
    }
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return ms;
}

// Kernel, memset and memcpy counts of one pipeline, from CUPTI activity tracing (no hardware counters,
// so it works without admin rights). It runs in a separate pass after all the timings.
// Time of each kernel of a multi-kernel pipeline: CUDA events between its kernels on the same stream
// Case of the region-size sweep being measured ("" in the main benchmark), recorded in every result
std::string currentRegion;

struct StageTiming {
    std::string name;
    int batch{0};
    std::string stage;
    TimingStats timing;
    std::string region{currentRegion};
};
std::vector<StageTiming> breakdown;

void measureStages(const Context& ctx, const Implementation& impl, int batch) {
    const size_t numStages = impl.stages.size();
    std::vector<cudaEvent_t> events(numStages + 1);
    for (auto& e : events) {
        gpuErrchk(cudaEventCreate(&e));
    }
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        impl.run();
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
    }
    std::vector<std::vector<float>> ms(numStages, std::vector<float>(TIMED_ITERS));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        gpuErrchk(cudaEventRecord(events[0], ctx.stream));
        for (size_t s = 0; s < numStages; ++s) {
            impl.stages[s].second();
            gpuErrchk(cudaEventRecord(events[s + 1], ctx.stream));
        }
        gpuErrchk(cudaEventSynchronize(events[numStages]));
        for (size_t s = 0; s < numStages; ++s) {
            gpuErrchk(cudaEventElapsedTime(&ms[s][i], events[s], events[s + 1]));
        }
    }
    for (auto& e : events) {
        gpuErrchk(cudaEventDestroy(e));
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

struct LaunchCounts {
    size_t kernels{0}, memsets{0}, memcpys{0};
};

// GPU execution of one kernel, from its CUPTI activity record (nanosecond timestamps)
struct KernelRecord {
    uint64_t start{0}, end{0};
    unsigned gridX{0}, gridY{0}, gridZ{0}, blockX{0}, blockY{0}, blockZ{0};
};

namespace cupti {
LaunchCounts counts;
bool keepKernels = false;
std::vector<KernelRecord> kernels;

#define CUPTI_CHECK(expr)                                                                                    \
    do {                                                                                                     \
        const CUptiResult result_ = (expr);                                                                  \
        if (result_ != CUPTI_SUCCESS) {                                                                      \
            const char* message_ = nullptr;                                                                  \
            cuptiGetResultString(result_, &message_);                                                        \
            throw std::runtime_error(std::string("CUPTI error ") + (message_ ? message_ : "") + " in " + #expr); \
        }                                                                                                    \
    } while (0)

void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *size = 8 << 20;
    // CUPTI needs 8 byte alignment, which malloc guarantees (std::aligned_alloc does not exist on Windows)
    *buffer = static_cast<uint8_t*>(std::malloc(*size));
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            counts.kernels++;
            // CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL records are CUpti_ActivityKernel12 in CUDA 13.3 and 13 in 13.4; new
            // versions only append fields, so the ones read here have the same offsets in both (and in 9 to 11)
            const auto* k = reinterpret_cast<const CUpti_ActivityKernel12*>(record);
            const bool flushKernel = k->name && std::strstr(k->name, "flushL2Kernel");
            if (keepKernels && !flushKernel) {
                kernels.push_back({k->start, k->end, static_cast<unsigned>(k->gridX), static_cast<unsigned>(k->gridY),
                                   static_cast<unsigned>(k->gridZ), static_cast<unsigned>(k->blockX),
                                   static_cast<unsigned>(k->blockY), static_cast<unsigned>(k->blockZ)});
            }
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            counts.memsets++;
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            counts.memcpys++;
        }
    }
    std::free(buffer);
}

void init() {
    CUPTI_CHECK(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}
} // namespace cupti

LaunchCounts countLaunches(const Context& ctx, const Implementation& impl) {
    gpuErrchk(cudaDeviceSynchronize());
    CUPTI_CHECK(cuptiActivityFlushAll(0));
    cupti::counts = LaunchCounts{};
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    impl.run();
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    return cupti::counts;
}

// What the GPU actually does in one pipeline, from CUPTI kernel records over PROFILE_RUNS runs (median per run):
// the sum of the kernel execution times (GPU busy), the time from the first kernel start to the last kernel end,
// and the execution time and launch configuration of every kernel of the pipeline
constexpr int PROFILE_RUNS = 100;

struct KernelProfile {
    unsigned gridX, gridY, gridZ, blockThreads;
    double medianUs;
};

struct PipelineProfile {
    std::string name;
    int batch{0};
    size_t kernelsPerRun{0};
    double busyUs{0.}, spanUs{0.};
    std::vector<KernelProfile> kernels;
};

double medianOf(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

PipelineProfile profilePipeline(const Context& ctx, const Implementation& impl, int batch) {
    for (int i = 0; i < 10; ++i) {
        impl.run();
    }
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    cupti::kernels.clear();
    cupti::keepKernels = true;
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    for (int r = 0; r < PROFILE_RUNS; ++r) {
        impl.run();
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
    }
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    cupti::keepKernels = false;
    auto& k = cupti::kernels;
    std::sort(k.begin(), k.end(), [](const KernelRecord& a, const KernelRecord& b) { return a.start < b.start; });
    if (k.empty() || k.size() % PROFILE_RUNS != 0) {
        throw std::runtime_error("Unexpected number of kernel records for " + impl.name);
    }
    PipelineProfile profile{impl.name, batch, k.size() / PROFILE_RUNS};
    const size_t perRun = profile.kernelsPerRun;
    std::vector<double> busy, span;
    std::vector<std::vector<double>> perKernel(perRun);
    for (int r = 0; r < PROFILE_RUNS; ++r) {
        double b = 0.;
        for (size_t i = 0; i < perRun; ++i) {
            const KernelRecord& rec = k[r * perRun + i];
            b += (rec.end - rec.start) * 1e-3;
            perKernel[i].push_back((rec.end - rec.start) * 1e-3);
        }
        busy.push_back(b);
        span.push_back((k[r * perRun + perRun - 1].end - k[r * perRun].start) * 1e-3);
    }
    profile.busyUs = medianOf(busy);
    profile.spanUs = medianOf(span);
    for (size_t i = 0; i < perRun; ++i) {
        const KernelRecord& rec = k[i];
        profile.kernels.push_back({rec.gridX, rec.gridY, rec.gridZ, rec.blockX * rec.blockY * rec.blockZ,
                                   medianOf(perKernel[i])});
    }
    return profile;
}

std::vector<PipelineProfile> profiles;

struct Result {
    std::string name;
    int batch{0};
    TimingStats timing;
    // Mean over the timed iterations of (this implementation's time / FKL BVF's time) in the same iteration
    double meanSpeedupOfBvf{NAN};
    // GPU time only: sum of the execution times of the pipeline's kernels (CUPTI), per execution
    TimingStats gpuTiming;
    double meanGpuSpeedupOfBvf{NAN};
    // GPU time only with L2 flushed before every kernel
    TimingStats gpuFlushedTiming;
    double meanGpuFlushedSpeedupOfBvf{NAN};
    ErrorStats errorVsOwnReference;
    ErrorStats errorVsFloatReference;
    ErrorStats errorVsNppConventionReference;
    std::string reference;
    LaunchCounts launches;
    std::string region{currentRegion};
};

// The implementations in the comparison, all with the same intermediates (8 bit RGB frame, then float3), and FKL BVF
// last, which has no intermediates in memory and keeps float3 from the color conversion to the write
template <size_t N>
std::vector<Implementation> makeImplementations(Context& ctx, std::shared_ptr<OpenCVState>& ocvState) {
    std::vector<Implementation> impls;
    impls.push_back(makeOpenCV<N>(ctx, ocvState));
    impls.push_back(makeNPP<N>(ctx));
    impls.push_back(makeFklVfHf<N, true>(ctx));
    impls.push_back(makeFklBvf<N>(ctx));
    return impls;
}

// Validation, interleaved timing with per-execution speedups, and per kernel breakdown, for batch N. With timing false,
// validation only.
template <size_t N>
void runBatch(Context& ctx, std::vector<Result>& results, bool timing = true) {
    std::cout << "=== N = " << N << std::endl;
    const auto refFloat = cpuReference<N, false>(ctx);
    const auto ref8bit = cpuReference<N, true>(ctx);
    const auto refNppConvention = nppConventionReference8bit<N>(ctx);
    std::shared_ptr<OpenCVState> ocvState;
    const auto impls = makeImplementations<N>(ctx, ocvState);
    const size_t bvf = impls.size() - 1;
    std::vector<Result> batchResults;
    for (const auto& impl : impls) {
        Result r;
        r.name = impl.name;
        r.batch = static_cast<int>(N);
        impl.fillNaN();
        impl.run();
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
        const auto output = impl.download();
        r.reference = impl.eightBitRGB ? "fkl_cpu_8bit_rgb" : "fkl_cpu_float";
        r.errorVsOwnReference = compare(output, impl.eightBitRGB ? ref8bit : refFloat);
        r.errorVsFloatReference = compare(output, refFloat);
        r.errorVsNppConventionReference = compare(output, refNppConvention);
        batchResults.push_back(r);
    }
    const auto checkFloat3Frame = [&](const Implementation& float3Frame) {
        float3Frame.fillNaN();
        float3Frame.run();
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
        const ErrorStats float3FrameError = compare(float3Frame.download(), refFloat);
        if (float3FrameError.max() > TOLERANCE) {
            throw std::runtime_error("fkl_vf_hf_float3_frame does not match the float reference");
        }
    };
    if (!timing) {
        for (const Result& r : batchResults) {
            std::cout << std::left << std::setw(12) << r.name << std::right << "  max err " << std::scientific
                      << std::setprecision(2) << r.errorVsOwnReference.max() << " (" << r.reference << "), vs float "
                      << r.errorVsFloatReference.max() << ", vs NPP sampling " << r.errorVsNppConventionReference.max()
                      << std::defaultfloat << std::endl;
            results.push_back(r);
        }
        checkFloat3Frame(makeFklVfHf<N, false>(ctx));
        return;
    }
    const auto ms = measureInterleaved(ctx, impls);
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
    // Per kernel breakdown of the multi-kernel pipelines, and of VF + HF with a float3 frame for reference
    for (const auto& impl : impls) {
        if (!impl.stages.empty()) {
            measureStages(ctx, impl, static_cast<int>(N));
        }
    }
    const auto float3Frame = makeFklVfHf<N, false>(ctx);
    checkFloat3Frame(float3Frame);
    measureStages(ctx, float3Frame, static_cast<int>(N));
}

// Kernel counts of every implementation for batch N, after all the timings
template <size_t N>
void countBatch(Context& ctx, std::vector<Result>& results) {
    std::shared_ptr<OpenCVState> ocvState;
    for (const auto& impl : makeImplementations<N>(ctx, ocvState)) {
        const LaunchCounts launches = countLaunches(ctx, impl);
        for (auto& r : results) {
            if (r.name == impl.name && r.batch == static_cast<int>(N)) {
                r.launches = launches;
            }
        }
        std::cout << std::left << std::setw(10) << impl.name << " N " << std::right << std::setw(3) << N
                  << "  kernels " << std::setw(4) << launches.kernels << "  memsets " << launches.memsets
                  << "  memcpys " << launches.memcpys << std::endl;
    }
}

// Compares NPP's 8 bit NV12->RGB with FKL's float conversion, to show how NPP rounds and samples chroma
void conversionDiagnostic(const Context& ctx, std::ostream& os) {
    cv::cuda::GpuMat rgb8(FRAME_H, FRAME_W, CV_8UC3);
    nv12ToRgb8(ctx, rgb8.data, static_cast<int>(rgb8.step));
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    cv::Mat host;
    rgb8.download(host);
    size_t diffRound = 0, diffTrunc = 0, total = 0;
    int maxRound = 0, maxTrunc = 0;
    for (int y = 0; y < FRAME_H; ++y) {
        const uchar* row = host.ptr<uchar>(y);
        for (int x = 0; x < FRAME_W; ++x) {
            const fk::uchar3 yuv = ctx.frameHost->readAt(fk::Point{x, y, 0});
            const fk::float3 rgb = YUVToRGB::exec(yuv);
            const float channels[3] = {rgb.x, rgb.y, rgb.z};
            for (int c = 0; c < 3; ++c) {
                const float clamped = std::min(255.f, std::max(0.f, channels[c]));
                const int rounded = static_cast<int>(std::floor(clamped + 0.5f));
                const int truncated = static_cast<int>(clamped);
                const int npp = row[3 * x + c];
                diffRound += npp != rounded;
                diffTrunc += npp != truncated;
                maxRound = std::max(maxRound, std::abs(npp - rounded));
                maxTrunc = std::max(maxTrunc, std::abs(npp - truncated));
                total++;
            }
        }
    }
    os << "NPP NV12->RGB (ColorTwist32f, BT.2020 full range matrix taken from FKL) vs FKL float conversion:\n"
       << "  vs round-to-nearest of FKL: " << diffRound << " of " << total << " channel values differ, max "
       << maxRound << " levels\n"
       << "  vs truncation of FKL:       " << diffTrunc << " of " << total << " channel values differ, max "
       << maxTrunc << " levels\n";
}

// System, versions and protocol of a run; workload replaces the main benchmark's workload description
void writePlatform(const std::filesystem::path& path, const Context& ctx, const std::string& diagnostic,
                   const std::string& workload = "") {
    int device = 0, runtime = 0, driver = 0;
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    gpuErrchk(cudaGetDeviceProperties(&prop, device));
    gpuErrchk(cudaRuntimeGetVersion(&runtime));
    gpuErrchk(cudaDriverGetVersion(&driver));
    const NppLibraryVersion* npp = nppGetLibVersion();
    const std::string opencvBuild = cv::getBuildInformation();
    std::ofstream(path.parent_path() / "opencv_build_information.txt") << opencvBuild;
    std::ofstream os(path);
    os << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ", " << prop.multiProcessorCount << " SMs, "
       << (prop.totalGlobalMem >> 20) << " MiB)\n"
       << "NVIDIA driver: " << bench::shell("nvidia-smi --query-gpu=driver_version --format=csv,noheader") << "\n"
       << "CUDA runtime: " << runtime / 1000 << "." << (runtime % 1000) / 10 << ", CUDA driver API: " << driver / 1000
       << "." << (driver % 1000) / 10 << "\n"
       << "nvcc: " << BENCH_NVCC_VERSION << "\n"
       << "nvcc flags: " << BENCH_NVCC_FLAGS << "\n"
       << "Host compiler: " << bench::hostCompiler() << "\n"
       << "OpenCV: " << cv::getVersionString() << " (opencv_contrib "
       << bench::buildInfoValue(opencvBuild, "Version control (extra):") << ", CUDA_ARCH_BIN "
       << bench::buildInfoValue(opencvBuild, "NVIDIA GPU arch:") << ", C++ compiler "
       << bench::buildInfoValue(opencvBuild, "C++ Compiler:") << "; full configuration in opencv_build_information.txt)\n"
       << "NPP: " << npp->major << "." << npp->minor << "." << npp->build << "\n"
       << "FKL: branch " << BENCH_FKL_BRANCH << ", commit " << BENCH_FKL_COMMIT << "\n"
       << "NPPDx: compared in fkl-vs-nppdx/ of this repository\n"
       << "CPU: " << bench::cpuName() << "\n"
       << "OS: " << bench::osName() << ", kernel " << bench::kernelVersion() << "\n"
       << "GPU clocks: not locked (locking needs root); application clocks as reported by nvidia-smi: "
       << bench::shell("nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem --format=csv,noheader") << "\n"
       << "\nWorkload: ";
    if (workload.empty()) {
        os << FRAME_W << "x" << FRAME_H << " NV12 frame (uniform random bytes, seed " << FRAME_SEED << "), N crops of "
           << CROP_W << "x" << CROP_H << " at uniform random positions (seed " << CROP_SEED
           << ", the first N of one sequence), NV12->RGB BT.2020 full range, bilinear resize to " << RES_W << "x"
           << RES_H << ", AddBorder " << BORDER_TOP << " rows top and " << BORDER_BOTTOM << " bottom (black), then "
           << "Mul (1/255), Sub (0.485, 0.456, 0.406), Div (0.229, 0.224, 0.225), float3 tensor of N planes " << OUT_W
           << "x" << OUT_H << "\n";
    } else {
        os << workload << "\n";
    }
    os << "Timing: " << WARMUP_ITERS << " warm-up then " << TIMED_ITERS << " timed iterations. In every iteration "
       << "OpenCV, NPP, FKL VF+HF and FKL BVF run one after the other, each timed with CUDA events on one stream "
       << "around the complete pipeline (all host enqueue work included) and followed by a synchronization. "
       << "FKL BVF's speedup over each baseline is computed per iteration (baseline time / BVF time) and averaged "
       << "(mean speedup, FKL's two-execution convention). Median, p10, p90 with linear interpolation.\n"
       << "Intermediates: OpenCV, NPP and FKL VF+HF all write an 8 bit RGB frame (NPP's NV12->RGB only outputs 8 or "
       << "16 bit integers and OpenCV CUDA has no NV12 conversion), then float3 crops, float3 resized crops and the "
       << "float3 output. FKL BVF has no intermediates in memory: float3 from the color conversion to the write.\n"
       << "GPU time only: a second pass with the same interleaved iterations and CUPTI kernel tracing on; the GPU "
       << "time of one execution is the sum of the execution times of its kernels (no host work, no launch gaps), "
       << "and FKL BVF's speedups are again computed per execution and averaged.\n"
       << "Kernel counts: CUPTI activity tracing of one pipeline, in a separate pass after all timings\n"
       << "\n" << diagnostic;
}

template <size_t... Is>
void runAllBatches(Context& ctx, std::vector<Result>& results, std::index_sequence<Is...>) {
    (runBatch<BATCHES[Is]>(ctx, results), ...);
}

template <size_t N>
void profileBatch(Context& ctx) {
    std::shared_ptr<OpenCVState> ocvState;
    auto impls = makeImplementations<N>(ctx, ocvState);
    impls.insert(impls.begin() + 3, makeFklVfHf<N, false>(ctx));
    for (const auto& impl : impls) {
        profiles.push_back(profilePipeline(ctx, impl, static_cast<int>(N)));
        const auto& p = profiles.back();
        std::cout << std::left << std::setw(24) << p.name << " N " << std::right << std::setw(3) << N << "  kernels "
                  << std::setw(3) << p.kernelsPerRun << "  GPU busy " << std::fixed << std::setprecision(2)
                  << std::setw(7) << p.busyUs << " us  first start to last end " << std::setw(7) << p.spanUs << " us"
                  << std::defaultfloat << std::endl;
    }
}

// Host time of enqueuing the FKL pipelines (std::chrono, median over TIMED_ITERS iterations, synchronizing after
// each one). FKL BVF is split into building the nine IOps, BackFuser::fuse_back and executeOperations, which then
// only launches; FKL VF+HF is split per kernel, enqueued back to back as in its run().
struct HostTiming {
    std::string name;
    int batch{0};
    std::string step;
    TimingStats timing;
};
std::vector<HostTiming> hostTimings;

template <size_t N>
void hostOverheadBatch(Context& ctx) {
    using Clock = std::chrono::steady_clock;
    const auto us = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<float, std::milli>(b - a).count();
    };
    const Geometry g = ctx.geom;
    const auto crops = firstCrops<N>(ctx);
    fk::Tensor<fk::float3> out(g.outW(), g.outH(), N, 1, fk::MemType::Device);
    std::vector<float> build, fuse, launch, total;
    for (int i = 0; i < WARMUP_ITERS + TIMED_ITERS; ++i) {
        const auto t0 = Clock::now();
        const auto read = fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice);
        const auto convert = YUVToRGB::build();
        const auto crop = fk::Crop<>::build(crops);
        const auto resize =
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(g.resW, g.resH));
        const auto border = fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black);
        const auto mul = fk::Mul<fk::float3>::build(mulValue);
        const auto sub = fk::Sub<fk::float3>::build(subValue);
        const auto div = fk::Div<fk::float3>::build(divValue);
        const auto write = fk::TensorWrite<fk::float3>::build(out.ptr());
        const auto t1 = Clock::now();
        const auto fused = fk::BackFuser::fuse_back(read, convert, crop, resize, border, mul, sub, div, write);
        const auto t2 = Clock::now();
        fk::apply([&](const auto&... iOps) { fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream, iOps...); }, fused);
        const auto t3 = Clock::now();
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
        if (i >= WARMUP_ITERS) {
            build.push_back(us(t0, t1));
            fuse.push_back(us(t1, t2));
            launch.push_back(us(t2, t3));
            total.push_back(us(t0, t3));
        }
    }
    hostTimings.push_back({"fkl_bvf", static_cast<int>(N), "1_build_iops", computeStats(build)});
    hostTimings.push_back({"fkl_bvf", static_cast<int>(N), "2_fuse_back", computeStats(fuse)});
    hostTimings.push_back({"fkl_bvf", static_cast<int>(N), "3_launch", computeStats(launch)});
    hostTimings.push_back({"fkl_bvf", static_cast<int>(N), "total_host_enqueue", computeStats(total)});

    const auto vfhf = makeFklVfHf<N, true>(ctx);
    std::vector<std::vector<float>> stageTimes(vfhf.stages.size());
    std::vector<float> vfhfTotal;
    for (int i = 0; i < WARMUP_ITERS + TIMED_ITERS; ++i) {
        Clock::time_point t = Clock::now();
        const Clock::time_point first = t;
        for (size_t k = 0; k < vfhf.stages.size(); ++k) {
            vfhf.stages[k].second();
            const Clock::time_point next = Clock::now();
            if (i >= WARMUP_ITERS) {
                stageTimes[k].push_back(us(t, next));
            }
            t = next;
        }
        if (i >= WARMUP_ITERS) {
            vfhfTotal.push_back(us(first, t));
        }
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
    }
    for (size_t k = 0; k < vfhf.stages.size(); ++k) {
        hostTimings.push_back({vfhf.name, static_cast<int>(N), "enqueue_" + vfhf.stages[k].first, computeStats(stageTimes[k])});
    }
    hostTimings.push_back({vfhf.name, static_cast<int>(N), "total_host_enqueue", computeStats(vfhfTotal)});
    for (const auto& h : hostTimings) {
        if (h.batch == static_cast<int>(N)) {
            std::cout << std::left << std::setw(10) << h.name << " N " << std::right << std::setw(3) << N << "  "
                      << std::left << std::setw(48) << h.step << std::right << std::fixed << std::setprecision(2)
                      << std::setw(7) << h.timing.median << " us" << std::defaultfloat << std::endl;
        }
    }
}

// Target for Nsight Compute: warms up one pipeline, then runs it once more inside the NVTX range "measure", so that
// ncu --nvtx --nvtx-include "measure/" profiles exactly one execution of it
template <size_t N>
void ncuTarget(Context& ctx, const std::string& name) {
    std::shared_ptr<OpenCVState> ocvState;
    auto impls = makeImplementations<N>(ctx, ocvState);
    impls.push_back(makeFklVfHf<N, false>(ctx));
    const auto impl = std::find_if(impls.begin(), impls.end(), [&](const Implementation& i) { return i.name == name; });
    if (impl == impls.end()) {
        throw std::runtime_error("Unknown implementation " + name);
    }
    for (int i = 0; i < 3; ++i) {
        impl->run();
    }
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    nvtxRangePushA("measure");
    impl->run();
    gpuErrchk(cudaStreamSynchronize(ctx.stream));
    nvtxRangePop();
}

template <size_t... Is>
void ncuTargetDispatch(Context& ctx, const std::string& name, size_t batch, std::index_sequence<Is...>) {
    bool found = false;
    ((BATCHES[Is] == batch ? (ncuTarget<BATCHES[Is]>(ctx, name), found = true) : false), ...);
    if (!found) {
        throw std::runtime_error("Batch size " + std::to_string(batch) + " is not one of the benchmarked ones");
    }
}

template <size_t... Is>
void hostOverheadAllBatches(Context& ctx, std::index_sequence<Is...>) {
    (hostOverheadBatch<BATCHES[Is]>(ctx), ...);
}

template <size_t... Is>
void profileAllBatches(Context& ctx, std::index_sequence<Is...>) {
    (profileBatch<BATCHES[Is]>(ctx), ...);
}

// GPU time only benchmark: the same interleaved iterations as the CPU + GPU one, with CUPTI kernel tracing on. The
// GPU time of one execution is the sum of the execution times of its kernels: no host work and no launch gaps.
template <size_t N>
void gpuOnlyBatch(Context& ctx, std::vector<Result>& results, L2Flusher* flusher = nullptr) {
    std::shared_ptr<OpenCVState> ocvState;
    const auto impls = makeImplementations<N>(ctx, ocvState);
    const size_t bvf = impls.size() - 1;
    std::vector<size_t> kernelsPerRun(impls.size());
    for (size_t k = 0; k < impls.size(); ++k) {
        kernelsPerRun[k] = countLaunches(ctx, impls[k]).kernels;
    }
    activeFlusher = flusher;
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        for (const auto& impl : impls) {
            impl.run();
            gpuErrchk(cudaStreamSynchronize(ctx.stream));
        }
    }
    CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    cupti::kernels.clear();
    cupti::keepKernels = true;
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        for (const auto& impl : impls) {
            impl.run();
            gpuErrchk(cudaStreamSynchronize(ctx.stream));
        }
        if (i % 100 == 99) {
            CUPTI_CHECK(cuptiActivityFlushAll(0));
        }
    }
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    cupti::keepKernels = false;
    activeFlusher = nullptr;
    auto& records = cupti::kernels;
    std::sort(records.begin(), records.end(), [](const KernelRecord& a, const KernelRecord& b) { return a.start < b.start; });
    size_t expected = 0;
    for (const size_t n : kernelsPerRun) {
        expected += n * TIMED_ITERS;
    }
    if (records.size() != expected) {
        throw std::runtime_error("GPU only benchmark: expected " + std::to_string(expected) + " kernel records, got " +
                                 std::to_string(records.size()));
    }
    // The executions are sequential, so in start time order the records are iteration by iteration, implementation
    // by implementation
    std::vector<std::vector<float>> ms(impls.size(), std::vector<float>(TIMED_ITERS));
    size_t idx = 0;
    for (int i = 0; i < TIMED_ITERS; ++i) {
        for (size_t k = 0; k < impls.size(); ++k) {
            uint64_t ns = 0;
            for (size_t j = 0; j < kernelsPerRun[k]; ++j, ++idx) {
                ns += records[idx].end - records[idx].start;
            }
            ms[k][i] = static_cast<float>(ns * 1e-6);
        }
    }
    records.clear();
    for (size_t k = 0; k < impls.size(); ++k) {
        auto r = std::find_if(results.begin(), results.end(),
                              [&](const Result& x) { return x.name == impls[k].name && x.batch == static_cast<int>(N); });
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
        timing = computeStats(ms[k]);
        if (k != bvf) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += ms[k][i] / ms[bvf][i];
            }
            speedup = sum / TIMED_ITERS;
        }
        std::cout << (flusher ? "GPU only, L2 flushed " : "GPU only ") << std::left << std::setw(12) << r->name
                  << std::right << " N " << std::setw(3) << N << std::fixed << std::setprecision(2) << "  median "
                  << std::setw(7) << timing.median << " us  p10 " << std::setw(7) << timing.p10 << "  p90 "
                  << std::setw(7) << timing.p90;
        if (k != bvf) {
            std::cout << "  FKL BVF mean speedup " << std::setw(6) << speedup << "x";
        }
        std::cout << std::defaultfloat << std::endl;
    }
}

template <size_t... Is>
void gpuOnlyAllBatches(Context& ctx, std::vector<Result>& results, std::index_sequence<Is...>,
                       L2Flusher* flusher = nullptr) {
    (gpuOnlyBatch<BATCHES[Is]>(ctx, results, flusher), ...);
}

template <size_t... Is>
void countAllBatches(Context& ctx, std::vector<Result>& results, std::index_sequence<Is...>) {
    (countBatch<BATCHES[Is]>(ctx, results), ...);
}

L2Flusher makeL2Flusher(const Context& ctx) {
    int device = 0;
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    gpuErrchk(cudaGetDeviceProperties(&prop, device));
    L2Flusher flusher;
    const size_t bytes = 2 * static_cast<size_t>(prop.l2CacheSize);
    flusher.count = bytes / sizeof(int4);
    gpuErrchk(cudaMalloc(&flusher.buffer, bytes));
    gpuErrchk(cudaMemset(flusher.buffer, 0, bytes));
    gpuErrchk(cudaMalloc(&flusher.sink, sizeof(int)));
    flusher.stream = ctx.stream;
    flusher.blocks = prop.multiProcessorCount * 8;
    std::cout << "Flushing " << bytes / (1 << 20) << " MiB (2x L2) before every kernel" << std::endl;
    return flusher;
}

void freeL2Flusher(L2Flusher& flusher) {
    gpuErrchk(cudaFree(flusher.buffer));
    gpuErrchk(cudaFree(flusher.sink));
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

// ------------------------------------------------------------------------------------------------
// Region-size sweep: how the BVF advantage depends on how much of the producer's output (the RGB frame) is used.
// N = 16 regions, one at the origin of each cell of a 4x4 grid of 960x540 cells (they never overlap), resized
// exactly 2x in both axes, then the same 16 + 16 border rows, normalization and float3 write as the main benchmark.
// The regions grow until they cover almost the whole frame.
// ------------------------------------------------------------------------------------------------
constexpr size_t REGION_N = 16;
constexpr int REGION_GRID = 4;
constexpr int REGION_CELL_W = FRAME_W / REGION_GRID;
constexpr int REGION_CELL_H = FRAME_H / REGION_GRID;

struct RegionCase {
    const char* name;
    Geometry geom;
    long long pixels() const { return static_cast<long long>(geom.cropW) * geom.cropH; }
    double coveragePercent() const {
        return 100. * static_cast<double>(REGION_N * pixels()) / (static_cast<double>(FRAME_W) * FRAME_H);
    }
};

const std::array<RegionCase, 4> REGION_CASES{{
    {"A", Geometry{32, 16, 64, 32}},
    {"B", Geometry{112, 96, 224, 192}},
    {"C", Geometry{320, 304, 640, 608}},
    {"D", Geometry{960, 524, 1920, 1048}},
}};

void setRegionCase(Context& ctx, const RegionCase& c) {
    const Geometry& g = c.geom;
    if (g.cropW > REGION_CELL_W || g.cropH > REGION_CELL_H || g.resW != 2 * g.cropW || g.resH != 2 * g.cropH) {
        throw std::runtime_error(std::string("Invalid region case ") + c.name);
    }
    ctx.geom = g;
    for (size_t i = 0; i < REGION_N; ++i) {
        ctx.crops[i] = fk::Rect(static_cast<int>(i % REGION_GRID) * REGION_CELL_W,
                                static_cast<int>(i / REGION_GRID) * REGION_CELL_H, g.cropW, g.cropH);
    }
    currentRegion = c.name;
}

const RegionCase& regionCase(const std::string& name) {
    for (const auto& c : REGION_CASES) {
        if (name == c.name) {
            return c;
        }
    }
    throw std::runtime_error("Unknown region case " + name + ", use A, B, C or D");
}

// Bytes of every buffer each implementation reads or writes in global memory for one execution, from the sizes in the
// code and the operations it runs (an operation that reads or writes a whole buffer counts it once). The sizes are
// logical (width x height x element): row padding of pitched allocations is neither read nor written.
struct BufferTraffic {
    std::string implementation, buffer, element;
    long long width, height, planes, bufferBytes, bytesWritten, bytesRead;
};

std::vector<BufferTraffic> bufferTraffic(const Geometry& g) {
    const long long n = REGION_N;
    const long long frameNV12 = static_cast<long long>(FRAME_W) * FRAME_H * 3 / 2;
    const long long cropPixels = n * g.cropW * g.cropH;
    const long long resizedPixels = n * g.resW * g.resH;
    const long long outPixels = n * g.outW() * g.outH();
    constexpr long long F3 = 3 * sizeof(float);
    std::vector<BufferTraffic> rows;
    const auto multiKernel = [&](const std::string& impl, bool float3Frame, int outputWrites) {
        const long long rgbElement = float3Frame ? F3 : 3;
        const long long rgbFrame = static_cast<long long>(FRAME_W) * FRAME_H * rgbElement;
        rows.push_back({impl, "nv12_frame (input)", "uint8", FRAME_W, FRAME_H * 3 / 2, 1, frameNV12, 0, frameNV12});
        rows.push_back({impl, "rgb_frame", float3Frame ? "float3" : "uchar3", FRAME_W, FRAME_H, 1, rgbFrame, rgbFrame,
                        cropPixels * rgbElement});
        rows.push_back({impl, "crops", "float3", g.cropW, g.cropH, n, cropPixels * F3, cropPixels * F3,
                        cropPixels * F3});
        rows.push_back({impl, "resized_crops", "float3", g.resW, g.resH, n, resizedPixels * F3, resizedPixels * F3,
                        resizedPixels * F3});
        // OpenCV and NPP write the output with the border copy, then multiply, subtract and divide it in place
        rows.push_back({impl, "output", "float3", g.outW(), g.outH(), n, outPixels * F3, outputWrites * outPixels * F3,
                        (outputWrites - 1) * outPixels * F3});
    };
    multiKernel("opencv_cuda", false, 4);
    multiKernel("npp", false, 4);
    multiKernel("fkl_vf_hf", false, 1);
    multiKernel("fkl_vf_hf_float3_frame", true, 1);
    // BVF reads the NV12 samples of the regions (the 2x bilinear resize uses every pixel of each region) and
    // writes the output once
    rows.push_back({"fkl_bvf", "nv12_frame (input)", "uint8", FRAME_W, FRAME_H * 3 / 2, 1, frameNV12, 0,
                    cropPixels * 3 / 2});
    rows.push_back({"fkl_bvf", "output", "float3", g.outW(), g.outH(), n, outPixels * F3, outPixels * F3, 0});
    return rows;
}

// Two-execution CSVs of the sweep, one per baseline and timing, keyed by the source pixels of one region
void writeRegionTwoExecutionCSVs(const std::filesystem::path& outDir, const std::vector<Result>& results) {
    const std::string parArch(fk::toStrView(fk::defaultParArch));
    for (const std::string timingName : {"", "gpu_only_", "gpu_only_l2_flushed_"}) {
        for (const std::string baseline : {"opencv_cuda", "npp", "fkl_vf_hf"}) {
            std::ofstream two(outDir / (baseline + "_vs_fkl_bvf_region_sweep_" + timingName + parArch + ".csv"));
            writeTwoExecutionHeader(two, "Source pixels per region", baseline);
            for (const auto& c : REGION_CASES) {
                const auto find = [&](const std::string& name) {
                    return std::find_if(results.begin(), results.end(),
                                        [&](const Result& r) { return r.name == name && r.region == c.name; });
                };
                const auto b = find(baseline), f = find("fkl_bvf");
                if (b == results.end() || f == results.end()) {
                    continue;
                }
                if (timingName.empty()) {
                    writeTwoExecutionRow(two, c.pixels(), b->timing, f->timing, b->meanSpeedupOfBvf);
                } else if (timingName == "gpu_only_") {
                    writeTwoExecutionRow(two, c.pixels(), b->gpuTiming, f->gpuTiming, b->meanGpuSpeedupOfBvf);
                } else {
                    writeTwoExecutionRow(two, c.pixels(), b->gpuFlushedTiming, f->gpuFlushedTiming,
                                         b->meanGpuFlushedSpeedupOfBvf);
                }
            }
        }
    }
}

void writeRegionSweep(const std::filesystem::path& outDir, const std::vector<Result>& results, bool validateOnly) {
    std::ofstream summary(outDir / "region_sweep_cases.csv");
    summary << "case,regions,crop_w,crop_h,resize_w,resize_h,output_w,output_h,source_pixels_per_region,"
               "source_pixels_all_regions,frame_coverage_percent,region_positions\n";
    for (const auto& c : REGION_CASES) {
        const Geometry& g = c.geom;
        summary << c.name << "," << REGION_N << "," << g.cropW << "," << g.cropH << "," << g.resW << "," << g.resH << ","
                << g.outW() << "," << g.outH() << "," << c.pixels() << "," << REGION_N * c.pixels() << ","
                << std::fixed << std::setprecision(4) << c.coveragePercent() << std::defaultfloat
                << ",origin of each 960x540 cell of a 4x4 grid\n";
    }
    std::ofstream traffic(outDir / "region_sweep_buffers.csv");
    traffic << "case,implementation,buffer,element,width,height,planes,logical_bytes,bytes_written,bytes_read\n";
    for (const auto& c : REGION_CASES) {
        for (const auto& b : bufferTraffic(c.geom)) {
            traffic << c.name << "," << b.implementation << "," << b.buffer << "," << b.element << "," << b.width << ","
                    << b.height << "," << b.planes << "," << b.bufferBytes << "," << b.bytesWritten << ","
                    << b.bytesRead << "\n";
        }
    }
    std::ofstream validation(outDir / "region_sweep_validation.csv");
    validation << "case,implementation,reference,max_abs_err_r,max_abs_err_g,max_abs_err_b,mean_abs_err,"
                  "non_finite_values,max_abs_err_vs_float_reference,max_abs_err_vs_npp_sampling_8bit_reference,"
                  "within_tolerance_1e-3\n";
    for (const auto& r : results) {
        const auto& e = r.errorVsOwnReference;
        validation << r.region << "," << r.name << "," << r.reference << "," << std::scientific << std::setprecision(3)
                   << e.maxPerChannel[0] << "," << e.maxPerChannel[1] << "," << e.maxPerChannel[2] << "," << e.mean
                   << "," << e.nonFinite << "," << r.errorVsFloatReference.max() << ","
                   << r.errorVsNppConventionReference.max() << "," << (e.max() <= TOLERANCE ? "yes" : "no")
                   << std::defaultfloat << "\n";
    }
    if (validateOnly) {
        return;
    }
    std::ofstream times(outDir / "region_sweep_times.csv");
    times << "case,source_pixels_per_region,implementation,kernels_launched,"
             "cpu_gpu_median_us,cpu_gpu_p10_us,cpu_gpu_p90_us,cpu_gpu_mean_us,fkl_bvf_mean_speedup_cpu_gpu,"
             "gpu_only_median_us,gpu_only_p10_us,gpu_only_p90_us,gpu_only_mean_us,fkl_bvf_mean_speedup_gpu_only,"
             "gpu_l2_flushed_median_us,gpu_l2_flushed_p10_us,gpu_l2_flushed_p90_us,gpu_l2_flushed_mean_us,"
             "fkl_bvf_mean_speedup_gpu_l2_flushed,iterations\n";
    for (const auto& r : results) {
        const RegionCase& c = regionCase(r.region);
        times << r.region << "," << c.pixels() << "," << r.name << "," << r.launches.kernels << std::fixed
              << std::setprecision(2);
        for (const auto& [t, speedup] : std::array<std::pair<const TimingStats*, double>, 3>{
                 {{&r.timing, r.meanSpeedupOfBvf}, {&r.gpuTiming, r.meanGpuSpeedupOfBvf},
                  {&r.gpuFlushedTiming, r.meanGpuFlushedSpeedupOfBvf}}}) {
            times << "," << t->median << "," << t->p10 << "," << t->p90 << "," << t->mean << ",";
            if (!std::isnan(speedup)) {
                times << std::setprecision(3) << speedup << std::setprecision(2);
            }
        }
        times << "," << TIMED_ITERS << std::defaultfloat << "\n";
    }
    std::ofstream kernels(outDir / "region_sweep_kernels.csv");
    kernels << "case,implementation,kernels,memsets,memcpys\n";
    for (const auto& r : results) {
        kernels << r.region << "," << r.name << "," << r.launches.kernels << "," << r.launches.memsets << ","
                << r.launches.memcpys << "\n";
    }
    std::ofstream stages(outDir / "region_sweep_kernel_breakdown.csv");
    stages << "case,implementation,kernel,median_us,p10_us,p90_us\n";
    for (const auto& b : breakdown) {
        if (!b.region.empty()) {
            stages << b.region << "," << b.name << "," << b.stage << "," << std::fixed << std::setprecision(2)
                   << b.timing.median << "," << b.timing.p10 << "," << b.timing.p90 << std::defaultfloat << "\n";
        }
    }
    writeRegionTwoExecutionCSVs(outDir, results);
}

// The whole sweep: validation, CPU + GPU timing and per kernel breakdown for every case, then (with CUPTI, after all
// of those) GPU time only, GPU time only with the L2 flushed before every kernel, and kernel counts. The passes and
// their order are the main benchmark's.
void regionSweep(Context& ctx, const std::filesystem::path& outDir, bool validateOnly) {
    std::vector<std::vector<Result>> perCase(REGION_CASES.size());
    for (size_t c = 0; c < REGION_CASES.size(); ++c) {
        setRegionCase(ctx, REGION_CASES[c]);
        const Geometry& g = ctx.geom;
        std::cout << "=== Region case " << REGION_CASES[c].name << ": " << REGION_N << " crops of " << g.cropW << "x"
                  << g.cropH << " -> " << g.resW << "x" << g.resH << " -> " << g.outW() << "x" << g.outH() << ", "
                  << std::fixed << std::setprecision(4) << REGION_CASES[c].coveragePercent() << std::defaultfloat
                  << "% of the frame" << std::endl;
        runBatch<REGION_N>(ctx, perCase[c], !validateOnly);
    }
    if (!validateOnly) {
        cupti::init();
        std::cout << "=== GPU time only (CUPTI kernel execution times, after all CPU + GPU timings)" << std::endl;
        for (size_t c = 0; c < REGION_CASES.size(); ++c) {
            setRegionCase(ctx, REGION_CASES[c]);
            gpuOnlyBatch<REGION_N>(ctx, perCase[c]);
        }
        L2Flusher flusher = makeL2Flusher(ctx);
        for (size_t c = 0; c < REGION_CASES.size(); ++c) {
            setRegionCase(ctx, REGION_CASES[c]);
            gpuOnlyBatch<REGION_N>(ctx, perCase[c], &flusher);
        }
        freeL2Flusher(flusher);
        std::cout << "=== Kernel counts (CUPTI activity tracing, after all timings)" << std::endl;
        for (size_t c = 0; c < REGION_CASES.size(); ++c) {
            setRegionCase(ctx, REGION_CASES[c]);
            countBatch<REGION_N>(ctx, perCase[c]);
        }
    }
    currentRegion.clear();
    std::vector<Result> results;
    for (const auto& r : perCase) {
        results.insert(results.end(), r.begin(), r.end());
    }
    writeRegionSweep(outDir, results, validateOnly);
    std::ostringstream workload;
    workload << "region-size sweep. " << FRAME_W << "x" << FRAME_H << " NV12 frame (uniform random bytes, seed "
             << FRAME_SEED << "), N = " << REGION_N << " regions, one at the origin of each " << REGION_CELL_W << "x"
             << REGION_CELL_H << " cell of a " << REGION_GRID << "x" << REGION_GRID << " grid, bilinear resize exactly 2x";
    for (const auto& c : REGION_CASES) {
        workload << "; case " << c.name << ": " << c.geom.cropW << "x" << c.geom.cropH << " -> " << c.geom.resW << "x"
                 << c.geom.resH << " -> " << c.geom.outW() << "x" << c.geom.outH() << ", " << std::fixed
                 << std::setprecision(4) << c.coveragePercent() << std::defaultfloat << "% of the frame";
    }
    workload << ". NV12->RGB BT.2020 full range, AddBorder " << BORDER_TOP << " rows top and " << BORDER_BOTTOM
             << " bottom (black), Mul (1/255), Sub (0.485, 0.456, 0.406), Div (0.229, 0.224, 0.225), float3 tensor "
             << "of " << REGION_N << " planes. "
             << (validateOnly ? "Validation only."
                              : "All passes in one process: CPU + GPU for every case, then (CUPTI) GPU only for every "
                                "case, then GPU only with the L2 flushed before every kernel (a kernel that reads twice "
                                "the L2 size) for every case, then kernel counts.");
    std::ostringstream diagnostic;
    conversionDiagnostic(ctx, diagnostic);
    writePlatform(outDir / "region_sweep_platform.txt", ctx, diagnostic.str(), workload.str());
}

} // namespace bench

int main(int argc, char** argv) {
    using namespace bench;
    try {
        std::filesystem::path outDir = argc > 1 ? argv[1] : "results";
        std::filesystem::create_directories(outDir);

        Context ctx;
        initContext(ctx);

        if (argc > 2 && std::string(argv[2]) == "--gpu-only-l2-flushed") {
            // GPU time only with the L2 flushed before every kernel of every pipeline (no intermediate result or
            // input can be assumed to be in L2). Separate from the main run, which it does not touch.
            L2Flusher flusher = makeL2Flusher(ctx);
            cupti::init();
            std::vector<Result> flushed;
            gpuOnlyAllBatches(ctx, flushed, std::make_index_sequence<BATCHES.size()>{}, &flusher);
            std::ofstream csv(outDir / "gpu_time_l2_flushed.csv");
            csv << "implementation,N,gpu_median_us,gpu_p10_us,gpu_p90_us,gpu_mean_us,iterations,kernels_launched,"
                   "fkl_bvf_mean_gpu_speedup\n";
            for (const auto& r : flushed) {
                csv << r.name << "," << r.batch << "," << std::fixed << std::setprecision(2) << r.gpuFlushedTiming.median
                    << "," << r.gpuFlushedTiming.p10 << "," << r.gpuFlushedTiming.p90 << ","
                    << r.gpuFlushedTiming.mean << "," << TIMED_ITERS << "," << r.launches.kernels << ",";
                if (!std::isnan(r.meanGpuFlushedSpeedupOfBvf)) {
                    csv << std::setprecision(3) << r.meanGpuFlushedSpeedupOfBvf;
                }
                csv << "\n";
            }
            const std::string parArch(fk::toStrView(fk::defaultParArch));
            for (const std::string baseline : {"opencv_cuda", "npp", "fkl_vf_hf"}) {
                std::ofstream two(outDir / (baseline + "_vs_fkl_bvf_gpu_only_l2_flushed_" + parArch + ".csv"));
                writeTwoExecutionHeader(two, "Batch size", baseline);
                for (const auto& b : flushed) {
                    if (b.name != baseline) {
                        continue;
                    }
                    const auto f = std::find_if(flushed.begin(), flushed.end(), [&](const Result& r) {
                        return r.name == "fkl_bvf" && r.batch == b.batch;
                    });
                    writeTwoExecutionRow(two, b.batch, b.gpuFlushedTiming, f->gpuFlushedTiming,
                                         b.meanGpuFlushedSpeedupOfBvf);
                }
            }
            freeL2Flusher(flusher);
            std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
            return 0;
        }
        if (argc > 4 && std::string(argv[2]) == "--ncu-target") {
            ncuTargetDispatch(ctx, argv[3], std::stoul(argv[4]), std::make_index_sequence<BATCHES.size()>{});
            return 0;
        }
        if (argc > 2 && (std::string(argv[2]) == "--region-sweep" || std::string(argv[2]) == "--region-sweep-validate")) {
            // Region-size sweep (N = 16, region sizes A to D); --region-sweep-validate only validates
            regionSweep(ctx, outDir, std::string(argv[2]) == "--region-sweep-validate");
            std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
            return 0;
        }
        if (argc > 4 && std::string(argv[2]) == "--region-ncu-target") {
            // One execution of one pipeline of one case of the region-size sweep, for Nsight Compute
            setRegionCase(ctx, regionCase(argv[4]));
            ncuTarget<REGION_N>(ctx, argv[3]);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--host-overhead") {
            hostOverheadAllBatches(ctx, std::make_index_sequence<BATCHES.size()>{});
            std::ofstream csv(outDir / "host_overhead.csv");
            csv << "implementation,N,step,host_median_us,host_p10_us,host_p90_us\n";
            for (const auto& h : hostTimings) {
                csv << h.name << "," << h.batch << "," << h.step << "," << std::fixed << std::setprecision(2)
                    << h.timing.median << "," << h.timing.p10 << "," << h.timing.p90 << "\n";
            }
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--profile-kernels") {
            // GPU execution time of every kernel (CUPTI), without touching the latency results
            cupti::init();
            profileAllBatches(ctx, std::make_index_sequence<BATCHES.size()>{});
            std::ofstream summary(outDir / "kernel_gpu_times.csv");
            summary << "implementation,N,kernels_per_pipeline,gpu_busy_us,first_kernel_start_to_last_kernel_end_us\n";
            std::ofstream details(outDir / "kernel_gpu_times_fkl.csv");
            details << "implementation,N,kernel,grid_x,grid_y,grid_z,threads_per_block,total_threads,gpu_time_us\n";
            for (const auto& p : profiles) {
                summary << p.name << "," << p.batch << "," << p.kernelsPerRun << "," << std::fixed
                        << std::setprecision(2) << p.busyUs << "," << p.spanUs << "\n";
                if (p.name.rfind("fkl_", 0) == 0) {
                    for (size_t i = 0; i < p.kernels.size(); ++i) {
                        const auto& k = p.kernels[i];
                        const size_t threads = static_cast<size_t>(k.gridX) * k.gridY * k.gridZ * k.blockThreads;
                        details << p.name << "," << p.batch << "," << i + 1 << "," << k.gridX << "," << k.gridY << ","
                                << k.gridZ << "," << k.blockThreads << "," << threads << "," << std::fixed
                                << std::setprecision(2) << k.medianUs << "\n";
                    }
                }
            }
            std::cout << "Kernel profiles written to " << std::filesystem::absolute(outDir) << std::endl;
            return 0;
        }

        if (argc > 2) {
            throw std::runtime_error(std::string("Unknown mode or missing arguments: ") + argv[2] +
                                     " (see the modes in README.md)");
        }

        std::ostringstream diagnostic;
        conversionDiagnostic(ctx, diagnostic);
        std::cout << diagnostic.str();

        std::vector<Result> results;
        runAllBatches(ctx, results, std::make_index_sequence<BATCHES.size()>{});
        std::cout << "=== Kernel counts (CUPTI activity tracing, after all timings)" << std::endl;
        cupti::init();
        std::cout << "=== GPU time only (CUPTI kernel execution times, after all CPU + GPU timings)" << std::endl;
        gpuOnlyAllBatches(ctx, results, std::make_index_sequence<BATCHES.size()>{});
        countAllBatches(ctx, results, std::make_index_sequence<BATCHES.size()>{});

        std::ofstream latency(outDir / "latency.csv");
        latency << "implementation,N,median_us,p10_us,p90_us,iterations,max_abs_err,kernels_launched,mean_us,"
                   "fkl_bvf_mean_speedup\n";
        std::ofstream validation(outDir / "validation.csv");
        validation << "implementation,N,reference,max_abs_err_r,max_abs_err_g,max_abs_err_b,mean_abs_err,"
                      "non_finite_values,max_abs_err_vs_float_reference,max_abs_err_vs_npp_sampling_8bit_reference,"
                      "within_tolerance_1e-3\n";
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
            const auto& e = r.errorVsOwnReference;
            validation << r.name << "," << r.batch << "," << r.reference << "," << std::scientific
                       << std::setprecision(3) << e.maxPerChannel[0] << "," << e.maxPerChannel[1] << ","
                       << e.maxPerChannel[2] << "," << e.mean << "," << e.nonFinite << ","
                       << r.errorVsFloatReference.max() << "," << r.errorVsNppConventionReference.max() << ","
                       << (e.max() <= TOLERANCE ? "yes" : "no") << "\n";
            kernels << r.name << "," << r.batch << "," << r.launches.kernels << "," << r.launches.memsets << ","
                    << r.launches.memcpys << "\n";
        }
        // One CSV per baseline in FKL's two-execution format (times in ms), for both benchmarks:
        // <baseline>_vs_fkl_bvf_GPU_NVIDIA.csv (CPU + GPU) and <baseline>_vs_fkl_bvf_gpu_only_GPU_NVIDIA.csv
        const std::string parArch(fk::toStrView(fk::defaultParArch));
        for (const bool gpuOnly : {false, true}) {
            for (const std::string baseline : {"opencv_cuda", "npp", "fkl_vf_hf"}) {
                std::ofstream two(outDir / (baseline + "_vs_fkl_bvf_" + (gpuOnly ? "gpu_only_" : "") + parArch + ".csv"));
                writeTwoExecutionHeader(two, "Batch size", baseline);
                for (const auto& b : results) {
                    if (b.name != baseline) {
                        continue;
                    }
                    const auto f = std::find_if(results.begin(), results.end(), [&](const Result& r) {
                        return r.name == "fkl_bvf" && r.batch == b.batch;
                    });
                    writeTwoExecutionRow(two, b.batch, gpuOnly ? b.gpuTiming : b.timing, gpuOnly ? f->gpuTiming : f->timing,
                                         gpuOnly ? b.meanGpuSpeedupOfBvf : b.meanSpeedupOfBvf);
                }
            }
        }
        std::ofstream gpuTime(outDir / "gpu_time.csv");
        gpuTime << "implementation,N,gpu_median_us,gpu_p10_us,gpu_p90_us,gpu_mean_us,iterations,kernels_launched,"
                   "fkl_bvf_mean_gpu_speedup\n";
        for (const auto& r : results) {
            gpuTime << r.name << "," << r.batch << "," << std::fixed << std::setprecision(2) << r.gpuTiming.median << ","
                    << r.gpuTiming.p10 << "," << r.gpuTiming.p90 << "," << r.gpuTiming.mean << "," << TIMED_ITERS << ","
                    << r.launches.kernels << ",";
            if (!std::isnan(r.meanGpuSpeedupOfBvf)) {
                gpuTime << std::setprecision(3) << r.meanGpuSpeedupOfBvf;
            }
            gpuTime << "\n";
        }
        std::ofstream stages(outDir / "kernel_breakdown.csv");
        stages << "implementation,N,kernel,median_us,p10_us,p90_us\n";
        for (const auto& b : breakdown) {
            stages << b.name << "," << b.batch << "," << b.stage << "," << std::fixed << std::setprecision(2)
                   << b.timing.median << "," << b.timing.p10 << "," << b.timing.p90 << "\n";
        }
        writePlatform(outDir / "platform.txt", ctx, diagnostic.str());
        std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
