// Latency and correctness benchmark for the Backwards Vertical Fusion paper, Figure 1 pipeline:
// ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
//
// Implementations:
//   opencv_cuda        NPP NV12->RGB, then per crop cv::cuda convertTo, resize, copyMakeBorder,
//                      then multiply, subtract, divide on the whole output tensor
//   npp                NV12->RGB, per crop 8u->32f, one batched resize (nppiResize for N = 1, since
//                      nppiResizeBatch rejects a batch of one), per crop CopyConstBorder,
//                      then MulC, SubC, DivC on the whole output tensor
//   fkl_vf_hf          four FKL kernels with the intermediates in DRAM (VF + HF, no BVF)
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

#include <algorithm>
#include <array>
#include <cmath>
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

void nv12ToRgb8(const Context& ctx, Npp8u* dst, int dstStep) {
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
    auto out = std::make_shared<fk::Tensor<fk::float3>>(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const auto crops = firstCrops<N>(ctx);
    Implementation impl{"fkl_bvf", false};
    impl.run = [&ctx, out, crops]() {
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
    impl.download = [out]() { return downloadDevice(out->ptr().data, N * OUT_W * OUT_H * 3); };
    impl.fillNaN = [out]() { gpuErrchk(cudaMemset(out->ptr().data, 0xFF, N * OUT_W * OUT_H * sizeof(fk::float3))); };
    return impl;
}

template <size_t N>
std::array<fk::RawPtr<fk::ND::_2D, fk::float3>, N> planeViews(const fk::Tensor<fk::float3>& tensor, int width,
                                                               int height) {
    const auto raw = tensor.ptr();
    if (raw.dims.pitch != width * sizeof(fk::float3)) {
        throw std::runtime_error("Expected a contiguous Tensor");
    }
    std::array<fk::RawPtr<fk::ND::_2D, fk::float3>, N> views{};
    for (size_t i = 0; i < N; ++i) {
        views[i].data = raw.data + i * width * height;
        views[i].dims = fk::PtrDims<fk::ND::_2D>(width, height, width * sizeof(fk::float3));
    }
    return views;
}

template <size_t N>
Implementation makeFklVfHf(Context& ctx) {
    auto rgb = std::make_shared<fk::Ptr2D<fk::float3>>(FRAME_W, FRAME_H, 0, fk::MemType::Device);
    auto cropped = std::make_shared<fk::Tensor<fk::float3>>(CROP_W, CROP_H, N, 1, fk::MemType::Device);
    auto resized = std::make_shared<fk::Tensor<fk::float3>>(RES_W, RES_H, N, 1, fk::MemType::Device);
    auto out = std::make_shared<fk::Tensor<fk::float3>>(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const auto crops = firstCrops<N>(ctx);
    const auto croppedViews = planeViews<N>(*cropped, CROP_W, CROP_H);
    const auto resizedViews = planeViews<N>(*resized, RES_W, RES_H);
    Implementation impl{"fkl_vf_hf", false};
    impl.run = [&ctx, rgb, cropped, resized, out, crops, croppedViews, resizedViews]() {
        // Kernel 1: full frame NV12 -> RGB
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
            YUVToRGB::build(),
            fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(rgb->ptr()));
        // Kernel 2: batched crop
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(rgb->ptr()),
            fk::Crop<>::build(crops),
            fk::TensorWrite<fk::float3>::build(cropped->ptr()));
        // Kernel 3: batched resize
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(croppedViews),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::TensorWrite<fk::float3>::build(resized->ptr()));
        // Kernel 4: batched AddBorder + Mul + Sub + Div + Write
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(resizedViews),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
            fk::Mul<fk::float3>::build(mulValue),
            fk::Sub<fk::float3>::build(subValue),
            fk::Div<fk::float3>::build(divValue),
            fk::TensorWrite<fk::float3>::build(out->ptr()));
    };
    impl.download = [out]() { return downloadDevice(out->ptr().data, N * OUT_W * OUT_H * 3); };
    impl.fillNaN = [out]() { gpuErrchk(cudaMemset(out->ptr().data, 0xFF, N * OUT_W * OUT_H * sizeof(fk::float3))); };
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
    auto st = std::make_shared<OpenCVState>();
    st->rgb8.create(FRAME_H, FRAME_W, CV_8UC3);
    for (size_t i = 0; i < N; ++i) {
        const fk::Rect& r = ctx.crops[i];
        st->rects.emplace_back(r.x, r.y, r.width, r.height);
        st->cropped.emplace_back(CROP_H, CROP_W, CV_32FC3);
        st->resized.emplace_back(RES_H, RES_W, CV_32FC3);
    }
    cv::cuda::createContinuous(static_cast<int>(N) * OUT_H, OUT_W, CV_32FC3, st->tensor);
    stateOut = st;
    Implementation impl{"opencv_cuda", true};
    impl.run = [&ctx, st]() {
        nv12ToRgb8(ctx, st->rgb8.data, static_cast<int>(st->rgb8.step));
        for (size_t i = 0; i < N; ++i) {
            st->rgb8(st->rects[i]).convertTo(st->cropped[i], CV_32F, ctx.cvStream);
            cv::cuda::resize(st->cropped[i], st->resized[i], cv::Size(RES_W, RES_H), 0., 0., cv::INTER_LINEAR,
                             ctx.cvStream);
            cv::cuda::GpuMat plane = st->tensor.rowRange(static_cast<int>(i) * OUT_H, static_cast<int>(i + 1) * OUT_H);
            cv::cuda::copyMakeBorder(st->resized[i], plane, BORDER_TOP, BORDER_BOTTOM, 0, 0, cv::BORDER_CONSTANT,
                                     cv::Scalar(0., 0., 0.), ctx.cvStream);
        }
        cv::cuda::multiply(st->tensor, cv::Scalar(MUL[0], MUL[1], MUL[2]), st->tensor, 1., -1, ctx.cvStream);
        cv::cuda::subtract(st->tensor, cv::Scalar(SUB[0], SUB[1], SUB[2]), st->tensor, cv::noArray(), -1, ctx.cvStream);
        cv::cuda::divide(st->tensor, cv::Scalar(DIV[0], DIV[1], DIV[2]), st->tensor, 1., -1, ctx.cvStream);
    };
    impl.download = [st]() { return downloadDevice(st->tensor.data, N * OUT_W * OUT_H * 3); };
    impl.fillNaN = [st]() { gpuErrchk(cudaMemset(st->tensor.data, 0xFF, N * OUT_W * OUT_H * 3 * sizeof(float))); };
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
    constexpr int cropStep = CROP_W * 3 * sizeof(Npp32f);
    constexpr int outStep = OUT_W * 3 * sizeof(Npp32f);
    constexpr size_t cropFloats = CROP_W * CROP_H * 3;
    constexpr size_t resizedFloats = RES_W * RES_H * 3;
    constexpr size_t outFloats = OUT_W * OUT_H * 3;
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
        src[i] = NppiImageDescriptor{st->cropped + i * cropFloats, cropStep, NppiSize{CROP_W, CROP_H}};
        dst[i] = NppiImageDescriptor{st->resized + i * resizedFloats, outStep, NppiSize{RES_W, RES_H}};
        rois[i] = NppiResizeBatchROI_Advanced{NppiRect{0, 0, CROP_W, CROP_H}, NppiRect{0, 0, RES_W, RES_H}};
        st->rects.push_back(ctx.crops[i]);
    }
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->srcDescriptors), N * sizeof(NppiImageDescriptor)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->dstDescriptors), N * sizeof(NppiImageDescriptor)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&st->rois), N * sizeof(NppiResizeBatchROI_Advanced)));
    gpuErrchk(cudaMemcpy(st->srcDescriptors, src.data(), N * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(st->dstDescriptors, dst.data(), N * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(st->rois, rois.data(), N * sizeof(NppiResizeBatchROI_Advanced), cudaMemcpyHostToDevice));

    Implementation impl{"npp", true};
    impl.run = [&ctx, st]() {
        static const Npp32f zero[3] = {0.f, 0.f, 0.f};
        nv12ToRgb8(ctx, st->rgb8, st->rgbStep);
        for (size_t i = 0; i < N; ++i) {
            const fk::Rect& r = st->rects[i];
            const Npp8u* cropSrc = st->rgb8 + static_cast<size_t>(r.y) * st->rgbStep + r.x * 3;
            NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(cropSrc, st->rgbStep, st->cropped + i * cropFloats, cropStep,
                                                NppiSize{CROP_W, CROP_H}, ctx.npp));
        }
        if constexpr (N == 1) {
            // nppiResizeBatch rejects a batch of one image with NPP_BAD_ARGUMENT_ERROR
            NPP_CHECK(nppiResize_32f_C3R_Ctx(st->cropped, cropStep, NppiSize{CROP_W, CROP_H}, NppiRect{0, 0, CROP_W, CROP_H},
                                             st->resized, outStep, NppiSize{RES_W, RES_H}, NppiRect{0, 0, RES_W, RES_H},
                                             NPPI_INTER_LINEAR, ctx.npp));
        } else {
            NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(RES_W, RES_H, st->srcDescriptors, st->dstDescriptors, st->rois,
                                                           static_cast<unsigned int>(N), NPPI_INTER_LINEAR, ctx.npp));
        }
        for (size_t i = 0; i < N; ++i) {
            NPP_CHECK(nppiCopyConstBorder_32f_C3R_Ctx(st->resized + i * resizedFloats, outStep, NppiSize{RES_W, RES_H},
                                                      st->tensor + i * outFloats, outStep, NppiSize{OUT_W, OUT_H},
                                                      BORDER_TOP, 0, zero, ctx.npp));
        }
        const NppiSize tensorSize{OUT_W, static_cast<int>(N) * OUT_H};
        NPP_CHECK(nppiMulC_32f_C3IR_Ctx(MUL, st->tensor, outStep, tensorSize, ctx.npp));
        NPP_CHECK(nppiSubC_32f_C3IR_Ctx(SUB, st->tensor, outStep, tensorSize, ctx.npp));
        NPP_CHECK(nppiDivC_32f_C3IR_Ctx(DIV, st->tensor, outStep, tensorSize, ctx.npp));
    };
    impl.download = [st]() { return downloadDevice(st->tensor, N * outFloats); };
    impl.fillNaN = [st]() { gpuErrchk(cudaMemset(st->tensor, 0xFF, N * outFloats * sizeof(Npp32f))); };
    return impl;
}

// References on FKL's CPU backend. The float reference is the Figure 1 chain itself. The 8 bit reference
// quantizes the RGB values to 8 bit right after the color conversion, like the OpenCV and NPP pipelines do.
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
    return std::vector<float>(data, data + N * OUT_W * OUT_H * 3);
}

// Plain C++ reference with NPP's bilinear sampling convention. FKL and OpenCV CUDA sample at src = dst * scale.
// NPP's nppiResize / nppiResizeBatch 32f linear sample at src = dst * scale - 0.25, clamped to the image
// (measured with linear ramps at upscale factors 2, 3.2 and 5; it is not the pixel-center convention,
// which would be dst * scale - 0.34375 here). Same 8 bit RGB, border and pointwise tail as the 8 bit reference.
constexpr float NPP_SAMPLE_OFFSET = -0.25f;

template <size_t N>
std::vector<float> nppConventionReference8bit(const Context& ctx) {
    std::vector<float> out(N * OUT_W * OUT_H * 3);
    const float scaleX = static_cast<float>(CROP_W) / RES_W;
    const float scaleY = static_cast<float>(CROP_H) / RES_H;
    for (size_t i = 0; i < N; ++i) {
        const fk::Rect& r = ctx.crops[i];
        float crop[CROP_H][CROP_W][3];
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                const fk::float3 rgb = YUVToRGB::exec(ctx.frameHost->readAt(fk::Point{r.x + x, r.y + y, 0}));
                const float channels[3] = {rgb.x, rgb.y, rgb.z};
                for (int c = 0; c < 3; ++c) {
                    crop[y][x][c] = std::floor(std::min(255.f, std::max(0.f, channels[c])) + 0.5f);
                }
            }
        }
        for (int y = 0; y < OUT_H; ++y) {
            for (int x = 0; x < OUT_W; ++x) {
                float* dst = &out[((i * OUT_H + y) * OUT_W + x) * 3];
                const int ry = y - BORDER_TOP;
                float value[3] = {0.f, 0.f, 0.f};
                if (ry >= 0 && ry < RES_H) {
                    const float sx = std::min(static_cast<float>(CROP_W - 1), std::max(0.f, x * scaleX + NPP_SAMPLE_OFFSET));
                    const float sy = std::min(static_cast<float>(CROP_H - 1), std::max(0.f, ry * scaleY + NPP_SAMPLE_OFFSET));
                    const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
                    const int x1 = std::min(x0 + 1, CROP_W - 1), y1 = std::min(y0 + 1, CROP_H - 1);
                    const float fx = sx - x0, fy = sy - y0;
                    for (int c = 0; c < 3; ++c) {
                        value[c] = (crop[y0][x0][c] * (1.f - fx) + crop[y0][x1][c] * fx) * (1.f - fy) +
                                   (crop[y1][x0][c] * (1.f - fx) + crop[y1][x1][c] * fx) * fy;
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

struct TimingStats {
    double median{0.}, p10{0.}, p90{0.};
};

double percentile(const std::vector<float>& sorted, double p) {
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

// Latency of one complete pipeline: CUDA events on the pipeline's stream around everything the host does to
// enqueue it, synchronizing after every iteration. Inputs are already in device memory.
TimingStats measure(const Context& ctx, const Implementation& impl) {
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        impl.run();
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
    }
    std::vector<float> ms(TIMED_ITERS);
    for (int i = 0; i < TIMED_ITERS; ++i) {
        gpuErrchk(cudaEventRecord(start, ctx.stream));
        impl.run();
        gpuErrchk(cudaEventRecord(stop, ctx.stream));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&ms[i], start, stop));
    }
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    std::sort(ms.begin(), ms.end());
    return {percentile(ms, 0.5) * 1000., percentile(ms, 0.1) * 1000., percentile(ms, 0.9) * 1000.};
}

// Kernel, memset and memcpy counts of one pipeline, from CUPTI activity tracing (no hardware counters,
// so it works without admin rights). It runs in a separate pass after all the timings.
struct LaunchCounts {
    size_t kernels{0}, memsets{0}, memcpys{0};
};

namespace cupti {
LaunchCounts counts;

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
    *buffer = static_cast<uint8_t*>(std::aligned_alloc(8, *size));
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            counts.kernels++;
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

struct Result {
    std::string name;
    int batch{0};
    TimingStats timing;
    ErrorStats errorVsOwnReference;
    ErrorStats errorVsFloatReference;
    ErrorStats errorVsNppConventionReference;
    std::string reference;
    LaunchCounts launches;
};

template <size_t N>
std::vector<Implementation> makeImplementations(Context& ctx, std::shared_ptr<OpenCVState>& ocvState) {
    std::vector<Implementation> impls;
    impls.push_back(makeOpenCV<N>(ctx, ocvState));
    impls.push_back(makeNPP<N>(ctx));
    impls.push_back(makeFklVfHf<N>(ctx));
    impls.push_back(makeFklBvf<N>(ctx));
    return impls;
}

// Validation and timing of every implementation for batch N
template <size_t N>
void runBatch(Context& ctx, std::vector<Result>& results) {
    std::cout << "=== N = " << N << std::endl;
    const auto refFloat = cpuReference<N, false>(ctx);
    const auto ref8bit = cpuReference<N, true>(ctx);
    const auto refNppConvention = nppConventionReference8bit<N>(ctx);
    std::shared_ptr<OpenCVState> ocvState;
    for (const auto& impl : makeImplementations<N>(ctx, ocvState)) {
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
        r.timing = measure(ctx, impl);
        std::cout << std::left << std::setw(10) << r.name << " median " << std::right << std::setw(9) << std::fixed
                  << std::setprecision(1) << r.timing.median << " us  p10 " << std::setw(9) << r.timing.p10
                  << "  p90 " << std::setw(9) << r.timing.p90 << "  max err " << std::scientific
                  << std::setprecision(2) << r.errorVsOwnReference.max() << " (" << r.reference << "), vs NPP-convention 8 bit "
                  << r.errorVsNppConventionReference.max()
                  << std::defaultfloat << std::endl;
        results.push_back(r);
    }
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

void writePlatform(const std::filesystem::path& path, const Context& ctx, const std::string& diagnostic) {
    int device = 0, runtime = 0, driver = 0;
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    gpuErrchk(cudaGetDeviceProperties(&prop, device));
    gpuErrchk(cudaRuntimeGetVersion(&runtime));
    gpuErrchk(cudaDriverGetVersion(&driver));
    const NppLibraryVersion* npp = nppGetLibVersion();
    std::ofstream os(path);
    os << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ", " << prop.multiProcessorCount << " SMs, "
       << (prop.totalGlobalMem >> 20) << " MiB)\n"
       << "NVIDIA driver: " << shell("nvidia-smi --query-gpu=driver_version --format=csv,noheader") << "\n"
       << "CUDA runtime: " << runtime / 1000 << "." << (runtime % 1000) / 10 << ", CUDA driver API: " << driver / 1000
       << "." << (driver % 1000) / 10 << "\n"
       << "nvcc: " << BENCH_NVCC_VERSION << "\n"
       << "nvcc flags: " << BENCH_NVCC_FLAGS << "\n"
       << "Host compiler: GCC " << __VERSION__ << "\n"
       << "OpenCV: " << cv::getVersionString() << " (with opencv_contrib 4.14.0, CUDA_ARCH_BIN=8.9, no DNN/OpenCL/python)\n"
       << "NPP: " << npp->major << "." << npp->minor << "." << npp->build << "\n"
       << "FKL: branch " << BENCH_FKL_BRANCH << ", commit " << BENCH_FKL_COMMIT << "\n"
       << "NPPDx: not installed, implementation (iii) skipped\n"
       << "CPU: " << shell("lscpu | sed -n 's/^Model name: *//p'") << "\n"
       << "OS: " << shell(". /etc/os-release && echo $PRETTY_NAME") << ", kernel " << shell("uname -r") << "\n"
       << "GPU clocks: not locked (locking needs root); application clocks as reported by nvidia-smi: "
       << shell("nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem --format=csv,noheader") << "\n"
       << "\nWorkload: " << FRAME_W << "x" << FRAME_H << " NV12 frame (uniform random bytes, seed " << FRAME_SEED
       << "), N crops of " << CROP_W << "x" << CROP_H << " at uniform random positions (seed " << CROP_SEED
       << ", the first N of one sequence), NV12->RGB BT.2020 full range, bilinear resize to " << RES_W << "x"
       << RES_H << ", AddBorder " << BORDER_TOP << " rows top and " << BORDER_BOTTOM << " bottom (black), then "
       << "Mul (1/255), Sub (0.485, 0.456, 0.406), Div (0.229, 0.224, 0.225), float3 tensor of N planes " << OUT_W
       << "x" << OUT_H << "\n"
       << "Timing: CUDA events on one stream around one complete pipeline (all host enqueue work included), "
       << WARMUP_ITERS << " warm-up iterations then " << TIMED_ITERS
       << " timed, synchronizing after each one; median, p10, p90 with linear interpolation\n"
       << "Kernel counts: CUPTI activity tracing of one pipeline, in a separate pass after all timings\n"
       << "\n" << diagnostic;
}

template <size_t... Is>
void runAllBatches(Context& ctx, std::vector<Result>& results, std::index_sequence<Is...>) {
    (runBatch<BATCHES[Is]>(ctx, results), ...);
}

template <size_t... Is>
void countAllBatches(Context& ctx, std::vector<Result>& results, std::index_sequence<Is...>) {
    (countBatch<BATCHES[Is]>(ctx, results), ...);
}

} // namespace bench

int main(int argc, char** argv) {
    using namespace bench;
    try {
        std::filesystem::path outDir = argc > 1 ? argv[1] : "results";
        std::filesystem::create_directories(outDir);

        Context ctx;
        initContext(ctx);

        std::ostringstream diagnostic;
        conversionDiagnostic(ctx, diagnostic);
        std::cout << diagnostic.str();

        std::vector<Result> results;
        runAllBatches(ctx, results, std::make_index_sequence<BATCHES.size()>{});
        std::cout << "=== Kernel counts (CUPTI activity tracing, after all timings)" << std::endl;
        cupti::init();
        countAllBatches(ctx, results, std::make_index_sequence<BATCHES.size()>{});

        std::ofstream latency(outDir / "latency.csv");
        latency << "implementation,N,median_us,p10_us,p90_us,iterations,max_abs_err,kernels_launched\n";
        std::ofstream validation(outDir / "validation.csv");
        validation << "implementation,N,reference,max_abs_err_r,max_abs_err_g,max_abs_err_b,mean_abs_err,"
                      "non_finite_values,max_abs_err_vs_float_reference,max_abs_err_vs_npp_sampling_8bit_reference,"
                      "within_tolerance_1e-3\n";
        std::ofstream kernels(outDir / "kernels.csv");
        kernels << "implementation,N,kernels,memsets,memcpys\n";
        for (const auto& r : results) {
            latency << r.name << "," << r.batch << "," << std::fixed << std::setprecision(2) << r.timing.median << ","
                    << r.timing.p10 << "," << r.timing.p90 << "," << TIMED_ITERS << "," << std::scientific
                    << std::setprecision(3) << r.errorVsOwnReference.max() << "," << r.launches.kernels << "\n";
            const auto& e = r.errorVsOwnReference;
            validation << r.name << "," << r.batch << "," << r.reference << "," << std::scientific
                       << std::setprecision(3) << e.maxPerChannel[0] << "," << e.maxPerChannel[1] << ","
                       << e.maxPerChannel[2] << "," << e.mean << "," << e.nonFinite << ","
                       << r.errorVsFloatReference.max() << "," << r.errorVsNppConventionReference.max() << ","
                       << (e.max() <= TOLERANCE ? "yes" : "no") << "\n";
            kernels << r.name << "," << r.batch << "," << r.launches.kernels << "," << r.launches.memsets << ","
                    << r.launches.memcpys << "\n";
        }
        writePlatform(outDir / "platform.txt", ctx, diagnostic.str());
        std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
