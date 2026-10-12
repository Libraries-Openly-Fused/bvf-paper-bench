// CPU benchmark of the Figure 1 pipeline: RPP's HOST backend vs FKL's CPU backends (BVF), the ROCm counterpart of
// fkl-vs-opencv-cpu/src/benchmark_opencv_cpu.cpp (which compares with OpenCV).
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// The workload of common/bench_geometry.h: one 3840x2160 NV12 frame in host memory, N crops, each resized (bilinear)
// and given black rows on top and bottom, then (x * mul - sub) / div written as float3 into an N-plane tensor. BT.601
// limited range, as on the GPU.
//
//   rpp*:   RPP has no NV12 conversion on its HOST backend (rppt_yuv_to_rgb is HIP only), so every RPP HOST pipeline
//           starts with FKL's CPU conversion of the full frame to 8 bit RGB (the first kernel of the GPU benchmark's
//           FKL VF+HF: normalize, add half a level, saturate, denormalize to uchar3), on one thread (ParArch::CPU) for
//           the 1 thread runs and on FKL's thread pool (ParArch::CPU_MULTITHREAD) for the all threads runs. Then the
//           same RPP variants as the GPU benchmark (rocm/src/rocm_bench.hip), with RPP_HOST_BACKEND:
//             rpp                 rppt_crop_mirror_normalize 8u -> 32f (one batched call, all batch images aliasing
//                                 the RGB frame), memset of the border rows of the output, rppt_resize into the rows of
//                                 the output planes below the top border, rppt_multiply_scalar,
//                                 rppt_tensor_subtract_tensor, rppt_tensor_divide_tensor
//             rpp_npp_like        the resized crops in their own buffer, copied into the output rows by rppt_crop
//             rpp_fused           memset of the border rows, rppt_resize_mirror_normalize (crop + 8u -> 32f + resize
//                                 + Mul), Sub, Div
//             rpp_normalize       rpp with Mul + Sub + Div in one rppt_normalize (out of place), without the memset
//             rpp_fused_normalize rppt_resize_mirror_normalize (crop + 8u -> 32f + resize), rppt_normalize
//           Only the border rows need zeroing (every pipeline writes all the interior rows), and only where the
//           pipeline modifies them in place: the normalize variants never write the border rows of their input
//           tensor, which is zeroed once when it is allocated.
//           RPP's HOST backend parallelizes over the batch images with OpenMP; it is timed with 1 thread and with all
//           the hardware threads (numThreads of rppCreate).
// Two comparisons, each with the same number of threads on both sides, as in fkl-vs-opencv-cpu: RPP with 1 thread
// against fkl, and RPP with all the hardware threads against fkl mt.
//   fkl:    one executeOperations<TransformDPP<ParArch::CPU>> call with the whole chain (BVF), on one thread.
//   fkl mt: the same call with TransformDPP<ParArch::CPU_MULTITHREAD> on FKL's thread pool (one worker per hardware
//           thread), in chunks of one output row of one crop.
//
// Both libraries' thread pools spin for a while after finishing before their workers go to sleep, so every timed run
// is preceded by an idle gap that lets them fall asleep, as in fkl-vs-opencv-cpu/src/benchmark_opencv_cpu.cpp.

// rpp.h includes HIP's headers, which declare uchar3, float3... in the global namespace: included first, so that FKL
// does not export its own vector types there (this file uses fk:: vector types only)
#include <rpp/rpp.h>
#include <rpp/rpp_version.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/execution_model/cpu_multithread.h>

#include "bench_geometry.h"
#include "bench_platform.h"
#include "bench_validation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define RPP_CHECK(expr)                                                                                     \
    do {                                                                                                    \
        const RppStatus status_ = (expr);                                                                   \
        if (status_ != RPP_SUCCESS) {                                                                       \
            throw std::runtime_error(std::string("RPP error ") + std::to_string(status_) + " in " + #expr); \
        }                                                                                                   \
    } while (0)

namespace rppcpu {

// Frame, crop, resize, border and output sizes, batch sizes, seeds and crop positions, shared by every benchmark
using namespace bench::geometry;
// BENCH_WARMUP_ITERATIONS and BENCH_ITERATIONS come from CMake: 100 and 1000 for the paper's results
constexpr int WARMUP_ITERS = BENCH_WARMUP_ITERATIONS;
constexpr int ITERS = BENCH_ITERATIONS;
static_assert(WARMUP_ITERS >= 0 && ITERS >= 1, "at least one timed iteration");
constexpr auto IDLE_GAP = std::chrono::milliseconds(1);

constexpr float MUL[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
constexpr float SUB[3] = {0.485f, 0.456f, 0.406f};
constexpr float DIV[3] = {0.229f, 0.224f, 0.225f};
constexpr float QUANTIZATION_OFFSET = 0.5f / 255.f;
// RPP variants within this max abs error of the RPP-convention reference are numerically equivalent; the fastest of
// them is the RPP pipeline of the benchmark (selection pass)
constexpr float RPP_EQUIVALENCE = 1e-5f;
// Rows of RPP buffers are padded to a multiple of 8 pixels (RPP's vectorized kernels load and store 8 pixels), and
// every RPP buffer gets slack at its end
constexpr int RPP_PIXEL_ALIGN = 8;
constexpr size_t RPP_SLACK_FLOATS = 1 << 14;
constexpr int alignUp(int v, int a) { return (v + a - 1) / a * a; }
// RPP 3.2.0's HOST rppt_crop_mirror_normalize (8u -> 32f, packed RGB) writes one float past the end of every row (the
// constant 1.0, from its 48-float AVX store) when the ROI width is a multiple of 16, as with 32 pixel crops. Inside a
// crop the next row overwrites it, but the last row's lands on the first float of the next crop, which another OpenMP
// thread may already have written. So every crop of the crop buffer is followed by a spare row.
constexpr int CROP_SPARE_ROWS = 1;
// RPP's HOST rppt_resize (F32) reads 12 bytes before its source and rppt_resize_mirror_normalize (8u) 3 bytes before
// the first row of its ROI: the buffers they read start after some slack. (On the RGB frame, the crops are at least
// CROP_MARGIN pixels from the edges.)
constexpr size_t RPP_FRONT_SLACK_FLOATS = 16;
// The 8 bit ramps that measure rppt_resize_mirror_normalize's sampling positions: value = RAMP_STEP_8BIT * position,
// the largest power of two that keeps the longest crop side within 255 (exact in float, and divided back exactly)
constexpr float rampStep8bit() {
    float step = 128.f;
    while (step > 1.f && step * static_cast<float>(std::max(CROP_W, CROP_H) - 1) > 255.f) {
        step /= 2.f;
    }
    return step;
}
constexpr float RAMP_STEP_8BIT = rampStep8bit();
static_assert(RAMP_STEP_8BIT * static_cast<float>(std::max(CROP_W, CROP_H) - 1) <= 255.f,
              "The 8 bit sampling ramps need crop sides of at most 256 pixels");

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
// FKL's NV12 -> RGB conversion: BT.601, limited range, as in every benchmark of the repository
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Limited, fk::ColorPrimitives::bt601>;

struct Workload {
    std::unique_ptr<NV12Image> frame;
    std::array<fk::Rect, MAX_BATCH> crops{};
};

void initWorkload(Workload& w) {
    w.frame = std::make_unique<NV12Image>(FRAME_W, FRAME_H, fk::MemType::Host);
    auto data = w.frame->getData();
    generateFrame([&](int x, int y, uint8_t value) { data.at(x, y) = value; });
    const auto crops = makeCrops();
    for (size_t i = 0; i < MAX_BATCH; ++i) {
        w.crops[i] = fk::Rect(crops[i].x, crops[i].y, crops[i].width, crops[i].height);
    }
}

// ------------------------------------------------------------------------------------------------
// FKL's conversion of the full frame to 8 bit RGB, the first step of every RPP HOST pipeline
// ------------------------------------------------------------------------------------------------
template <fk::ParArch PA>
struct FklRgb8 {
    fk::Ptr2D<fk::uchar3> rgb{FRAME_W, FRAME_H, 0, fk::MemType::Host};
    fk::Stream_<PA> stream;

    void run(const NV12Image& frame) {
        fk::executeOperations<fk::TransformDPP<PA>>(stream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(frame),
            YUVToRGB::build(),
            fk::NormalizePixel<fk::float3, fk::ColorDepth::p8bit>::build(),
            fk::Add<fk::float3>::build(fk::make_set<fk::float3>(QUANTIZATION_OFFSET)),
            fk::SaturateDenormalizePixel<fk::float3, fk::uchar3, fk::ColorDepth::p8bit>::build(),
            fk::PerThreadWrite<fk::ND::_2D, fk::uchar3>::build(rgb.ptr()));
        stream.sync();
    }
    uint8_t* data() { return reinterpret_cast<uint8_t*>(rgb.ptr().data); }
    unsigned pitch() const { return static_cast<unsigned>(rgb.ptr().dims.pitch); }
};

// ------------------------------------------------------------------------------------------------
// RPP HOST pipelines. Buffers, descriptors and parameter tensors are created once, outside of the timing. HOST handles
// are created with batch size 1: the HOST kernels iterate over the descriptors' n, and the handle's batch size only
// sizes a scratch buffer (99532800 floats per image, 39.8 GB at N = 100) that none of these calls use.
// ------------------------------------------------------------------------------------------------
enum class Variant { Rpp, NppLike, Fused, Normalize, FusedNormalize };
constexpr std::array<Variant, 5> VARIANTS{Variant::Rpp, Variant::NppLike, Variant::Fused, Variant::Normalize,
                                          Variant::FusedNormalize};

std::string variantName(Variant v) {
    switch (v) {
    case Variant::Rpp: return "rpp_per_op";
    case Variant::NppLike: return "rpp_npp_like";
    case Variant::Fused: return "rpp_fused";
    case Variant::Normalize: return "rpp_normalize";
    case Variant::FusedNormalize: return "rpp_fused_normalize";
    }
    return "?";
}

template <fk::ParArch PA>
struct RppHost {
    const Workload& w;
    size_t n;
    Variant variant;
    unsigned threads;
    rppHandle_t handle{};
    FklRgb8<PA> rgb8;
    int cropPitch{alignUp(CROP_W, RPP_PIXEL_ALIGN)}, resizedPitch{alignUp(RES_W, RPP_PIXEL_ALIGN)};
    // Floats from one crop to the next: its rows and a spare row (CROP_SPARE_ROWS)
    size_t cropStride{static_cast<size_t>(cropPitch) * (CROP_H + CROP_SPARE_ROWS) * 3};
    std::vector<float> cropped, resized, tensorA, tensorB;
    float* crops{nullptr}; // the first crop, RPP_FRONT_SLACK_FLOATS into cropped
    float* output{nullptr};
    std::vector<RpptROI> cropRois, resizeRois, resizedRois;
    std::vector<RpptImagePatch> dstSizes;
    std::vector<float> cmnOffset, cmnMultiplier, rmnMean, rmnStdDevMul, rmnStdDevOne;
    std::vector<Rpp32u> mirror;
    std::vector<float> mulScalar;
    std::vector<RpptROI3D> volumeRois;
    std::vector<float> subRows, divRows, normMean, normStdDevInit, normStdDev;
    std::vector<Rpp32u> rowsRoi, rowRoi;
    RpptDesc frameBatchDesc, cropDesc, resizedDesc, outInteriorDesc;
    RpptGenericDesc volumeDesc{}, rowsDesc{}, rowDesc{};
    std::vector<std::pair<std::string, std::function<void()>>> ops;

    RppHost(const Workload& workload, size_t batch, Variant v, unsigned numThreads)
        : w(workload), n(batch), variant(v), threads(numThreads) {
        RPP_CHECK(static_cast<RppStatus>(rppCreate(&handle, 1, threads, nullptr, RppBackend::RPP_HOST_BACKEND)));
        const unsigned N = static_cast<unsigned>(n);
        cropped.assign(RPP_FRONT_SLACK_FLOATS + n * cropStride + RPP_SLACK_FLOATS, 0.f);
        crops = cropped.data() + RPP_FRONT_SLACK_FLOATS;
        resized.assign(n * resizedPitch * RES_H * 3 + RPP_SLACK_FLOATS, 0.f);
        tensorA.assign(n * OUT_FLOATS + BORDER_TOP * OUT_W * 3 + RPP_SLACK_FLOATS, 0.f);
        tensorB.assign(n * OUT_FLOATS + RPP_SLACK_FLOATS, 0.f);
        cropRois.resize(n);
        resizeRois.resize(n);
        resizedRois.resize(n);
        dstSizes.resize(n);
        mirror.assign(n, 0);
        cmnOffset.assign(n * 3, 0.f);
        cmnMultiplier.assign(n * 3, 1.f);
        rmnMean.assign(n * 3, 0.f);
        rmnStdDevMul.resize(n * 3);
        rmnStdDevOne.assign(n * 3, 1.f);
        for (size_t i = 0; i < n; ++i) {
            const fk::Rect& r = w.crops[i];
            cropRois[i].xywhROI = {{static_cast<int>(r.x), static_cast<int>(r.y)}, CROP_W, CROP_H};
            resizeRois[i].xywhROI = {{0, 0}, CROP_W, CROP_H};
            resizedRois[i].xywhROI = {{0, 0}, RES_W, RES_H};
            dstSizes[i] = {RES_W, RES_H};
            for (int c = 0; c < 3; ++c) {
                rmnStdDevMul[i * 3 + c] = 1.f / MUL[c];
            }
        }
        // The output tensor as N samples (one per plane): RPP's HOST backend parallelizes over the samples (dims[0])
        const Rpp32u cols = OUT_W * 3;
        mulScalar.assign(n, MUL[0]);
        volumeRois.resize(n);
        for (auto& r : volumeRois) {
            r.xyzwhdROI = {{0, 0, 0}, OUT_W, OUT_H, 1};
        }
        for (size_t i = 0; i < n; ++i) {
            for (Rpp32u k = 0; k < cols; ++k) {
                subRows.push_back(SUB[k % 3]); // one row of the per channel constants per sample, broadcast over its rows
                divRows.push_back(DIV[k % 3]);
            }
            for (const Rpp32u v : {0u, 0u, static_cast<Rpp32u>(OUT_H), cols}) {
                rowsRoi.push_back(v);
            }
            for (const Rpp32u v : {0u, 0u, 1u, cols}) {
                rowRoi.push_back(v);
            }
        }
        for (Rpp32u k = 0; k < cols; ++k) { // rppt_normalize: one row of parameters shared by all the samples
            normMean.push_back(SUB[k % 3] / MUL[k % 3]);
            normStdDevInit.push_back(DIV[k % 3] / MUL[k % 3]);
        }
        normStdDev = normStdDevInit;
        frameBatchDesc = RpptDesc(N, 3, FRAME_H, FRAME_W, RpptDataType::U8, RpptLayout::NHWC, 0,
                                  RpptStrides{0, 1, rgb8.pitch(), 3});
        cropDesc = RpptDesc(N, 3, CROP_H, CROP_W, RpptDataType::F32, RpptLayout::NHWC, 0,
                            RpptStrides{static_cast<Rpp32u>(cropStride), 1, static_cast<Rpp32u>(cropPitch * 3), 3});
        resizedDesc = RpptDesc(N, 3, RES_H, RES_W, RpptDataType::F32, RpptLayout::NHWC, 0,
                               RpptStrides{static_cast<Rpp32u>(resizedPitch * RES_H * 3), 1,
                                           static_cast<Rpp32u>(resizedPitch * 3), 3});
        outInteriorDesc = RpptDesc(N, 3, RES_H, RES_W, RpptDataType::F32, RpptLayout::NHWC,
                                   static_cast<Rpp32u>(BORDER_TOP * OUT_W * 3 * sizeof(float)),
                                   RpptStrides{static_cast<Rpp32u>(OUT_FLOATS), 1, static_cast<Rpp32u>(OUT_W * 3), 3});
        RpptGenericDesc& vd = volumeDesc;
        vd.numDims = 5; vd.offsetInBytes = 0; vd.dataType = RpptDataType::F32; vd.layout = RpptLayout::NDHWC;
        vd.dims[0] = N; vd.dims[1] = 1; vd.dims[2] = OUT_H; vd.dims[3] = OUT_W; vd.dims[4] = 3;
        vd.strides[4] = 1; vd.strides[3] = 3; vd.strides[2] = OUT_W * 3; vd.strides[1] = OUT_FLOATS;
        vd.strides[0] = OUT_FLOATS;
        RpptGenericDesc& a = rowsDesc;
        a.numDims = 3; a.offsetInBytes = 0; a.dataType = RpptDataType::F32; a.layout = RpptLayout::NDHWC;
        a.dims[0] = N; a.dims[1] = OUT_H; a.dims[2] = cols; a.strides[2] = 1; a.strides[1] = cols; a.strides[0] = OUT_FLOATS;
        rowDesc = a;
        rowDesc.dims[1] = 1;
        rowDesc.strides[0] = cols;
        buildOps();
    }

    ~RppHost() { rppDestroy(handle, RppBackend::RPP_HOST_BACKEND); }

    void cropConvert() {
        RPP_CHECK(rppt_crop_mirror_normalize(rgb8.data(), &frameBatchDesc, crops, &cropDesc, cmnOffset.data(),
                                             cmnMultiplier.data(), mirror.data(), cropRois.data(), RpptRoiType::XYWH,
                                             handle, RppBackend::RPP_HOST_BACKEND));
    }
    // The border rows of every output plane (RPP has no constant border); the interior rows are written by every
    // pipeline. Only for the variants that modify the border rows in place
    void zeroBorders() {
        for (size_t i = 0; i < n; ++i) {
            float* plane = tensorA.data() + i * OUT_FLOATS;
            std::memset(plane, 0, static_cast<size_t>(BORDER_TOP) * OUT_W * 3 * sizeof(float));
            std::memset(plane + static_cast<size_t>(BORDER_TOP + RES_H) * OUT_W * 3, 0,
                        static_cast<size_t>(BORDER_BOTTOM) * OUT_W * 3 * sizeof(float));
        }
    }
    void resizeInto(float* src, RpptDesc* srcDesc, float* dst, RpptDesc* dstDesc) {
        RPP_CHECK(rppt_resize(src, srcDesc, dst, dstDesc, dstSizes.data(), RpptInterpolationType::BILINEAR,
                              resizeRois.data(), RpptRoiType::XYWH, handle, RppBackend::RPP_HOST_BACKEND));
    }
    void copyIntoOutput() {
        RPP_CHECK(rppt_crop(resized.data(), &resizedDesc, tensorA.data(), &outInteriorDesc, resizedRois.data(),
                            RpptRoiType::XYWH, handle, RppBackend::RPP_HOST_BACKEND));
    }
    void resizeMirrorNormalize(bool mul) {
        RPP_CHECK(rppt_resize_mirror_normalize(rgb8.data(), &frameBatchDesc, tensorA.data(), &outInteriorDesc,
                                               dstSizes.data(), RpptInterpolationType::BILINEAR, rmnMean.data(),
                                               mul ? rmnStdDevMul.data() : rmnStdDevOne.data(), mirror.data(),
                                               cropRois.data(), RpptRoiType::XYWH, handle, RppBackend::RPP_HOST_BACKEND));
    }
    void multiply() {
        RPP_CHECK(rppt_multiply_scalar(tensorA.data(), &volumeDesc, tensorA.data(), &volumeDesc, mulScalar.data(),
                                       volumeRois.data(), RpptRoi3DType::XYZWHD, handle, RppBackend::RPP_HOST_BACKEND));
    }
    void subtract() {
        RPP_CHECK(rppt_tensor_subtract_tensor(tensorA.data(), subRows.data(), &rowsDesc, &rowDesc, tensorA.data(),
                                              &rowsDesc, RPP_BROADCAST_ENABLE, rowsRoi.data(), rowRoi.data(), handle,
                                              RppBackend::RPP_HOST_BACKEND));
    }
    void divide() {
        RPP_CHECK(rppt_tensor_divide_tensor(tensorA.data(), divRows.data(), &rowsDesc, &rowDesc, tensorA.data(),
                                            &rowsDesc, RPP_BROADCAST_ENABLE, rowsRoi.data(), rowRoi.data(), handle,
                                            RppBackend::RPP_HOST_BACKEND));
    }
    // RPP's HOST rppt_normalize replaces the caller's stdDev values by scale / stdDev and zeroes its destination before
    // reading its source: the stdDev row is restored before every call (192 floats), and it runs out of place
    void normalize() {
        std::copy(normStdDevInit.begin(), normStdDevInit.end(), normStdDev.begin());
        RPP_CHECK(rppt_normalize(tensorA.data(), &rowsDesc, tensorB.data(), &rowsDesc, 1, normMean.data(),
                                 normStdDev.data(), 0, 1.f, 0.f, rowsRoi.data(), handle, RppBackend::RPP_HOST_BACKEND));
    }

    void buildOps() {
        const auto add = [&](const std::string& name, std::function<void()> f) { ops.emplace_back(name, std::move(f)); };
        add("fkl_nv12_to_rgb8_full_frame", [this]() { rgb8.run(*w.frame); });
        switch (variant) {
        case Variant::Rpp:
            add("rppt_crop_mirror_normalize_8u32f_batch", [this]() { cropConvert(); });
            add("memset_output_border_rows", [this]() { zeroBorders(); });
            add("rppt_resize_batch_into_output_rows",
                [this]() { resizeInto(crops, &cropDesc, tensorA.data(), &outInteriorDesc); });
            add("rppt_multiply_scalar", [this]() { multiply(); });
            add("rppt_tensor_subtract_tensor", [this]() { subtract(); });
            add("rppt_tensor_divide_tensor", [this]() { divide(); });
            output = tensorA.data();
            break;
        case Variant::NppLike:
            add("rppt_crop_mirror_normalize_8u32f_batch", [this]() { cropConvert(); });
            add("rppt_resize_batch", [this]() { resizeInto(crops, &cropDesc, resized.data(), &resizedDesc); });
            add("memset_output_border_rows", [this]() { zeroBorders(); });
            add("rppt_crop_f32_batch_into_output_rows", [this]() { copyIntoOutput(); });
            add("rppt_multiply_scalar", [this]() { multiply(); });
            add("rppt_tensor_subtract_tensor", [this]() { subtract(); });
            add("rppt_tensor_divide_tensor", [this]() { divide(); });
            output = tensorA.data();
            break;
        case Variant::Fused:
            // The HOST rppt_resize_mirror_normalize zeroes nStride bytes (OUT_H / 4 rows) from the first interior row of
            // each destination plane, never past the plane, then writes the interior rows; Sub and Div modify the
            // border rows in place, so they are zeroed again
            add("memset_output_border_rows", [this]() { zeroBorders(); });
            add("rppt_resize_mirror_normalize_8u32f_batch_mul_into_output_rows", [this]() { resizeMirrorNormalize(true); });
            add("rppt_tensor_subtract_tensor", [this]() { subtract(); });
            add("rppt_tensor_divide_tensor", [this]() { divide(); });
            output = tensorA.data();
            break;
        case Variant::Normalize:
            // rppt_normalize writes tensorB: tensorA's border rows stay zero from its allocation
            add("rppt_crop_mirror_normalize_8u32f_batch", [this]() { cropConvert(); });
            add("rppt_resize_batch_into_output_rows",
                [this]() { resizeInto(crops, &cropDesc, tensorA.data(), &outInteriorDesc); });
            add("rppt_normalize_mul_sub_div", [this]() { normalize(); });
            output = tensorB.data();
            break;
        case Variant::FusedNormalize:
            // As rpp_normalize: tensorA's border rows stay zero from its allocation
            add("rppt_resize_mirror_normalize_8u32f_batch_into_output_rows", [this]() { resizeMirrorNormalize(false); });
            add("rppt_normalize_mul_sub_div", [this]() { normalize(); });
            output = tensorB.data();
            break;
        }
    }

    void run() {
        for (const auto& op : ops) {
            op.second();
        }
    }

    std::vector<float> outputValues() const { return std::vector<float>(output, output + n * OUT_FLOATS); }
};

// ------------------------------------------------------------------------------------------------
// FKL CPU backends, BVF: one executeOperations call, on one thread (ParArch::CPU) or on FKL's thread pool
// (ParArch::CPU_MULTITHREAD)
// ------------------------------------------------------------------------------------------------
template <size_t N, fk::ParArch PA = fk::ParArch::CPU>
struct FklCpu {
    const Workload& w;
    std::array<fk::Rect, N> crops{};
    fk::Tensor<fk::float3> out{OUT_W, OUT_H, N, 1, fk::MemType::Host};
    fk::Stream_<PA> stream;

    explicit FklCpu(const Workload& workload) : w(workload) { std::copy_n(w.crops.begin(), N, crops.begin()); }

    void run() {
        fk::executeOperations<fk::TransformDPP<PA>>(stream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*w.frame),
            YUVToRGB::build(),
            fk::Crop<>::build(crops),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, fk::float3{0.f, 0.f, 0.f}),
            fk::Mul<fk::float3>::build(fk::float3{MUL[0], MUL[1], MUL[2]}),
            fk::Sub<fk::float3>::build(fk::float3{SUB[0], SUB[1], SUB[2]}),
            fk::Div<fk::float3>::build(fk::float3{DIV[0], DIV[1], DIV[2]}),
            fk::TensorWrite<fk::float3>::build(out.ptr()));
        stream.sync();
    }

    std::vector<float> output() const {
        const float* data = reinterpret_cast<const float*>(out.ptr().data);
        return std::vector<float>(data, data + N * OUT_FLOATS);
    }
};

// ------------------------------------------------------------------------------------------------
// Sample positions of RPP HOST's bilinear resampling for the benchmark's geometry (CROP_W x CROP_H -> RES_W x RES_H),
// measured with linear ramps, as the paper measured NPP's: pixel centers, src = (dst + 0.5) * scale - 0.5. rppt_resize
// (F32) clamps them to [0, size - 2] except the last output columns: its AVX bilinear load, when the left neighbour
// is past size - 3, only replaces the right neighbour by the left one, so positions >= size - 1 read pixel size - 1.
// rppt_resize_mirror_normalize (8u -> 32f) clamps them to [0, size - 1]. Both are measured (--resize-convention).
// ------------------------------------------------------------------------------------------------
struct Sampling {
    std::vector<float> x, y; // source position of every output column and row
};

Sampling measureHostSampling(bool resizeMirrorNormalize) {
    rppHandle_t handle{};
    RPP_CHECK(static_cast<RppStatus>(rppCreate(&handle, 1, 1, nullptr, RppBackend::RPP_HOST_BACKEND)));
    const int pitch = alignUp(CROP_W, RPP_PIXEL_ALIGN);
    Sampling s{std::vector<float>(RES_W), std::vector<float>(RES_H)};
    for (const bool vertical : {false, true}) {
        // Ramp along one axis: the position (F32) or RAMP_STEP_8BIT * position (8 bit), both exact; the output divided
        // back is the position
        const float step = resizeMirrorNormalize ? RAMP_STEP_8BIT : 1.f;
        std::vector<float> inFBuffer(RPP_FRONT_SLACK_FLOATS + static_cast<size_t>(pitch) * CROP_H * 3 + RPP_SLACK_FLOATS, 0.f);
        std::vector<Rpp8u> inUBuffer(RPP_FRONT_SLACK_FLOATS + static_cast<size_t>(pitch) * CROP_H * 3 + RPP_SLACK_FLOATS, 0);
        float* inF = inFBuffer.data() + RPP_FRONT_SLACK_FLOATS;
        Rpp8u* inU = inUBuffer.data() + RPP_FRONT_SLACK_FLOATS;
        for (int y = 0; y < CROP_H; ++y)
            for (int x = 0; x < CROP_W; ++x)
                for (int c = 0; c < 3; ++c) {
                    const float v = step * (vertical ? y : x);
                    inF[(y * pitch + x) * 3 + c] = v;
                    inU[(y * pitch + x) * 3 + c] = static_cast<Rpp8u>(v);
                }
        std::vector<float> out(static_cast<size_t>(RES_W) * RES_H * 3 + RPP_SLACK_FLOATS, 0.f);
        RpptDesc src(1, 3, CROP_H, CROP_W, resizeMirrorNormalize ? RpptDataType::U8 : RpptDataType::F32, RpptLayout::NHWC,
                     0, RpptStrides{static_cast<Rpp32u>(pitch * CROP_H * 3), 1, static_cast<Rpp32u>(pitch * 3), 3});
        RpptDesc dst(1, 3, RES_H, RES_W, RpptDataType::F32, RpptLayout::NHWC);
        RpptROI roi;
        roi.xywhROI = {{0, 0}, CROP_W, CROP_H};
        RpptImagePatch size{RES_W, RES_H};
        if (resizeMirrorNormalize) {
            float mean[3] = {0.f, 0.f, 0.f}, stdDev[3] = {1.f, 1.f, 1.f};
            Rpp32u mirror[1] = {0};
            RPP_CHECK(rppt_resize_mirror_normalize(inU, &src, out.data(), &dst, &size, RpptInterpolationType::BILINEAR,
                                                   mean, stdDev, mirror, &roi, RpptRoiType::XYWH, handle,
                                                   RppBackend::RPP_HOST_BACKEND));
        } else {
            RPP_CHECK(rppt_resize(inF, &src, out.data(), &dst, &size, RpptInterpolationType::BILINEAR, &roi,
                                  RpptRoiType::XYWH, handle, RppBackend::RPP_HOST_BACKEND));
        }
        if (vertical) {
            for (int p = 0; p < RES_H; ++p) s.y[p] = out[(p * RES_W + 1) * 3] / step;
        } else {
            for (int p = 0; p < RES_W; ++p) s.x[p] = out[(1 * RES_W + p) * 3] / step;
        }
    }
    rppDestroy(handle, RppBackend::RPP_HOST_BACKEND);
    return s;
}

// ------------------------------------------------------------------------------------------------
// Independent host references: the same chain in plain C++ with FKL's conversion matrix.
//  - FKL conventions: float RGB, sample at src = dst * scale, clamped to [0, size - 1]
//  - RPP HOST conventions: RGB rounded to 8 bit (what FKL's conversion step writes), sample at the measured positions
//  - the paper's 8 bit reference: RGB rounded to 8 bit, FKL's sampling
// ------------------------------------------------------------------------------------------------
enum class Convention { FklFloat, Rpp8bit, Fkl8bit };

std::vector<float> hostReference(const Workload& w, size_t n, Convention conv, const Sampling* sampling = nullptr) {
    std::vector<float> out(n * OUT_FLOATS);
    const float sx = static_cast<float>(CROP_W) / RES_W;
    const float sy = static_cast<float>(CROP_H) / RES_H;
    const bool eightBit = conv != Convention::FklFloat;
    const bool rppSampling = conv == Convention::Rpp8bit;
    for (size_t i = 0; i < n; ++i) {
        const fk::Rect& r = w.crops[i];
        float crop[CROP_H][CROP_W][3];
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                const fk::uchar3 yuv = w.frame->readAt(fk::Point{static_cast<int>(r.x) + x, static_cast<int>(r.y) + y, 0});
                const fk::float3 rgb = YUVToRGB::exec(yuv);
                const float c3[3] = {rgb.x, rgb.y, rgb.z};
                for (int c = 0; c < 3; ++c) {
                    crop[y][x][c] = eightBit ? std::floor(std::clamp(c3[c], 0.f, 255.f) + 0.5f) : c3[c];
                }
            }
        }
        for (int y = 0; y < OUT_H; ++y) {
            for (int x = 0; x < OUT_W; ++x) {
                const int ry = y - BORDER_TOP;
                float v[3] = {0.f, 0.f, 0.f};
                if (ry >= 0 && ry < RES_H) {
                    float fx = rppSampling ? sampling->x[x] : std::clamp(x * sx, 0.f, static_cast<float>(CROP_W - 1));
                    float fy = rppSampling ? sampling->y[ry] : std::clamp(ry * sy, 0.f, static_cast<float>(CROP_H - 1));
                    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
                    const int x1 = std::min(x0 + 1, CROP_W - 1), y1 = std::min(y0 + 1, CROP_H - 1);
                    const float ax = fx - x0, ay = fy - y0;
                    for (int c = 0; c < 3; ++c) {
                        v[c] = (crop[y0][x0][c] * (1.f - ax) + crop[y0][x1][c] * ax) * (1.f - ay) +
                               (crop[y1][x0][c] * (1.f - ax) + crop[y1][x1][c] * ax) * ay;
                    }
                }
                float* dst = &out[((i * OUT_H + y) * OUT_W + x) * 3];
                for (int c = 0; c < 3; ++c) {
                    dst[c] = ((v[c] * MUL[c]) - SUB[c]) / DIV[c];
                }
            }
        }
    }
    return out;
}

float maxAbsError(const std::vector<float>& a, const std::vector<float>& b) {
    float e = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::abs(a[i] - b[i]);
        e = std::isfinite(d) ? std::max(e, d) : INFINITY;
    }
    return e;
}

// ------------------------------------------------------------------------------------------------
// Two-execution timing in FKL's format: in every iteration RPP and then FKL, each timed with steady_clock after an
// idle gap
// ------------------------------------------------------------------------------------------------
struct Stats {
    double mean{0.}, variance{0.}, max{0.}, min{0.}, median{0.}, p10{0.}, p90{0.};
};

double percentile(const std::vector<double>& sorted, double p) {
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

Stats stats(std::vector<double> ms) {
    Stats s;
    double sum = 0.;
    for (const double v : ms) sum += v;
    s.mean = sum / ms.size();
    double sq = 0.;
    for (const double v : ms) sq += (v - s.mean) * (v - s.mean);
    s.variance = ms.size() > 1 ? sq / (ms.size() - 1) : 0.;
    std::sort(ms.begin(), ms.end());
    s.min = ms.front();
    s.max = ms.back();
    s.median = percentile(ms, 0.5);
    s.p10 = percentile(ms, 0.1);
    s.p90 = percentile(ms, 0.9);
    return s;
}

struct Row {
    size_t batch;
    Stats rpp, fkl;
    double meanSpeedup;
};

// FKL's pool workers spin FKL_POOL_SPIN after their last chunk before they sleep (fkl/thread-pool/include/ThreadPool.h).
// The RPP all-threads pipeline starts with FKL's conversion on that pool, then RPP's OpenMP team runs on the same
// hardware threads: the pool's spin is turned off before every timed run of it, and restored to FKL's default before
// every timed run of FKL, both outside of the timing (before the idle gap)
constexpr auto FKL_POOL_SPIN = std::chrono::microseconds(100);
void fklPoolSpin(bool on) {
    fk::cpuThreadPool().setSpinTime(on ? std::chrono::nanoseconds(FKL_POOL_SPIN) : std::chrono::nanoseconds(0));
}

Row timeTwo(size_t batch, const std::function<void()>& rpp, const std::function<void()>& fkl,
            const std::function<void()>& beforeRpp = {}, const std::function<void()>& beforeFkl = {}) {
    using Clock = std::chrono::steady_clock;
    const auto time = [](const std::function<void()>& f, const std::function<void()>& before) {
        if (before) {
            before();
        }
        std::this_thread::sleep_for(IDLE_GAP);
        const auto t0 = Clock::now();
        f();
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        time(rpp, beforeRpp);
        time(fkl, beforeFkl);
    }
    std::vector<double> a(ITERS), b(ITERS);
    double speedup = 0.;
    for (int i = 0; i < ITERS; ++i) {
        a[i] = time(rpp, beforeRpp);
        b[i] = time(fkl, beforeFkl);
        speedup += a[i] / b[i];
    }
    return {batch, stats(a), stats(b), speedup / ITERS};
}

void writeCSV(const std::filesystem::path& path, const std::string& rppLabel, const std::vector<Row>& rows,
              const std::string& fklLabel) {
    std::ofstream f(path);
    f << "Batch size";
    for (const std::string& label : {rppLabel, fklLabel}) {
        f << ", " << label << " MeanTime, " << label << " TimeVariance, " << label << " MaxTime, " << label << " MinTime";
    }
    f << ", Mean Speedup";
    for (const std::string& label : {rppLabel, fklLabel}) {
        f << ", " << label << " MedianTime, " << label << " P10Time, " << label << " P90Time";
    }
    f << std::endl;
    for (const auto& r : rows) {
        f << r.batch;
        for (const Stats* s : {&r.rpp, &r.fkl}) {
            f << ", " << s->mean << ", " << s->variance << ", " << s->max << ", " << s->min;
        }
        f << ", " << r.meanSpeedup;
        for (const Stats* s : {&r.rpp, &r.fkl}) {
            f << ", " << s->median << ", " << s->p10 << ", " << s->p90;
        }
        f << std::endl;
    }
}

// The two comparisons of fkl-vs-opencv-cpu, each with the same number of threads on both sides: RPP 1 thread vs FKL
// CPU, RPP all threads vs FKL CPU_MULTITHREAD, each with the RPP pipeline that the selection pass chose for that thread
// count
enum Pairing { ONE_THREAD_VS_CPU, ALL_THREADS_VS_MT, PAIRINGS };

struct Check {
    size_t batch;
    std::string implementation;
    unsigned threads;
    float vsOwnReference, vsRppReference, vsFklFloatReference, vsFkl8bitReference;
};

// ------------------------------------------------------------------------------------------------
// Selection of the RPP pipeline, for each thread count (1 and all hardware threads). Every RPP variant is validated
// against its RPP-convention reference, twice (no library state may change between calls), and compared with
// rpp_per_op's output; the candidates are the variants within RPP_EQUIVALENCE of both (numerically equivalent to the
// one-call-per-operation pipeline). The candidates are timed round-robin, each run after the idle gap,
// SELECTION_WARMUP + SELECTION_ITERS iterations per N; the one with the lowest geometric mean of its medians over all N
// is the RPP pipeline of that thread count, the only one in the measurements.
// ------------------------------------------------------------------------------------------------
constexpr int SELECTION_WARMUP = 10;
constexpr int SELECTION_ITERS = 100;

struct SelectionRow {
    unsigned threads;
    size_t batch;
    std::string variant;
    float vsOwnReference, vsPerOp;
    bool stable, candidate;
    Stats timing;
};

double timeOnce(const std::function<void()>& f, const std::function<void()>& before = {}) {
    if (before) {
        before();
    }
    std::this_thread::sleep_for(IDLE_GAP);
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

template <size_t N, fk::ParArch PA>
void selectionConfig(Workload& w, unsigned threads, bool timing, const std::vector<float>& refResize,
                     const std::vector<float>& refRmn, const std::vector<float>& refFloat,
                     const std::vector<float>& ref8bit, std::vector<SelectionRow>& rows, std::vector<Check>& checks) {
    std::vector<std::unique_ptr<RppHost<PA>>> pipelines;
    for (const Variant v : VARIANTS) {
        pipelines.push_back(std::make_unique<RppHost<PA>>(w, N, v, threads));
    }
    std::vector<float> perOp; // rpp_per_op's output
    std::vector<SelectionRow> local;
    for (size_t k = 0; k < VARIANTS.size(); ++k) {
        const bool rmn = VARIANTS[k] == Variant::Fused || VARIANTS[k] == Variant::FusedNormalize;
        const auto& refOwn = rmn ? refRmn : refResize;
        pipelines[k]->run();
        const auto firstOut = pipelines[k]->outputValues();
        const float first = maxAbsError(firstOut, refOwn);
        pipelines[k]->run();
        const auto out = pipelines[k]->outputValues();
        const float second = maxAbsError(out, refOwn);
        // Same output twice, value by value (no library state may change from one call to the next)
        const bool sameOutput = out == firstOut;
        if (k == 0) {
            perOp = out;
        }
        SelectionRow r{threads, N, variantName(VARIANTS[k]), second, maxAbsError(out, perOp), sameOutput, false, {}};
        r.candidate = r.stable && r.vsOwnReference <= RPP_EQUIVALENCE && r.vsPerOp <= RPP_EQUIVALENCE;
        // vs own reference (rppt_resize's or rppt_resize_mirror_normalize's convention), vs the RPP-convention reference
        // of rppt_resize (the column of cpu_validation.csv), vs FKL's float and 8 bit references
        checks.push_back({N, r.variant, threads, second, maxAbsError(out, refResize), maxAbsError(out, refFloat),
                          maxAbsError(out, ref8bit)});
        std::printf("N %3zu | %-20s %2u thread(s): max err vs its RPP-convention reference %.2e (first run %.2e), vs "
                    "rpp_per_op %.2e, vs paper's 8 bit reference %.2e%s\n", N, r.variant.c_str(), threads, second, first,
                    r.vsPerOp, checks.back().vsFkl8bitReference, r.candidate ? "" : "  (not a candidate)");
        local.push_back(r);
    }
    if (timing) {
        std::vector<std::vector<double>> ms(VARIANTS.size());
        for (int i = 0; i < SELECTION_WARMUP + SELECTION_ITERS; ++i) {
            for (size_t k = 0; k < VARIANTS.size(); ++k) {
                if (local[k].candidate) {
                    const double t = timeOnce([&]() { pipelines[k]->run(); }, [] {
                        if constexpr (PA == fk::ParArch::CPU_MULTITHREAD) {
                            fklPoolSpin(false);
                        }
                    });
                    if (i >= SELECTION_WARMUP) {
                        ms[k].push_back(t);
                    }
                }
            }
        }
        fklPoolSpin(true);
        std::printf("N %3zu | RPP selection, %u thread(s):", N, threads);
        for (size_t k = 0; k < VARIANTS.size(); ++k) {
            if (local[k].candidate) {
                local[k].timing = stats(ms[k]);
                std::printf("  %s %.1f us", local[k].variant.c_str(), local[k].timing.median * 1000.);
            }
        }
        std::printf("\n");
    }
    rows.insert(rows.end(), local.begin(), local.end());
}

template <size_t N>
void selectionBatch(Workload& w, unsigned hardwareThreads, bool timing, const Sampling& resizeSampling,
                    const Sampling& rmnSampling, std::vector<SelectionRow>& rows, std::vector<Check>& checks) {
    const auto refRppResize = hostReference(w, N, Convention::Rpp8bit, &resizeSampling);
    const auto refRppRmn = hostReference(w, N, Convention::Rpp8bit, &rmnSampling);
    const auto refFloat = hostReference(w, N, Convention::FklFloat);
    const auto ref8bit = hostReference(w, N, Convention::Fkl8bit);
    FklCpu<N> fkl(w);
    FklCpu<N, fk::ParArch::CPU_MULTITHREAD> fklMT(w);
    fkl.run();
    fklMT.run();
    const auto fklOut = fkl.output();
    checks.push_back({N, "fkl_cpu", 1, maxAbsError(fklOut, refFloat), maxAbsError(fklOut, refRppResize),
                      maxAbsError(fklOut, refFloat), maxAbsError(fklOut, ref8bit)});
    const auto fklMTOut = fklMT.output();
    checks.push_back({N, "fkl_cpu_multithread", hardwareThreads, maxAbsError(fklMTOut, refFloat),
                      maxAbsError(fklMTOut, refRppResize), maxAbsError(fklMTOut, refFloat), maxAbsError(fklMTOut, ref8bit)});
    std::printf("N %3zu | FKL CPU max err vs FKL-convention reference %.2e, FKL CPU_MULTITHREAD %.2e, CPU_MULTITHREAD vs "
                "CPU %.2e\n", N, checks[checks.size() - 2].vsOwnReference, checks.back().vsOwnReference,
                maxAbsError(fklMTOut, fklOut));
    selectionConfig<N, fk::ParArch::CPU>(w, 1, timing, refRppResize, refRppRmn, refFloat, ref8bit, rows, checks);
    selectionConfig<N, fk::ParArch::CPU_MULTITHREAD>(w, hardwareThreads, timing, refRppResize, refRppRmn, refFloat,
                                                     ref8bit, rows, checks);
}

// Whether some RPP variant is a candidate (numerically equivalent to rpp_per_op and to its reference) at every N
bool hasRppCandidate(const std::vector<SelectionRow>& rows, unsigned threads) {
    for (const Variant v : VARIANTS) {
        bool seen = false, candidate = true;
        for (const auto& r : rows) {
            if (r.threads == threads && r.variant == variantName(v)) {
                seen = true;
                candidate = candidate && r.candidate;
            }
        }
        if (seen && candidate) {
            return true;
        }
    }
    return false;
}

Variant chooseRpp(const std::vector<SelectionRow>& rows, unsigned threads) {
    int best = -1;
    double bestScore = INFINITY;
    for (size_t k = 0; k < VARIANTS.size(); ++k) {
        double score = 0.;
        int count = 0;
        bool candidate = true;
        for (const auto& r : rows) {
            if (r.threads == threads && r.variant == variantName(VARIANTS[k])) {
                candidate = candidate && r.candidate;
                score += std::log(r.timing.median);
                ++count;
            }
        }
        if (count > 0 && candidate && score < bestScore) {
            bestScore = score;
            best = static_cast<int>(k);
        }
    }
    if (best < 0) {
        throw std::runtime_error("No RPP HOST variant is numerically equivalent to rpp_per_op");
    }
    std::printf("RPP pipeline with %u thread(s): %s\n", threads, variantName(VARIANTS[best]).c_str());
    return VARIANTS[best];
}

// The two comparisons for batch N, with the chosen RPP pipelines
template <size_t N>
void runBatch(Workload& w, std::array<std::vector<Row>, PAIRINGS>& rows, unsigned hardwareThreads, Variant oneThread,
              Variant allThreads) {
    FklCpu<N> fkl(w);
    FklCpu<N, fk::ParArch::CPU_MULTITHREAD> fklMT(w);
    RppHost<fk::ParArch::CPU> rpp1(w, N, oneThread, 1);
    RppHost<fk::ParArch::CPU_MULTITHREAD> rppAll(w, N, allThreads, hardwareThreads);
    rows[ONE_THREAD_VS_CPU].push_back(timeTwo(N, [&]() { rpp1.run(); }, [&]() { fkl.run(); }));
    rows[ALL_THREADS_VS_MT].push_back(timeTwo(N, [&]() { rppAll.run(); }, [&]() { fklMT.run(); },
                                              [] { fklPoolSpin(false); }, [] { fklPoolSpin(true); }));
    const Row& a = rows[ONE_THREAD_VS_CPU].back();
    const Row& c = rows[ALL_THREADS_VS_MT].back();
    std::printf("N %3zu | RPP 1 thread %9.1f us vs FKL CPU %8.1f us (BVF mean speedup %6.2fx) | RPP %u threads %9.1f us vs "
                "FKL CPU_MULTITHREAD %8.1f us (%6.2fx)\n", N, a.rpp.median * 1000., a.fkl.median * 1000., a.meanSpeedup,
                hardwareThreads, c.rpp.median * 1000., c.fkl.median * 1000., c.meanSpeedup);
}

using bench::shell;

// rppt_resize HOST sampling convention, from linear ramps (value = position, exact), as for the HIP backend
void resizeConvention(const std::filesystem::path& outDir) {
    std::ofstream csv(outDir / "rpp_host_resize_convention.csv");
    csv << "axis,src,dst,scale,dst_index,measured_src_position,s_times_p,pixel_center,pixel_center_clamped_to_0_size_minus_2\n";
    rppHandle_t handle{};
    RPP_CHECK(static_cast<RppStatus>(rppCreate(&handle, 1, 1, nullptr, RppBackend::RPP_HOST_BACKEND)));
    struct Case { bool vertical; int src, dst; };
    // The benchmark's crop -> resize in both axes, then other ratios
    for (const Case c : {Case{false, CROP_W, RES_W}, Case{true, CROP_H, RES_H}, Case{false, 20, 64}, Case{false, 10, 32},
                         Case{false, 8, 40}, Case{true, 20, 64}}) {
        const int sw = c.vertical ? 8 : c.src, sh = c.vertical ? c.src : 2;
        const int dw = c.vertical ? 8 : c.dst, dh = c.vertical ? c.dst : 2;
        std::vector<float> inBuffer(RPP_FRONT_SLACK_FLOATS + static_cast<size_t>(sw) * sh * 3 + RPP_SLACK_FLOATS);
        std::vector<float> out(static_cast<size_t>(dw) * dh * 3 + RPP_SLACK_FLOATS);
        float* in = inBuffer.data() + RPP_FRONT_SLACK_FLOATS;
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < sw; ++x)
                for (int ch = 0; ch < 3; ++ch) in[(y * sw + x) * 3 + ch] = static_cast<float>(c.vertical ? y : x);
        RpptDesc s(1, 3, sh, sw, RpptDataType::F32, RpptLayout::NHWC), d(1, 3, dh, dw, RpptDataType::F32, RpptLayout::NHWC);
        RpptROI roi;
        roi.xywhROI = {{0, 0}, sw, sh};
        RpptImagePatch size{static_cast<Rpp32u>(dw), static_cast<Rpp32u>(dh)};
        RPP_CHECK(rppt_resize(in, &s, out.data(), &d, &size, RpptInterpolationType::BILINEAR, &roi,
                              RpptRoiType::XYWH, handle, RppBackend::RPP_HOST_BACKEND));
        const float scale = static_cast<float>(c.src) / c.dst;
        double maxDev = 0.;
        for (int p = 0; p < c.dst; ++p) {
            const float measured = c.vertical ? out[(p * dw + 1) * 3] : out[(1 * dw + p) * 3];
            const float center = scale * (p + 0.5f) - 0.5f;
            const float clamped = std::min(static_cast<float>(c.src - 2), std::max(0.f, center));
            maxDev = std::max(maxDev, static_cast<double>(std::abs(measured - clamped)));
            csv << (c.vertical ? "y" : "x") << "," << c.src << "," << c.dst << "," << scale << "," << p << "," << measured
                << "," << scale * p << "," << center << "," << clamped << "\n";
        }
        std::printf("HOST %s %2d -> %2d: max |measured - clamp(s*(p+0.5)-0.5, 0, size-2)| = %.2e\n",
                    c.vertical ? "vertical  " : "horizontal", c.src, c.dst, maxDev);
    }
    rppDestroy(handle, RppBackend::RPP_HOST_BACKEND);
}

} // namespace rppcpu

int main(int argc, char** argv) {
    using namespace rppcpu;
    // RPP's HOST backend uses OpenMP (libomp), whose workers spin 200 ms after every parallel region by default: they
    // would still be spinning after the idle gap. 100 us, like FKL's thread pool. Set before any OpenMP call;
    // run_all.sh exports it too.
    setenv("KMP_BLOCKTIME", "100us", 0);
    try {
        if (argc == 2 && std::string(argv[1]) == "--batches") {
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
        const std::filesystem::path outDir = argc > 1 ? argv[1] : "results/rocm_cpu";
        std::filesystem::create_directories(outDir);
        const bool validateOnly = argc > 2 && std::string(argv[2]) == "--validate";
        if (argc > 2 && std::string(argv[2]) == "--resize-convention") {
            resizeConvention(outDir);
            return 0;
        }
        if (argc > 2 && !validateOnly) {
            throw std::runtime_error(std::string("Unknown mode ") + argv[2] + " (--validate or --resize-convention)");
        }
        // The all-threads runs use as many threads as FKL's thread pool has workers, on both sides, as fkl-vs-opencv-cpu
        // does (it can differ from hardware_concurrency() under an affinity mask or a CPU quota)
        const unsigned hardwareThreads = static_cast<unsigned>(fk::cpuThreadPool().size());

        Workload w;
        initWorkload(w);
        const Sampling resizeSampling = measureHostSampling(false);
        const Sampling rmnSampling = measureHostSampling(true);
        {
            std::ofstream os(outDir / "rpp_host_sampling_positions.csv");
            os << "axis,dst_index,rppt_resize_src_position,rppt_resize_mirror_normalize_src_position,pixel_center,"
                  "pixel_center_clamped_to_0_size_minus_2\n";
            for (int p = 0; p < RES_W + RES_H; ++p) {
                const bool vertical = p >= RES_W;
                const int i = vertical ? p - RES_W : p;
                const float scale = vertical ? static_cast<float>(CROP_H) / RES_H : static_cast<float>(CROP_W) / RES_W;
                const float center = (i + 0.5f) * scale - 0.5f;
                os << (vertical ? "y" : "x") << "," << i << "," << (vertical ? resizeSampling.y[i] : resizeSampling.x[i])
                   << "," << (vertical ? rmnSampling.y[i] : rmnSampling.x[i]) << "," << center << ","
                   << std::clamp(center, 0.f, static_cast<float>((vertical ? CROP_H : CROP_W) - 2)) << "\n";
            }
        }
        std::vector<SelectionRow> selection;
        std::vector<Check> checks;
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (selectionBatch<BATCHES[Is]>(w, hardwareThreads, !validateOnly, resizeSampling, rmnSampling, selection, checks),
             ...);
        }(std::make_index_sequence<BATCHES.size()>{});

        std::ofstream v(outDir / "cpu_validation.csv");
        v << "Batch size, Implementation, Threads, MaxAbsError vs own reference, MaxAbsError vs RPP-convention reference, "
             "MaxAbsError vs FKL-convention float reference, MaxAbsError vs paper's 8 bit reference (FKL sampling)"
          << std::endl;
        for (const auto& c : checks) {
            v << c.batch << ", " << c.implementation << ", " << c.threads << ", " << c.vsOwnReference << ", "
              << c.vsRppReference << ", " << c.vsFklFloatReference << ", " << c.vsFkl8bitReference << std::endl;
        }
        // FKL's outputs must be within TOLERANCE of the FKL float reference (common/bench_validation.h), and every
        // thread count needs an RPP pipeline numerically equivalent to rpp_per_op; the RPP outputs are reported
        bool failed = false;
        for (const auto& c : checks) {
            if (c.implementation.rfind("fkl_", 0) == 0 && !(c.vsOwnReference <= bench::validation::TOLERANCE)) {
                std::cerr << "ERROR: " << c.implementation << " at N = " << c.batch << ": max abs error " << c.vsOwnReference
                          << " against the FKL float reference, more than " << bench::validation::TOLERANCE << std::endl;
                failed = true;
            }
        }
        for (const unsigned t : {1u, hardwareThreads}) {
            if (!hasRppCandidate(selection, t)) {
                std::cerr << "ERROR: no RPP HOST variant is numerically equivalent to rpp_per_op at every N with " << t
                          << " thread(s)" << std::endl;
                failed = true;
            }
        }
        if (failed) {
            return 1;
        }
        if (validateOnly) {
            return 0;
        }
        const Variant oneThread = chooseRpp(selection, 1);
        const Variant allThreads = chooseRpp(selection, hardwareThreads);
        {
            std::ofstream sel(outDir / "cpu_rpp_selection.csv");
            sel << "threads,N,variant,max_abs_err_vs_its_rpp_reference,max_abs_err_vs_rpp_per_op,same_output_twice,"
                   "candidate,selection_median_ms,selection_p10_ms,selection_p90_ms,chosen,selection_iterations\n";
            for (const auto& r : selection) {
                const bool chosen = r.variant == variantName(r.threads == 1 ? oneThread : allThreads);
                sel << r.threads << "," << r.batch << "," << r.variant << "," << r.vsOwnReference << "," << r.vsPerOp << ","
                    << (r.stable ? "yes" : "no") << "," << (r.candidate ? "yes" : "no") << ",";
                if (r.candidate) {
                    sel << r.timing.median << "," << r.timing.p10 << "," << r.timing.p90;
                } else {
                    sel << ",,";
                }
                sel << "," << (chosen ? "yes" : "no") << "," << SELECTION_ITERS << "\n";
            }
        }
        std::array<std::vector<Row>, PAIRINGS> rows;
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (runBatch<BATCHES[Is]>(w, rows, hardwareThreads, oneThread, allThreads), ...);
        }(std::make_index_sequence<BATCHES.size()>{});
        const std::string all = std::to_string(hardwareThreads) + "threads";
        writeCSV(outDir / "benchmark_cpu_rpp_1thread_vs_fkl_bvf_CPU.csv", "RPP HOST 1 thread", rows[ONE_THREAD_VS_CPU],
                 "FKL CPU");
        writeCSV(outDir / ("benchmark_cpu_rpp_" + all + "_vs_fkl_bvf_CPU_MULTITHREAD.csv"), "RPP HOST " + all,
                 rows[ALL_THREADS_VS_MT], "FKL CPU_MULTITHREAD");
        std::ofstream plat(outDir / "cpu_platform.txt");
        plat << "CPU: " << bench::cpuName() << ", " << std::thread::hardware_concurrency() << " hardware threads\n"
             << "SIMD of the CPU: "
             << shell("grep -o -w -E 'avx|avx2|fma|avx512f|avx512bw|avx512vl' /proc/cpuinfo | sort -u | paste -sd' '") << "\n"
             << "Environment: " << (shell("grep -ci microsoft /proc/version 2>/dev/null") != "0" ? "WSL2" : "bare metal")
             << "\n"
             << "FKL: commit " << BENCH_FKL_COMMIT << ", CPU backend (single thread) and CPU_MULTITHREAD backend (thread "
             << "pool of " << fk::cpuThreadPool().size() << " workers, thread-pool " << BENCH_THREAD_POOL_COMMIT
             << ", one chunk per output row of each crop), compiled with " << BENCH_COMPILER << ", flags "
             << BENCH_CXX_FLAGS << "\n"
             << "RPP: " << RPP_VERSION_MAJOR << "." << RPP_VERSION_MINOR << "." << RPP_VERSION_PATCH
             << " HOST backend, as packaged (" << BENCH_RPP_LIB << "; ROCm 10.1 package amdrocm-rpp10.1, prebuilt by "
             << "AMD, OpenMP over the batch images), numThreads 1 and " << hardwareThreads << " (rppCreate; RPP's "
             << "default, numThreads 0, is the handle's batch size); OpenMP workers spin KMP_BLOCKTIME = "
             << (std::getenv("KMP_BLOCKTIME") ? std::getenv("KMP_BLOCKTIME") : "libomp default (200 ms)")
             << " after a parallel region (set like FKL's thread pool, which spins 100 us), so that the idle gap "
             << "before every timed run puts them to sleep; handles created with batch size 1 (the "
             << "HOST kernels iterate over the descriptors' n; the handle's batch size only sizes an unused scratch "
             << "buffer of 99532800 floats per image)\n"
             << "OS: " << bench::osName() << ", kernel " << bench::kernelVersion() << "\n"
             << "Workload: " << describe() << "\n"
             << "Pipeline: as the GPU benchmark, BT.601 limited range NV12 -> RGB. RPP's NV12 conversion (rppt_yuv_to_rgb) is "
                "HIP only, so the RPP HOST pipelines get their 8 bit RGB frame from FKL's CPU conversion (round to "
                "nearest after saturation): ParArch::CPU for the 1 thread runs, ParArch::CPU_MULTITHREAD for the "
             << hardwareThreads << " threads runs. RPP pipeline: " << variantName(oneThread) << " with 1 thread, "
             << variantName(allThreads) << " with " << hardwareThreads << " threads, the fastest RPP variant numerically "
             << "equivalent (within " << RPP_EQUIVALENCE << ") to rpp_per_op and to its RPP-convention reference, chosen "
             << "by a selection pass (every candidate timed round-robin, " << SELECTION_WARMUP << " warm-up + "
             << SELECTION_ITERS << " iterations per N, lowest geometric mean of the medians; cpu_rpp_selection.csv); only "
             << "that pipeline is measured afterwards. "
             << "Variants: rpp_per_op (crop_mirror_normalize 8u->32f, memset of the output's border rows, resize into "
                "the output rows, multiply_scalar, tensor_subtract_tensor, tensor_divide_tensor), rpp_npp_like (resized "
                "crops in their own buffer, copied by crop), rpp_fused (memset of the border rows, resize_mirror_normalize "
                "with Mul, Sub, Div), rpp_normalize (rpp_per_op with Mul + Sub + Div in one normalize, out of place, "
                "without the memset: its input's border rows stay zero), rpp_fused_normalize (resize_mirror_normalize, "
                "normalize, without the memset). RPP HOST behaviour handled in the pipelines: rppt_normalize "
                "replaces the caller's stdDev values by scale/stdDev (restored before every call, inside the timing) "
                "and zeroes its destination before reading its source (run out of place); rppt_resize_mirror_normalize "
                "zeroes nStride bytes (a quarter of the plane) from the first interior row of each float destination "
                "plane, never past the plane: harmless, the interior rows are written afterwards and the border rows "
                "are zero. "
                "FKL's thread pool does not spin after the conversion of the RPP all-threads pipeline (setSpinTime(0) "
                "before every timed run of it, outside of the timing; FKL's default 100 us before FKL's runs), so that its "
                "workers do not compete with RPP's OpenMP team. "
                "rppt_crop_mirror_normalize 8u->32f writes one float (1.0) past the end of every row when the ROI width "
                "is a multiple of 16: every crop in the crop buffer is followed by a spare row, so that this float never "
                "lands on the next crop, which another OpenMP thread may have written already; rppt_resize reads 12 bytes "
                "before its source and rppt_resize_mirror_normalize up to 3 bytes before and 13 after the rows of its "
                "ROIs: the crop buffer starts after some slack, and the crops are at least " << CROP_MARGIN << " pixels "
                "from the frame edges. "
                "multiply_scalar, tensor_subtract_tensor, tensor_divide_tensor and normalize see the output as N samples "
                "(RPP's HOST backend parallelizes over the samples), so at N = 1 they, like every RPP HOST call, run "
                "on one thread. "
                "RPP-convention reference: RGB rounded to 8 bit, bilinear sampling at the positions RPP HOST uses, "
                "measured with linear ramps for this geometry (rpp_host_sampling_positions.csv; 8 bit ramps of step "
             << RAMP_STEP_8BIT << " per pixel for rppt_resize_mirror_normalize): pixel centers, "
                "rppt_resize clamps them to [0, size - 2] except the last output columns (its AVX bilinear load only "
                "replaces the right neighbour when the left one is past size - 3, so positions >= size - 1 read pixel "
                "size - 1), rppt_resize_mirror_normalize clamps them to [0, size - 1]; "
                "so the rppt_resize-based variants (rpp_per_op, rpp_npp_like, rpp_normalize) and the "
                "rppt_resize_mirror_normalize-based ones (rpp_fused, rpp_fused_normalize) differ at the last columns and "
                "rows, each matching its own measured convention; the latter are therefore not numerically equivalent "
                "to rpp_per_op and not candidates of the selection.\n"
             << "Timing: steady_clock around each pipeline, " << WARMUP_ITERS << " warm-up and " << ITERS
             << " timed iterations, RPP and FKL one after the other in each, each run preceded by a "
             << std::chrono::milliseconds(IDLE_GAP).count()
             << " ms idle gap so that both thread pools are asleep; speedup computed per iteration and averaged.\n";
        std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
