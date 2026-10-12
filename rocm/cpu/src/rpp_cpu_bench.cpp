// CPU benchmark of the Figure 1 pipeline: RPP's HOST backend vs FKL's CPU backends (BVF), the ROCm counterpart of
// fkl-vs-opencv-cpu/src/benchmark_opencv_cpu.cpp (which compares with OpenCV).
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// The workload of common/bench_geometry.h: one 3840x2160 NV12 frame in host memory and N regions per iteration, new
// every iteration (the paper's build: 20x10, 32x16 and 48x24 by turns), each resized (bilinear) to 64x32 and given
// black rows on top and bottom, then (x * mul - sub) / div written as float3 into an N-plane tensor. BT.601 limited
// range, as on the GPU. Iteration t of every loop runs the first N regions of iteration t
// (bench::geometry::RegionSets), all generated before the first loop; every pipeline converts the regions of every
// iteration to its own argument format when it is created, so that inside the timing it only receives those of t:
// FKL builds the chain from them (Crop<>::build, the rest of the build and the fusion, as for any call), RPP points the
// ROI arguments of its calls at the ROI arrays of t.
//
//   rpp*:   RPP has no NV12 conversion on its HOST backend (rppt_yuv_to_rgb is HIP only), so every RPP HOST pipeline
//           starts with FKL's CPU conversion of the full frame to 8 bit RGB (the first kernel of the GPU benchmark's
//           FKL VF+HF: normalize, add half a level, saturate, denormalize to uchar3), on one thread (ParArch::CPU) for
//           the 1 thread runs and on FKL's thread pool (ParArch::CPU_MULTITHREAD) for the all threads runs. Then the
//           same RPP variants as the GPU benchmark (rocm/src/rocm_bench.hip), with RPP_HOST_BACKEND:
//             rpp                 rppt_crop_mirror_normalize 8u -> 32f (one batched call, all batch images aliasing
//                                 the RGB frame, into a crop buffer of one slot of the largest region size per region),
//                                 memset of the border rows of the output, rppt_resize into the rows of the output
//                                 planes below the top border, rppt_multiply_scalar, rppt_tensor_subtract_tensor,
//                                 rppt_tensor_divide_tensor
//             rpp_npp_like        the resized regions in their own buffer, copied into the output rows by rppt_crop
//             rpp_fused           memset of the border rows, rppt_resize_mirror_normalize (crop + 8u -> 32f + resize
//                                 + Mul), Sub, Div
//             rpp_normalize       rpp with Mul + Sub + Div in one rppt_normalize (out of place), without the memset
//             rpp_fused_normalize rppt_resize_mirror_normalize (crop + 8u -> 32f + resize), rppt_normalize
//           Every region has its own ROI (its size and, in the frame, its position); the crop buffer's descriptor has
//           the largest region size. Only the border rows need zeroing (every pipeline writes all the interior
//           rows), and only where the pipeline modifies them in place: the normalize variants never write the border
//           rows of their input tensor, which is zeroed once when it is allocated.
//           RPP's HOST backend parallelizes over the batch images with OpenMP; it is timed with 1 thread and with all
//           the hardware threads (numThreads of rppCreate).
// Two comparisons, each with the same number of threads on both sides, as in fkl-vs-opencv-cpu: RPP with 1 thread
// against fkl, and RPP with all the hardware threads against fkl mt.
//   fkl:    one executeOperations<TransformDPP<ParArch::CPU>> call with the whole chain (BVF), on one thread.
//   fkl mt: the same call with TransformDPP<ParArch::CPU_MULTITHREAD> on FKL's thread pool (one worker per hardware
//           thread), in chunks of one output row of one region.
//
// Both libraries' thread pools spin for a while after finishing before their workers go to sleep, so every timed run
// is preceded by an idle gap that lets them fall asleep, as in fkl-vs-opencv-cpu/src/benchmark_opencv_cpu.cpp.
// Before the timing of every N, the timed pipelines run the regions of the timed iterations that
// bench::validation::validatedIterations samples (the first, every 50th and the last), and their outputs are compared
// with host references of those regions: FKL must be within bench::validation::TOLERANCE of its float reference in
// every one of them.

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

// Frame, region, resize, border and output sizes, batch sizes, seeds and the regions of every iteration, shared by
// every benchmark
using namespace bench::geometry;
// BENCH_WARMUP_ITERATIONS and BENCH_ITERATIONS come from CMake: 100 and 1000 for the paper's results. Warm-up iteration
// i runs the regions of iteration i, timed iteration i those of iteration WARMUP_ITERS + i.
constexpr int WARMUP_ITERS = BENCH_WARMUP_ITERATIONS;
constexpr int ITERS = BENCH_ITERATIONS;
static_assert(WARMUP_ITERS >= 0 && ITERS >= 1, "at least one timed iteration");
constexpr auto IDLE_GAP = std::chrono::milliseconds(1);
// Selection pass of the RPP pipeline (selectionConfig): the candidates are timed round-robin, SELECTION_WARMUP +
// SELECTION_ITERS iterations per N, iteration i with the regions of iteration i
constexpr int SELECTION_WARMUP = 10;
constexpr int SELECTION_ITERS = 100;
// Stability check of the selection pass: the regions of iterations A, B, C and A again, three consecutive iterations,
// so that before the second run of A every slot of the crop buffer has held a region of every size
constexpr std::array<size_t, 4> STABILITY_SEQUENCE{0, 1, 2, 0};
constexpr size_t stabilitySets() {
    size_t sets = 0;
    for (const size_t t : STABILITY_SEQUENCE) {
        sets = std::max(sets, t + 1);
    }
    return sets;
}
// The iterations whose regions are generated (RegionSets), before anything runs: enough for the timed passes (warm-up
// and timed iterations, the validated ones among them), the selection pass and its stability check
constexpr size_t REGION_SET_COUNT = std::max({static_cast<size_t>(WARMUP_ITERS + ITERS),
                                              static_cast<size_t>(SELECTION_WARMUP + SELECTION_ITERS),
                                              stabilitySets()});

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
// constant 1.0, from its 48-float AVX store) when the ROI width is a multiple of 16, as with the 32 and 48 pixel wide
// regions. In a slot of the crop buffer (rows of CROP_PITCH = 48 pixels), a 32 pixel row's lands in the padding of
// the row, and a 48 pixel row's on the first float of the next row, which overwrites it; but the last row's would land
// on the first float of the next slot, which another OpenMP thread may already have written. So every slot of the crop
// buffer ends with a spare row.
constexpr int CROP_SPARE_ROWS = 1;
// The crop buffer of the RPP pipelines that crop before resizing: one slot per region, all of the largest region size
// (an RPP descriptor has one set of strides for the whole batch; the ROI of each image gives its size), rows padded to
// RPP_PIXEL_ALIGN pixels, each region at the top left of its slot, and the spare row
constexpr int CROP_PITCH = alignUp(MAX_REGION_W, RPP_PIXEL_ALIGN);
constexpr size_t CROP_SLOT_FLOATS = static_cast<size_t>(CROP_PITCH) * (MAX_REGION_H + CROP_SPARE_ROWS) * 3;
// RPP's HOST rppt_resize (F32) reads 12 bytes before its source and rppt_resize_mirror_normalize (8u) 3 bytes before
// the first row of its ROI: the buffers they read start after some slack. (On the RGB frame, the regions are at least
// REGION_MARGIN pixels from the edges.)
constexpr size_t RPP_FRONT_SLACK_FLOATS = 16;
// The longest region side of this build
constexpr int maxRegionSide() {
    int side = 0;
    for (const RegionSize s : REGION_SIZES) {
        side = std::max({side, s.width, s.height});
    }
    return side;
}
// The 8 bit ramps that measure rppt_resize_mirror_normalize's sampling positions: value = RAMP_STEP_8BIT * position,
// the largest power of two that keeps the longest region side within 255 (exact in float, and divided back exactly),
// the same for every region size (4 for the 48 pixel wide regions)
constexpr float rampStep8bit() {
    float step = 128.f;
    while (step > 1.f && step * static_cast<float>(maxRegionSide() - 1) > 255.f) {
        step /= 2.f;
    }
    return step;
}
constexpr float RAMP_STEP_8BIT = rampStep8bit();
static_assert(RAMP_STEP_8BIT * static_cast<float>(maxRegionSide() - 1) <= 255.f,
              "The 8 bit sampling ramps need region sides of at most 256 pixels");

using NV12Image = fk::Image<fk::PixelFormat::NV12>;
// FKL's NV12 -> RGB conversion: BT.601, limited range, as in every benchmark of the repository
using YUVToRGB = fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Limited, fk::ColorPrimitives::bt601>;

struct Workload {
    std::unique_ptr<NV12Image> frame;
    // The regions of iterations 0 to REGION_SET_COUNT - 1; batch N runs the first N of every iteration
    RegionSets sets{REGION_SET_COUNT};
};

void initWorkload(Workload& w) {
    w.frame = std::make_unique<NV12Image>(FRAME_W, FRAME_H, fk::MemType::Host);
    auto data = w.frame->getData();
    generateFrame([&](int x, int y, uint8_t value) { data.at(x, y) = value; });
}

// The index of a region's size in PAPER_REGION_SIZES
size_t paperSizeIndex(int width, int height) {
    for (size_t k = 0; k < PAPER_REGION_SIZES.size(); ++k) {
        if (PAPER_REGION_SIZES[k].width == width && PAPER_REGION_SIZES[k].height == height) {
            return k;
        }
    }
    throw std::runtime_error("A " + std::to_string(width) + "x" + std::to_string(height) +
                             " region is not of a paper size");
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
// RPP HOST pipelines. Buffers, descriptors and parameter tensors are created once, outside of the timing, and so are
// the ROIs of the regions of every iteration (XYWH, the format the HOST kernels use): RPP's HOST backend reads the ROIs
// on the host, never modifies them and caches nothing per handle, so run(t) only points the ROI arguments of its calls
// at the ROIs of iteration t. HOST handles are created with batch size 1: the HOST kernels iterate over the
// descriptors' n, and the handle's batch size only sizes a scratch buffer (99532800 floats per image, 39.8 GB at
// N = 100) that none of these calls use.
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

// The variants built on rppt_resize_mirror_normalize, whose sampling convention differs from rppt_resize's
bool usesResizeMirrorNormalize(Variant v) { return v == Variant::Fused || v == Variant::FusedNormalize; }

template <fk::ParArch PA>
struct RppHost {
    const Workload& w;
    size_t n;
    Variant variant;
    unsigned threads;
    rppHandle_t handle{};
    FklRgb8<PA> rgb8;
    int resizedPitch{alignUp(RES_W, RPP_PIXEL_ALIGN)};
    std::vector<float> cropped, resized, tensorA, tensorB;
    float* crops{nullptr}; // the first slot of the crop buffer, RPP_FRONT_SLACK_FLOATS into cropped
    float* output{nullptr};
    // The ROIs of every iteration, n per iteration: each region in the frame (crop_mirror_normalize and
    // resize_mirror_normalize) and in its slot of the crop buffer (resize)
    std::vector<RpptROI> allCropRois, allResizeRois;
    // The ROIs of the iteration being run, in the arrays above: set by run(t)
    RpptROI* cropRois{nullptr};
    RpptROI* resizeRois{nullptr};
    std::vector<RpptROI> resizedRois;
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
        cropped.assign(RPP_FRONT_SLACK_FLOATS + n * CROP_SLOT_FLOATS + RPP_SLACK_FLOATS, 0.f);
        crops = cropped.data() + RPP_FRONT_SLACK_FLOATS;
        resized.assign(n * resizedPitch * RES_H * 3 + RPP_SLACK_FLOATS, 0.f);
        tensorA.assign(n * OUT_FLOATS + BORDER_TOP * OUT_W * 3 + RPP_SLACK_FLOATS, 0.f);
        tensorB.assign(n * OUT_FLOATS + RPP_SLACK_FLOATS, 0.f);
        const size_t iterations = w.sets.iterations();
        allCropRois.resize(iterations * n);
        allResizeRois.resize(iterations * n);
        for (size_t t = 0; t < iterations; ++t) {
            for (size_t i = 0; i < n; ++i) {
                const Region& r = w.sets.at(t, i);
                allCropRois[t * n + i].xywhROI = {{r.x, r.y}, r.width, r.height};
                allResizeRois[t * n + i].xywhROI = {{0, 0}, r.width, r.height};
            }
        }
        setIteration(0);
        resizedRois.resize(n);
        dstSizes.resize(n);
        mirror.assign(n, 0);
        cmnOffset.assign(n * 3, 0.f);
        cmnMultiplier.assign(n * 3, 1.f);
        rmnMean.assign(n * 3, 0.f);
        rmnStdDevMul.resize(n * 3);
        rmnStdDevOne.assign(n * 3, 1.f);
        for (size_t i = 0; i < n; ++i) {
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
        // The crop buffer: N slots of the largest region size. Its h and w are the largest sizes: RPP's HOST rppt_resize
        // clips its ROIs to its source descriptor, without an error
        cropDesc = RpptDesc(N, 3, MAX_REGION_H, MAX_REGION_W, RpptDataType::F32, RpptLayout::NHWC, 0,
                            RpptStrides{static_cast<Rpp32u>(CROP_SLOT_FLOATS), 1, static_cast<Rpp32u>(CROP_PITCH * 3), 3});
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

    // The ROI arguments of the calls: the ROIs of iteration t
    void setIteration(size_t t) {
        cropRois = allCropRois.data() + t * n;
        resizeRois = allResizeRois.data() + t * n;
    }
    // Whether the ROIs of iteration t still hold its regions (RPP must not modify them: they are passed again)
    bool roisIntact(size_t t) const {
        for (size_t i = 0; i < n; ++i) {
            const Region& r = w.sets.at(t, i);
            const RpptRoiXywh& c = allCropRois[t * n + i].xywhROI;
            const RpptRoiXywh& s = allResizeRois[t * n + i].xywhROI;
            if (c.xy.x != r.x || c.xy.y != r.y || c.roiWidth != r.width || c.roiHeight != r.height || s.xy.x != 0 ||
                s.xy.y != 0 || s.roiWidth != r.width || s.roiHeight != r.height) {
                return false;
            }
        }
        return true;
    }

    void cropConvert() {
        RPP_CHECK(rppt_crop_mirror_normalize(rgb8.data(), &frameBatchDesc, crops, &cropDesc, cmnOffset.data(),
                                             cmnMultiplier.data(), mirror.data(), cropRois, RpptRoiType::XYWH, handle,
                                             RppBackend::RPP_HOST_BACKEND));
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
        RPP_CHECK(rppt_resize(src, srcDesc, dst, dstDesc, dstSizes.data(), RpptInterpolationType::BILINEAR, resizeRois,
                              RpptRoiType::XYWH, handle, RppBackend::RPP_HOST_BACKEND));
    }
    void copyIntoOutput() {
        RPP_CHECK(rppt_crop(resized.data(), &resizedDesc, tensorA.data(), &outInteriorDesc, resizedRois.data(),
                            RpptRoiType::XYWH, handle, RppBackend::RPP_HOST_BACKEND));
    }
    void resizeMirrorNormalize(bool mul) {
        RPP_CHECK(rppt_resize_mirror_normalize(rgb8.data(), &frameBatchDesc, tensorA.data(), &outInteriorDesc,
                                               dstSizes.data(), RpptInterpolationType::BILINEAR, rmnMean.data(),
                                               mul ? rmnStdDevMul.data() : rmnStdDevOne.data(), mirror.data(),
                                               cropRois, RpptRoiType::XYWH, handle, RppBackend::RPP_HOST_BACKEND));
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

    // Iteration t: the ROI arguments point at the ROIs of its regions, then the calls
    void run(size_t t) {
        setIteration(t);
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
    // The first N regions of every iteration as FKL rectangles, converted when the pipeline is created
    std::vector<std::array<fk::Rect, N>> regions;
    fk::Tensor<fk::float3> out{OUT_W, OUT_H, N, 1, fk::MemType::Host};
    fk::Stream_<PA> stream;

    explicit FklCpu(const Workload& workload) : w(workload), regions(workload.sets.iterations()) {
        for (size_t t = 0; t < regions.size(); ++t) {
            for (size_t i = 0; i < N; ++i) {
                const Region& r = w.sets.at(t, i);
                regions[t][i] = fk::Rect(static_cast<unsigned>(r.x), static_cast<unsigned>(r.y), r.width, r.height);
            }
        }
    }

    // Iteration t: the chain is built from the regions of t (Crop<>::build gives every plane its region's size, and
    // Resize its own scale), fused and executed, all inside the call
    void run(size_t t) {
        fk::executeOperations<fk::TransformDPP<PA>>(stream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*w.frame),
            YUVToRGB::build(),
            fk::Crop<>::build(regions[t]),
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
// Sample positions of RPP HOST's bilinear resampling for the benchmark's resizes (every region size of this build ->
// RES_W x RES_H), measured with linear ramps, as the paper measured NPP's: pixel centers, src = (dst + 0.5) * scale - 0.5.
// rppt_resize (F32) clamps them to [0, size - 2] except the last output columns: its AVX bilinear load, when the left
// neighbour is past size - 3, only replaces the right neighbour by the left one, so positions >= size - 1 read pixel
// size - 1. rppt_resize_mirror_normalize (8u -> 32f) clamps them to [0, size - 1]. Both are measured
// (--resize-convention).
// They are measured with every size of this build in one batched call, as in the pipelines, in the crop buffer's
// layout (rows of CROP_PITCH pixels, slots of MAX_REGION_H + CROP_SPARE_ROWS rows, each region at the top left of its
// slot, a MAX_REGION_W x MAX_REGION_H descriptor: what rppt_resize reads in the benchmark), and
// rppt_resize_mirror_normalize also in the frame's layout (one 8 bit image that every batch image aliases, the regions
// REGION_MARGIN pixels apart and from its edges: what it reads in the benchmark), which must give the same outputs.
// Everything outside the ROIs holds a sentinel, and every measurement is made with two different sentinels: the two
// outputs must be identical, value by value, or RPP would be using values outside its ROIs (in the benchmark: the frame
// around a region, or what a larger region of an earlier iteration left in a slot of the crop buffer).
// ------------------------------------------------------------------------------------------------
struct Sampling {
    std::vector<float> x, y; // source position of every output column and row
};

// The sampling positions of every region size of this build (REGION_SIZES)
struct SamplingTable {
    std::array<Sampling, REGION_SIZE_COUNT> bySize;
    const Sampling& of(int width, int height) const {
        for (size_t s = 0; s < REGION_SIZE_COUNT; ++s) {
            if (REGION_SIZES[s].width == width && REGION_SIZES[s].height == height) {
                return bySize[s];
            }
        }
        throw std::runtime_error("No RPP HOST sampling positions measured for " + std::to_string(width) + "x" +
                                 std::to_string(height) + " regions");
    }
};

enum class SamplingLayout { CropSlots, AliasedFrame };

// The output planes (RES_W x RES_H float3, one per size of this build) of rppt_resize (F32) or
// rppt_resize_mirror_normalize (8u) for linear ramps along one axis in a region of every size, in one layout, with
// two sentinels outside the ROIs: throws if the two outputs differ
std::vector<float> rampOutputs(rppHandle_t handle, bool resizeMirrorNormalize, SamplingLayout layout, bool vertical) {
    constexpr size_t n = REGION_SIZE_COUNT;
    const char* call = resizeMirrorNormalize ? "rppt_resize_mirror_normalize" : "rppt_resize";
    const bool slots = layout == SamplingLayout::CropSlots;
    // The 8 bit frame: the regions side by side, REGION_MARGIN pixels apart and from the edges, each at the top left of
    // a MAX_REGION_W x MAX_REGION_H box
    const int frameW = REGION_MARGIN + static_cast<int>(n) * (MAX_REGION_W + REGION_MARGIN);
    const int frameH = MAX_REGION_H + 2 * REGION_MARGIN;
    const auto regionX = [](size_t s) { return REGION_MARGIN + static_cast<int>(s) * (MAX_REGION_W + REGION_MARGIN); };
    // Index of channel 0 of pixel (x, y) of region s, in values from the start of the source
    const auto at = [&](size_t s, int x, int y) {
        return slots ? s * CROP_SLOT_FLOATS + (static_cast<size_t>(y) * CROP_PITCH + x) * 3
                     : (static_cast<size_t>(REGION_MARGIN + y) * frameW + regionX(s) + x) * 3;
    };
    const size_t srcValues = slots ? n * CROP_SLOT_FLOATS : static_cast<size_t>(frameW) * frameH * 3;
    const size_t bufferValues = RPP_FRONT_SLACK_FLOATS + srcValues + RPP_SLACK_FLOATS;
    std::vector<RpptROI> rois(n);
    for (size_t s = 0; s < n; ++s) {
        rois[s].xywhROI = {{slots ? 0 : regionX(s), slots ? 0 : REGION_MARGIN}, REGION_SIZES[s].width,
                           REGION_SIZES[s].height};
    }
    const RpptDataType type = resizeMirrorNormalize ? RpptDataType::U8 : RpptDataType::F32;
    RpptDesc src = slots ? RpptDesc(n, 3, MAX_REGION_H, MAX_REGION_W, type, RpptLayout::NHWC, 0,
                                    RpptStrides{static_cast<Rpp32u>(CROP_SLOT_FLOATS), 1,
                                                static_cast<Rpp32u>(CROP_PITCH * 3), 3})
                         : RpptDesc(n, 3, frameH, frameW, type, RpptLayout::NHWC, 0,
                                    RpptStrides{0, 1, static_cast<Rpp32u>(frameW * 3), 3});
    RpptDesc dst(n, 3, RES_H, RES_W, RpptDataType::F32, RpptLayout::NHWC);
    const size_t outFloats = n * RES_W * RES_H * 3;
    // Ramp along one axis: the position (F32) or RAMP_STEP_8BIT * position (8 bit), both exact
    const float step = resizeMirrorNormalize ? RAMP_STEP_8BIT : 1.f;
    const std::array<float, 2> sentinels = resizeMirrorNormalize ? std::array<float, 2>{0.f, 255.f}
                                                                 : std::array<float, 2>{-1000.f, 1000.f};
    std::array<std::vector<float>, 2> outputs;
    for (size_t k = 0; k < sentinels.size(); ++k) {
        std::vector<float> inFBuffer(resizeMirrorNormalize ? 0 : bufferValues, sentinels[k]);
        std::vector<Rpp8u> inUBuffer(resizeMirrorNormalize ? bufferValues : 0, static_cast<Rpp8u>(sentinels[k]));
        float* inF = inFBuffer.data() + RPP_FRONT_SLACK_FLOATS;
        Rpp8u* inU = inUBuffer.data() + RPP_FRONT_SLACK_FLOATS;
        for (size_t s = 0; s < n; ++s) {
            for (int y = 0; y < REGION_SIZES[s].height; ++y) {
                for (int x = 0; x < REGION_SIZES[s].width; ++x) {
                    const float v = step * static_cast<float>(vertical ? y : x);
                    for (int c = 0; c < 3; ++c) {
                        if (resizeMirrorNormalize) {
                            inU[at(s, x, y) + c] = static_cast<Rpp8u>(v);
                        } else {
                            inF[at(s, x, y) + c] = v;
                        }
                    }
                }
            }
        }
        std::vector<float> out(outFloats + RPP_SLACK_FLOATS, 0.f);
        std::vector<RpptImagePatch> sizes(n, RpptImagePatch{RES_W, RES_H});
        if (resizeMirrorNormalize) {
            std::vector<float> mean(n * 3, 0.f), stdDev(n * 3, 1.f);
            std::vector<Rpp32u> mirror(n, 0);
            RPP_CHECK(rppt_resize_mirror_normalize(inU, &src, out.data(), &dst, sizes.data(),
                                                   RpptInterpolationType::BILINEAR, mean.data(), stdDev.data(),
                                                   mirror.data(), rois.data(), RpptRoiType::XYWH, handle,
                                                   RppBackend::RPP_HOST_BACKEND));
        } else {
            RPP_CHECK(rppt_resize(inF, &src, out.data(), &dst, sizes.data(), RpptInterpolationType::BILINEAR,
                                  rois.data(), RpptRoiType::XYWH, handle, RppBackend::RPP_HOST_BACKEND));
        }
        out.resize(outFloats);
        outputs[k] = std::move(out);
    }
    if (outputs[0] != outputs[1]) {
        throw std::runtime_error(std::string("RPP HOST ") + call + " uses values outside its ROIs: its " +
                                 (vertical ? "vertical" : "horizontal") + " ramps in the " +
                                 (slots ? "crop buffer's" : "frame's") + " layout give different outputs with the " +
                                 "sentinels " + std::to_string(sentinels[0]) + " and " + std::to_string(sentinels[1]) +
                                 " outside the ROIs");
    }
    for (float& v : outputs[0]) {
        v /= step; // divided back exactly: the position
    }
    return outputs[0];
}

SamplingTable measureHostSampling(bool resizeMirrorNormalize) {
    const char* call = resizeMirrorNormalize ? "rppt_resize_mirror_normalize" : "rppt_resize";
    rppHandle_t handle{};
    RPP_CHECK(static_cast<RppStatus>(rppCreate(&handle, 1, 1, nullptr, RppBackend::RPP_HOST_BACKEND)));
    SamplingTable table;
    for (Sampling& s : table.bySize) {
        s = Sampling{std::vector<float>(RES_W), std::vector<float>(RES_H)};
    }
    const size_t planeFloats = static_cast<size_t>(RES_W) * RES_H * 3;
    for (const bool vertical : {false, true}) {
        const auto out = rampOutputs(handle, resizeMirrorNormalize, SamplingLayout::CropSlots, vertical);
        if (resizeMirrorNormalize &&
            rampOutputs(handle, resizeMirrorNormalize, SamplingLayout::AliasedFrame, vertical) != out) {
            throw std::runtime_error(std::string("RPP HOST ") + call + ": the " + (vertical ? "vertical" : "horizontal") +
                                     " ramps give different outputs in the crop buffer's and in the frame's layout");
        }
        for (size_t s = 0; s < REGION_SIZE_COUNT; ++s) {
            const float* plane = out.data() + s * planeFloats;
            const int size = vertical ? REGION_SIZES[s].height : REGION_SIZES[s].width;
            std::vector<float>& axis = vertical ? table.bySize[s].y : table.bySize[s].x;
            for (size_t p = 0; p < axis.size(); ++p) {
                axis[p] = vertical ? plane[(p * RES_W + 1) * 3] : plane[(1 * RES_W + p) * 3];
                // The references index the region with these positions
                if (!(axis[p] >= 0.f && axis[p] <= static_cast<float>(size - 1))) {
                    throw std::runtime_error(std::string("RPP HOST ") + call + ": measured sampling position " +
                                             std::to_string(axis[p]) + " outside [0, " + std::to_string(size - 1) + "]");
                }
            }
        }
    }
    rppDestroy(handle, RppBackend::RPP_HOST_BACKEND);
    return table;
}

// ------------------------------------------------------------------------------------------------
// Independent host references: the same chain in plain C++ with FKL's conversion matrix, every region with its own
// size, scale and clamps.
//  - FKL conventions: float RGB, sample at src = dst * scale, clamped to [0, size - 1]
//  - RPP HOST conventions: RGB rounded to 8 bit (what FKL's conversion step writes), sample at the positions measured
//    for the region's size
//  - the paper's 8 bit reference: RGB rounded to 8 bit, FKL's sampling
// ------------------------------------------------------------------------------------------------
enum class Convention { FklFloat, Rpp8bit, Fkl8bit };

// The reference output of the first n regions of iteration t
std::vector<float> hostReference(const Workload& w, size_t t, size_t n, Convention conv,
                                 const SamplingTable* sampling = nullptr) {
    std::vector<float> out(n * OUT_FLOATS);
    const bool eightBit = conv != Convention::FklFloat;
    const bool rppSampling = conv == Convention::Rpp8bit;
    for (size_t i = 0; i < n; ++i) {
        const Region& r = w.sets.at(t, i);
        const int cw = r.width, ch = r.height;
        const float sx = static_cast<float>(cw) / RES_W;
        const float sy = static_cast<float>(ch) / RES_H;
        const Sampling* positions = rppSampling ? &sampling->of(cw, ch) : nullptr;
        float crop[MAX_REGION_H][MAX_REGION_W][3];
        for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
                const fk::uchar3 yuv = w.frame->readAt(fk::Point{r.x + x, r.y + y, 0});
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
                    const float fx = positions ? positions->x[x] : std::clamp(x * sx, 0.f, static_cast<float>(cw - 1));
                    const float fy = positions ? positions->y[ry] : std::clamp(ry * sy, 0.f, static_cast<float>(ch - 1));
                    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
                    const int x1 = std::min(x0 + 1, cw - 1), y1 = std::min(y0 + 1, ch - 1);
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

float maxAbsError(const float* a, const float* b, size_t count) {
    float e = 0.f;
    for (size_t i = 0; i < count; ++i) {
        const float d = std::abs(a[i] - b[i]);
        e = std::isfinite(d) ? std::max(e, d) : INFINITY;
    }
    return e;
}

float maxAbsError(const std::vector<float>& a, const std::vector<float>& b) {
    return maxAbsError(a.data(), b.data(), a.size());
}

// The references of the first n regions of iteration t
struct References {
    std::vector<float> rppResize, rppRmn, fklFloat, fkl8bit;
};

References references(const Workload& w, size_t t, size_t n, const SamplingTable& resizeSampling,
                      const SamplingTable& rmnSampling) {
    return {hostReference(w, t, n, Convention::Rpp8bit, &resizeSampling),
            hostReference(w, t, n, Convention::Rpp8bit, &rmnSampling), hostReference(w, t, n, Convention::FklFloat),
            hostReference(w, t, n, Convention::Fkl8bit)};
}

// One output checked against the references of its regions (one row of cpu_validation.csv)
struct Check {
    size_t batch;
    std::string implementation;
    unsigned threads;
    size_t iteration;  // the output is of the regions of this iteration
    std::string check; // "selection": the stability check of the selection pass; "sampled": a validated timed iteration
    bench::validation::Errors own; // vs its own reference
    float vsOwnReference, vsRppReference, vsFklFloatReference, vsFkl8bitReference;
    // max abs error vs its own reference over the regions of each of PAPER_REGION_SIZES (-1: no region of that size)
    std::array<float, PAPER_REGION_SIZES.size()> ownBySize;
};

Check makeCheck(const Workload& w, size_t n, size_t t, const std::string& check, const std::string& implementation,
                unsigned threads, const std::vector<float>& out, const std::vector<float>& own, const References& refs) {
    Check c{n, implementation, threads, t, check, bench::validation::compare(out, own), maxAbsError(out, own),
            maxAbsError(out, refs.rppResize), maxAbsError(out, refs.fklFloat), maxAbsError(out, refs.fkl8bit), {}};
    c.ownBySize.fill(-1.f);
    for (size_t i = 0; i < n; ++i) {
        const Region& r = w.sets.at(t, i);
        float& e = c.ownBySize[paperSizeIndex(r.width, r.height)];
        e = std::max(e, maxAbsError(out.data() + i * OUT_FLOATS, own.data() + i * OUT_FLOATS, OUT_FLOATS));
    }
    return c;
}

// "20x10 1.2e-06, 48x24 3.4e-07": the errors of the sizes with regions
std::string bySizeText(const std::array<float, PAPER_REGION_SIZES.size()>& errors) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(1);
    bool first = true;
    for (size_t k = 0; k < errors.size(); ++k) {
        if (errors[k] >= 0.f) {
            os << (first ? "" : ", ") << PAPER_REGION_SIZES[k].width << "x" << PAPER_REGION_SIZES[k].height << " "
               << errors[k];
            first = false;
        }
    }
    return os.str();
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

// Warm-up iteration i runs the regions of iteration i, timed iteration i those of iteration WARMUP_ITERS + i; RPP and
// FKL run the same regions in every iteration
Row timeTwo(size_t batch, const std::function<void(size_t)>& rpp, const std::function<void(size_t)>& fkl,
            const std::function<void()>& beforeRpp = {}, const std::function<void()>& beforeFkl = {}) {
    using Clock = std::chrono::steady_clock;
    const auto time = [](const std::function<void(size_t)>& f, size_t t, const std::function<void()>& before) {
        if (before) {
            before();
        }
        std::this_thread::sleep_for(IDLE_GAP);
        const auto t0 = Clock::now();
        f(t);
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        const size_t t = static_cast<size_t>(i);
        time(rpp, t, beforeRpp);
        time(fkl, t, beforeFkl);
    }
    std::vector<double> a(ITERS), b(ITERS);
    double speedup = 0.;
    for (int i = 0; i < ITERS; ++i) {
        const size_t t = static_cast<size_t>(WARMUP_ITERS + i);
        a[i] = time(rpp, t, beforeRpp);
        b[i] = time(fkl, t, beforeFkl);
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

// ------------------------------------------------------------------------------------------------
// Selection of the RPP pipeline, for each thread count (1 and all hardware threads). Every RPP variant runs the regions
// of iterations A, B, C and A again (STABILITY_SEQUENCE): its two outputs of A must be identical, value by value (no
// library state may change between calls, and no value that the regions of B and C leave in its buffers may be used),
// and the second one within RPP_EQUIVALENCE of its RPP-convention reference for A and of rpp_per_op's output for A:
// those variants are the candidates (numerically equivalent to the one-call-per-operation pipeline). The candidates
// are timed round-robin, each run after the idle gap, SELECTION_WARMUP + SELECTION_ITERS iterations per N, iteration i
// with the regions of iteration i; the one with the lowest geometric mean of its medians over all N is the RPP
// pipeline of that thread count, the only one in the measurements.
// ------------------------------------------------------------------------------------------------
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
void selectionConfig(const Workload& w, unsigned threads, bool timing, const References& refA,
                     std::vector<SelectionRow>& rows, std::vector<Check>& checks) {
    constexpr size_t A = STABILITY_SEQUENCE.front();
    static_assert(STABILITY_SEQUENCE.back() == A, "The stability sequence runs the regions of A first and last");
    std::vector<std::unique_ptr<RppHost<PA>>> pipelines;
    for (const Variant v : VARIANTS) {
        pipelines.push_back(std::make_unique<RppHost<PA>>(w, N, v, threads));
    }
    std::vector<float> perOp; // rpp_per_op's output for the regions of A
    std::vector<SelectionRow> local;
    for (size_t k = 0; k < VARIANTS.size(); ++k) {
        const auto& refOwn = usesResizeMirrorNormalize(VARIANTS[k]) ? refA.rppRmn : refA.rppResize;
        std::vector<float> firstOut;
        for (size_t s = 0; s < STABILITY_SEQUENCE.size(); ++s) {
            pipelines[k]->run(STABILITY_SEQUENCE[s]);
            if (s == 0) {
                firstOut = pipelines[k]->outputValues();
            }
        }
        const auto out = pipelines[k]->outputValues();
        const float first = maxAbsError(firstOut, refOwn);
        const bool sameOutput = out == firstOut;
        if (k == 0) {
            perOp = out;
        }
        // vs own reference (rppt_resize's or rppt_resize_mirror_normalize's convention), vs the RPP-convention reference
        // of rppt_resize (the column of cpu_validation.csv), vs FKL's float and 8 bit references
        checks.push_back(makeCheck(w, N, A, "selection", variantName(VARIANTS[k]), threads, out, refOwn, refA));
        const Check& c = checks.back();
        SelectionRow r{threads, N, c.implementation, c.vsOwnReference, maxAbsError(out, perOp), sameOutput, false, {}};
        r.candidate = r.stable && r.vsOwnReference <= RPP_EQUIVALENCE && r.vsPerOp <= RPP_EQUIVALENCE;
        std::printf("N %3zu | %-20s %2u thread(s), regions of iteration %zu: max err vs its RPP-convention reference "
                    "%.2e (%s; first run %.2e, same output after iterations 1 and 2: %s), vs rpp_per_op %.2e, vs "
                    "paper's 8 bit reference %.2e%s\n", N, r.variant.c_str(), threads, A, r.vsOwnReference,
                    bySizeText(c.ownBySize).c_str(), first, sameOutput ? "yes" : "no", r.vsPerOp, c.vsFkl8bitReference,
                    r.candidate ? "" : "  (not a candidate)");
        local.push_back(r);
    }
    if (timing) {
        std::vector<std::vector<double>> ms(VARIANTS.size());
        for (int i = 0; i < SELECTION_WARMUP + SELECTION_ITERS; ++i) {
            const size_t t = static_cast<size_t>(i);
            for (size_t k = 0; k < VARIANTS.size(); ++k) {
                if (local[k].candidate) {
                    const double ms_ = timeOnce([&]() { pipelines[k]->run(t); }, [] {
                        if constexpr (PA == fk::ParArch::CPU_MULTITHREAD) {
                            fklPoolSpin(false);
                        }
                    });
                    if (i >= SELECTION_WARMUP) {
                        ms[k].push_back(ms_);
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
void selectionBatch(const Workload& w, unsigned hardwareThreads, bool timing, const SamplingTable& resizeSampling,
                    const SamplingTable& rmnSampling, std::vector<SelectionRow>& rows, std::vector<Check>& checks) {
    const References refA = references(w, STABILITY_SEQUENCE.front(), N, resizeSampling, rmnSampling);
    selectionConfig<N, fk::ParArch::CPU>(w, 1, timing, refA, rows, checks);
    selectionConfig<N, fk::ParArch::CPU_MULTITHREAD>(w, hardwareThreads, timing, refA, rows, checks);
}

// The RPP variants that are candidates (numerically equivalent to rpp_per_op and to their references) at every N
std::vector<Variant> candidatesAtEveryN(const std::vector<SelectionRow>& rows, unsigned threads) {
    std::vector<Variant> variants;
    for (const Variant v : VARIANTS) {
        bool seen = false, candidate = true;
        for (const auto& r : rows) {
            if (r.threads == threads && r.variant == variantName(v)) {
                seen = true;
                candidate = candidate && r.candidate;
            }
        }
        if (seen && candidate) {
            variants.push_back(v);
        }
    }
    return variants;
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

// The timed iterations whose outputs are validated, as "0, 50, ..., 950, 999"
std::string sampledText() {
    const auto k = bench::validation::validatedIterations(ITERS);
    std::ostringstream os;
    for (size_t i = 0; i < k.size(); ++i) {
        if (k.size() > 5 && i >= 2 && i + 2 < k.size()) {
            if (i == 2) {
                os << ", ...";
            }
            continue;
        }
        os << (i ? ", " : "") << k[i];
    }
    return os.str();
}

// Batch N: the timed pipelines (the RPP pipeline of each thread count, FKL CPU and FKL CPU_MULTITHREAD), validated
// before the timing on the regions of the sampled timed iterations, then, if timing, the two comparisons. oneThread
// and allThreads are the RPP pipelines of each thread count: the one chosen by the selection pass (the first is timed),
// or, with --validate, every candidate. Returns whether the FKL outputs are within TOLERANCE of their reference in
// every validated iteration.
template <size_t N>
bool runBatch(const Workload& w, unsigned hardwareThreads, const std::vector<Variant>& oneThread,
              const std::vector<Variant>& allThreads, const SamplingTable& resizeSampling,
              const SamplingTable& rmnSampling, bool timing, std::vector<Check>& checks,
              std::array<std::vector<Row>, PAIRINGS>& rows) {
    FklCpu<N> fkl(w);
    FklCpu<N, fk::ParArch::CPU_MULTITHREAD> fklMT(w);
    std::vector<std::unique_ptr<RppHost<fk::ParArch::CPU>>> rpp1;
    std::vector<std::unique_ptr<RppHost<fk::ParArch::CPU_MULTITHREAD>>> rppAll;
    for (const Variant v : oneThread) {
        rpp1.push_back(std::make_unique<RppHost<fk::ParArch::CPU>>(w, N, v, 1));
    }
    for (const Variant v : allThreads) {
        rppAll.push_back(std::make_unique<RppHost<fk::ParArch::CPU_MULTITHREAD>>(w, N, v, hardwareThreads));
    }

    // Validation, outside of the timing: the regions of timed iteration k are those of iteration WARMUP_ITERS + k
    const size_t firstCheck = checks.size();
    float mtVsCpu = 0.f;
    const auto validateRpp = [&](auto& p, size_t t, const References& refs) {
        p.run(t);
        if (!p.roisIntact(t)) {
            throw std::runtime_error(variantName(p.variant) + " modified the ROIs of iteration " + std::to_string(t));
        }
        checks.push_back(makeCheck(w, N, t, "sampled", variantName(p.variant), p.threads, p.outputValues(),
                                   usesResizeMirrorNormalize(p.variant) ? refs.rppRmn : refs.rppResize, refs));
    };
    const auto sampled = bench::validation::validatedIterations(ITERS);
    for (const size_t k : sampled) {
        const size_t t = static_cast<size_t>(WARMUP_ITERS) + k;
        const References refs = references(w, t, N, resizeSampling, rmnSampling);
        fkl.run(t);
        fklMT.run(t);
        const auto fklOut = fkl.output();
        const auto fklMTOut = fklMT.output();
        checks.push_back(makeCheck(w, N, t, "sampled", "fkl_cpu", 1, fklOut, refs.fklFloat, refs));
        checks.push_back(makeCheck(w, N, t, "sampled", "fkl_cpu_multithread", hardwareThreads, fklMTOut, refs.fklFloat,
                                   refs));
        mtVsCpu = std::max(mtVsCpu, maxAbsError(fklMTOut, fklOut));
        for (auto& p : rpp1) {
            validateRpp(*p, t, refs);
        }
        for (auto& p : rppAll) {
            validateRpp(*p, t, refs);
        }
    }
    // Summary per implementation over the validated iterations
    struct Summary {
        std::string implementation;
        unsigned threads;
        bench::validation::Errors own;
        std::array<float, PAPER_REGION_SIZES.size()> ownBySize;
        float vsFkl8bit;
    };
    std::vector<Summary> summaries;
    bool fklWithinTolerance = true;
    for (size_t c = firstCheck; c < checks.size(); ++c) {
        const Check& k = checks[c];
        if (k.implementation.rfind("fkl_", 0) == 0 && !(k.vsOwnReference <= bench::validation::TOLERANCE)) {
            fklWithinTolerance = false;
        }
        auto s = std::find_if(summaries.begin(), summaries.end(), [&](const Summary& x) {
            return x.implementation == k.implementation && x.threads == k.threads;
        });
        if (s == summaries.end()) {
            summaries.push_back({k.implementation, k.threads, k.own, k.ownBySize, k.vsFkl8bitReference});
            continue;
        }
        s->own = bench::validation::combine(s->own, k.own);
        for (size_t i = 0; i < s->ownBySize.size(); ++i) {
            s->ownBySize[i] = std::max(s->ownBySize[i], k.ownBySize[i]);
        }
        s->vsFkl8bit = std::max(s->vsFkl8bit, k.vsFkl8bitReference);
    }
    for (const Summary& s : summaries) {
        std::printf("N %3zu | %-20s %2u thread(s), %zu validated iterations: max err vs its %s reference %.2e (mean "
                    "%.2e; %s), vs paper's 8 bit reference %.2e\n", N, s.implementation.c_str(), s.threads,
                    sampled.size(), s.implementation.rfind("fkl_", 0) == 0 ? "FKL float" : "RPP-convention",
                    s.own.max(), s.own.mean, bySizeText(s.ownBySize).c_str(), s.vsFkl8bit);
    }
    std::printf("N %3zu | FKL CPU_MULTITHREAD vs FKL CPU: max abs difference %.2e\n", N, mtVsCpu);
    if (!timing || !fklWithinTolerance) {
        return fklWithinTolerance;
    }

    RppHost<fk::ParArch::CPU>& r1 = *rpp1.front();
    RppHost<fk::ParArch::CPU_MULTITHREAD>& rAll = *rppAll.front();
    rows[ONE_THREAD_VS_CPU].push_back(timeTwo(N, [&](size_t t) { r1.run(t); }, [&](size_t t) { fkl.run(t); }));
    rows[ALL_THREADS_VS_MT].push_back(timeTwo(N, [&](size_t t) { rAll.run(t); }, [&](size_t t) { fklMT.run(t); },
                                              [] { fklPoolSpin(false); }, [] { fklPoolSpin(true); }));
    const Row& a = rows[ONE_THREAD_VS_CPU].back();
    const Row& c = rows[ALL_THREADS_VS_MT].back();
    std::printf("N %3zu | RPP 1 thread %9.1f us vs FKL CPU %8.1f us (BVF mean speedup %6.2fx) | RPP %u threads %9.1f us vs "
                "FKL CPU_MULTITHREAD %8.1f us (%6.2fx)\n", N, a.rpp.median * 1000., a.fkl.median * 1000., a.meanSpeedup,
                hardwareThreads, c.rpp.median * 1000., c.fkl.median * 1000., c.meanSpeedup);
    return true;
}

using bench::shell;

// rppt_resize HOST sampling convention for the benchmark's resizes (every region size of this build, along each axis),
// from linear ramps (value = position, exact) in tight buffers, as for the HIP backend
void resizeConvention(const std::filesystem::path& outDir) {
    std::ofstream csv(outDir / "rpp_host_resize_convention.csv");
    csv << "axis,src,dst,scale,dst_index,measured_src_position,s_times_p,pixel_center,pixel_center_clamped_to_0_size_minus_2\n";
    rppHandle_t handle{};
    RPP_CHECK(static_cast<RppStatus>(rppCreate(&handle, 1, 1, nullptr, RppBackend::RPP_HOST_BACKEND)));
    struct Case { bool vertical; int src, dst; };
    std::vector<Case> cases;
    for (const RegionSize s : REGION_SIZES) {
        cases.push_back({false, s.width, RES_W});
    }
    for (const RegionSize s : REGION_SIZES) {
        cases.push_back({true, s.height, RES_H});
    }
    for (const Case c : cases) {
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
    // The validation rows, written on every exit path once the run has started, also when an exception ends it (so that
    // an earlier run's cpu_validation.csv never stays in the results directory)
    std::vector<Check> checks;
    std::filesystem::path validationFile;
    const auto writeValidation = [&]() {
        if (validationFile.empty()) {
            return;
        }
        std::ofstream v(validationFile);
        v << "Batch size, Implementation, Threads, Iteration, Check, MaxAbsError vs own reference, MaxAbsError vs "
             "RPP-convention reference, MaxAbsError vs FKL-convention float reference, MaxAbsError vs paper's 8 bit "
             "reference (FKL sampling)";
        for (const RegionSize s : PAPER_REGION_SIZES) {
            v << ", MaxAbsError vs own reference " << s.width << "x" << s.height << " regions";
        }
        v << std::endl;
        for (const auto& c : checks) {
            v << c.batch << ", " << c.implementation << ", " << c.threads << ", " << c.iteration << ", " << c.check
              << ", " << c.vsOwnReference << ", " << c.vsRppReference << ", " << c.vsFklFloatReference << ", "
              << c.vsFkl8bitReference;
            for (const float e : c.ownBySize) {
                v << ", ";
                if (e >= 0.f) {
                    v << e;
                }
            }
            v << std::endl;
        }
    };
    try {
        if (argc == 2 && std::string(argv[1]) == "--batches") {
            for (size_t i = 0; i < BATCHES.size(); ++i) {
                std::cout << (i ? " " : "") << BATCHES[i];
            }
            std::cout << std::endl;
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--workload") {
            // The workload (frame, region sizes and positions, batch sizes, seeds), for the consistency check of
            // rocm/scripts/run_all.sh
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
        validationFile = outDir / "cpu_validation.csv";
        writeValidation();

        // The frame and the regions of every iteration of every loop, before anything runs
        Workload w;
        initWorkload(w);
        writeRegionStats((outDir / "regions.csv").string(), w.sets, WARMUP_ITERS, ITERS);
        const SamplingTable resizeSampling = measureHostSampling(false);
        const SamplingTable rmnSampling = measureHostSampling(true);
        {
            std::ofstream os(outDir / "rpp_host_sampling_positions.csv");
            os << "src_w,src_h,axis,src,dst,dst_index,rppt_resize_src_position,rppt_resize_mirror_normalize_src_position,"
                  "pixel_center,pixel_center_clamped_to_0_size_minus_2\n";
            for (const RegionSize size : REGION_SIZES) {
                const Sampling& resize = resizeSampling.of(size.width, size.height);
                const Sampling& rmn = rmnSampling.of(size.width, size.height);
                for (int p = 0; p < RES_W + RES_H; ++p) {
                    const bool vertical = p >= RES_W;
                    const int i = vertical ? p - RES_W : p;
                    const int src = vertical ? size.height : size.width, dst = vertical ? RES_H : RES_W;
                    const float center = (i + 0.5f) * static_cast<float>(src) / static_cast<float>(dst) - 0.5f;
                    os << size.width << "," << size.height << "," << (vertical ? "y" : "x") << "," << src << "," << dst
                       << "," << i << "," << (vertical ? resize.y[i] : resize.x[i]) << ","
                       << (vertical ? rmn.y[i] : rmn.x[i]) << "," << center << ","
                       << std::clamp(center, 0.f, static_cast<float>(src - 2)) << "\n";
                }
            }
        }
        std::printf("RPP HOST sampling positions measured for %s regions (rppt_resize and rppt_resize_mirror_normalize, "
                    "the same outputs with two sentinel values outside the ROIs)\n", regionSizeList().c_str());

        std::vector<SelectionRow> selection;
        // Selection pass: the stability check of every RPP variant and, unless validating only, the timing of the
        // candidates
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (selectionBatch<BATCHES[Is]>(w, hardwareThreads, !validateOnly, resizeSampling, rmnSampling, selection, checks),
             ...);
        }(std::make_index_sequence<BATCHES.size()>{});
        // Every thread count needs an RPP pipeline numerically equivalent to rpp_per_op
        bool failed = false;
        for (const unsigned t : {1u, hardwareThreads}) {
            if (candidatesAtEveryN(selection, t).empty()) {
                std::cerr << "ERROR: no RPP HOST variant is numerically equivalent to rpp_per_op at every N with " << t
                          << " thread(s)" << std::endl;
                failed = true;
            }
        }
        if (failed) {
            writeValidation();
            return 1;
        }
        // The RPP pipelines validated (and timed) with each thread count: the one the selection pass chose or, with
        // --validate (no timing to choose by), every candidate
        Variant oneThread = VARIANTS[0], allThreads = VARIANTS[0];
        std::vector<Variant> oneThreadValidated, allThreadsValidated;
        if (validateOnly) {
            oneThreadValidated = candidatesAtEveryN(selection, 1);
            allThreadsValidated = candidatesAtEveryN(selection, hardwareThreads);
        } else {
            oneThread = chooseRpp(selection, 1);
            allThreads = chooseRpp(selection, hardwareThreads);
            oneThreadValidated = {oneThread};
            allThreadsValidated = {allThreads};
            std::ofstream sel(outDir / "cpu_rpp_selection.csv");
            sel << "threads,N,variant,max_abs_err_vs_its_rpp_reference,max_abs_err_vs_rpp_per_op,same_output_iterations";
            for (const size_t t : STABILITY_SEQUENCE) {
                sel << "_" << t;
            }
            sel << ",candidate,selection_median_ms,selection_p10_ms,selection_p90_ms,chosen,selection_iterations\n";
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
        // Validation of the sampled timed iterations and, unless validating only, the timing, N by N; after an FKL
        // output outside its tolerance, the remaining N are only validated
        std::array<std::vector<Row>, PAIRINGS> rows;
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((failed = !runBatch<BATCHES[Is]>(w, hardwareThreads, oneThreadValidated, allThreadsValidated, resizeSampling,
                                              rmnSampling, !validateOnly && !failed, checks, rows) || failed),
             ...);
        }(std::make_index_sequence<BATCHES.size()>{});
        writeValidation();
        // FKL's outputs must be within TOLERANCE of the FKL float reference (common/bench_validation.h) in every
        // validated iteration; the RPP outputs are reported
        for (const auto& c : checks) {
            if (c.implementation.rfind("fkl_", 0) == 0 && !(c.vsOwnReference <= bench::validation::TOLERANCE)) {
                std::cerr << "ERROR: " << c.implementation << " at N = " << c.batch << ", regions of iteration "
                          << c.iteration << ": max abs error " << c.vsOwnReference
                          << " against the FKL float reference, more than " << bench::validation::TOLERANCE << std::endl;
            }
        }
        if (failed) {
            return 1;
        }
        if (validateOnly) {
            return 0;
        }
        const std::string all = std::to_string(hardwareThreads) + "threads";
        writeCSV(outDir / "benchmark_cpu_rpp_1thread_vs_fkl_bvf_CPU.csv", "RPP HOST 1 thread", rows[ONE_THREAD_VS_CPU],
                 "FKL CPU");
        writeCSV(outDir / ("benchmark_cpu_rpp_" + all + "_vs_fkl_bvf_CPU_MULTITHREAD.csv"), "RPP HOST " + all,
                 rows[ALL_THREADS_VS_MT], "FKL CPU_MULTITHREAD");
        const size_t validated = bench::validation::validatedIterations(ITERS).size();
        std::ofstream plat(outDir / "cpu_platform.txt");
        plat << "CPU: " << bench::cpuName() << ", " << std::thread::hardware_concurrency() << " hardware threads\n"
             << "SIMD of the CPU: "
             << shell("grep -o -w -E 'avx|avx2|fma|avx512f|avx512bw|avx512vl' /proc/cpuinfo | sort -u | paste -sd' '") << "\n"
             << "Environment: " << (shell("grep -ci microsoft /proc/version 2>/dev/null") != "0" ? "WSL2" : "bare metal")
             << "\n"
             << "FKL: commit " << BENCH_FKL_COMMIT << ", CPU backend (single thread) and CPU_MULTITHREAD backend (thread "
             << "pool of " << fk::cpuThreadPool().size() << " workers, thread-pool " << BENCH_THREAD_POOL_COMMIT
             << ", one chunk per output row of each region), compiled with " << BENCH_COMPILER << ", flags "
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
             << "by a selection pass (each variant run on the regions of iterations 0, 1, 2 and 0 again, so that every "
             << "slot of the crop buffer has held a region of every size: the two outputs of iteration 0 identical value "
             << "by value, the second within " << RPP_EQUIVALENCE << " of its reference and of rpp_per_op's; then every "
             << "candidate timed round-robin, " << SELECTION_WARMUP << " warm-up + " << SELECTION_ITERS
             << " iterations per N, iteration i with the regions of iteration i, lowest geometric mean of the medians; "
             << "cpu_rpp_selection.csv); only that pipeline is measured afterwards. "
             << "Variants: rpp_per_op (crop_mirror_normalize 8u->32f, memset of the output's border rows, resize into "
                "the output rows, multiply_scalar, tensor_subtract_tensor, tensor_divide_tensor), rpp_npp_like (resized "
                "regions in their own buffer, copied by crop), rpp_fused (memset of the border rows, "
                "resize_mirror_normalize with Mul, Sub, Div), rpp_normalize (rpp_per_op with Mul + Sub + Div in one "
                "normalize, out of place, without the memset: its input's border rows stay zero), rpp_fused_normalize "
                "(resize_mirror_normalize, normalize, without the memset). Every RPP call gets one ROI per region (XYWH: "
                "its size and, in the frame, its position). RPP HOST behaviour handled in the pipelines: rppt_normalize "
                "replaces the caller's stdDev values by scale/stdDev (restored before every call, inside the timing) "
                "and zeroes its destination before reading its source (run out of place); rppt_resize_mirror_normalize "
                "zeroes nStride bytes (a quarter of the plane) from the first interior row of each float destination "
                "plane, never past the plane: harmless, the interior rows are written afterwards and the border rows "
                "are zero; rppt_resize clips its ROIs to its source descriptor, without an error: the crop buffer's "
                "descriptor is "
             << MAX_REGION_W << "x" << MAX_REGION_H << ", the largest region size. "
                "FKL's thread pool does not spin after the conversion of the RPP all-threads pipeline (setSpinTime(0) "
                "before every timed run of it, outside of the timing; FKL's default 100 us before FKL's runs), so that its "
                "workers do not compete with RPP's OpenMP team. "
                "The crop buffer has one slot per region of the largest region size, "
             << MAX_REGION_W << "x" << MAX_REGION_H << " with rows of " << CROP_PITCH
             << " pixels, each region at the top left of its slot; rppt_crop_mirror_normalize 8u->32f writes one float "
                "(1.0) past the end of every row when the ROI width is a multiple of 16 (the 32 and 48 pixel wide "
                "regions), so every slot ends with a spare row, and this float never lands on the next slot, which "
                "another OpenMP thread may have written already; rppt_resize reads 12 bytes before its source and "
                "rppt_resize_mirror_normalize up to 3 bytes before and 13 after the rows of its ROIs: the crop buffer "
                "starts after some slack, and the regions are at least " << REGION_MARGIN << " pixels from the frame "
                "edges. "
                "multiply_scalar, tensor_subtract_tensor, tensor_divide_tensor and normalize see the output as N samples "
                "(RPP's HOST backend parallelizes over the samples), so at N = 1 they, like every RPP HOST call, run "
                "on one thread. "
                "RPP-convention reference: RGB rounded to 8 bit, bilinear sampling at the positions RPP HOST uses, "
                "measured with linear ramps for every region size of this build ("
             << regionSizeList() << "; rpp_host_sampling_positions.csv; 8 bit ramps of step " << RAMP_STEP_8BIT
             << " per pixel for rppt_resize_mirror_normalize), every region with the positions of its size, its own "
                "scale and its own clamps. The positions are measured with all the sizes in one batched call, in the "
                "layout of the crop buffer above (what rppt_resize reads) and, for rppt_resize_mirror_normalize, also "
                "in the frame's (an 8 bit image that every batch image aliases, what it reads; the same outputs in "
                "both), each with two different sentinel values outside the ROIs: the outputs are identical, so "
                "neither call uses a value outside its ROIs (the frame around a region, or what a larger region of an "
                "earlier iteration left in a slot). Pixel centers; "
                "rppt_resize clamps them to [0, size - 2] except the last output columns (its AVX bilinear load only "
                "replaces the right neighbour when the left one is past size - 3, so positions >= size - 1 read pixel "
                "size - 1), rppt_resize_mirror_normalize clamps them to [0, size - 1]; "
                "so the rppt_resize-based variants (rpp_per_op, rpp_npp_like, rpp_normalize) and the "
                "rppt_resize_mirror_normalize-based ones (rpp_fused, rpp_fused_normalize) differ at the last columns and "
                "rows, each matching its own measured convention; the latter are therefore not numerically equivalent "
                "to rpp_per_op and not candidates of the selection.\n"
             << "Timing: steady_clock around each pipeline, " << WARMUP_ITERS << " warm-up and " << ITERS
             << " timed iterations, RPP and FKL one after the other in each, both with the regions of the same "
             << "iteration (warm-up iteration i: the regions of iteration i; timed iteration i: those of iteration "
             << WARMUP_ITERS << " + i; regions.csv has their statistics), each run preceded by a "
             << std::chrono::milliseconds(IDLE_GAP).count()
             << " ms idle gap so that both thread pools are asleep; speedup computed per iteration and averaged. "
                "Inside the timed region: FKL builds the whole chain from the regions of the iteration (Crop<>::build "
                "of its N regions, every plane with its region's size and Resize with its own scale), fuses it and "
                "executes it; RPP points the ROI arguments of its calls at the ROIs of the iteration and runs FKL's "
                "conversion of the full frame to 8 bit RGB and its calls. Outside: the generation of the regions of "
                "every iteration and their conversion to FKL rectangles and RPP ROIs (once per pipeline, when it is "
                "created, before its first run), the idle gaps, the thread pool spin settings and the validation.\n"
             << "Validation: before the timing of every N, the four timed pipelines (RPP 1 thread, RPP " << hardwareThreads
             << " threads, FKL CPU, FKL CPU_MULTITHREAD) run the regions of the timed iterations k = " << sampledText()
             << " (" << validated << (validated == 1 ? " iteration" : " iterations") << ": the first, every "
             << bench::validation::VALIDATION_STRIDE << "th and the last), those of iterations " << WARMUP_ITERS
             << " + k, and their outputs are compared with host references of those regions: FKL against the FKL float "
             << "reference, within " << bench::validation::TOLERANCE << " in every validated iteration (otherwise the "
             << "program fails); the RPP pipelines against their own RPP-convention reference, and everything against "
             << "the paper's 8 bit reference (reported). cpu_validation.csv has one row per N, implementation, thread "
             << "count and iteration of the regions (check sampled), and one per N, RPP variant and thread count for the "
             << "selection pass's stability check, on the regions of iteration " << STABILITY_SEQUENCE.front()
             << " (check selection), with the errors per region size.\n";
        std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        writeValidation();
        return 1;
    }
    return 0;
}
