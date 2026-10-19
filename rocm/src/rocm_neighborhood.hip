// Experiment 3 of the Backwards Vertical Fusion paper on an AMD GPU: a neighborhood operation, the K x K box blur
// (mean of the window centred on the value, per channel, replicating the edge of the crop in every pass), on fixed-size
// crops of the frame, without the resize and the border of Figure 1:
// ReadNV12 -> YUVtoRGB -> Crop 64x32 -> Box K x K -> Mul -> Sub -> Div -> Write (N planes of 64x32 float3)
// with the frame and batch sizes of common/bench_geometry.h and the crops (cropRegionSets: 64x32 at random even
// positions, new every iteration) and parameters of common/bench_neighborhood.h.
//
// Part (b), the library comparison (K = BOX_K, one pass, every N of BATCHES):
//   rpp_u8box_cmn         rppt_yuv_to_rgb (full frame), rppt_box_filter U8 on the frame with the crops as per-image
//                         XYWH ROIs (crop + box in one call, into N 64x32 U8 images), rppt_crop_mirror_normalize
//                         8u->32f with the whole normalization (per channel multiplier Mul / Div and offset -Sub / Div)
//                         into the output tensor
//   rpp_u8box_normalize   the same with rppt_crop_mirror_normalize 8u->32f converting only, then rppt_normalize
//                         ((x * Mul - Sub) / Div per channel) on the whole tensor
//   rpp_f32box_cmn        rppt_yuv_to_rgb, rppt_crop_mirror_normalize 8u->32f of the crops (aliased frame, XYWH ROIs)
//                         with the whole normalization, rppt_box_filter F32 of the normalized crops into the output
//   rpp_f32box_normalize  the same with rppt_crop_mirror_normalize converting only, then rppt_normalize
//   <rpp variant>_crop    each of the four with its NV12 -> RGB conversion of only the N crops of the iteration (one
//                         rppt_yuv_to_rgb per crop, the planes' pointers offset to the crop origin, into the RGB frame at
//                         the crop's position) instead of the full frame; the calls after it are the same. The group's
//                         RPP pipeline is the fastest of the eight within tolerance and equivalent to its RPP-convention
//                         reference
//   fkl_dpp_<W>x<H>       one FKL kernel: MorphologyDPP's executeBoxBlur<passes> (every pass a mean: an Add reducer and
//                         a Mul 1 / K^2 finalizer) batched over the crops, reading the BVF chain ReadNV12 -> YUVtoRGB
//                         -> Crop and writing through Mul -> Sub -> Div -> Write, one for every tile of DPP_TILES
// Part (a), the micro-experiment (FKL only, K of PART_A_KS, 1 or 2 passes, N of PART_A_BATCHES), in two variants:
// on the crops as above (operation box), and with the resize (operation resize_box): Figure 1's chain on Experiment 1's
// regions (common/bench_geometry.h), ReadNV12 -> YUVtoRGB -> Crop -> Resize 64x32 -> Box -> Mul -> Sub -> Div -> Write
//   fkl_bvf_naive         one FKL kernel with the naive ReadBack operation BoxBVF<K> of common/bench_neighborhood_ops.h
//                         (twice for two passes) in the BVF chain after the Crop (or the Resize)
//   fkl_dpp_<W>x<H>       as in part (b), with executeBoxBlur<passes>, its input chain ending at the Crop (or the Resize)
// In every group (part, operation, K, passes, N) all of them are validated, then timed one at a time (the baselines
// first, each RPP variant in its full-frame form and then its _crop form, the DPP tiles last): each one's complete loop
// (warm-up and timed iterations) before the next, never interleaved per iteration, iteration t of every loop on the crops
// of iteration t. The RPP variant and the DPP tile of a group (chosen) are the fastest (median of the CPU + GPU pass) of
// those that pass their checks; gpu_time*.csv also name the fastest RPP variant in their own metric
// (rpp_chosen_in_this_metric). Every one is recorded. The speedups are per iteration (baseline / chosen DPP tile, the runs of the same iteration), averaged.
//
// The harness (context, timing, rocprofiler-sdk tool, cache flusher) is the one of rocm_bench.hip, which this file
// includes. No HIP graphs anywhere.

#define BENCH_NO_MAIN
#include "rocm_bench.hip"

#include <fused_kernel/algorithms/image_processing/morphology.h>

#include <thread>

#include "bench_neighborhood.h"
#include "bench_neighborhood_ops.h"

namespace bench::nb {

using neighborhood::BOX_K;
using neighborhood::DPP_TILES;
using neighborhood::PLANE_FLOATS;
using neighborhood::PLANE_H;
using neighborhood::PLANE_PIXELS;
using neighborhood::PLANE_W;

// RPP's HIP box filter refuses a source descriptor with offsetInBytes < 12 * (kernelSize / 2) (RPP_ERROR_LOW_OFFSET;
// rppt_tensor_filter_augmentations.cpp in RPP 3.2.0): its sources (the RGB frame, and the converted crops of the F32
// variants) start this many bytes into their allocation, given to RPP through the descriptor's offsetInBytes
constexpr size_t RPP_LEAD_BYTES = 256;
constexpr int RPP_MAX_KERNEL = 9; // the largest kernelSize of RPP's HIP box filter
static_assert(RPP_LEAD_BYTES >= 12 * (RPP_MAX_KERNEL / 2), "RPP's HIP box filter needs the lead bytes");
static_assert(PLANE_W % RPP_PIXEL_ALIGN == 0, "RPP's kernels write the plane rows 8 pixels at a time");
// rppt_normalize reuses Experiment 1's rows of per column constants (OUT_W float3)
static_assert(PLANE_W == OUT_W, "the normalize constants are rows of OUT_W float3");

// One group of the experiment: every implementation of a group computes the same output
struct Group {
    char part{'b'};
    int k{0};
    int passes{1};
    int batch{0};
    bool resize{false}; // part (a) with the resize: Experiment 1's regions, resized to 64x32 before the box
    std::string operation() const { return resize ? "resize_box" : "box"; }
    std::string label() const {
        return std::string(1, part) + "_" + operation() + "_k" + std::to_string(k) + "_p" + std::to_string(passes) +
               "_n" + std::to_string(batch);
    }
};

// The regions of part (a) with the resize: Experiment 1's (common/bench_geometry.h, the default RegionSets), while
// ctx.regions holds the crops of every other group
std::unique_ptr<RegionSets> resizeRegions;

template <bool R>
const RegionSets& regionsOf(const Context& ctx) {
    return R ? *resizeRegions : *ctx.regions;
}

// The first N regions of iteration t as FKL rectangles, and those of every iteration
template <size_t N>
std::array<fk::Rect, N> rectsOf(const RegionSets& sets, size_t t) {
    std::array<fk::Rect, N> rects{};
    for (size_t i = 0; i < N; ++i) {
        const Region& r = sets.at(t, i);
        rects[i] = fk::Rect(static_cast<uint>(r.x), static_cast<uint>(r.y), r.width, r.height);
    }
    return rects;
}

template <size_t N>
std::shared_ptr<const std::vector<std::array<fk::Rect, N>>> rectSetsOf(const RegionSets& sets) {
    auto all = std::make_shared<std::vector<std::array<fk::Rect, N>>>(sets.iterations());
    for (size_t t = 0; t < all->size(); ++t) {
        (*all)[t] = rectsOf<N>(sets, t);
    }
    return all;
}

using ResizeTo64x32 = fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>;

// ------------------------------------------------------------------------------------------------
// References (common/bench_neighborhood.h): the crops blurred, then normalized.
// The float reference: FKL's CPU float chain ReadNV12 -> YUVtoRGB -> Crop, then boxReference (sums in double, no
// rounding, the normalization in float); FKL is checked against it with TOLERANCE. RPP's frame is 8 bit RGB, saturated:
// it is checked with TOLERANCE_8BIT (one level of 8 bit RGB) against the same chain with the RGB saturated and rounded
// to 8 bits (eightBitReference).
// The RPP-convention references: each RPP variant's own arithmetic on RPP's 8 bit frame (ctx.rppRgb8, the output of
// rppt_yuv_to_rgb), operation by operation, as RPP 3.2.0's HIP kernels compute it:
//   rppt_box_filter U8   the window sum of 8 bit values (exact in float) times RPP's float 1 / K^2, to 8 bits with
//                        v_cvt_pk_u8_f32 (round to nearest even, saturated): rint(float(sum) * scale)
//                        (box_filter.cpp: box_filter_row_hip_compute, rpp_hip_pack_float24_pln3_and_store24_pkd3)
//   rppt_box_filter F32  the window summed in float from 0, row after row, left to right, times RPP's float 1 / K^2
//   rppt_crop_mirror_normalize 8u->32f   fmaf(x, multiplier, offset) per channel (crop_mirror_normalize.cpp)
//   rppt_normalize       fmaf(x - mean, 1 * (1 / stdDev), 0) (normalize.cpp), mean = Sub / Mul and stdDev = Div / Mul
//                        in float, as rocm_bench.hip uploads them
// They are bit-exact by construction; an RPP variant is equivalent when it is within RPP_EQUIVALENCE of its own.
// ------------------------------------------------------------------------------------------------
static_assert(sizeof(fk::float3) == 3 * sizeof(float), "the planes are copied as float3");

inline size_t at(size_t plane, int y, int x, int c) {
    return (plane * PLANE_PIXELS + static_cast<size_t>(y) * PLANE_W + x) * 3 + c;
}

// The crops of iteration t on FKL's CPU float chain ReadNV12 -> YUVtoRGB -> Crop (RGB values, not saturated), resized
// to 64x32 (bilinear, aspect ratio ignored) for part (a) with the resize
template <size_t N, bool R = false>
std::vector<double> fklFloatCrops(const Context& ctx, size_t t) {
    const auto crops = rectsOf<N>(regionsOf<R>(ctx), t);
    fk::Stream_<fk::ParArch::CPU> cpuStream;
    fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, N, 1, fk::MemType::Host);
    if constexpr (R) {
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameHost),
            YUVToRGB::build(),
            fk::Crop<>::build(crops),
            ResizeTo64x32::build(fk::Size(PLANE_W, PLANE_H)),
            fk::TensorWrite<fk::float3>::build(out.ptr()));
    } else {
        fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameHost),
            YUVToRGB::build(),
            fk::Crop<>::build(crops),
            fk::TensorWrite<fk::float3>::build(out.ptr()));
    }
    cpuStream.sync();
    const float* data = reinterpret_cast<const float*>(out.ptr().data);
    return std::vector<double>(data, data + N * PLANE_FLOATS);
}

template <size_t N, bool R = false>
std::vector<float> floatReference(const Context& ctx, size_t t, int k, int passes) {
    return neighborhood::boxReference(fklFloatCrops<N, R>(ctx, t), N, k, passes, neighborhood::Rounding::None, MUL,
                                      SUB, DIV);
}

// The reference of RPP's tolerance check (its frame is 8 bit RGB): FKL's conversion saturated and rounded to nearest
// 8 bit (Experiment 1's fkl_cpu_8bit_rgb), the box in double, its output rounded half up to 8 bits for the U8 box
// (RPP's U8 box rounds to nearest; a 5 x 5 window sum of 8 bit values is never a half level from an integer), then the
// normalization in float
template <size_t N>
std::vector<float> eightBitReference(const Context& ctx, size_t t, int k, int passes, bool roundedBox) {
    std::vector<double> rgb = fklFloatCrops<N>(ctx, t);
    for (auto& v : rgb) {
        v = neighborhood::round8bit(v, neighborhood::Rounding::HalfUp);
    }
    return neighborhood::boxReference(rgb, N, k, passes,
                                      roundedBox ? neighborhood::Rounding::HalfUp : neighborhood::Rounding::None, MUL,
                                      SUB, DIV);
}

// The float 1 / K^2 of RPP 3.2.0's HIP box filter (box_filter.cpp: kernelSize<K>InverseSquare); 0.04f is 1.0f / 25
float rppBoxScale(int k) {
    switch (k) {
        case 3: return 0.1111111f;
        case 5: return 0.04f;
        case 7: return 0.02040816f;
        case 9: return 0.01234568f;
        default: throw std::runtime_error("RPP's HIP box filter has no kernel size " + std::to_string(k));
    }
}

// The RPP variants of part (b): which box (U8 on the frame, or F32 on the converted crops) and where the
// normalization is (in rppt_crop_mirror_normalize, or rppt_normalize after the box)
enum class RppBox { U8, F32 };
enum class RppNorm { InCmn, Normalize };
struct RppVariant {
    const char* name;
    RppBox box;
    RppNorm norm;
};
constexpr std::array<RppVariant, 4> RPP_VARIANTS_NB{{{"rpp_u8box_cmn", RppBox::U8, RppNorm::InCmn},
                                                     {"rpp_u8box_normalize", RppBox::U8, RppNorm::Normalize},
                                                     {"rpp_f32box_cmn", RppBox::F32, RppNorm::InCmn},
                                                     {"rpp_f32box_normalize", RppBox::F32, RppNorm::Normalize}}};

// The variant of an RPP implementation: its name, or its name with the per-crop conversion's "_crop" suffix
const RppVariant& rppVariant(const std::string& name) {
    const std::string suffix = "_crop";
    const bool crop = name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    const std::string base = crop ? name.substr(0, name.size() - suffix.size()) : name;
    for (const auto& v : RPP_VARIANTS_NB) {
        if (base == v.name) {
            return v;
        }
    }
    throw std::runtime_error("Unknown RPP variant " + name);
}

// rppt_crop_mirror_normalize's per channel multiplier and offset for the whole normalization: x * Mul / Div - Sub / Div
float cmnMultiplier(int c) { return MUL[c] / DIV[c]; }
float cmnOffset(int c) { return -SUB[c] / DIV[c]; }

template <size_t N>
std::vector<float> rppConventionReference(const Context& ctx, size_t t, const RppVariant& variant) {
    if (ctx.rppRgb8.size() != static_cast<size_t>(FRAME_W) * FRAME_H * 3) {
        throw std::runtime_error("RPP's RGB frame (ctx.rppRgb8) is not available");
    }
    const int k = BOX_K, anchor = k / 2;
    const float scale = rppBoxScale(k);
    const bool normInCmn = variant.norm == RppNorm::InCmn;
    std::vector<float> m(3), o(3), mean(3), invStdDev(3);
    for (int c = 0; c < 3; ++c) {
        m[c] = normInCmn ? cmnMultiplier(c) : 1.f;
        o[c] = normInCmn ? cmnOffset(c) : 0.f;
        mean[c] = SUB[c] / MUL[c];
        invStdDev[c] = 1.f * (1.f / (DIV[c] / MUL[c]));
    }
    // The 8 bit crops
    std::vector<uint8_t> rgb(N * PLANE_FLOATS);
    for (size_t i = 0; i < N; ++i) {
        const Region& r = ctx.regions->at(t, i);
        for (int y = 0; y < PLANE_H; ++y) {
            for (int x = 0; x < PLANE_W; ++x) {
                for (int c = 0; c < 3; ++c) {
                    rgb[at(i, y, x, c)] = ctx.rppRgb8[(static_cast<size_t>(r.y + y) * FRAME_W + (r.x + x)) * 3 + c];
                }
            }
        }
    }
    const auto clampY = [](int v) { return std::clamp(v, 0, PLANE_H - 1); };
    const auto clampX = [](int v) { return std::clamp(v, 0, PLANE_W - 1); };
    std::vector<float> out(N * PLANE_FLOATS);
    if (variant.box == RppBox::U8) {
        for (size_t i = 0; i < N; ++i) {
            for (int y = 0; y < PLANE_H; ++y) {
                for (int x = 0; x < PLANE_W; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        int sum = 0;
                        for (int dy = -anchor; dy <= anchor; ++dy) {
                            for (int dx = -anchor; dx <= anchor; ++dx) {
                                sum += rgb[at(i, clampY(y + dy), clampX(x + dx), c)];
                            }
                        }
                        const float boxed = std::clamp(std::rint(static_cast<float>(sum) * scale), 0.f, 255.f);
                        out[at(i, y, x, c)] = std::fmaf(boxed, m[c], o[c]);
                    }
                }
            }
        }
    } else {
        std::vector<float> converted(N * PLANE_FLOATS);
        for (size_t j = 0; j < converted.size(); ++j) {
            converted[j] = std::fmaf(static_cast<float>(rgb[j]), m[j % 3], o[j % 3]);
        }
        for (size_t i = 0; i < N; ++i) {
            for (int y = 0; y < PLANE_H; ++y) {
                for (int x = 0; x < PLANE_W; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        float sum = 0.f;
                        for (int dy = -anchor; dy <= anchor; ++dy) {
                            for (int dx = -anchor; dx <= anchor; ++dx) {
                                sum += converted[at(i, clampY(y + dy), clampX(x + dx), c)];
                            }
                        }
                        out[at(i, y, x, c)] = sum * scale;
                    }
                }
            }
        }
    }
    if (!normInCmn) {
        for (size_t j = 0; j < out.size(); ++j) {
            const int c = static_cast<int>(j % 3);
            out[j] = std::fmaf(out[j] - mean[c], invStdDev[c], 0.f);
        }
    }
    return out;
}

// ------------------------------------------------------------------------------------------------
// FKL
// ------------------------------------------------------------------------------------------------
template <size_t N>
std::shared_ptr<fk::Tensor<fk::float3>> planeTensor() {
    return std::make_shared<fk::Tensor<fk::float3>>(PLANE_W, PLANE_H, N, 1, fk::MemType::Device);
}

template <size_t N>
void planeIo(Implementation& impl, Context& ctx, const std::shared_ptr<fk::Tensor<fk::float3>>& out) {
    impl.download = [&ctx, out]() { return downloadDevice(ctx.stream, out->ptr().data, N * PLANE_FLOATS); };
    impl.fillNaN = [&ctx, out]() { poisonDevice(ctx.stream, out->ptr().data, N * PLANE_FLOATS); };
}

// The BVF chain before the box: ReadNV12 -> YUVtoRGB -> Crop, and the resize for part (a) with the resize
template <bool R, typename Rects>
auto chainBeforeBox(const Context& ctx, const Rects& rects) {
    const auto crop =
        fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice).then(YUVToRGB::build()).then(fk::Crop<>::build(rects));
    if constexpr (R) {
        return crop.then(ResizeTo64x32::build(fk::Size(PLANE_W, PLANE_H)));
    } else {
        return crop;
    }
}

// MorphologyDPP's box blur with the BVF chain up to the Crop (or the Resize) as its input and the pointwise tail and the
// write as its output, batched over the crops (grid z), with a W x H tile and a K x K mask, P passes
template <size_t N, int K, int P, bool R, int TW, int TH>
Implementation makeFklDpp(Context& ctx) {
    auto out = planeTensor<N>();
    const auto sets = rectSetsOf<N>(regionsOf<R>(ctx));
    Implementation impl{"fkl_dpp_" + std::to_string(TW) + "x" + std::to_string(TH), "fkl_cpu_float"};
    impl.run = [&ctx, out, sets](size_t t) {
        using Details = fk::MorphologyDPPDetails<fk::float3, TW, TH, K, K>;
        const Details details{PLANE_W, PLANE_H, K, K, K / 2, K / 2};
        const auto input = chainBeforeBox<R>(ctx, (*sets)[t]);
        const auto output = fk::Mul<fk::float3>::build(mulValue)
                                .then(fk::Sub<fk::float3>::build(subValue))
                                .then(fk::Div<fk::float3>::build(divValue))
                                .then(fk::TensorWrite<fk::float3>::build(out->ptr()));
        beforeKernel();
        if (!fk::executeBoxBlur<P>(details, input, output, *ctx.fkStream)) {
            throw std::runtime_error("MorphologyDPP rejected its details");
        }
    };
    planeIo<N>(impl, ctx, out);
    return impl;
}

// The naive operation of common/bench_neighborhood_ops.h in the BVF chain: every output value reads K x K cropped
// (or resized) values (K^4 for two passes), each of them the color conversion of its NV12 values (four of them for a
// resized value)
template <size_t N, int K, int P, bool R>
Implementation makeFklNaive(Context& ctx) {
    static_assert(P == 1 || P == 2, "part (a) runs one or two passes");
    auto out = planeTensor<N>();
    const auto sets = rectSetsOf<N>(regionsOf<R>(ctx));
    Implementation impl{"fkl_bvf_naive", "fkl_cpu_float"};
    impl.run = [&ctx, out, sets](size_t t) {
        beforeKernel();
        if constexpr (R && P == 1) {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                YUVToRGB::build(),
                fk::Crop<>::build((*sets)[t]),
                ResizeTo64x32::build(fk::Size(PLANE_W, PLANE_H)),
                fk::BoxBVF<K>::build(),
                fk::Mul<fk::float3>::build(mulValue),
                fk::Sub<fk::float3>::build(subValue),
                fk::Div<fk::float3>::build(divValue),
                fk::TensorWrite<fk::float3>::build(out->ptr()));
        } else if constexpr (R) {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                YUVToRGB::build(),
                fk::Crop<>::build((*sets)[t]),
                ResizeTo64x32::build(fk::Size(PLANE_W, PLANE_H)),
                fk::BoxBVF<K>::build(),
                fk::BoxBVF<K>::build(),
                fk::Mul<fk::float3>::build(mulValue),
                fk::Sub<fk::float3>::build(subValue),
                fk::Div<fk::float3>::build(divValue),
                fk::TensorWrite<fk::float3>::build(out->ptr()));
        } else if constexpr (P == 1) {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                YUVToRGB::build(),
                fk::Crop<>::build((*sets)[t]),
                fk::BoxBVF<K>::build(),
                fk::Mul<fk::float3>::build(mulValue),
                fk::Sub<fk::float3>::build(subValue),
                fk::Div<fk::float3>::build(divValue),
                fk::TensorWrite<fk::float3>::build(out->ptr()));
        } else {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                YUVToRGB::build(),
                fk::Crop<>::build((*sets)[t]),
                fk::BoxBVF<K>::build(),
                fk::BoxBVF<K>::build(),
                fk::Mul<fk::float3>::build(mulValue),
                fk::Sub<fk::float3>::build(subValue),
                fk::Div<fk::float3>::build(divValue),
                fk::TensorWrite<fk::float3>::build(out->ptr()));
        }
    };
    planeIo<N>(impl, ctx, out);
    return impl;
}

// ------------------------------------------------------------------------------------------------
// RPP (HIP backend), on Experiment 1's RppState (rocm_bench.hip) for its handle (batch size N), output tensor,
// conversion-only crop_mirror_normalize parameters, mirror flags and rppt_normalize constants; plus the RGB frame
// RPP_LEAD_BYTES into its own allocation (the box filter's source), the crops of every iteration (XYWH on the frame,
// in pinned memory, uploaded by every run), the fixed XYWH ROIs of the whole 64x32 images (uploaded once), the 8 bit
// boxed crops (U8 variants) or the converted crops RPP_LEAD_BYTES into their allocation (F32 variants), and the
// crop_mirror_normalize parameters of the whole normalization. The output is the first N planes of 64x32 float3 of
// RppState's tensor, contiguous.
// ------------------------------------------------------------------------------------------------
struct RppBox8 {
    std::shared_ptr<RppState> st;
    uint8_t* rgb{nullptr};       // the RGB frame at RPP_LEAD_BYTES
    uint8_t* boxed{nullptr};     // U8: N boxed crops of 64x32
    uint8_t* converted{nullptr}; // F32: N converted crops of 64x32 at RPP_LEAD_BYTES
    RpptROI* crops{nullptr};     // device: the crops of the iteration
    RpptROI* cropsAll{nullptr};  // pinned: the crops of every iteration
    RpptROI* planeRois{nullptr}; // device: (0, 0, 64, 32) of every image
    float* normMultiplier{nullptr};
    float* normOffset{nullptr};
    RpptDesc frameDesc, boxedDesc, convertedDesc, outDesc;
    ~RppBox8() {
        (void)hipFree(rgb);
        (void)hipFree(boxed);
        (void)hipFree(converted);
        (void)hipFree(crops);
        (void)hipHostFree(cropsAll);
        (void)hipFree(planeRois);
        (void)hipHostFree(normMultiplier);
        (void)hipHostFree(normOffset);
    }
};

// conversion: the NV12 -> RGB conversion of the full frame (rppt_yuv_to_rgb once), or of only the N crops of the
// iteration (one rppt_yuv_to_rgb per crop, rocm_bench.hip's rppYuvToRgbCrop, into the same RGB frame at the crops'
// positions; name with "_crop"). The calls after it are the same: the box reads the crops of the frame through their
// ROIs and replicates each crop's own edge (REPLICATE), so a crop needs no pixel outside it.
template <size_t N>
Implementation makeRppBox(Context& ctx, const RppVariant& variant, RppConversion conversion) {
    constexpr int K = BOX_K;
    auto st = makeRppState(ctx, N, false, RppRois::None);
    auto p = std::make_shared<RppBox8>();
    p->st = st;
    p->rgb = deviceAlloc<uint8_t>(RPP_LEAD_BYTES + static_cast<size_t>(FRAME_W) * FRAME_H * 3 + RPP_SLACK_BYTES);
    const size_t iterations = ctx.regions->iterations();
    p->cropsAll = hostAlloc<RpptROI>(iterations * N);
    for (size_t t = 0; t < iterations; ++t) {
        for (size_t i = 0; i < N; ++i) {
            const Region& r = ctx.regions->at(t, i);
            if (r.width != PLANE_W || r.height != PLANE_H) {
                throw std::runtime_error("The crops must be " + std::to_string(PLANE_W) + "x" + std::to_string(PLANE_H));
            }
            p->cropsAll[t * N + i].xywhROI = {{r.x, r.y}, r.width, r.height};
        }
    }
    p->crops = deviceAlloc<RpptROI>(N * sizeof(RpptROI));
    std::vector<RpptROI> whole(N);
    for (auto& r : whole) {
        r.xywhROI = {{0, 0}, PLANE_W, PLANE_H};
    }
    p->planeRois = deviceAlloc<RpptROI>(N * sizeof(RpptROI));
    gpuErrchk(hipMemcpyAsync(p->planeRois, whole.data(), N * sizeof(RpptROI), hipMemcpyHostToDevice, ctx.stream));
    p->normMultiplier = hostAlloc<float>(N * 3);
    p->normOffset = hostAlloc<float>(N * 3);
    for (size_t i = 0; i < N; ++i) {
        for (int c = 0; c < 3; ++c) {
            p->normMultiplier[i * 3 + c] = cmnMultiplier(c);
            p->normOffset[i * 3 + c] = cmnOffset(c);
        }
    }
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    // Every batch image aliases the RGB frame (zero batch stride), each with its crop as ROI
    p->frameDesc = st->frameBatchDesc;
    p->frameDesc.offsetInBytes = static_cast<Rpp32u>(RPP_LEAD_BYTES);
    const RpptStrides planes{static_cast<Rpp32u>(PLANE_FLOATS), 1, static_cast<Rpp32u>(PLANE_W * 3), 3};
    p->boxedDesc = RpptDesc(static_cast<Rpp32u>(N), 3, PLANE_H, PLANE_W, RpptDataType::U8, RpptLayout::NHWC, 0, planes);
    p->convertedDesc = RpptDesc(static_cast<Rpp32u>(N), 3, PLANE_H, PLANE_W, RpptDataType::F32, RpptLayout::NHWC,
                                static_cast<Rpp32u>(RPP_LEAD_BYTES), planes);
    p->outDesc = RpptDesc(static_cast<Rpp32u>(N), 3, PLANE_H, PLANE_W, RpptDataType::F32, RpptLayout::NHWC, 0, planes);
    if (variant.box == RppBox::U8) {
        p->boxed = deviceAlloc<uint8_t>(N * PLANE_FLOATS + RPP_SLACK_BYTES);
    } else {
        p->converted = deviceAlloc<uint8_t>(RPP_LEAD_BYTES + N * PLANE_FLOATS * sizeof(float) + RPP_SLACK_BYTES);
    }
    // rppt_normalize on the rows of the N planes: N * PLANE_H rows of PLANE_W float3
    const Rpp32u rows = static_cast<Rpp32u>(N * PLANE_H), cols = static_cast<Rpp32u>(PLANE_W * 3);
    if (st->rowsRoi[3] != cols) {
        throw std::runtime_error("unexpected rppt_normalize row length");
    }
    st->rowsDesc.dims[1] = rows;
    st->rowsDesc.strides[0] = rows * cols;
    st->rowsRoi[2] = rows;

    const bool normInCmn = variant.norm == RppNorm::InCmn;
    float* multiplier = normInCmn ? p->normMultiplier : st->cmnMultiplier;
    float* offset = normInCmn ? p->normOffset : st->cmnOffset;
    const std::string k = std::to_string(K);
    const std::string cmnWhat = normInCmn ? "_normalizing" : "_converting";
    std::vector<RppStage> stages;
    stages.push_back({"hipMemcpyAsync_rois_of_the_iteration", [&ctx, p](size_t t) {
                          beforeKernel();
                          gpuErrchk(hipMemcpyAsync(p->crops, p->cropsAll + t * N, N * sizeof(RpptROI),
                                                   hipMemcpyHostToDevice, ctx.stream));
                      }});
    if (conversion == RppConversion::Frame) {
        stages.push_back({"rppt_yuv_to_rgb_full_frame", [&ctx, p](size_t) {
                              RppState& s = *p->st;
                              beforeKernel();
                              RPP_CHECK(rppt_yuv_to_rgb(ctx.y, ctx.uv, &s.nv12Desc, p->rgb + RPP_LEAD_BYTES, &s.rgbDesc,
                                                        ctx.pitch, ctx.pitch, FRAME_W * 3, FRAME_W, FRAME_H,
                                                        RpptColorStandard_BT601, RpptColorRange_STUDIO, s.handle,
                                                        RppBackend::RPP_HIP_BACKEND));
                          }});
    } else {
        // The calls of every iteration's crops, computed now (rocm_bench.hip's rppCropCalls), as the ROIs are
        auto calls = std::make_shared<const std::vector<RppCropCall>>(rppCropCalls(ctx, p->rgb + RPP_LEAD_BYTES, N));
        stages.push_back({"rppt_yuv_to_rgb_per_crop", [&ctx, p, calls](size_t t) {
                              RppState& s = *p->st;
                              for (size_t i = 0; i < N; ++i) {
                                  beforeKernel();
                                  rppYuvToRgbCrop(ctx, s.nv12Desc, s.rgbDesc, s.handle, (*calls)[t * N + i]);
                              }
                          }});
    }
    if (variant.box == RppBox::U8) {
        stages.push_back({"rppt_box_filter_u8_" + k + "x" + k + "_crops_of_the_frame", [p](size_t) {
                              beforeKernel();
                              RPP_CHECK(rppt_box_filter(p->rgb, &p->frameDesc, p->boxed, &p->boxedDesc, K,
                                                        RpptImageBorderType::REPLICATE, p->crops, RpptRoiType::XYWH,
                                                        p->st->handle, RppBackend::RPP_HIP_BACKEND));
                          }});
        stages.push_back({"rppt_crop_mirror_normalize_8u32f" + cmnWhat, [p, multiplier, offset](size_t) {
                              RppState& s = *p->st;
                              beforeKernel();
                              RPP_CHECK(rppt_crop_mirror_normalize(p->boxed, &p->boxedDesc, s.tensor, &p->outDesc,
                                                                   offset, multiplier, s.mirror, p->planeRois,
                                                                   RpptRoiType::XYWH, s.handle,
                                                                   RppBackend::RPP_HIP_BACKEND));
                          }});
    } else {
        stages.push_back({"rppt_crop_mirror_normalize_8u32f" + cmnWhat + "_crops_of_the_frame",
                          [p, multiplier, offset](size_t) {
                              RppState& s = *p->st;
                              beforeKernel();
                              RPP_CHECK(rppt_crop_mirror_normalize(p->rgb, &p->frameDesc, p->converted,
                                                                   &p->convertedDesc, offset, multiplier, s.mirror,
                                                                   p->crops, RpptRoiType::XYWH, s.handle,
                                                                   RppBackend::RPP_HIP_BACKEND));
                          }});
        stages.push_back({"rppt_box_filter_f32_" + k + "x" + k, [p](size_t) {
                              RppState& s = *p->st;
                              beforeKernel();
                              RPP_CHECK(rppt_box_filter(p->converted, &p->convertedDesc, s.tensor, &p->outDesc, K,
                                                        RpptImageBorderType::REPLICATE, p->planeRois, RpptRoiType::XYWH,
                                                        s.handle, RppBackend::RPP_HIP_BACKEND));
                          }});
    }
    if (!normInCmn) {
        stages.push_back(normalizeStage(st));
    }

    // Checked against the float reference with TOLERANCE_8BIT, and against its own RPP-convention reference
    Implementation impl{conversion == RppConversion::Frame ? std::string(variant.name)
                                                           : std::string(variant.name) + "_crop",
                        variant.box == RppBox::U8 ? "fkl_cpu_8bit_rgb_rounded_box" : "fkl_cpu_8bit_rgb"};
    for (size_t i = 0; i < stages.size(); ++i) {
        stages[i].first = std::to_string(i + 1) + "_" + stages[i].first;
    }
    impl.stages = std::move(stages);
    impl.run = [stages = impl.stages](size_t t) {
        for (const auto& stage : stages) {
            stage.second(t);
        }
    };
    impl.download = [&ctx, p]() { return downloadDevice(ctx.stream, p->st->tensor, N * PLANE_FLOATS); };
    // Before a per-crop pipeline is validated, its RGB frame holds RGB_SENTINEL everywhere (rocm_bench.hip): a call
    // reading a pixel no crop converted would give an output off the reference
    impl.fillNaN = [&ctx, p, conversion]() {
        poisonDevice(ctx.stream, p->st->tensor, N * PLANE_FLOATS);
        if (conversion == RppConversion::Crop) {
            gpuErrchk(hipMemsetAsync(p->rgb + RPP_LEAD_BYTES, RGB_SENTINEL, static_cast<size_t>(FRAME_W) * FRAME_H * 3,
                                     ctx.stream));
        }
    };
    return impl;
}

// ------------------------------------------------------------------------------------------------
// Groups
// ------------------------------------------------------------------------------------------------
template <size_t N, int K, int P, bool R, size_t... Is>
void addDppTiles(Context& ctx, std::vector<Implementation>& impls, std::index_sequence<Is...>) {
    (impls.push_back(makeFklDpp<N, K, P, R, DPP_TILES[Is].width, DPP_TILES[Is].height>(ctx)), ...);
}

// Part (b): the RPP variants, then the DPP tiles; part (a): the naive FKL pipeline, then the DPP tiles
template <char PART, size_t N, int K, int P, bool R>
std::vector<Implementation> makeGroup(Context& ctx) {
    static_assert(PART == 'a' || (K == BOX_K && P == 1 && !R), "part (b) is one pass of the BOX_K x BOX_K box");
    std::vector<Implementation> impls;
    if constexpr (PART == 'b') {
        for (const auto& v : RPP_VARIANTS_NB) {
            impls.push_back(makeRppBox<N>(ctx, v, RppConversion::Frame));
            impls.push_back(makeRppBox<N>(ctx, v, RppConversion::Crop));
        }
    } else {
        impls.push_back(makeFklNaive<N, K, P, R>(ctx));
    }
    addDppTiles<N, K, P, R>(ctx, impls, std::make_index_sequence<DPP_TILES.size()>{});
    return impls;
}

bool isDpp(const std::string& name) { return name.rfind("fkl_dpp_", 0) == 0; }
bool isRpp(const std::string& name) { return name.rfind("rpp_", 0) == 0; }

// One implementation of one group. Result (rocm_bench.hip) keeps the timings, errors and counts; its
// meanSpeedupOfBvf fields hold the mean speedup of the group's chosen DPP tile over this implementation here.
struct GroupResult {
    Group group;
    Result r;
    bool chosen{false};
    bool equivalent{false}; // RPP: within RPP_EQUIVALENCE of its RPP-convention reference in every validated iteration
};
std::vector<GroupResult> groupResults;
// The chosen DPP tile of every group (label -> implementation), from the CPU + GPU pass; the profiling child gets
// them in a file
std::map<std::string, std::string> chosenDpp;

std::string flatName(const Group& g, const std::string& impl) { return g.label() + "/" + impl; }

template <char PART, size_t N, int K, int P, bool R>
void runGroup(Context& ctx, bool timing) {
    const Group g{PART, K, P, static_cast<int>(N), R};
    std::cout << "=== " << g.label() << std::endl;
    progress("implementations");
    const auto impls = makeGroup<PART, N, K, P, R>(ctx);
    std::vector<GroupResult> rs(impls.size());
    for (size_t k = 0; k < impls.size(); ++k) {
        rs[k].group = g;
        rs[k].r.name = impls[k].name;
        rs[k].r.batch = static_cast<int>(N);
        rs[k].r.reference = impls[k].reference;
        rs[k].equivalent = isRpp(impls[k].name);
    }
    const auto validated = validation::validatedIterations(TIMED_ITERS);
    std::vector<std::vector<float>> firstOutputs(impls.size());
    for (size_t v = 0; v < validated.size(); ++v) {
        const size_t t = static_cast<size_t>(WARMUP_ITERS) + validated[v];
        progress("float reference of iteration " + std::to_string(t));
        const auto refFloat = floatReference<N, R>(ctx, t, K, P);
        const auto ref8bit = PART == 'b' ? eightBitReference<N>(ctx, t, K, P, false) : std::vector<float>{};
        const auto ref8bitRoundedBox = PART == 'b' ? eightBitReference<N>(ctx, t, K, P, true) : std::vector<float>{};
        std::vector<std::vector<float>> outputs(impls.size());
        for (size_t k = 0; k < impls.size(); ++k) {
            const Implementation& impl = impls[k];
            progress("validating " + impl.name + " on iteration " + std::to_string(t));
            impl.fillNaN();
            impl.run(t);
            const hipError_t launchError = hipGetLastError();
            if (launchError != hipSuccess) {
                throw std::runtime_error(g.label() + " " + impl.name + ": HIP error after the pipeline: " +
                                         hipGetErrorString(launchError));
            }
            gpuErrchk(hipStreamSynchronize(ctx.stream));
            const auto output = impl.download();
            outputs[k] = output;
            if (v == 0) {
                firstOutputs[k] = output;
            }
            Result::IterationErrors e;
            e.iteration = t;
            e.vsFloat = compare(output, refFloat);
            e.own = e.vsFloat;
            if (isRpp(impl.name)) {
                e.own = compare(output, rppVariant(impl.name).box == RppBox::U8 ? ref8bitRoundedBox : ref8bit);
                e.vsRppConvention = compare(output, rppConventionReference<N>(ctx, t, rppVariant(impl.name)));
                rs[k].equivalent = rs[k].equivalent && e.vsRppConvention.nonFinite == 0 &&
                                   e.vsRppConvention.max() <= RPP_EQUIVALENCE;
            }
            rs[k].r.addIteration(e);
        }
        // Every <variant>_crop must give exactly its full-frame twin's output (rocm_bench.hip)
        requireCropFormsMatchTheirFrameForms(impls, outputs, t);
    }
    if (!timing) {
        for (auto& gr : rs) {
            std::cout << "  " << std::left << std::setw(22) << gr.r.name << std::right << "  max err " << std::scientific
                      << std::setprecision(2) << gr.r.errorVsOwnReference.max() << " (" << gr.r.reference << ")";
            if (isRpp(gr.r.name)) {
                std::cout << ", vs its RPP-convention reference " << gr.r.errorVsRppConventionReference.max();
            }
            std::cout << std::defaultfloat << (gr.r.withinTolerance() ? "" : "  NOT within its tolerance")
                      << (isRpp(gr.r.name) && !gr.equivalent ? "  NOT equivalent to its RPP-convention reference" : "")
                      << std::endl;
            groupResults.push_back(gr);
        }
        return;
    }
    progress("CPU + GPU timing");
    const auto ms = measureSequential(ctx, impls);
    // The output after the timed iterations must still be the validated one (as in rocm_bench.hip)
    const size_t firstValidated = static_cast<size_t>(WARMUP_ITERS) + validated.front();
    for (size_t k = 0; k < impls.size(); ++k) {
        impls[k].fillNaN();
        impls[k].run(firstValidated);
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        const ErrorStats after = compare(impls[k].download(), firstOutputs[k]);
        if (after.nonFinite != 0 || after.max() > 1e-6f) {
            throw std::runtime_error(g.label() + " " + impls[k].name +
                                     ": the output after the timed iterations differs from the validated one");
        }
    }
    // The chosen DPP tile (fastest within its tolerance) and RPP variant (fastest within its tolerance and equivalent
    // to its RPP-convention reference)
    size_t dpp = impls.size();
    for (size_t k = 0; k < impls.size(); ++k) {
        rs[k].r.timing = computeStats(ms[k]);
        rs[k].r.ms = ms[k];
    }
    for (size_t k = 0; k < impls.size(); ++k) {
        if (isDpp(impls[k].name) && rs[k].r.withinTolerance() &&
            (dpp == impls.size() || rs[k].r.timing.median < rs[dpp].r.timing.median)) {
            dpp = k;
        }
    }
    if (dpp == impls.size()) {
        throw std::runtime_error(g.label() + ": no DPP tile is within its tolerance");
    }
    rs[dpp].chosen = true;
    chosenDpp[g.label()] = impls[dpp].name;
    size_t rpp = impls.size();
    for (size_t k = 0; k < impls.size(); ++k) {
        if (isRpp(impls[k].name) && rs[k].equivalent && rs[k].r.withinTolerance() &&
            (rpp == impls.size() || rs[k].r.timing.median < rs[rpp].r.timing.median)) {
            rpp = k;
        }
    }
    if (PART == 'b' && rpp == impls.size()) {
        throw std::runtime_error(g.label() + ": no RPP variant is within its tolerance and equivalent to its "
                                             "RPP-convention reference");
    }
    for (size_t k = 0; k < impls.size(); ++k) {
        GroupResult& gr = rs[k];
        gr.chosen = gr.chosen || k == rpp;
        if (k != dpp) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += static_cast<double>(ms[k][i]) / ms[dpp][i];
            }
            gr.r.meanSpeedupOfBvf = sum / TIMED_ITERS;
        }
        std::cout << "  " << std::left << std::setw(22) << gr.r.name << std::right << (gr.chosen ? " *" : "  ")
                  << std::fixed << std::setprecision(1) << " median " << std::setw(8) << gr.r.timing.median
                  << " us  mean " << std::setw(8) << gr.r.timing.mean << " us  p10 " << std::setw(8) << gr.r.timing.p10
                  << "  p90 " << std::setw(8) << gr.r.timing.p90;
        if (k != dpp) {
            std::cout << "  " << impls[dpp].name << " mean speedup " << std::setprecision(2) << gr.r.meanSpeedupOfBvf
                      << "x";
        }
        std::cout << "  max err " << std::scientific << std::setprecision(2) << gr.r.errorVsOwnReference.max() << " ("
                  << gr.r.reference << ")" << std::defaultfloat << std::endl;
        groupResults.push_back(gr);
    }
    // Per operation breakdown of the RPP pipelines
    for (const auto& impl : impls) {
        if (!impl.stages.empty()) {
            Implementation labelled = impl;
            labelled.name = flatName(g, impl.name);
            measureStages(ctx, labelled, static_cast<int>(N));
        }
    }
}

// Part (a) with or without the resize: every K, every number of passes, every N
template <bool R, typename F>
void forEachPartAGroup(F&& f) {
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
        ([&]<size_t KI>() {
            constexpr int K = neighborhood::PART_A_KS[KI];
            [&]<size_t... Ps>(std::index_sequence<Ps...>) {
                ([&]<size_t PI>() {
                    constexpr int P = neighborhood::PART_A_PASSES[PI];
                    [&]<size_t... Ns>(std::index_sequence<Ns...>) {
                        (f.template operator()<'a', neighborhood::PART_A_BATCHES[Ns], K, P, R>(), ...);
                    }(std::make_index_sequence<neighborhood::PART_A_BATCHES.size()>{});
                }.template operator()<Ps>(), ...);
            }(std::make_index_sequence<neighborhood::PART_A_PASSES.size()>{});
        }.template operator()<Ks>(), ...);
    }(std::make_index_sequence<neighborhood::PART_A_KS.size()>{});
}

// Every group of the experiment: f.template operator()<part, N, K, passes, resize>() for part (b), then part (a)
// without the resize, then part (a) with the resize
template <typename F>
void forEachGroup(F&& f) {
    // Part (b): every N, one pass of the BOX_K x BOX_K box
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (f.template operator()<'b', BATCHES[Is], BOX_K, 1, false>(), ...);
    }(std::make_index_sequence<BATCHES.size()>{});
    forEachPartAGroup<false>(f);
    forEachPartAGroup<true>(f);
}

// ------------------------------------------------------------------------------------------------
// Profiling child: kernel counts, GPU time only and GPU time only with the caches flushed, of every implementation of
// every group, with the speedups over the parent's chosen DPP tiles. Rows in rocm_bench.hip's format, with the
// implementation named <group>/<implementation>.
// ------------------------------------------------------------------------------------------------
std::vector<Result> profileRows;

Result& profileRow(const Group& g, const std::string& impl) {
    const std::string name = flatName(g, impl);
    auto r = std::find_if(profileRows.begin(), profileRows.end(),
                          [&](const Result& x) { return x.name == name && x.batch == g.batch; });
    if (r == profileRows.end()) {
        Result fresh;
        fresh.name = name;
        fresh.batch = g.batch;
        profileRows.push_back(fresh);
        r = profileRows.end() - 1;
    }
    return *r;
}

template <char PART, size_t N, int K, int P, bool R>
void countGroup(Context& ctx) {
    const Group g{PART, K, P, static_cast<int>(N), R};
    const size_t t = static_cast<size_t>(WARMUP_ITERS);
    for (const auto& impl : makeGroup<PART, N, K, P, R>(ctx)) {
        impl.run(t); // warm-up: code object loading is not counted
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        profileRow(g, impl.name).launches = countLaunches(ctx, impl, t);
        for (const auto& stage : impl.stages) {
            rocprof::counts = LaunchCounts{};
            ROCP_CHECK(rocprofiler_start_context(rocprof::apiContext));
            stage.second(t);
            gpuErrchk(hipStreamSynchronize(ctx.stream));
            ROCP_CHECK(rocprofiler_stop_context(rocprof::apiContext));
            operationCounts.push_back({flatName(g, impl.name), static_cast<int>(N), stage.first, rocprof::counts});
        }
    }
}

// rocm_bench.hip's gpuOnlyBatch for the implementations of one group, with the speedups over the chosen DPP tile
template <char PART, size_t N, int K, int P, bool R>
void gpuOnlyGroup(Context& ctx, CacheFlusher* flusher) {
    const Group g{PART, K, P, static_cast<int>(N), R};
    const auto impls = makeGroup<PART, N, K, P, R>(ctx);
    const auto chosen = chosenDpp.find(g.label());
    if (chosen == chosenDpp.end()) {
        throw std::runtime_error("No chosen DPP tile for " + g.label());
    }
    const size_t dpp = static_cast<size_t>(std::find_if(impls.begin(), impls.end(), [&](const Implementation& i) {
                                               return i.name == chosen->second;
                                           }) - impls.begin());
    if (dpp == impls.size()) {
        throw std::runtime_error("The chosen DPP tile " + chosen->second + " is not in " + g.label());
    }
    std::vector<size_t> kernelsPerRun(impls.size());
    for (size_t k = 0; k < impls.size(); ++k) {
        impls[k].run(static_cast<size_t>(WARMUP_ITERS));
        gpuErrchk(hipStreamSynchronize(ctx.stream));
        kernelsPerRun[k] = countLaunches(ctx, impls[k], static_cast<size_t>(WARMUP_ITERS)).kernels;
    }
    rocprof::flushKernels();
    {
        std::lock_guard<std::mutex> lock(rocprof::mutex);
        rocprof::kernels.clear();
    }
    // Each implementation's complete loop in turn, the chosen DPP tile among the DPP tiles after the baselines
    activeFlusher = flusher;
    const std::vector<Window> windows = runGpuOnlyLoops(ctx, impls);
    // rocprofiler-sdk writes a dispatch record after the kernel completes, on its own thread: flush until the records
    // of every execution (at least kernelsPerRun of each) have arrived, for at most 2 s
    size_t expected = 0;
    for (const size_t n : kernelsPerRun) {
        expected += n * static_cast<size_t>(TIMED_ITERS);
    }
    for (int attempt = 0; attempt < 400; ++attempt) {
        rocprof::flushKernels();
        size_t received = 0;
        {
            std::lock_guard<std::mutex> lock(rocprof::mutex);
            for (const auto& r : rocprof::kernels) {
                received += !rocprof::isFlushKernel(r.kernelId) && !rocprof::isRuntimeBlitKernel(r.kernelId);
            }
        }
        if (received >= expected) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
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
                throw std::runtime_error("GPU only benchmark: " + g.label() + " " + impls[k].name + " execution " +
                                         std::to_string(i) + " has " + std::to_string(kernelCount[k][i]) +
                                         " kernel records, expected " + std::to_string(kernelsPerRun[k]) + " (" +
                                         std::to_string(unassigned) + " kernel records outside every execution)");
            }
        }
    }
    for (size_t k = 0; k < impls.size(); ++k) {
        Result& r = profileRow(g, impls[k].name);
        r.launches.kernels = kernelsPerRun[k];
        TimingStats& timing = flusher ? r.gpuFlushedTiming : r.gpuTiming;
        double& speedup = flusher ? r.meanGpuFlushedSpeedupOfBvf : r.meanGpuSpeedupOfBvf;
        (flusher ? r.gpuFlushedMeasured : r.gpuMeasured) = true;
        timing = computeStats(ms[k]);
        (flusher ? r.gpuFlushedMs : r.gpuMs) = ms[k];
        (flusher ? r.gpuFlushedBlitTiming : r.gpuBlitTiming) = computeStats(blitMs[k]);
        std::vector<float> counts = blitCount[k];
        std::sort(counts.begin(), counts.end());
        (flusher ? r.gpuFlushedBlitDispatches : r.gpuBlitDispatches) = percentile(counts, 0.5);
        if (k != dpp) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += static_cast<double>(ms[k][i]) / ms[dpp][i];
            }
            speedup = sum / TIMED_ITERS;
        }
        std::cout << (flusher ? "GPU only, L2 + MALL flushed " : "GPU only ") << g.label() << " " << std::left
                  << std::setw(22) << impls[k].name << std::right << std::fixed << std::setprecision(2) << "  median "
                  << std::setw(8) << timing.median << " us  p10 " << std::setw(8) << timing.p10 << "  p90 "
                  << std::setw(8) << timing.p90;
        if (k != dpp) {
            std::cout << "  " << impls[dpp].name << " mean speedup " << std::setw(7) << speedup << "x";
        }
        std::cout << std::defaultfloat << std::endl;
    }
}

// rocm_bench.hip's kernelTracingAvailable, waiting for the probe's dispatch record (rocprofiler-sdk writes it after the
// kernel completes, on its own thread) for at most 2 s instead of checking right after one flush
bool kernelTracingWorks(Context& ctx) {
    rocprof::requireConfigured();
    const auto probe = makeFklBvf<1>(ctx);
    probe.run(0);
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    rocprof::flushKernels();
    {
        std::lock_guard<std::mutex> lock(rocprof::mutex);
        rocprof::kernels.clear();
    }
    ROCP_CHECK(rocprofiler_start_context(rocprof::kernelContext));
    probe.run(1);
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    ROCP_CHECK(rocprofiler_stop_context(rocprof::kernelContext));
    bool available = false;
    for (int attempt = 0; attempt < 400 && !available; ++attempt) {
        rocprof::flushKernels();
        {
            std::lock_guard<std::mutex> lock(rocprof::mutex);
            available = !rocprof::kernels.empty();
        }
        if (!available) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    std::lock_guard<std::mutex> lock(rocprof::mutex);
    rocprof::kernels.clear();
    return available;
}

void writeChoices(const std::filesystem::path& path) {
    std::ofstream os(path);
    for (const auto& [label, impl] : chosenDpp) {
        os << label << "," << impl << "\n";
    }
}

void readChoices(const std::filesystem::path& path) {
    std::ifstream is(path);
    std::string line;
    while (std::getline(is, line)) {
        const auto comma = line.find(',');
        if (comma != std::string::npos) {
            chosenDpp[line.substr(0, comma)] = line.substr(comma + 1);
        }
    }
    if (chosenDpp.empty()) {
        throw std::runtime_error("No chosen DPP tiles in " + path.string());
    }
}

void profilePass(Context& ctx, const std::filesystem::path& outDir, const std::filesystem::path& rowsFile,
                 const std::filesystem::path& choicesFile) {
    readChoices(choicesFile);
    std::cout << "=== Kernel counts (HIP runtime API tracing)" << std::endl;
    forEachGroup([&]<char PART, size_t N, int K, int P, bool R>() { countGroup<PART, N, K, P, R>(ctx); });
    writeOperationCounts(outDir / "kernel_counts_per_operation.csv");
    const bool tracing = kernelTracingWorks(ctx);
    if (tracing) {
        std::cout << "=== GPU time only (kernel dispatch tracing)" << std::endl;
        forEachGroup([&]<char PART, size_t N, int K, int P, bool R>() { gpuOnlyGroup<PART, N, K, P, R>(ctx, nullptr); });
        CacheFlusher flusher = makeCacheFlusher(ctx);
        forEachGroup([&]<char PART, size_t N, int K, int P, bool R>() { gpuOnlyGroup<PART, N, K, P, R>(ctx, &flusher); });
        freeCacheFlusher(flusher);
    } else {
        std::cout << "=== GPU time only: NOT MEASURED (no kernel dispatch records from rocprofiler-sdk on this system)"
                  << std::endl;
    }
    writeProfileRows(rowsFile, profileRows, tracing);
}

bool runProfilingChild(const std::filesystem::path& outDir) {
    const auto rows = std::filesystem::absolute(outDir / ".profiling_pass.csv");
    const auto choices = std::filesystem::absolute(outDir / ".chosen_dpp_tiles.csv");
    writeChoices(choices);
    const std::string exe = std::filesystem::read_symlink("/proc/self/exe").string();
    const auto quote = [](const std::string& value) {
        std::string quoted = "'";
        for (const char c : value) {
            quoted += c == '\'' ? std::string("'\\''") : std::string(1, c);
        }
        return quoted + "'";
    };
    const std::string command = "BENCH_ROCPROFILER=1 " + quote(exe) + " " + quote(outDir.string()) + " --profile-pass " +
                                quote(rows.string()) + " " + quote(choices.string());
    std::cout << "=== Profiling pass in a child process: " << command << std::endl;
    std::filesystem::remove(rows);
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("The profiling pass failed");
    }
    for (auto& gr : groupResults) {
        Result fresh;
        fresh.name = flatName(gr.group, gr.r.name);
        fresh.batch = gr.r.batch;
        profileRows.push_back(fresh);
    }
    const bool tracing = mergeProfileRows(rows, profileRows);
    for (auto& gr : groupResults) {
        const std::string name = flatName(gr.group, gr.r.name);
        const auto p = std::find_if(profileRows.begin(), profileRows.end(),
                                    [&](const Result& x) { return x.name == name && x.batch == gr.r.batch; });
        Result& r = gr.r;
        r.launches = p->launches;
        r.gpuMeasured = p->gpuMeasured;
        r.gpuTiming = p->gpuTiming;
        r.meanGpuSpeedupOfBvf = p->meanGpuSpeedupOfBvf;
        r.gpuFlushedMeasured = p->gpuFlushedMeasured;
        r.gpuFlushedTiming = p->gpuFlushedTiming;
        r.meanGpuFlushedSpeedupOfBvf = p->meanGpuFlushedSpeedupOfBvf;
        r.gpuBlitTiming = p->gpuBlitTiming;
        r.gpuBlitDispatches = p->gpuBlitDispatches;
        r.gpuFlushedBlitTiming = p->gpuFlushedBlitTiming;
        r.gpuFlushedBlitDispatches = p->gpuFlushedBlitDispatches;
        r.gpuMs = p->gpuMs;
        r.gpuFlushedMs = p->gpuFlushedMs;
    }
    std::filesystem::remove(rows);
    std::filesystem::remove(choices);
    return tracing;
}

// ------------------------------------------------------------------------------------------------
// Output files
// ------------------------------------------------------------------------------------------------
void writeGroupColumns(std::ostream& os, const GroupResult& gr) {
    os << gr.group.part << "," << gr.group.operation() << "," << gr.group.k << "," << gr.group.passes << "," << gr.r.batch << "," << gr.r.name
       << "," << (gr.chosen ? "yes" : "no");
}

constexpr const char* GROUP_HEADER = "part,operation,K,passes,N,implementation,chosen";

void writeLatency(const std::filesystem::path& path) {
    std::ofstream os(path);
    os << GROUP_HEADER << ",median_us,p10_us,p90_us,mean_us,iterations,max_abs_err,reference,within_tolerance,"
                          "rpp_equivalent,kernels,memsets,memcpys,chosen_dpp_mean_speedup\n";
    for (const auto& gr : groupResults) {
        const Result& r = gr.r;
        writeGroupColumns(os, gr);
        os << "," << std::fixed << std::setprecision(2) << r.timing.median << "," << r.timing.p10 << "," << r.timing.p90
           << "," << r.timing.mean << "," << TIMED_ITERS << "," << std::scientific << std::setprecision(3)
           << r.errorVsOwnReference.max() << std::defaultfloat << "," << r.reference << ","
           << (r.withinTolerance() ? "yes" : "no") << "," << (isRpp(r.name) ? (gr.equivalent ? "yes" : "no") : "")
           << "," << r.launches.kernels << "," << r.launches.memsets << "," << r.launches.memcpys << ",";
        if (!std::isnan(r.meanSpeedupOfBvf)) {
            os << std::fixed << std::setprecision(3) << r.meanSpeedupOfBvf << std::defaultfloat;
        }
        os << "\n";
    }
}

// The RPP pipeline of every group of part (b) in this GPU time metric: the fastest RPP variant by its median in it
// (the first in order on a tie), among those within their tolerance and equivalent to their RPP-convention reference;
// the chosen column keeps the CPU + GPU choice (the chosen DPP tile is the speedups' denominator in every metric)
std::map<std::string, std::string> rppChosenInGpuMetric(bool flushed) {
    std::map<std::string, std::string> best;
    std::map<std::string, double> bestMedian;
    for (const auto& gr : groupResults) {
        if (!isRpp(gr.r.name) || !gr.equivalent || !gr.r.withinTolerance()) {
            continue;
        }
        const double m = (flushed ? gr.r.gpuFlushedTiming : gr.r.gpuTiming).median;
        const std::string label = gr.group.label();
        if (!bestMedian.count(label) || m < bestMedian[label]) {
            bestMedian[label] = m;
            best[label] = gr.r.name;
        }
    }
    return best;
}

void writeGpuTime(const std::filesystem::path& path, bool flushed) {
    std::ofstream os(path);
    os << GROUP_HEADER << ",gpu_median_us,gpu_p10_us,gpu_p90_us,gpu_mean_us,iterations,kernels_launched,"
                          "chosen_dpp_mean_gpu_speedup,runtime_blit_median_us,runtime_blit_p10_us,runtime_blit_p90_us,"
                          "runtime_blit_mean_us,runtime_blit_dispatches_per_execution,rpp_chosen_in_this_metric\n";
    const auto rppChosen = rppChosenInGpuMetric(flushed);
    for (const auto& gr : groupResults) {
        const Result& r = gr.r;
        const TimingStats& t = flushed ? r.gpuFlushedTiming : r.gpuTiming;
        const double s = flushed ? r.meanGpuFlushedSpeedupOfBvf : r.meanGpuSpeedupOfBvf;
        const TimingStats& b = flushed ? r.gpuFlushedBlitTiming : r.gpuBlitTiming;
        writeGroupColumns(os, gr);
        os << "," << std::fixed << std::setprecision(2) << t.median << "," << t.p10 << "," << t.p90 << "," << t.mean
           << "," << TIMED_ITERS << "," << r.launches.kernels << ",";
        if (!std::isnan(s)) {
            os << std::setprecision(3) << s;
        }
        os << std::setprecision(2) << "," << b.median << "," << b.p10 << "," << b.p90 << "," << b.mean << ","
           << std::setprecision(1) << (flushed ? r.gpuFlushedBlitDispatches : r.gpuBlitDispatches) << std::defaultfloat
           << ",";
        if (isRpp(r.name)) {
            const auto c = rppChosen.find(gr.group.label());
            os << (c != rppChosen.end() && c->second == r.name ? "yes" : "no");
        }
        os << "\n";
    }
}

// The per-iteration sibling of latency.csv (pass 0), gpu_time.csv (1) or gpu_time_l2_flushed.csv (2): one row per
// implementation of every group and timed iteration with the iteration (its crops), the implementation's time and the
// group's chosen DPP tile's time in that iteration, in us, and their ratio, which the chosen_dpp_mean_*speedup column
// averages (none on the chosen DPP tile's own rows)
void writePerIteration(const std::filesystem::path& path, int pass) {
    const auto times = [pass](const Result& r) -> const std::vector<float>& {
        return pass == 0 ? r.ms : pass == 1 ? r.gpuMs : r.gpuFlushedMs;
    };
    std::ofstream os(path);
    os << GROUP_HEADER
       << (pass == 0 ? ",iteration,time_us,chosen_dpp_time_us,chosen_dpp_speedup\n"
                     : ",iteration,gpu_time_us,chosen_dpp_gpu_time_us,chosen_dpp_gpu_speedup\n");
    for (const auto& gr : groupResults) {
        const auto dpp = std::find_if(groupResults.begin(), groupResults.end(), [&](const GroupResult& x) {
            return x.group.label() == gr.group.label() && x.chosen && isDpp(x.r.name);
        });
        if (dpp == groupResults.end()) {
            throw std::runtime_error("No chosen DPP tile for " + gr.group.label());
        }
        const std::vector<float>& ms = times(gr.r);
        const std::vector<float>& dppMs = times(dpp->r);
        if (ms.size() != dppMs.size()) {
            throw std::runtime_error(gr.group.label() + " " + gr.r.name + ": per-iteration times of different lengths");
        }
        for (size_t i = 0; i < ms.size(); ++i) {
            writeGroupColumns(os, gr);
            os << "," << WARMUP_ITERS + i << "," << std::fixed << std::setprecision(3) << ms[i] * 1000. << ","
               << dppMs[i] * 1000. << "," << std::defaultfloat << std::setprecision(6);
            if (&*dpp != &gr) {
                os << static_cast<double>(ms[i]) / dppMs[i];
            }
            os << "\n";
        }
    }
}

void writeValidation(const std::filesystem::path& path) {
    std::ofstream os(path);
    os << GROUP_HEADER << ",iteration,reference,max_abs_err_r,max_abs_err_g,max_abs_err_b,mean_abs_err,"
                          "non_finite_values,max_abs_err_vs_float_reference,values_beyond_1e-3,values,tolerance,"
                          "within_tolerance,max_abs_err_vs_rpp_convention_reference\n";
    for (const auto& gr : groupResults) {
        const Result& r = gr.r;
        for (const auto& it : r.errorsPerIteration) {
            const auto& e = it.own;
            writeGroupColumns(os, gr);
            os << "," << it.iteration << "," << r.reference << "," << std::scientific << std::setprecision(3)
               << e.maxPerChannel[0] << "," << e.maxPerChannel[1] << "," << e.maxPerChannel[2] << "," << e.mean << ","
               << e.nonFinite << "," << it.vsFloat.max() << "," << e.beyond << "," << e.values << "," << r.tolerance()
               << "," << (r.withinTolerance(e) ? "yes" : "no") << ",";
            if (isRpp(r.name)) {
                os << it.vsRppConvention.max();
            }
            os << std::defaultfloat << "\n";
        }
    }
}

// The workload, for --workload and the platform file
std::string workloadNb() {
    std::ostringstream os;
    os << FRAME_W << "x" << FRAME_H << " NV12 frame (uniform random limited-range values, Y in [" << LUMA_MIN << ", "
       << LUMA_MAX << "], U and V in [" << CHROMA_MIN << ", " << CHROMA_MAX << "], seed " << FRAME_SEED
       << "), NV12->RGB BT.601 limited range, N in {" << batchList(", ") << "} crops of " << PLANE_W << "x" << PLANE_H
       << " per iteration at uniform random even positions at least " << REGION_MARGIN
       << " pixels from the frame edges, new every iteration (warm-up and timed) and precomputed before the loop (seed "
       << REGION_SEED << ", batch N uses the first N of every iteration; common/bench_neighborhood.h cropRegionSets), "
       << "box blur K x K (mean of the window centred on the value, per channel, replicating the edge of the crop in "
       << "every pass; part (b) K = " << BOX_K << ", one pass), Mul (1/255), Sub (0.485, 0.456, 0.406), Div (0.229, "
       << "0.224, 0.225), output N planes of " << PLANE_W << "x" << PLANE_H << " float3, no border; no resize, except "
       << "part (a)'s resize_box groups: Experiment 1's regions (N in {1, 16}) resized to " << PLANE_W << "x" << PLANE_H
       << " (bilinear, aspect ratio ignored) before the box";
    return os.str();
}

void writePlatformNb(const std::filesystem::path& path, const std::string& diagnostic, bool kernelTracing,
                     const std::string& profilingError) {
    int device = 0, runtime = 0, driver = 0;
    gpuErrchk(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    gpuErrchk(hipGetDeviceProperties(&prop, device));
    gpuErrchk(hipRuntimeGetVersion(&runtime));
    gpuErrchk(hipDriverGetVersion(&driver));
    const MallSize mall = mallSize();
    const std::string rocmVersion = shell(std::string("cat ") + BENCH_ROCM_PATH + "/.info/version 2>/dev/null");
    const std::string driverVersion =
        shell("amd-smi static --gpu 0 --driver 2>/dev/null | sed -n 's/^ *VERSION: *//p' | head -1");
    const std::string perfLevel =
        shell("amd-smi metric --gpu 0 --perf-level 2>/dev/null | sed -n 's/^ *PERF_LEVEL: *//p' | head -1");
    const std::string gfxClock = shell("amd-smi metric --gpu 0 --clock 2>/dev/null | awk '$1 == \"GFX_0:\" {f=1; next} "
                                       "f && $1 == \"CLK:\" {print $2 \" MHz\"; exit}'");
    std::ofstream os(path);
    os << "Experiment 3 (neighborhood operation: box blur of fixed-size crops), AMD\n"
       << "GPU: " << prop.name << " (" << prop.gcnArchName << "), " << (prop.totalGlobalMem >> 20) << " MiB, L2 "
       << prop.l2CacheSize / 1024 << " KiB, MALL " << mall.kib << " KiB (" << mall.source << "), "
       << (onWsl() ? "WSL2" : "bare metal") << ", kernel driver " << (driverVersion.empty() ? "unknown" : driverVersion)
       << "\n"
       << "ROCm: " << rocmVersion << " (" << BENCH_ROCM_PATH << "), HIP runtime " << runtime << ", HIP driver " << driver
       << "; RPP " << RPP_VERSION_MAJOR << "." << RPP_VERSION_MINOR << "." << RPP_VERSION_PATCH << " (" << BENCH_RPP_LIB
       << ")\n"
       << "Compiler: " << __clang_version__ << "\nHIP flags: " << BENCH_HIP_FLAGS << "\n"
       << "FKL: " << BENCH_FKL_COMMIT
       << (std::string(BENCH_FKL_BRANCH) == "HEAD" ? std::string() : std::string(", branch ") + BENCH_FKL_BRANCH) << "\n"
       << "bvf-paper-bench: " << BENCH_REPO_COMMIT << "\n"
       << "CPU: " << cpuName() << "\nOS: " << osName() << ", kernel " << kernelVersion() << "\n"
       << "GPU clocks: not changed by the benchmark. Performance level: "
       << (perfLevel.empty() ? std::string("not reported") : perfLevel) << "; GFX clock at the end of the run "
       << (gfxClock.empty() ? std::string("not reported") : gfxClock) << "\n\n"
       << "Workload: " << workloadNb() << ".\n"
       << "Pipeline: ReadNV12 -> YUVtoRGB -> Crop " << PLANE_W << "x" << PLANE_H << " -> Box K x K -> Mul -> Sub -> Div "
       << "-> Write.\n"
       << "Groups: part (b) K = " << BOX_K << ", one pass, N = " << batchList(", ")
       << ": the RPP variants and the FKL MorphologyDPP tiles; part (a) K = 3, 5, 7, one and two passes, N = 1, 16: "
       << "fkl_bvf_naive (BoxBVF<K> of common/bench_neighborhood_ops.h, twice for two passes) and the DPP tiles, on the "
       << "crops (operation box) and with the resize (operation resize_box: Figure 1's chain on Experiment 1's regions, "
       << describe(false, "1, 16") << ": ReadNV12 -> YUVtoRGB -> Crop -> Resize " << PLANE_W << "x" << PLANE_H
       << " -> Box K x K -> Mul -> Sub -> Div -> Write; regions_resize.csv). DPP tiles: ";
    for (size_t i = 0; i < DPP_TILES.size(); ++i) {
        os << (i ? ", " : "") << DPP_TILES[i].width << "x" << DPP_TILES[i].height;
    }
    os << " (MorphologyDPPDetails<float3, W, H, K, K>, executeBoxBlur<passes>: every pass a mean, an Add reducer and a "
       << "Mul 1 / K^2 finalizer).\n"
       << "Timing (CPU + GPU): as rocm_bench.hip: " << WARMUP_ITERS << " warm-up then " << TIMED_ITERS
       << " timed iterations per implementation; in every group each implementation runs its complete loop before the "
       << "next one starts (the RPP variants or fkl_bvf_naive first, the DPP tiles last; never two implementations at "
       << "the same time, never interleaved per iteration), iteration t of every loop on the crops of iteration t, each "
       << "execution between two HIP events on one stream and followed by a synchronization. Chosen per group: the DPP "
       << "tile with "
       << "the lowest median among those within the float tolerance, and the RPP variant with the lowest median among "
       << "those within their tolerance and within " << RPP_EQUIVALENCE << " of their RPP-convention reference "
       << "(chosen = yes in the CSV files; every implementation is recorded). Speedups: per iteration, implementation "
       << "time / chosen DPP time of the same iteration (the same crops), averaged over the timed iterations (the chosen "
       << "DPP's own row has none; every pair in latency_per_iteration.csv, gpu_time_per_iteration.csv and "
       << "gpu_time_l2_flushed_per_iteration.csv).\n"
       << "Validation: as rocm_bench.hip, on the timed iterations " << validation::VALIDATION_STRIDE << " apart and the "
       << "last, into NaN-poisoned outputs. FKL against the float reference: FKL's CPU float chain ReadNV12 -> YUVtoRGB -> "
       << "Crop, then bench::neighborhood::boxReference (the box in double, no rounding, then the normalization in float; "
       << "the box is linear and the normalization rounds, so the order matters: the references blur, then normalize, "
       << "the order of the pipeline and of every implementation but rpp_f32box_cmn, which normalizes in "
       << "crop_mirror_normalize before its F32 box and is held to its tolerance against these references and exactly to "
       << "its own RPP-convention reference, in its own order), tolerance " << TOLERANCE << ". RPP, whose frame is 8 bit RGB, against the same chain with the "
       << "RGB saturated and rounded to nearest 8 bit (fkl_cpu_8bit_rgb, as in Experiment 1), and for the U8 box its "
       << "output rounded half up to 8 bit (fkl_cpu_8bit_rgb_rounded_box), tolerance " << TOLERANCE_8BIT
       << " (one level of 8 bit RGB); its error against the float reference, whose RGB is not saturated, is recorded "
       << "too (max_abs_err_vs_float_reference). RPP also against its RPP-convention reference: RPP 3.2.0's HIP "
       << "arithmetic, operation by operation, on RPP's 8 bit frame (the U8 box rint(float(window sum) * RPP's float "
       << "1 / K^2); the F32 box the window summed in float, row after row, times that scale; crop_mirror_normalize "
       << "fmaf(x, multiplier, offset); rppt_normalize fmaf(x - mean, 1 / stdDev, 0)), equivalent within "
       << RPP_EQUIVALENCE << " (max_abs_err_vs_rpp_convention_reference in validation.csv). After the timing, the first "
       << "validated iteration again (every value within 1e-6).\n"
       << "RPP (HIP backend): every variant starts with the hipMemcpyAsync of the crops of its iteration (XYWH ROIs on "
       << "the frame, from pinned memory) and the NV12->RGB conversion (1 kernel and 8 hipMemcpyToSymbol per "
       << "rppt_yuv_to_rgb call) into an RGB frame that starts " << RPP_LEAD_BYTES << " bytes into its allocation "
       << "(descriptor offsetInBytes; RPP's HIP box filter refuses an offset below 12 * (kernelSize / 2) bytes). The "
       << "conversion is of the full frame (one rppt_yuv_to_rgb), or in the variants named <variant>_crop of only the N "
       << "crops (one rppt_yuv_to_rgb per crop: rppt_yuv_to_rgb takes no ROI tensor, so each crop is an ROI in NPP's "
       << "sense, the Y and UV pointers offset to the crop origin, the frame's pitches, the crop's width and height, "
       << "written at the crop's position in the RGB frame; exact, see the per-crop conversion check at the end of "
       << "this file). The calls "
       << "after the conversion are the same in both forms: the box reads the crops through their ROIs and replicates "
       << "each crop's own edge, so it uses no pixel outside the crop (checked by the sentinel fill and the twin check "
       << "below). Before a _crop variant is validated, its RGB "
       << "frame is filled with the byte " << static_cast<int>(RGB_SENTINEL) << ", and its output must be exactly its "
       << "full-frame twin's in every validated iteration (otherwise the run stops). The group's RPP pipeline is the "
       << "fastest of the eight variants in each metric: by the CPU + GPU median in the chosen column, and by the GPU "
       << "time only (and flushed) median in gpu_time*.csv's rpp_chosen_in_this_metric column (the first in order on "
       << "a tie, so the full-frame form when a variant's two forms tie; the GPU time only medians leave out the HIP "
       << "runtime's blit kernels, among them the 8 hipMemcpyToSymbol uploads of every rppt_yuv_to_rgb call, whose "
       << "time and dispatches are in the runtime_blit_* columns); the speedups of every metric are over the DPP tile "
       << "chosen by the CPU + GPU "
       << "median. rpp_u8box_cmn: rppt_box_filter U8 "
       << "on the frame with the crops as per-image ROIs (every batch image aliasing the frame; the box replicates the "
       << "ROI edge and writes each crop at (0, 0) of its own 64x32 U8 image: crop + box in one call), then "
       << "rppt_crop_mirror_normalize 8u->32f of the boxed crops (whole-image XYWH ROIs, uploaded once) with the whole "
       << "normalization, multiplier Mul / Div and offset -Sub / Div per channel, into the output tensor. "
       << "rpp_u8box_normalize: the same with rppt_crop_mirror_normalize converting only (multiplier 1, offset 0), then "
       << "rppt_normalize (rows of " << PLANE_W << " float3, per column mean Sub / Mul and stdDev Div / Mul). "
       << "rpp_f32box_cmn: rppt_crop_mirror_normalize 8u->32f of the crops of the frame with the whole normalization "
       << "into converted crops that start " << RPP_LEAD_BYTES << " bytes into their allocation, then rppt_box_filter "
       << "F32 of the whole images into the output tensor. rpp_f32box_normalize: the same with the conversion only, "
       << "then rppt_normalize. No memset. One RPP handle per pipeline with batch size N.\n"
       << "Kernel counts, GPU time only and GPU time only with L2 + MALL flushed: as rocm_bench.hip, in a profiling child "
       << "process, every implementation of every group, with the same sequential loops"
       << (kernelTracing ? "" : onWsl() ? " (GPU time only NOT MEASURED on WSL2)" : " (GPU time only NOT MEASURED)") << ".\n";
    if (!profilingError.empty()) {
        os << "PROFILING PASS FAILED (" << profilingError << "): kernel counts and GPU time only are not measured\n";
    }
    os << "Floor of the CPU + GPU metric (this run): empty bracket " << std::fixed << std::setprecision(2)
       << eventFloor.empty.median << " us, one empty kernel " << eventFloor.oneKernel.median << " us, four empty kernels "
       << eventFloor.fourKernels.median << " us (medians)" << std::defaultfloat << "\n\n"
       << diagnostic;
}

} // namespace bench::nb

int main(int argc, char** argv) {
    using namespace bench;
    using namespace bench::nb;
    try {
        if (argc == 2 && std::string(argv[1]) == "--workload") {
            std::cout << workloadNb() << std::endl;
            return 0;
        }
        const std::filesystem::path outDir = argc > 1 ? argv[1] : "results_neighborhood";
        std::filesystem::create_directories(outDir);
        Context ctx;
        initContext(ctx);
        // The experiment's crops instead of Experiment 1's regions (the same in the profiling child)
        ctx.regions = std::make_unique<RegionSets>(neighborhood::cropRegionSets(REGION_ITERATIONS));
        resizeRegions = std::make_unique<RegionSets>(REGION_ITERATIONS);
        if (argc > 4 && std::string(argv[2]) == "--profile-pass") {
            nb::profilePass(ctx, outDir, argv[3], argv[4]);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "--check-kernel-tracing") {
            const bool tracing = nb::kernelTracingWorks(ctx);
            std::cout << "Kernel dispatch tracing: " << (tracing ? "available" : "NOT available") << std::endl;
            return tracing ? 0 : 3;
        }
        const bool validateOnly = argc > 2 && std::string(argv[2]) == "--validate";
        if (argc > 2 && !validateOnly) {
            throw std::runtime_error(std::string("Unknown mode or missing arguments: ") + argv[2]);
        }
        computeRppRgb8(ctx);
        std::ostringstream diagnostic;
        conversionDiagnostic(ctx, diagnostic);
        perCropConversionCheck(ctx, diagnostic);
        std::cout << diagnostic.str();
        const auto fklFailed = []() {
            return std::any_of(groupResults.begin(), groupResults.end(), [](const GroupResult& gr) {
                return gr.r.name.rfind("fkl_", 0) == 0 && !gr.r.withinTolerance();
            });
        };
        if (validateOnly) {
            forEachGroup([&]<char PART, size_t N, int K, int P, bool R>() { runGroup<PART, N, K, P, R>(ctx, false); });
            writeValidation(outDir / "validation.csv");
            if (fklFailed()) {
                std::cerr << "ERROR: an FKL output is not within its tolerance" << std::endl;
                return 1;
            }
            return 0;
        }
        measureEventFloor(ctx);
        forEachGroup([&]<char PART, size_t N, int K, int P, bool R>() { runGroup<PART, N, K, P, R>(ctx, true); });
        bool tracing = false;
        std::string profilingError;
        try {
            tracing = nb::runProfilingChild(outDir);
        } catch (const std::exception& e) {
            profilingError = e.what();
            std::cerr << "ERROR in the profiling pass: " << profilingError << std::endl;
        }
        writeLatency(outDir / "latency.csv");
        writePerIteration(outDir / "latency_per_iteration.csv", 0);
        writeValidation(outDir / "validation.csv");
        if (tracing) {
            writeGpuTime(outDir / "gpu_time.csv", false);
            writeGpuTime(outDir / "gpu_time_l2_flushed.csv", true);
            writePerIteration(outDir / "gpu_time_per_iteration.csv", 1);
            writePerIteration(outDir / "gpu_time_l2_flushed_per_iteration.csv", 2);
        } else {
            // The GPU time only files of an earlier run in this directory, which this run does not rewrite (as in
            // rocm_bench.hip)
            for (const char* file : {"gpu_time.csv", "gpu_time_l2_flushed.csv", "gpu_time_per_iteration.csv",
                                     "gpu_time_l2_flushed_per_iteration.csv"}) {
                std::filesystem::remove(outDir / file);
            }
        }
        writeRegionStats((outDir / "regions.csv").string(), *ctx.regions, WARMUP_ITERS, TIMED_ITERS);
        writeRegionStats((outDir / "regions_resize.csv").string(), *resizeRegions, WARMUP_ITERS, TIMED_ITERS);
        {
            std::ofstream stages(outDir / "kernel_breakdown.csv");
            stages << "implementation,N,kernel,median_us,p10_us,p90_us\n";
            for (const auto& b : breakdown) {
                stages << b.name << "," << b.batch << "," << b.stage << "," << std::fixed << std::setprecision(2)
                       << b.timing.median << "," << b.timing.p10 << "," << b.timing.p90 << "\n";
            }
        }
        writePlatformNb(outDir / "platform.txt", diagnostic.str(), tracing, profilingError);
        std::cout << "Results written to " << std::filesystem::absolute(outDir) << std::endl;
        if (fklFailed()) {
            std::cerr << "ERROR: an FKL output is not within its tolerance, see validation.csv" << std::endl;
            return 1;
        }
        if (!profilingError.empty()) {
            std::cerr << "ERROR: the profiling pass failed; the CPU + GPU results were written" << std::endl;
            return 2;
        }
        // GPU time only is part of the experiment's results: its absence is an error unless allowed, as in
        // rocm/scripts/run_all.sh
        const char* allowNoTrace = std::getenv("ALLOW_NO_KERNEL_TRACE");
        if (!tracing && !(allowNoTrace && std::string(allowNoTrace) == "1")) {
            std::cerr << "ERROR: GPU time only was not measured (no kernel dispatch records in the profiling pass); the "
                         "CPU + GPU results were written (ALLOW_NO_KERNEL_TRACE=1 to accept it)" << std::endl;
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
