// Experiment 3 of the Backwards Vertical Fusion paper on an AMD GPU: a neighborhood operation (erode, or open = erode
// then dilate; K x K, centred, replicate at the 64x32 edge in every pass) after the resize of the Figure 1 pipeline,
// without AddBorder:
// ReadNV12 -> YUVtoRGB -> Crop -> Resize 64x32 -> Morph -> Mul -> Sub -> Div -> Write (N planes of 64x32 float3)
// with the frame, region sizes, batch sizes and the regions of every iteration of common/bench_geometry.h, and the
// parameters of common/bench_neighborhood.h.
//
// Part (b), the library comparison (K = MORPH_K, every N of BATCHES, erode and open):
//   rpp_normalize        rppt_yuv_to_rgb (full frame), rppt_crop_mirror_normalize 8u->32f (batched), rppt_resize F32
//                        bilinear into a float3 plane buffer, rppt_erode (open: then rppt_dilate) batched over the N
//                        planes into the output tensor, rppt_normalize (Mul + Sub + Div) on the whole tensor
//   rpp_fused_normalize  rppt_yuv_to_rgb, rppt_resize_mirror_normalize 8u->32f (crop + convert + resize, batched) into
//                        the plane buffer, then the same morphology and rppt_normalize
//   fkl_dpp_<W>x<H>      one FKL kernel: MorphologyDPP<float3, W, H, K, K> batched over the regions, reading the BVF
//                        chain ReadNV12 -> YUVtoRGB -> Crop -> Resize and writing through Mul -> Sub -> Div -> Write,
//                        one for every tile of DPP_TILES
// Part (a), the micro-experiment (FKL only, K of PART_A_KS, 1 or 2 passes, N of PART_A_BATCHES):
//   fkl_bvf_naive        one FKL kernel with the naive ReadBack operations of common/bench_neighborhood_ops.h
//                        (ErodeBVF, then DilateBVF for the open) in the BVF chain after the Resize
//   fkl_dpp_<W>x<H>      as in part (b)
// In every group (part, operation, K, N) all of them are validated and timed in the same interleaved iterations; the
// RPP variant and the DPP tile of a group are the fastest (median of the CPU + GPU pass) of those that pass their
// check, and every one is recorded. The speedups are per iteration (baseline / chosen DPP tile), averaged.
//
// The harness (context, references of Experiment 1, timing, rocprofiler-sdk tool, cache flusher) is the one of
// rocm_bench.hip, which this file includes. No HIP graphs anywhere.

#define BENCH_NO_MAIN
#include "rocm_bench.hip"

#include <fused_kernel/algorithms/image_processing/morphology.h>

#include <thread>

#include "bench_neighborhood.h"
#include "bench_neighborhood_ops.h"

namespace bench::nb {

using neighborhood::DPP_TILES;
using neighborhood::MORPH_K;
using neighborhood::MorphOp;
using neighborhood::PLANE_FLOATS;
using neighborhood::PLANE_H;
using neighborhood::PLANE_W;

// The batched MorphologyDPP (FKL PR #393), which processes every plane of its input (grid z) and sizes its staging per
// number of passes (STAGE_ELEMENTS<passes>): an FKL without it (63782f6) would process the first region only
template <typename Details>
concept BatchedMorphologyDetails = requires { Details::template STAGE_ELEMENTS<1>; };
static_assert(BatchedMorphologyDetails<fk::MorphologyDPPDetails<fk::float3>>,
              "rocm_neighborhood needs FKL with the batched MorphologyDPP: the commit of the repository's fkl submodule");

// RPP's HIP morphology refuses a source descriptor with offsetInBytes < 12 * (kernelSize / 2) (RPP_ERROR_LOW_OFFSET;
// rppt_tensor_morphological_operations.cpp in RPP 3.2.0): its sources (the resized planes, and the eroded planes of
// the open) start this many bytes into their allocation, given to RPP through the descriptor's offsetInBytes
constexpr size_t RPP_LEAD_BYTES = 256;
constexpr int RPP_MAX_KERNEL = 9; // the largest kernelSize of RPP's HIP morphology
static_assert(RPP_LEAD_BYTES >= 12 * (RPP_MAX_KERNEL / 2), "RPP's HIP morphology needs the lead bytes");
static_assert(PLANE_W % RPP_PIXEL_ALIGN == 0, "RPP's kernels write the plane rows 8 pixels at a time");
// rppt_normalize reuses Experiment 1's rows of per column constants (OUT_W float3)
static_assert(PLANE_W == OUT_W, "the normalize constants are rows of OUT_W float3");

// One group of the experiment: every implementation of a group computes the same output
struct Group {
    char part{'b'};
    MorphOp op{MorphOp::Erode};
    int k{0};
    int batch{0};
    std::string label() const {
        return std::string(1, part) + "_" + neighborhood::name(op) + "_k" + std::to_string(k) + "_n" + std::to_string(batch);
    }
};

// ------------------------------------------------------------------------------------------------
// References. The Figure 1 chain without AddBorder (FKL's CPU backend, float; or RPP's sampling convention on RPP's
// own 8 bit frame, as in Experiment 1), normalized, then the morphology of every plane with hostMorphPlane: x ->
// (x * mul - sub) / div is monotone non-decreasing per channel (mul, div > 0) in IEEE round-to-nearest, so per channel
// min and max commute with it exactly, and morphology after the normalization is the morphology before it.
// ------------------------------------------------------------------------------------------------
static_assert(sizeof(fk::float3) == 3 * sizeof(float), "the planes are copied as float3");

void morphPlanes(std::vector<float>& planes, int k, MorphOp op) {
    const size_t count = planes.size() / PLANE_FLOATS;
    std::vector<fk::float3> in(count * PLANE_W * PLANE_H), out(in.size());
    std::memcpy(in.data(), planes.data(), planes.size() * sizeof(float));
    neighborhood::hostMorph(in.data(), out.data(), count, k, op);
    std::memcpy(planes.data(), out.data(), planes.size() * sizeof(float));
}

template <size_t N>
std::vector<float> floatReference(const Context& ctx, size_t t, int k, MorphOp op) {
    const auto crops = fklRegions<N>(ctx, t);
    fk::Stream_<fk::ParArch::CPU> cpuStream;
    fk::Tensor<fk::float3> out(PLANE_W, PLANE_H, N, 1, fk::MemType::Host);
    fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU>>(cpuStream,
        fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameHost),
        YUVToRGB::build(),
        fk::Crop<>::build(crops),
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(PLANE_W, PLANE_H)),
        fk::Mul<fk::float3>::build(mulValue),
        fk::Sub<fk::float3>::build(subValue),
        fk::Div<fk::float3>::build(divValue),
        fk::TensorWrite<fk::float3>::build(out.ptr()));
    cpuStream.sync();
    const float* data = reinterpret_cast<const float*>(out.ptr().data);
    std::vector<float> planes(data, data + N * PLANE_FLOATS);
    morphPlanes(planes, k, op);
    return planes;
}

// rppConventionReference8bit of rocm_bench.hip (RPP's 8 bit frame, RPP's bilinear sampling) without the border
template <size_t N>
std::vector<float> rppConventionReference(const Context& ctx, size_t t, int k, MorphOp op) {
    if (ctx.rppRgb8.size() != static_cast<size_t>(FRAME_W) * FRAME_H * 3) {
        throw std::runtime_error("RPP's RGB frame (ctx.rppRgb8) is not available");
    }
    std::vector<float> out(N * PLANE_FLOATS);
    for (size_t i = 0; i < N; ++i) {
        const Region& r = ctx.regions->at(t, i);
        const int cropW = r.width, cropH = r.height;
        const float scaleX = static_cast<float>(cropW) / PLANE_W;
        const float scaleY = static_cast<float>(cropH) / PLANE_H;
        const auto at = [&](int y, int x, int c) {
            return static_cast<float>(ctx.rppRgb8[(static_cast<size_t>(r.y + y) * FRAME_W + (r.x + x)) * 3 + c]);
        };
        for (int y = 0; y < PLANE_H; ++y) {
            for (int x = 0; x < PLANE_W; ++x) {
                float* dst = &out[((i * PLANE_H + y) * PLANE_W + x) * 3];
                const float sx = std::min(static_cast<float>(cropW - 2), std::max(0.f, (x + 0.5f) * scaleX - 0.5f));
                const float sy = std::min(static_cast<float>(cropH - 2), std::max(0.f, (y + 0.5f) * scaleY - 0.5f));
                const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
                const int x1 = std::min(x0 + 1, cropW - 1), y1 = std::min(y0 + 1, cropH - 1);
                const float fx = sx - x0, fy = sy - y0;
                for (int c = 0; c < 3; ++c) {
                    const float value = (at(y0, x0, c) * (1.f - fx) + at(y0, x1, c) * fx) * (1.f - fy) +
                                        (at(y1, x0, c) * (1.f - fx) + at(y1, x1, c) * fx) * fy;
                    dst[c] = (value * MUL[c] - SUB[c]) / DIV[c];
                }
            }
        }
    }
    morphPlanes(out, k, op);
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

// MorphologyDPP with the BVF chain up to the Resize as its input and the pointwise tail and the write as its output,
// batched over the regions (grid z), with a W x H tile and a K x K maximum mask
template <size_t N, int K, MorphOp OP, int TW, int TH>
Implementation makeFklDpp(Context& ctx) {
    auto out = planeTensor<N>();
    const auto sets = fklRegionSets<N>(ctx);
    Implementation impl{"fkl_dpp_" + std::to_string(TW) + "x" + std::to_string(TH), "fkl_cpu_float"};
    impl.run = [&ctx, out, sets](size_t t) {
        using Details = fk::MorphologyDPPDetails<fk::float3, TW, TH, K, K>;
        const Details details{PLANE_W, PLANE_H, K, K, K / 2, K / 2};
        const auto input =
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice)
                .then(YUVToRGB::build())
                .then(fk::Crop<>::build((*sets)[t]))
                .then(fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(
                    fk::Size(PLANE_W, PLANE_H)));
        const auto output = fk::Mul<fk::float3>::build(mulValue)
                                .then(fk::Sub<fk::float3>::build(subValue))
                                .then(fk::Div<fk::float3>::build(divValue))
                                .then(fk::TensorWrite<fk::float3>::build(out->ptr()));
        beforeKernel();
        bool launched = false;
        if constexpr (OP == MorphOp::Erode) {
            launched = fk::executeErode(details, input, output, *ctx.fkStream);
        } else {
            launched = fk::executeOpen(details, input, output, *ctx.fkStream);
        }
        if (!launched) {
            throw std::runtime_error("MorphologyDPP rejected its details");
        }
    };
    planeIo<N>(impl, ctx, out);
    return impl;
}

// The naive operations of common/bench_neighborhood_ops.h in the BVF chain: every output value reads K x K resized
// values (K^4 for the open), each of them 4 bilinear taps of the color conversion
template <size_t N, int K, MorphOp OP>
Implementation makeFklNaive(Context& ctx) {
    auto out = planeTensor<N>();
    const auto sets = fklRegionSets<N>(ctx);
    Implementation impl{"fkl_bvf_naive", "fkl_cpu_float"};
    impl.run = [&ctx, out, sets](size_t t) {
        const auto resize =
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(PLANE_W, PLANE_H));
        beforeKernel();
        if constexpr (OP == MorphOp::Erode) {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                YUVToRGB::build(),
                fk::Crop<>::build((*sets)[t]),
                resize,
                fk::ErodeBVF<K>::build(),
                fk::Mul<fk::float3>::build(mulValue),
                fk::Sub<fk::float3>::build(subValue),
                fk::Div<fk::float3>::build(divValue),
                fk::TensorWrite<fk::float3>::build(out->ptr()));
        } else {
            fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
                fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
                YUVToRGB::build(),
                fk::Crop<>::build((*sets)[t]),
                resize,
                fk::ErodeBVF<K>::build(),
                fk::DilateBVF<K>::build(),
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
// RPP (HIP backend), on Experiment 1's RppState (rocm_bench.hip): its frame, crop buffer, ROIs of every iteration,
// parameter tensors and handle (batch size N), plus the float3 planes of the resized regions and the planes of the
// eroded ones (open), each RPP_LEAD_BYTES into its allocation, and the fixed XYWH ROIs of the whole planes for the
// morphology (uploaded once, outside the timing: in XYWH, the format rppt_erode and rppt_dilate work in). The output
// is the first N planes of 64x32 float3 of RppState's tensor, contiguous.
// ------------------------------------------------------------------------------------------------
struct RppPlanes {
    std::shared_ptr<RppState> st;
    uint8_t* resized{nullptr};
    uint8_t* eroded{nullptr};
    RpptROI* planeRois{nullptr};
    RpptDesc leadDesc, outDesc;
    ~RppPlanes() {
        (void)hipFree(resized);
        (void)hipFree(eroded);
        (void)hipFree(planeRois);
    }
};

const std::array<std::string, 2> RPP_VARIANTS_NB{"rpp_normalize", "rpp_fused_normalize"};

template <size_t N, int K, MorphOp OP>
Implementation makeRppPlanes(Context& ctx, const std::string& variant) {
    const bool fused = variant == "rpp_fused_normalize";
    if (!fused && variant != "rpp_normalize") {
        throw std::runtime_error("Unknown RPP variant " + variant);
    }
    auto st = makeRppState(ctx, N, false, fused ? RppRois::ResizeMirrorNormalize : RppRois::CropAndResize);
    auto p = std::make_shared<RppPlanes>();
    p->st = st;
    const size_t bytes = RPP_LEAD_BYTES + N * PLANE_FLOATS * sizeof(float) + RPP_SLACK_BYTES;
    p->resized = deviceAlloc<uint8_t>(bytes);
    if (OP == MorphOp::Open) {
        p->eroded = deviceAlloc<uint8_t>(bytes);
    }
    std::vector<RpptROI> rois(N);
    for (auto& r : rois) {
        r.xywhROI = {{0, 0}, PLANE_W, PLANE_H};
    }
    p->planeRois = deviceAlloc<RpptROI>(N * sizeof(RpptROI));
    gpuErrchk(hipMemcpyAsync(p->planeRois, rois.data(), N * sizeof(RpptROI), hipMemcpyHostToDevice, ctx.stream));
    gpuErrchk(hipStreamSynchronize(ctx.stream));
    const RpptStrides strides{static_cast<Rpp32u>(PLANE_FLOATS), 1, static_cast<Rpp32u>(PLANE_W * 3), 3};
    p->leadDesc = RpptDesc(static_cast<Rpp32u>(N), 3, PLANE_H, PLANE_W, RpptDataType::F32, RpptLayout::NHWC,
                           static_cast<Rpp32u>(RPP_LEAD_BYTES), strides);
    p->outDesc = RpptDesc(static_cast<Rpp32u>(N), 3, PLANE_H, PLANE_W, RpptDataType::F32, RpptLayout::NHWC, 0, strides);
    // rppt_normalize on the rows of the N planes: N * PLANE_H rows of PLANE_W float3
    const Rpp32u rows = static_cast<Rpp32u>(N * PLANE_H), cols = static_cast<Rpp32u>(PLANE_W * 3);
    if (st->rowsRoi[3] != cols) {
        throw std::runtime_error("unexpected rppt_normalize row length");
    }
    st->rowsDesc.dims[1] = rows;
    st->rowsDesc.strides[0] = rows * cols;
    st->rowsRoi[2] = rows;

    std::vector<RppStage> stages{yuvToRgbStage(ctx, st)};
    if (fused) {
        stages.push_back({"rppt_resize_mirror_normalize_8u32f_batch_into_planes", [p](size_t) {
                              RppState& s = *p->st;
                              beforeKernel();
                              RPP_CHECK(rppt_resize_mirror_normalize(s.rgb8, &s.frameBatchDesc, p->resized, &p->leadDesc,
                                                                     s.dstSizes, RpptInterpolationType::BILINEAR,
                                                                     s.rmnMean, s.rmnStdDevOne, s.mirror, s.cropRoisLtrb,
                                                                     RpptRoiType::LTRB, s.handle,
                                                                     RppBackend::RPP_HIP_BACKEND));
                          }});
    } else {
        stages.push_back(cropConvertStage(st));
        stages.push_back({"rppt_resize_batch_into_planes", [p](size_t) {
                              RppState& s = *p->st;
                              beforeKernel();
                              RPP_CHECK(rppt_resize(s.cropped, &s.cropDesc, p->resized, &p->leadDesc, s.dstSizes,
                                                    RpptInterpolationType::BILINEAR, s.resizeRois, RpptRoiType::LTRB,
                                                    s.handle, RppBackend::RPP_HIP_BACKEND));
                          }});
    }
    const std::string k = std::to_string(K);
    if constexpr (OP == MorphOp::Erode) {
        stages.push_back({"rppt_erode_" + k + "x" + k + "_batch", [p](size_t) {
                              RppState& s = *p->st;
                              beforeKernel();
                              RPP_CHECK(rppt_erode(p->resized, &p->leadDesc, s.tensor, &p->outDesc, K, p->planeRois,
                                                   RpptRoiType::XYWH, s.handle, RppBackend::RPP_HIP_BACKEND));
                          }});
    } else {
        stages.push_back({"rppt_erode_" + k + "x" + k + "_batch_into_planes", [p](size_t) {
                              RppState& s = *p->st;
                              beforeKernel();
                              RPP_CHECK(rppt_erode(p->resized, &p->leadDesc, p->eroded, &p->leadDesc, K, p->planeRois,
                                                   RpptRoiType::XYWH, s.handle, RppBackend::RPP_HIP_BACKEND));
                          }});
        stages.push_back({"rppt_dilate_" + k + "x" + k + "_batch", [p](size_t) {
                              RppState& s = *p->st;
                              beforeKernel();
                              RPP_CHECK(rppt_dilate(p->eroded, &p->leadDesc, s.tensor, &p->outDesc, K, p->planeRois,
                                                    RpptRoiType::XYWH, s.handle, RppBackend::RPP_HIP_BACKEND));
                          }});
    }
    stages.push_back(normalizeStage(st));

    Implementation impl{variant, "rpp_sampling_8bit"};
    stages.insert(stages.begin(), uploadRoisStage(ctx, st));
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
    impl.fillNaN = [&ctx, p]() { poisonDevice(ctx.stream, p->st->tensor, N * PLANE_FLOATS); };
    return impl;
}

// ------------------------------------------------------------------------------------------------
// Groups
// ------------------------------------------------------------------------------------------------
template <size_t N, int K, MorphOp OP, size_t... Is>
void addDppTiles(Context& ctx, std::vector<Implementation>& impls, std::index_sequence<Is...>) {
    (impls.push_back(makeFklDpp<N, K, OP, DPP_TILES[Is].width, DPP_TILES[Is].height>(ctx)), ...);
}

// Part (b): the RPP variants, then the DPP tiles; part (a): the naive FKL pipeline, then the DPP tiles
template <char PART, size_t N, int K, MorphOp OP>
std::vector<Implementation> makeGroup(Context& ctx) {
    std::vector<Implementation> impls;
    if constexpr (PART == 'b') {
        for (const auto& v : RPP_VARIANTS_NB) {
            impls.push_back(makeRppPlanes<N, K, OP>(ctx, v));
        }
    } else {
        impls.push_back(makeFklNaive<N, K, OP>(ctx));
    }
    addDppTiles<N, K, OP>(ctx, impls, std::make_index_sequence<DPP_TILES.size()>{});
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
    bool equivalent{false}; // RPP: within RPP_EQUIVALENCE of the RPP-convention reference in every validated iteration
};
std::vector<GroupResult> groupResults;
// The chosen DPP tile of every group (label -> implementation), from the CPU + GPU pass; the profiling child gets
// them in a file
std::map<std::string, std::string> chosenDpp;

std::string flatName(const Group& g, const std::string& impl) { return g.label() + "/" + impl; }

template <char PART, size_t N, int K, MorphOp OP>
void runGroup(Context& ctx, bool timing) {
    const Group g{PART, OP, K, static_cast<int>(N)};
    std::cout << "=== " << g.label() << std::endl;
    progress("implementations");
    const auto impls = makeGroup<PART, N, K, OP>(ctx);
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
        progress("references of iteration " + std::to_string(t));
        const auto refFloat = floatReference<N>(ctx, t, K, OP);
        const auto refRpp = PART == 'b' ? rppConventionReference<N>(ctx, t, K, OP) : std::vector<float>{};
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
            if (v == 0) {
                firstOutputs[k] = output;
            }
            Result::IterationErrors e;
            e.iteration = t;
            e.vsFloat = compare(output, refFloat);
            if (PART == 'b') {
                e.vsRppConvention = compare(output, refRpp);
            }
            e.own = impl.reference == "fkl_cpu_float" ? e.vsFloat : e.vsRppConvention;
            rs[k].r.addIteration(e);
            if (isRpp(impl.name)) {
                rs[k].equivalent = rs[k].equivalent && e.vsRppConvention.nonFinite == 0 &&
                                   e.vsRppConvention.max() <= RPP_EQUIVALENCE;
            }
        }
    }
    if (!timing) {
        for (auto& gr : rs) {
            std::cout << "  " << std::left << std::setw(20) << gr.r.name << std::right << "  max err " << std::scientific
                      << std::setprecision(2) << gr.r.errorVsOwnReference.max() << " (" << gr.r.reference << ")"
                      << std::defaultfloat << (gr.r.withinTolerance() ? "" : "  NOT within its tolerance")
                      << (isRpp(gr.r.name) && !gr.equivalent ? "  NOT equivalent to the RPP-convention reference" : "")
                      << std::endl;
            groupResults.push_back(gr);
        }
        return;
    }
    progress("CPU + GPU timing");
    const auto ms = measureInterleaved(ctx, impls);
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
    // The chosen DPP tile (fastest within its tolerance) and RPP variant (fastest equivalent one)
    size_t dpp = impls.size();
    for (size_t k = 0; k < impls.size(); ++k) {
        rs[k].r.timing = computeStats(ms[k]);
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
        if (isRpp(impls[k].name) && rs[k].equivalent &&
            (rpp == impls.size() || rs[k].r.timing.median < rs[rpp].r.timing.median)) {
            rpp = k;
        }
    }
    if (PART == 'b' && rpp == impls.size()) {
        throw std::runtime_error(g.label() + ": no RPP variant is equivalent to the RPP-convention reference");
    }
    for (size_t k = 0; k < impls.size(); ++k) {
        GroupResult& gr = rs[k];
        gr.chosen = gr.chosen || k == rpp;
        if (k != dpp) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += ms[k][i] / ms[dpp][i];
            }
            gr.r.meanSpeedupOfBvf = sum / TIMED_ITERS;
        }
        std::cout << "  " << std::left << std::setw(20) << gr.r.name << std::right << (gr.chosen ? " *" : "  ")
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

// Every group of the experiment, part (b) first: f.template operator()<part, N, K, operation>() for every group
template <typename F>
void forEachGroup(F&& f) {
    // Part (b): every N, erode then open
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((f.template operator()<'b', BATCHES[Is], MORPH_K, MorphOp::Erode>(),
          f.template operator()<'b', BATCHES[Is], MORPH_K, MorphOp::Open>()), ...);
    }(std::make_index_sequence<BATCHES.size()>{});
    // Part (a): every K, 1 then 2 passes, every N
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
        ([&]<size_t KI>() {
            constexpr int K = neighborhood::PART_A_KS[KI];
            [&]<size_t... Ns>(std::index_sequence<Ns...>) {
                ((f.template operator()<'a', neighborhood::PART_A_BATCHES[Ns], K, MorphOp::Erode>()), ...);
                ((f.template operator()<'a', neighborhood::PART_A_BATCHES[Ns], K, MorphOp::Open>()), ...);
            }(std::make_index_sequence<neighborhood::PART_A_BATCHES.size()>{});
        }.template operator()<Ks>(), ...);
    }(std::make_index_sequence<neighborhood::PART_A_KS.size()>{});
}
static_assert(neighborhood::PART_A_PASSES.size() == 2 && neighborhood::PART_A_PASSES[0] == 1 &&
                  neighborhood::PART_A_PASSES[1] == 2,
              "part (a) runs 1 pass (erode) and 2 passes (open)");

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

template <char PART, size_t N, int K, MorphOp OP>
void countGroup(Context& ctx) {
    const Group g{PART, OP, K, static_cast<int>(N)};
    const size_t t = static_cast<size_t>(WARMUP_ITERS);
    for (const auto& impl : makeGroup<PART, N, K, OP>(ctx)) {
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
template <char PART, size_t N, int K, MorphOp OP>
void gpuOnlyGroup(Context& ctx, CacheFlusher* flusher) {
    const Group g{PART, OP, K, static_cast<int>(N)};
    const auto impls = makeGroup<PART, N, K, OP>(ctx);
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
    activeFlusher = flusher;
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        for (const auto& impl : impls) {
            impl.run(static_cast<size_t>(i));
            gpuErrchk(hipStreamSynchronize(ctx.stream));
        }
    }
    rocprof::flushKernels();
    {
        std::lock_guard<std::mutex> lock(rocprof::mutex);
        rocprof::kernels.clear();
    }
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
            impls[k].run(static_cast<size_t>(WARMUP_ITERS + i));
            gpuErrchk(hipStreamSynchronize(ctx.stream));
            ROCP_CHECK(rocprofiler_get_timestamp(&w.end));
            windows.push_back(w);
        }
        if (i % 100 == 99) {
            rocprof::flushKernels();
        }
    }
    ROCP_CHECK(rocprofiler_stop_context(rocprof::kernelContext));
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
        (flusher ? r.gpuFlushedBlitTiming : r.gpuBlitTiming) = computeStats(blitMs[k]);
        std::vector<float> counts = blitCount[k];
        std::sort(counts.begin(), counts.end());
        (flusher ? r.gpuFlushedBlitDispatches : r.gpuBlitDispatches) = percentile(counts, 0.5);
        if (k != dpp) {
            double sum = 0.;
            for (int i = 0; i < TIMED_ITERS; ++i) {
                sum += ms[k][i] / ms[dpp][i];
            }
            speedup = sum / TIMED_ITERS;
        }
        std::cout << (flusher ? "GPU only, L2 + MALL flushed " : "GPU only ") << g.label() << " " << std::left
                  << std::setw(20) << impls[k].name << std::right << std::fixed << std::setprecision(2) << "  median "
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
    forEachGroup([&]<char PART, size_t N, int K, MorphOp OP>() { countGroup<PART, N, K, OP>(ctx); });
    writeOperationCounts(outDir / "kernel_counts_per_operation.csv");
    const bool tracing = kernelTracingWorks(ctx);
    if (tracing) {
        std::cout << "=== GPU time only (kernel dispatch tracing)" << std::endl;
        forEachGroup([&]<char PART, size_t N, int K, MorphOp OP>() { gpuOnlyGroup<PART, N, K, OP>(ctx, nullptr); });
        CacheFlusher flusher = makeCacheFlusher(ctx);
        forEachGroup([&]<char PART, size_t N, int K, MorphOp OP>() { gpuOnlyGroup<PART, N, K, OP>(ctx, &flusher); });
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
    }
    std::filesystem::remove(rows);
    std::filesystem::remove(choices);
    return tracing;
}

// ------------------------------------------------------------------------------------------------
// Output files
// ------------------------------------------------------------------------------------------------
void writeGroupColumns(std::ostream& os, const GroupResult& gr) {
    os << gr.group.part << "," << neighborhood::name(gr.group.op) << "," << gr.group.k << "," << neighborhood::passes(gr.group.op)
       << "," << gr.r.batch << "," << gr.r.name << "," << (gr.chosen ? "yes" : "no");
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

void writeGpuTime(const std::filesystem::path& path, bool flushed) {
    std::ofstream os(path);
    os << GROUP_HEADER << ",gpu_median_us,gpu_p10_us,gpu_p90_us,gpu_mean_us,iterations,kernels_launched,"
                          "chosen_dpp_mean_gpu_speedup,runtime_blit_median_us,runtime_blit_p10_us,runtime_blit_p90_us,"
                          "runtime_blit_mean_us,runtime_blit_dispatches_per_execution\n";
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
           << "\n";
    }
}

void writeValidation(const std::filesystem::path& path) {
    std::ofstream os(path);
    os << GROUP_HEADER << ",iteration,reference,max_abs_err_r,max_abs_err_g,max_abs_err_b,mean_abs_err,"
                          "non_finite_values,max_abs_err_vs_float_reference,values_beyond_1e-3,values,tolerance,"
                          "within_tolerance,max_abs_err_vs_rpp_sampling_8bit_reference\n";
    for (const auto& gr : groupResults) {
        const Result& r = gr.r;
        for (const auto& it : r.errorsPerIteration) {
            const auto& e = it.own;
            writeGroupColumns(os, gr);
            os << "," << it.iteration << "," << r.reference << "," << std::scientific << std::setprecision(3)
               << e.maxPerChannel[0] << "," << e.maxPerChannel[1] << "," << e.maxPerChannel[2] << "," << e.mean << ","
               << e.nonFinite << "," << it.vsFloat.max() << "," << e.beyond << "," << e.values << "," << r.tolerance()
               << "," << (r.withinTolerance(e) ? "yes" : "no") << ",";
            if (gr.group.part == 'b') {
                os << it.vsRppConvention.max();
            }
            os << std::defaultfloat << "\n";
        }
    }
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
    os << "Experiment 3 (neighborhood operations), AMD\n"
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
       << "Workload: the frame, regions and batch sizes of common/bench_geometry.h (" << describe()
       << "), without its border and its square output: each region is resized (bilinear, aspect ratio ignored) to "
       << PLANE_W << "x" << PLANE_H << ", then the neighborhood operation (K x K, centred anchor, replicate at the " << PLANE_W << "x"
       << PLANE_H << " edge in every pass; erode = per channel min, open = min then max), then Mul (1/255), Sub (0.485, "
       << "0.456, 0.406), Div (0.229, 0.224, 0.225), float3 tensor of N planes " << PLANE_W << "x" << PLANE_H
       << " (no AddBorder).\n"
       << "Groups: part (b) K = " << MORPH_K << ", erode and open, N = " << batchList(", ")
       << ": rpp_normalize, rpp_fused_normalize and the FKL MorphologyDPP tiles; part (a) K = 3, 5, 7, erode and open, N "
       << "= 1, 16: fkl_bvf_naive (ErodeBVF / DilateBVF of common/bench_neighborhood_ops.h) and the DPP tiles. DPP tiles: ";
    for (size_t i = 0; i < DPP_TILES.size(); ++i) {
        os << (i ? ", " : "") << DPP_TILES[i].width << "x" << DPP_TILES[i].height;
    }
    os << " (MorphologyDPPDetails<float3, W, H, K, K>).\n"
       << "Timing (CPU + GPU): as rocm_bench.hip: " << WARMUP_ITERS << " warm-up then " << TIMED_ITERS
       << " timed iterations; iteration t runs every implementation of the group on the regions of iteration t, each "
       << "between two HIP events on one stream and followed by a synchronization. Chosen per group: the DPP tile with "
       << "the lowest median among those within the float tolerance, and the RPP variant with the lowest median among "
       << "those within " << RPP_EQUIVALENCE << " of the RPP-convention reference (chosen = yes in the CSV files; every "
       << "implementation is recorded). Speedups: per iteration, implementation time / chosen DPP time, averaged over "
       << "the timed iterations (the chosen DPP's own row has none).\n"
       << "Validation: as rocm_bench.hip, on the timed iterations " << validation::VALIDATION_STRIDE << " apart and the "
       << "last, into NaN-poisoned outputs: FKL against FKL's CPU float chain (Figure 1 without AddBorder) followed by "
       << "the morphology of every plane (bench::neighborhood::hostMorphPlane; morphology after the normalization equals "
       << "morphology before it, because the normalization is monotone non-decreasing per channel), tolerance "
       << TOLERANCE << "; RPP against the same morphology of the RPP-convention reference of rocm_bench.hip (RPP's 8 bit "
       << "frame, RPP's bilinear sampling) without the border, tolerance " << TOLERANCE_8BIT << ". After the timing, "
       << "the first validated iteration again (every value within 1e-6).\n"
       << "RPP (HIP backend): rpp_normalize = rppt_yuv_to_rgb (full frame; 1 kernel and 8 hipMemcpyToSymbol per call), "
       << "rppt_crop_mirror_normalize 8u->32f (batched, the batch images aliasing the RGB frame, XYWH ROIs), "
       << "rppt_resize F32 bilinear (batched, LTRB ROIs) into N float3 planes of " << PLANE_W << "x" << PLANE_H
       << ", rppt_erode (open: rppt_erode into a second plane buffer, then rppt_dilate) batched over the N planes into "
       << "the output tensor, rppt_normalize (Mul + Sub + Div, rows of " << PLANE_W << " float3, per column mean 255*sub "
       << "and stdDev 255*div). rpp_fused_normalize = rppt_yuv_to_rgb, rppt_resize_mirror_normalize 8u->32f (crop + "
       << "convert + resize, mean 0, stdDev 1, batched, LTRB ROIs; it zeroes its destination and synchronizes the stream "
       << "inside the call) into the planes, then the same morphology and rppt_normalize. The morphology sources start "
       << RPP_LEAD_BYTES << " bytes into their allocations (descriptor offsetInBytes; RPP's HIP morphology refuses an "
       << "offset below 12 * (kernelSize / 2) bytes), the morphology ROIs are the whole planes in XYWH, uploaded once. "
       << "Every RPP pipeline starts with the hipMemcpyAsync of the ROIs of its iteration, as in rocm_bench.hip; no "
       << "memset of the output (no border), the only memset is the one inside rppt_resize_mirror_normalize "
       << "(rpp_fused_normalize). One RPP handle per pipeline with batch size N.\n"
       << "Kernel counts, GPU time only and GPU time only with L2 + MALL flushed: as rocm_bench.hip, in a profiling child "
       << "process, every implementation of every group"
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
            std::cout << describe() << "; Experiment 3 instead: no border, output N planes of " << PLANE_W << "x"
                      << PLANE_H << " float3, neighborhood operation after the resize (K = " << MORPH_K
                      << " in part (b))" << std::endl;
            return 0;
        }
        const std::filesystem::path outDir = argc > 1 ? argv[1] : "results_neighborhood";
        std::filesystem::create_directories(outDir);
        Context ctx;
        initContext(ctx);
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
        std::cout << diagnostic.str();
        const auto fklFailed = []() {
            return std::any_of(groupResults.begin(), groupResults.end(), [](const GroupResult& gr) {
                return gr.r.name.rfind("fkl_", 0) == 0 && !gr.r.withinTolerance();
            });
        };
        if (validateOnly) {
            forEachGroup([&]<char PART, size_t N, int K, MorphOp OP>() { runGroup<PART, N, K, OP>(ctx, false); });
            writeValidation(outDir / "validation.csv");
            if (fklFailed()) {
                std::cerr << "ERROR: an FKL output is not within its tolerance" << std::endl;
                return 1;
            }
            return 0;
        }
        measureEventFloor(ctx);
        forEachGroup([&]<char PART, size_t N, int K, MorphOp OP>() { runGroup<PART, N, K, OP>(ctx, true); });
        bool tracing = false;
        std::string profilingError;
        try {
            tracing = nb::runProfilingChild(outDir);
        } catch (const std::exception& e) {
            profilingError = e.what();
            std::cerr << "ERROR in the profiling pass: " << profilingError << std::endl;
        }
        writeLatency(outDir / "latency.csv");
        writeValidation(outDir / "validation.csv");
        if (tracing) {
            writeGpuTime(outDir / "gpu_time.csv", false);
            writeGpuTime(outDir / "gpu_time_l2_flushed.csv", true);
        }
        writeRegionStats((outDir / "regions.csv").string(), *ctx.regions, WARMUP_ITERS, TIMED_ITERS);
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
