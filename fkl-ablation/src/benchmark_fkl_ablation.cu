// FKL's own ablation of the Figure 1 pipeline of the Backwards Vertical Fusion paper: FKL with vertical and horizontal
// fusion but without BVF (four kernels with the intermediates in memory) against FKL with BVF (one kernel), and the
// host time FKL spends enqueuing each of them (--host-overhead).
//   ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// with the frame, sizes, batch sizes and crop positions of common/bench_geometry.h and NV12 -> RGB in BT.601 limited
// range. The harness (validation, timing, modes, output files) is common/bench_gpu.h.
//
//   fkl_vf_hf   four FKL kernels with the intermediates in DRAM (VF + HF, no BVF), with the intermediates of the
//               OpenCV and NPP pipelines: 8 bit RGB frame, then float3 crops, float3 resized crops and the output
//   fkl_bvf     one FKL kernel for the whole chain (BVF)

#include "bench_gpu.h"

namespace bench {

// Views of the N planes of a contiguous tensor of planes of planeWidth x planeHeight, each view of width x height at the
// top-left corner of its plane, with the pitch of the plane
template <size_t N, typename T>
std::array<fk::RawPtr<fk::ND::_2D, T>, N> planeViews(const fk::Tensor<T>& tensor, int planeWidth, int planeHeight,
                                                     const std::array<fk::Size, N>& sizes) {
    const auto raw = tensor.ptr();
    if (raw.dims.pitch != planeWidth * sizeof(T)) {
        throw std::runtime_error("Expected a contiguous Tensor");
    }
    std::array<fk::RawPtr<fk::ND::_2D, T>, N> views{};
    for (size_t i = 0; i < N; ++i) {
        views[i].data = raw.data + i * planeWidth * planeHeight;
        views[i].dims = fk::PtrDims<fk::ND::_2D>(sizes[i].width, sizes[i].height, planeWidth * sizeof(T));
    }
    return views;
}

// VF + HF without BVF: four kernels with the intermediates in DRAM, the same intermediates as the OpenCV and NPP
// pipelines: an 8 bit RGB frame (NPP's NV12 -> RGB and OpenCV's cudacodec one can only write 8 or 16 bit integers),
// quantized as in the 8 bit reference (normalize, add half a level, __saturatef, denormalize), then
// float3 crops (like their per crop convertTo / Convert_8u32f) in planes of the largest region size, float3 resized
// crops and the float3 output. The crop kernel is one batch of N crops of different sizes; an FKL batch runs the
// threads of its largest plane in every plane (the largest region of the iteration, 48x24 as soon as N >= 3), so for
// a smaller region the threads beyond it read the RGB frame next to it (inside the frame: positions are drawn for
// 48x24) and write the rest of its plane, which the resize does not read: the resize reads every crop through a view
// of its own size.
template <size_t N>
Implementation makeFklVfHf(Context& ctx) {
    auto rgb = std::make_shared<fk::Ptr2D<fk::uchar3>>(FRAME_W, FRAME_H, 0, fk::MemType::Device);
    auto cropped = std::make_shared<fk::Tensor<fk::float3>>(MAX_REGION_W, MAX_REGION_H, N, 1, fk::MemType::Device);
    auto resized = std::make_shared<fk::Tensor<fk::float3>>(RES_W, RES_H, N, 1, fk::MemType::Device);
    auto out = std::make_shared<fk::Tensor<fk::float3>>(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    const auto regions = fklRegionSets<N>(ctx);
    // The views of the crops of every iteration, made before the loops like the rectangles; the resized crops all have
    // the same size
    auto croppedViews = std::make_shared<std::vector<std::array<fk::RawPtr<fk::ND::_2D, fk::float3>, N>>>();
    for (const auto& rects : *regions) {
        std::array<fk::Size, N> sizes{};
        for (size_t i = 0; i < N; ++i) {
            sizes[i] = fk::Size(rects[i].width, rects[i].height);
        }
        croppedViews->push_back(planeViews<N>(*cropped, MAX_REGION_W, MAX_REGION_H, sizes));
    }
    std::array<fk::Size, N> resizedSizes{};
    resizedSizes.fill(fk::Size(RES_W, RES_H));
    const auto resizedViews = planeViews<N>(*resized, RES_W, RES_H, resizedSizes);
    Implementation impl{"fkl_vf_hf", Reference::Fkl8bit};
    impl.stages.emplace_back("1_nv12_to_rgb8_full_frame", [&ctx, rgb](size_t) {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice),
            YUVToRGB::build(),
            fk::NormalizePixel<fk::float3, fk::ColorDepth::p8bit>::build(),
            fk::Add<fk::float3>::build(fk::make_set<fk::float3>(QUANTIZATION_OFFSET)),
            fk::SaturateDenormalizePixel<fk::float3, fk::uchar3, fk::ColorDepth::p8bit>::build(),
            fk::PerThreadWrite<fk::ND::_2D, fk::uchar3>::build(rgb->ptr()));
    });
    impl.stages.emplace_back("2_crop_to_float3", [&ctx, rgb, cropped, regions](size_t t) {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(rgb->ptr()),
            fk::Crop<>::build((*regions)[t]),
            fk::Cast<fk::uchar3, fk::float3>::build(),
            fk::TensorWrite<fk::float3>::build(cropped->ptr()));
    });
    impl.stages.emplace_back("3_resize", [&ctx, resized, croppedViews](size_t t) {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::PerThreadRead<fk::ND::_2D, fk::float3>::build((*croppedViews)[t]),
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
            fk::TensorWrite<fk::float3>::build(resized->ptr()));
    });
    impl.stages.emplace_back("4_addborder_mul_sub_div_write", [&ctx, out, resizedViews](size_t) {
        beforeKernel();
        fk::executeOperations<fk::TransformDPP<>>(*ctx.fkStream,
            fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(resizedViews),
            fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
            fk::Mul<fk::float3>::build(mulValue),
            fk::Sub<fk::float3>::build(subValue),
            fk::Div<fk::float3>::build(divValue),
            fk::TensorWrite<fk::float3>::build(out->ptr()));
    });
    impl.run = runStages(impl.stages);
    impl.download = [out]() { return downloadDevice(out->ptr().data, N * OUT_FLOATS); };
    impl.fillNaN = [out]() { gpuErrchk(cudaMemset(out->ptr().data, 0xFF, N * OUT_FLOATS * sizeof(float))); };
    return impl;
}

// Host time of enqueuing the FKL pipelines (std::chrono, median, p10 and p90 over the TIMED_ITERS iterations, which have
// different regions, synchronizing after each one), on the regions of every iteration as in the timed loops. FKL BVF
// is split into building the nine IOps, BackFuser::fuse_back and executeOperations, which then only launches; FKL
// VF+HF is split per kernel, enqueued back to back as in its run().
struct HostTiming {
    std::string name;
    int batch{0};
    std::string step;
    TimingStats timing;
};

template <size_t N>
void hostOverheadBatch(Context& ctx, std::vector<HostTiming>& hostTimings) {
    using Clock = std::chrono::steady_clock;
    // In milliseconds, like the CUDA event times: computeStats() takes milliseconds and reports microseconds
    const auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<float, std::milli>(b - a).count();
    };
    const auto regions = fklRegionSets<N>(ctx);
    fk::Tensor<fk::float3> out(OUT_W, OUT_H, N, 1, fk::MemType::Device);
    std::vector<float> build, fuse, launch, total;
    for (int i = 0; i < WARMUP_ITERS + TIMED_ITERS; ++i) {
        const auto t0 = Clock::now();
        const auto read = fk::ReadYUV<fk::PixelFormat::NV12>::build(*ctx.frameDevice);
        const auto convert = YUVToRGB::build();
        const auto crop = fk::Crop<>::build((*regions)[i]);
        const auto resize =
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H));
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
            build.push_back(ms(t0, t1));
            fuse.push_back(ms(t1, t2));
            launch.push_back(ms(t2, t3));
            total.push_back(ms(t0, t3));
        }
    }
    hostTimings.push_back({"fkl_bvf", static_cast<int>(N), "1_build_iops", computeStats(build)});
    hostTimings.push_back({"fkl_bvf", static_cast<int>(N), "2_fuse_back", computeStats(fuse)});
    hostTimings.push_back({"fkl_bvf", static_cast<int>(N), "3_launch", computeStats(launch)});
    hostTimings.push_back({"fkl_bvf", static_cast<int>(N), "total_host_enqueue", computeStats(total)});

    const auto vfhf = makeFklVfHf<N>(ctx);
    std::vector<std::vector<float>> stageTimes(vfhf.stages.size());
    std::vector<float> vfhfTotal;
    for (int i = 0; i < WARMUP_ITERS + TIMED_ITERS; ++i) {
        Clock::time_point t = Clock::now();
        const Clock::time_point first = t;
        for (size_t k = 0; k < vfhf.stages.size(); ++k) {
            vfhf.stages[k].second(static_cast<size_t>(i));
            const Clock::time_point next = Clock::now();
            if (i >= WARMUP_ITERS) {
                stageTimes[k].push_back(ms(t, next));
            }
            t = next;
        }
        if (i >= WARMUP_ITERS) {
            vfhfTotal.push_back(ms(first, t));
        }
        gpuErrchk(cudaStreamSynchronize(ctx.stream));
    }
    for (size_t k = 0; k < vfhf.stages.size(); ++k) {
        hostTimings.push_back(
            {vfhf.name, static_cast<int>(N), "enqueue_" + vfhf.stages[k].first, computeStats(stageTimes[k])});
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

struct FklAblation {
    static constexpr const char* title = "FKL VF+HF vs FKL BVF (fkl-ablation)";

    template <size_t N>
    static std::vector<Implementation> implementations(Context& ctx) {
        std::vector<Implementation> impls;
        impls.push_back(makeFklVfHf<N>(ctx));
        return impls;
    }

    static void describe(std::ostream& os, const std::filesystem::path&) {
        os << "FKL VF+HF pipeline: four executeOperations calls (vertical and horizontal fusion, no BVF), with the "
              "intermediates of the OpenCV and NPP pipelines in memory: an 8 bit RGB frame (quantized as in the 8 bit "
              "reference), float3 crops (planes of the largest region size), float3 resized crops, and AddBorder + "
              "Mul + Sub + Div fused into the write of the float3 output. The crop kernel is one FKL batch of N "
              "crops of different sizes, which runs the threads of the iteration's largest region in every plane "
              "(48x24 as soon as N >= 3): for a smaller region, the threads beyond it write the rest of its plane, "
              "which the resize does not read (it reads every crop through a view of its own size). Per iteration, "
              "inside the timed region: the build() calls of the four chains with the iteration's rectangles and "
              "crop views.\n";
    }

    static std::string diagnostic(const Context&) { return ""; }

    // --host-overhead: host time to enqueue the FKL pipelines (host_overhead.csv)
    static bool extraMode(Context& ctx, const std::string& mode, const std::filesystem::path& outDir) {
        if (mode != "--host-overhead") {
            return false;
        }
        std::vector<HostTiming> hostTimings;
        forEachBatch([&]<size_t N>() { hostOverheadBatch<N>(ctx, hostTimings); });
        std::ofstream csv(outDir / "host_overhead.csv");
        csv << "implementation,N,step,host_median_us,host_p10_us,host_p90_us\n";
        for (const auto& h : hostTimings) {
            csv << h.name << "," << h.batch << "," << h.step << "," << std::fixed << std::setprecision(2)
                << h.timing.median << "," << h.timing.p10 << "," << h.timing.p90 << "\n";
        }
        return true;
    }
};

} // namespace bench

int main(int argc, char** argv) {
    return bench::gpuBenchMain<bench::FklAblation>(argc, argv);
}
