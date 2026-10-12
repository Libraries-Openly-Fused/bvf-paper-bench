// Prints the C++ type of the fused read operation that executeOperations builds for the Figure 1 chain
// (ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder, after BackFuser::fuse_back), for the paper's
// Builder section.

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/utils/type_to_string.h>

#if !defined(_MSC_VER)
#include <cxxabi.h>
#endif

#include "bench_geometry.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <typeinfo>

template <typename T>
std::string demangled() {
#if defined(_MSC_VER)
    // MSVC's type names are not mangled
    return typeid(T).name();
#else
    int status = 0;
    char* name = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
    std::string result = status == 0 ? name : typeid(T).name();
    std::free(name);
    return result;
#endif
}

int main() {
    using namespace bench::geometry;
    constexpr size_t N = MAX_BATCH;
    std::array<fk::Rect, N> crops{};
    for (size_t i = 0; i < N; ++i) {
        const RegionSize s = regionSize(0, i);
        crops[i] = fk::Rect(static_cast<int>(i), static_cast<int>(i), s.width, s.height);
    }
    const fk::Image<fk::PixelFormat::NV12> frame(FRAME_W, FRAME_H);
    const fk::Tensor<fk::float3> output(OUT_W, OUT_H, N);
    const fk::float3 black{0.f, 0.f, 0.f};

    const auto fused = fk::BackFuser::fuse_back(
        fk::ReadYUV<fk::PixelFormat::NV12>::build(frame),
        fk::ConvertYUVToRGB<fk::ColorDepth::p8bit, fk::ColorRange::Limited, fk::ColorPrimitives::bt601>::build(),
        fk::Crop<>::build(crops),
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR>::build(fk::Size(RES_W, RES_H)),
        fk::AddBorder<>::build(BORDER_TOP, BORDER_BOTTOM, 0, 0, black),
        fk::Mul<fk::float3>::build(fk::float3{1.f, 1.f, 1.f}),
        fk::Sub<fk::float3>::build(fk::float3{0.f, 0.f, 0.f}),
        fk::Div<fk::float3>::build(fk::float3{1.f, 1.f, 1.f}),
        fk::TensorWrite<fk::float3>::build(output.ptr()));
    using FusedReadIOp = std::decay_t<decltype(fk::get<0>(fused))>;
    using FusedReadOperation = typename FusedReadIOp::Operation;

    std::cout << "N = " << N << " crops\n\n"
              << "fk::typeToString of the fused read IOp:\n" << fk::typeToString<FusedReadIOp>() << "\n\n"
              << "fk::typeToString of its Operation:\n" << fk::typeToString<FusedReadOperation>() << "\n\n"
              << "Demangled compiler type of the fused read IOp:\n" << demangled<FusedReadIOp>() << "\n\n"
              << "sizeof(fused read IOp) = " << sizeof(FusedReadIOp) << " bytes\n";
    return 0;
}
