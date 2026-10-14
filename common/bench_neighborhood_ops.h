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

// The morphology of the neighborhood experiment (bench_neighborhood.h) written as plain Backwards Vertical Fusion
// operations: ErodeBVF<K> and DilateBVF<K> are ReadBack operations, like Crop and Resize. Each output value reads the
// K x K window of the operation before it, so the whole chain before it runs K x K times per output value: an opening
// (DilateBVF after ErodeBVF) runs it K^4 times, and a bilinear Resize before them reads four values of its own source
// for each of those. This is the case that FKL's MorphologyDPP avoids by staging the values in shared memory.
//
//   executeOperations<TransformDPP<>>(stream, ReadYUV<...>::build(frame), YUVToRGB::build(), Crop<>::build(regions),
//                                     Resize<...>::build(Size(PLANE_W, PLANE_H)), ErodeBVF<5>::build(),
//                                     DilateBVF<5>::build(), Mul..., Sub..., Div..., TensorWrite...);
//
// The window is centered on the output value, and its coordinates are clamped to the size of the operation before
// (NumElems): every operation replicates the border of the image it reads, so DilateBVF after ErodeBVF replicates the
// border of the eroded image, as MorphologyDPP does in every pass. The minimum and the maximum are per channel for
// vector types. With a batched read (Crop of an array of regions), z selects the region, as in every BVF chain.

#ifndef BVF_BENCH_NEIGHBORHOOD_OPS_H
#define BVF_BENCH_NEIGHBORHOOD_OPS_H

#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>

// Inside namespace fk, as FKL's own operations: the DECLARE_*_PARENT macros use FKL's names unqualified
namespace fk {

enum class MorphReduction { Min, Max };

// No runtime parameters: the mask size is the template parameter K and the anchor is its center
struct MorphBVFParams {};

template <int K, MorphReduction R, typename BackIOp_ = NullType>
struct MorphBVF {
    static_assert(K > 0 && K % 2 == 1, "MorphBVF needs an odd mask size, centered on the output value");
    static_assert(isAnyCompleteReadType<BackIOp_>, "The BackIOp_ must be a complete Read type");

private:
    using SelfType = MorphBVF<K, R, BackIOp_>;

    FK_HOST_DEVICE_FUSE int clamp(const int value, const int high) {
        return value < 0 ? 0 : (value > high ? high : value);
    }

public:
    FK_STATIC_STRUCT(MorphBVF, SelfType)
    using Parent = ReadBackOperation<typename BackIOp_::Operation::OutputType, MorphBVFParams, BackIOp_,
                                     typename BackIOp_::Operation::OutputType, MorphBVF<K, R, BackIOp_>>;
    DECLARE_READBACK_PARENT

    FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params, const BackIOp& backIOp) {
        (void)params;
        const int lastX = static_cast<int>(NumElems::x(thread, backIOp)) - 1;
        const int lastY = static_cast<int>(NumElems::y(thread, backIOp)) - 1;
        constexpr int ANCHOR = K / 2;
        OutputType value = BackIOp::Operation::exec(
            Point{clamp(thread.x - ANCHOR, lastX), clamp(thread.y - ANCHOR, lastY), thread.z}, backIOp);
        for (int ky = 0; ky < K; ++ky) {
            const int y = clamp(thread.y + ky - ANCHOR, lastY);
            for (int kx = 0; kx < K; ++kx) {
                if (kx == 0 && ky == 0) {
                    continue;
                }
                const OutputType sample =
                    BackIOp::Operation::exec(Point{clamp(thread.x + kx - ANCHOR, lastX), y, thread.z}, backIOp);
                if constexpr (R == MorphReduction::Min) {
                    value = cxp::min::f(value, sample);
                } else {
                    value = cxp::max::f(value, sample);
                }
            }
        }
        return value;
    }

    // The size of the operation before: the window moves over it, it does not crop it
    FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
        return NumElems::x(thread, opData.backIOp);
    }
    FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
        return NumElems::y(thread, opData.backIOp);
    }
    FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
        return 1;
    }
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
        return {num_elems_x(Point{0, 0, 0}, opData), num_elems_y(Point{0, 0, 0}, opData),
                num_elems_z(Point{0, 0, 0}, opData)};
    }
};

// The version users build, with no operation before it yet: fusion (then(), executeOperations) gives it the
// operation before, as with Crop<> and Resize<>
template <int K, MorphReduction R>
struct MorphBVF<K, R, NullType> {
private:
    using SelfType = MorphBVF<K, R, NullType>;

public:
    FK_STATIC_STRUCT(MorphBVF, SelfType)
    using Parent = IncompleteReadBackOperation<NullType, MorphBVFParams, NullType, NullType, MorphBVF<K, R, NullType>>;
    DECLARE_INCOMPLETEREADBACK_PARENT

    FK_HOST_FUSE auto build() { return InstantiableType{{MorphBVFParams{}, NullType{}}}; }

    template <typename BackIOp>
    FK_HOST_FUSE auto build(const BackIOp& backIOp, const InstantiableType& iOp) {
        return MorphBVF<K, R, BackIOp>::build(iOp.params, backIOp);
    }
};

// Erosion and dilation with a K x K mask anchored at its center
template <int K>
using ErodeBVF = MorphBVF<K, MorphReduction::Min>;
template <int K>
using DilateBVF = MorphBVF<K, MorphReduction::Max>;

} // namespace fk

#endif // BVF_BENCH_NEIGHBORHOOD_OPS_H
