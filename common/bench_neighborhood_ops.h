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

// The box blur of the neighborhood experiment (bench_neighborhood.h) written as a plain Backwards Vertical Fusion
// operation: BoxBVF<K> is a ReadBack operation, like Crop and Resize. Each output value reads the K x K window of the
// operation before it, so the whole chain before it runs K x K times per output value: two passes (BoxBVF after
// BoxBVF) run it K^4 times. This is the case that FKL's MorphologyDPP avoids by staging the values in shared memory.
//
//   executeOperations<TransformDPP<>>(stream, ReadYUV<...>::build(frame), YUVToRGB::build(), Crop<>::build(regions),
//                                     BoxBVF<5>::build(), Mul..., Sub..., Div..., TensorWrite...);
//
// The window is centered on the output value, and its coordinates are clamped to the size of the operation before
// (NumElems): it replicates the edge of the image it reads (the crop), so a second BoxBVF replicates the edge of the
// blurred crop, as MorphologyDPP does in every pass. The value is the sum of the window times 1 / K^2, per channel for
// vector types. With a batched read (Crop of an array of regions), z selects the region, as in every BVF chain.

#ifndef BVF_BENCH_NEIGHBORHOOD_OPS_H
#define BVF_BENCH_NEIGHBORHOOD_OPS_H

#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>

// Inside namespace fk, as FKL's own operations: the DECLARE_*_PARENT macros use FKL's names unqualified
namespace fk {

// No runtime parameters: the box size is the template parameter K and the anchor is its center
struct BoxBVFParams {};

template <int K, typename BackIOp_ = NullType>
struct BoxBVF {
    static_assert(K > 0 && K % 2 == 1, "BoxBVF needs an odd box size, centered on the output value");
    static_assert(isAnyCompleteReadType<BackIOp_>, "The BackIOp_ must be a complete Read type");

private:
    using SelfType = BoxBVF<K, BackIOp_>;
    FK_HOST_DEVICE_FUSE int clamp(const int value, const int high) {
        return value < 0 ? 0 : (value > high ? high : value);
    }

public:
    FK_STATIC_STRUCT(BoxBVF, SelfType)
    using Parent = ReadBackOperation<typename BackIOp_::Operation::OutputType, BoxBVFParams, BackIOp_,
                                     typename BackIOp_::Operation::OutputType, BoxBVF<K, BackIOp_>>;
    DECLARE_READBACK_PARENT

    FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params, const BackIOp& backIOp) {
        (void)params;
        const int lastX = static_cast<int>(NumElems::x(thread, backIOp)) - 1;
        const int lastY = static_cast<int>(NumElems::y(thread, backIOp)) - 1;
        constexpr int ANCHOR = K / 2;
        OutputType sum = BackIOp::Operation::exec(
            Point{clamp(thread.x - ANCHOR, lastX), clamp(thread.y - ANCHOR, lastY), thread.z}, backIOp);
        for (int ky = 0; ky < K; ++ky) {
            const int y = clamp(thread.y + ky - ANCHOR, lastY);
            for (int kx = 0; kx < K; ++kx) {
                if (kx == 0 && ky == 0) {
                    continue;
                }
                sum = sum + BackIOp::Operation::exec(Point{clamp(thread.x + kx - ANCHOR, lastX), y, thread.z}, backIOp);
            }
        }
        return sum * (1.f / static_cast<float>(K * K));
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
template <int K>
struct BoxBVF<K, NullType> {
private:
    using SelfType = BoxBVF<K, NullType>;

public:
    FK_STATIC_STRUCT(BoxBVF, SelfType)
    using Parent = IncompleteReadBackOperation<NullType, BoxBVFParams, NullType, NullType, BoxBVF<K, NullType>>;
    DECLARE_INCOMPLETEREADBACK_PARENT

    FK_HOST_FUSE auto build() { return InstantiableType{{BoxBVFParams{}, NullType{}}}; }

    template <typename BackIOp>
    FK_HOST_FUSE auto build(const BackIOp& backIOp, const InstantiableType& iOp) {
        return BoxBVF<K, BackIOp>::build(iOp.params, backIOp);
    }
};

} // namespace fk

#endif // BVF_BENCH_NEIGHBORHOOD_OPS_H
