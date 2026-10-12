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

// What the GPU benchmarks need to call NPP: the error check, the stream context of the _Ctx functions (the only ones
// NPP 13 exports), and NPP's NV12 -> RGB conversion with BT.601 in limited range.

#ifndef BVF_BENCH_NPP_H
#define BVF_BENCH_NPP_H

#include <cuda_runtime.h>
#include <npp.h>

#include <stdexcept>
#include <string>

#define NPP_CHECK(expr)                                                                                     \
    do {                                                                                                    \
        const NppStatus status_ = (expr);                                                                   \
        if (status_ != NPP_SUCCESS) {                                                                       \
            throw std::runtime_error(std::string("NPP error ") + std::to_string(status_) + " in " + #expr); \
        }                                                                                                   \
    } while (0)

namespace bench::npp {

inline void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(error) + " in " + what);
    }
}

// The NppStreamContext of a stream, filled in as NPP's documentation describes (NPP has no function that does it)
inline NppStreamContext streamContext(cudaStream_t stream) {
    NppStreamContext ctx{};
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    ctx.hStream = stream;
    ctx.nCudaDeviceId = device;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    check(cudaStreamGetFlags(stream, &ctx.nStreamFlags), "cudaStreamGetFlags");
    return ctx;
}

// NV12 -> 8 bit RGB with BT.601 in limited range. NPP's plain nppiNV12ToRGB_8u_P2C3R does not implement BT.601 (measured
// with NPP 13.1: it computes the analog YUV formula R = Y + 1.140 V, G = Y - 0.394 U - 0.581 V, B = Y + 2.032 U and
// truncates), so the conversion is nppiNV12ToRGB_8u_ColorTwist32f_P2C3R with the BT.601 limited-range matrix, written
// here from the standard: Kr = 0.299, Kb = 0.114, Y scaled by 255/219 after subtracting 16 and U, V by 255/224 after
// subtracting 128. In NPP 13.1 the fourth column of the twist is added to the result (measured; the header describes
// it as added to the source), so it holds the offsets of the affine map.
inline void bt601LimitedTwist(Npp32f twist[3][4]) {
    constexpr double kr = 0.299, kb = 0.114, kg = 1. - kr - kb;
    constexpr double ys = 255. / 219., cs = 255. / 224.;
    constexpr double rv = 2. * (1. - kr), bu = 2. * (1. - kb);
    constexpr double gu = -2. * kb * (1. - kb) / kg, gv = -2. * kr * (1. - kr) / kg;
    const double m[3][3] = {{ys, 0., cs * rv}, {ys, cs * gu, cs * gv}, {ys, cs * bu, 0.}};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            twist[r][c] = static_cast<Npp32f>(m[r][c]);
        }
        twist[r][3] = static_cast<Npp32f>(-16. * m[r][0] - 128. * (m[r][1] + m[r][2]));
    }
}

inline void nv12ToRgb8Bt601Limited(const Npp8u* y, const Npp8u* uv, int pitch, Npp8u* dst, int dstStep, NppiSize size,
                                   const NppStreamContext& ctx) {
    Npp32f twist[3][4];
    bt601LimitedTwist(twist);
    const Npp8u* src[2] = {y, uv};
    int srcSteps[2] = {pitch, pitch};
    NPP_CHECK(nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(src, srcSteps, dst, dstStep, size, twist, ctx));
}

} // namespace bench::npp

#endif // BVF_BENCH_NPP_H
