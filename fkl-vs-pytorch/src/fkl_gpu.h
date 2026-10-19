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

// FKL's side of the GPU program (src/fkl_gpu.cu), compiled as CUDA with nvcc or as HIP, separately from PyTorch's side
// (src/benchmark_pytorch_gpu.cpp, compiled against LibTorch with the C++ compiler): FKL BVF on the GPU, the references,
// FKL's conversion coefficients and the description of the device and of the FKL build. Neither FKL's nor PyTorch's
// headers are included here, so each side sees only its own library.

#ifndef BVF_BENCH_PYTORCH_FKL_GPU_H
#define BVF_BENCH_PYTORCH_FKL_GPU_H

#include "bench_geometry.h"
#include "bench_pipeline.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bench::pytorch {

// FKL with Backwards Vertical Fusion on N regions: one executeOperations call, one kernel for the whole chain (the
// paper's Listing 2, as fkl_bvf of common/bench_gpu.h), with the frame already in device memory
class FklGpu {
public:
    virtual ~FklGpu() = default;
    // Enqueues the pipeline on the regions of iteration t on the stream given to makeFklGpu
    virtual void run(size_t t) = 0;
    // The N x OUT_H x OUT_W x 3 output, after waiting for the stream
    virtual std::vector<float> download() = 0;
    // Poisons the output (NaN), on the stream
    virtual void fillNaN() = 0;
};

// n is one of BATCHES; stream is the cudaStream_t (hipStream_t on ROCm) of PyTorch's stream, so that both libraries
// run on one stream; regions must outlive the result
std::unique_ptr<FklGpu> makeFklGpu(size_t n, void* stream, const geometry::RegionSets& regions);

// The NV12 frame of every benchmark (common/bench_geometry.h's generateFrame) in host memory: FRAME_H * 3 / 2 rows of
// FRAME_W bytes, *pitch bytes apart
const uint8_t* hostFrame(size_t* pitch);

// FKL's chain on its CPU backend, in float: the reference of FKL BVF (fkl_cpu_float of common/bench_gpu.h)
std::vector<float> fklCpuReference(size_t n, const geometry::RegionSets& regions, size_t t);
// The plain C++ reference of common/bench_reference.h with PyTorch's pixel-center sampling
std::vector<float> pixelCenterReference(size_t n, const geometry::RegionSets& regions, size_t t);

// FKL's NV12 -> RGB conversion as a matrix and offsets, for PyTorch's pipeline
pipeline::Conversion fklConversion();

// Launches an empty kernel, benchExecutionMarker, on the stream: in a trace of one library's loop, the GPU time pass
// launches it after every execution, and splits the trace there
void launchExecutionMarker(void* stream);
// The name of that kernel, as traces show it (a substring)
constexpr const char* EXECUTION_MARKER = "benchExecutionMarker";

// "GPU_NVIDIA" or "GPU_AMD": FKL's name of the GPU backend, for the file names
std::string parArchName();
// The GPU, the GPU runtime and the compiler and flags of FKL's code, for the platform file
std::string describeGpuAndFkl();

// The PCI bus ID of the current device (cudaDeviceGetPCIBusId / hipDeviceGetPCIBusId), as nvidia-smi --id takes it
std::string gpuPciBusId();

} // namespace bench::pytorch

#endif // BVF_BENCH_PYTORCH_FKL_GPU_H
