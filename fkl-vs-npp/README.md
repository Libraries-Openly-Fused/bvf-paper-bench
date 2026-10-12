# FKL vs NPP

Part of the [BVF paper benchmarks](../README.md): the Figure 1 pipeline with NVIDIA NPP and with FKL, on the
[workload](../README.md#workload) of every benchmark of this repository (NV12 to RGB in BT.601 limited range).

| Implementation | What runs |
|---|---|
| `npp` | `nppiNV12ToRGB_8u_ColorTwist32f_P2C3R` with the BT.601 limited range matrix of the standard ([`common/bench_npp.h`](../common/bench_npp.h); NPP's plain NV12 conversion is not BT.601) of the whole frame into an 8 bit RGB frame, then per region `nppiConvert_8u32f_C3R` (float3 crop, into a buffer of the largest region size), one `nppiResizeBatch_32f_C3R_Advanced` for all regions with their own sizes (`nppiResize_32f_C3R` for N = 1, since the batched resize rejects a batch of one), per region `nppiCopyConstBorder_32f_C3R` (NPP has no batched border), then `nppiMulC`, `nppiSubC` and `nppiDivC_32f_C3IR` on the whole float3 output: 2N + 5 kernels. The batched resize reads per-image sizes and ROIs from descriptors in device memory: in every iteration, inside the timed region, the source descriptors and ROIs of its regions are written into a pinned host buffer and uploaded with one `cudaMemcpyAsync` (one memory copy per execution for N > 1; the destination descriptors never change and are uploaded once) |
| `fkl_bvf` | FKL with BVF: one `executeOperations` call, one kernel for the whole chain, with no intermediates in memory |

NPP's bilinear resize samples at `dst * scale - 0.25` (measured with linear ramps, for upscales), not at
`dst * scale` like FKL's, so NPP is checked against the chain on FKL's CPU backend with the RGB rounded to 8 bits and
NPP's sampling. The timings, modes, output files and validation are those of every GPU benchmark on
[`common/bench_gpu.h`](../common/bench_gpu.h), described in the [main README](../README.md#the-gpu-benchmarks).

## Build and run

    cmake -S fkl-vs-npp -B build-npp -DCMAKE_BUILD_TYPE=Release
    cmake --build build-npp -j
    ./build-npp/benchmark_npp ../results/fkl-vs-npp                          # CPU + GPU and GPU only
    ./build-npp/benchmark_npp ../results/fkl-vs-npp --gpu-only-l2-flushed
    ./build-npp/benchmark_npp ../results/fkl-vs-npp --profile-kernels

The compile-time units are in [`src/compile_time.cu`](src/compile_time.cu) (`HEADERS`, `NPP`, `BVF`; with
`SET=npp scripts/compile_time.sh`, see the [main README](../README.md#compile-time)).
