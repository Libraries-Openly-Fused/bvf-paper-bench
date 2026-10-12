# FKL's ablation

Part of the [BVF paper benchmarks](../README.md): what Backwards Vertical Fusion adds to FKL on the Figure 1 pipeline,
on the [workload](../README.md#workload) of every benchmark of this repository (NV12 to RGB in BT.601 limited range).

| Implementation | What runs |
|---|---|
| `fkl_vf_hf` | FKL with vertical and horizontal fusion but without BVF: four `executeOperations` calls, with the intermediates of the OpenCV and NPP pipelines in memory: an 8 bit RGB frame (quantized as in the 8 bit reference), float3 crops, float3 resized crops, and AddBorder + Mul + Sub + Div fused into the write of the float3 output |
| `fkl_bvf` | FKL with BVF: one `executeOperations` call, one kernel for the whole chain, with no intermediates in memory |

The timings, modes, output files and validation are those of every GPU benchmark on
[`common/bench_gpu.h`](../common/bench_gpu.h), described in the [main README](../README.md#the-gpu-benchmarks).
`benchmark_fkl_ablation <dir> --host-overhead` also measures the host time FKL spends enqueuing each pipeline
(`host_overhead.csv`): for FKL BVF, building the nine operations, `BackFuser::fuse_back` and the launch; for FKL VF+HF,
each of its four calls. `bvf_types` prints the C++ type of the fused read operation FKL builds for the chain, for the
paper's Builder section.

## Build and run

    cmake -S fkl-ablation -B build-ablation -DCMAKE_BUILD_TYPE=Release
    cmake --build build-ablation -j
    ./build-ablation/benchmark_fkl_ablation ../results/fkl-ablation                    # CPU + GPU and GPU only
    ./build-ablation/benchmark_fkl_ablation ../results/fkl-ablation --gpu-only-l2-flushed
    ./build-ablation/benchmark_fkl_ablation ../results/fkl-ablation --host-overhead
    ./build-ablation/bvf_types

The compile-time units are in [`src/compile_time.cu`](src/compile_time.cu) (`HEADERS`, `BVF`, `BVF_CWD`, `VFHF`; with
`SET=ablation scripts/compile_time.sh`, see the [main README](../README.md#compile-time)).
