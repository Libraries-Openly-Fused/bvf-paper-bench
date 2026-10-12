# FKL vs NPPDx

Part of the [BVF paper benchmarks](../README.md) (until October 2026 the separate repository FKL-vs-NPPDx).
Benchmarks comparing the [Fused Kernel Library (FKL)](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary)
and NVIDIA [NPPDx](https://github.com/NVIDIA/NPPDx) on the image preprocessing pipeline of the Backwards Vertical
Fusion paper (Figure 1):

    ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write

One 3840x2160 NV12 frame in device memory, N crops of 20x10 pixels at fixed random positions, NV12 to RGB
(BT.2020, full range), bilinear resize to 64x32, 16 black rows added on top and bottom (64x64), then
`(x * mul - sub) / div` with ImageNet constants (`mul = 1/255`, `sub = (0.485, 0.456, 0.406)`,
`div = (0.229, 0.224, 0.225)`), written as float3 into a tensor of N planes. N is 1, 4, 16, 50 and 100.

Both libraries run the whole pipeline in **one kernel**:

- **FKL**: one `executeOperations` call with the nine operations (Backwards Vertical Fusion), from the `fkl`
  submodule of this repository (FKL main, which has `AddBorder`).
- **NPPDx**: one CUDA kernel built from NPPDx operations, one thread block per crop: NV12 ingest of the crop,
  `ColorConvert<yuv_bt2020, rgb>` and `Resize<5, 16, bilinear>` on the shared memory tile. AddBorder, Mul, Sub,
  Div and the float3 write are written by hand in the same kernel, because NPPDx 0.1.2 has no border
  operation, its affine channel map only takes integer constants, and it has no floating point output format.

## Results on an RTX 4090

Two benchmarks, with the same iterations: 1000 timed after 100 warm-up ones, NPPDx and FKL one after the other in
each, and FKL's mean speedup computed per iteration (NPPDx time / FKL time) and averaged, as in FKL's and
cvGPUSpeedup's benchmarks.

**CPU + GPU**: CUDA events around everything the host does to enqueue the kernel plus its execution. Median
latency in microseconds:

| N | NPPDx (us) | FKL (us) | Speedup |
|---|---|---|---|
| 1 | 19.4 | 3.1 | 5.9 |
| 4 | 18.4 | 3.4 | 5.3 |
| 16 | 19.1 | 4.1 | 4.4 |
| 50 | 34.6 | 7.9 | 4.5 |
| 100 | 61.1 | 11.4 | 5.3 |

**GPU only**: the GPU execution time of each kernel, from CUPTI kernel tracing (no host work). Median in
microseconds:

| N | NPPDx (us) | FKL (us) | Speedup |
|---|---|---|---|
| 1 | 16.4 | 1.38 | 12.0 |
| 4 | 16.5 | 1.47 | 11.3 |
| 16 | 17.2 | 1.60 | 10.7 |
| 50 | 32.8 | 2.66 | 12.4 |
| 100 | 58.9 | 3.78 | 15.5 |

**GPU only, L2 flushed**: like GPU only, but the L2 cache is flushed (a kernel reads twice the L2 size) before every
execution, as on a GPU that runs other kernels in between, so that the input cannot be assumed to be in L2. Run with
`benchmark_figure1_bvf <dir> --gpu-only-l2-flushed` (NPPDx with 256 threads per block). Median in microseconds:

| N | NPPDx (us) | FKL (us) | Speedup |
|---|---|---|---|
| 1 | 16.9 | 1.92 | 8.9 |
| 4 | 17.1 | 2.21 | 7.7 |
| 16 | 17.7 | 2.43 | 7.3 |
| 50 | 34.3 | 3.46 | 9.9 |
| 100 | 62.8 | 4.51 | 13.9 |

Most of the difference between the two benchmarks for FKL is host work: building and back-fusing the chain of
operations for N crops and launching a kernel with their parameters (about 1.7 us at N = 1 and 7 us at N = 100).

The CSV files and the platform description are in [`results/RTX4090`](results/RTX4090): NVIDIA GeForce RTX 4090,
driver 610.43.02, CUDA 13.3, GCC 15.2, AMD Ryzen 9 5950X, Ubuntu 26.04.1, FKL `b9f4281`, NPPDx v0.1.2.
GPU clocks were not locked.

**DRAM traffic** (Nsight Compute, `dram__bytes_read.sum` and `dram__bytes_write.sum` of one execution, see
`benchmark_figure1_bvf_dram_traffic_GPU_NVIDIA.csv`). With the L2 cache flushed before the kernel, NPPDx reads 77,
91 and 154 KB from DRAM for N = 1, 16 and 100, and FKL 9, 22 and 84 KB. With a warm cache both read nothing from
DRAM, and neither writes to DRAM while the kernel runs: the RTX 4090's 72 MB L2 is write-back and holds the
whole working set (the 12.4 MB NV12 frame and an output of at most 4.9 MB). Collected with
`--ncu-target <nppdx|fkl_bvf> <N>`, which runs one execution inside the NVTX range `measure`.

## Correctness and sampling conventions

Every run checks both outputs before timing them:

- FKL is compared with FKL's CPU backend running the same chain: max absolute error below 2e-6.
- NPPDx is compared with the same chain using pixel-center sampling in the resize: max absolute error below 2e-6.

The two libraries do not compute the same bilinear resize. FKL (like OpenCV's CUDA `resize`) samples the source at
`dst * scale`, while NPPDx samples at pixel centers, `(dst + 0.5) * scale - 0.5`, clamped to the image. Against
FKL's reference the NPPDx output differs by up to 5.4 in normalized units for this random content. Both
libraries replicate the NV12 chroma and neither clamps the RGB values after the conversion. Run
`benchmark_figure1_bvf <dir> --analyze` to see the comparison against each convention.

The crop positions are snapped to even coordinates: NPPDx takes the chroma of a crop at `(x / 2, y / 2)` relative
to the crop origin, and FKL relative to the frame, which only read the same samples for even origins.

## NPPDx configuration

The NPPDx kernel and its configuration (`NppdxFigure1`) are in `benchmarks/nppdx_figure1.h`.

NPPDx tiles must be multiples of its 4x2 pixel cells and of the 5:16 resize ratio, so the only valid tile for a
20x10 crop is the whole crop, one thread block per crop. All the operations run in the same kernel, which is
launched with the largest of their block sizes (the ingest's own recommended block covers the tile without the
resize halo, and would leave rows of the input tile unloaded). The benchmark tries NPPDx's recommended
configuration and `BlockDim` 64, 128, 256 and 512 for every N, checks each one, and compares FKL with the fastest
correct one (256 or 512 threads, which are within noise of each other). The sweep is in
`benchmark_figure1_bvf_nppdx_block_sweep_GPU_NVIDIA.csv`.

## Region size sweep

`benchmark_figure1_bvf <dir> --region-sweep` runs the same chain with N = 16 regions of growing size, each resized
exactly 2x (NPPDx ratio 1:2) and given the 16 black rows on top and bottom:

| Region | Source | Resized | Output | 16 regions cover |
|---|---|---|---|---|
| A | 32x16 | 64x32 | 64x64 | 0.0988% of the frame |
| B | 112x96 | 224x192 | 224x224 | 2.0741% |
| C | 320x304 | 640x608 | 640x640 | 18.7654% |
| D | 960x524 | 1920x1048 | 1920x1080 | 97.0370% |

The regions sit at the origins of a 4x4 grid of 960x540 cells (region i = row * 4 + column), so they never overlap.
Only a 32x16 region fits in one NPPDx tile: for the larger ones NPPDx rejects a whole-region tile (its output tile
does not fit in the shared memory of a block), so every region is split in tiles, several thread blocks per region.
For every size the benchmark checks three tiles with NPPDx's recommended block and with `BlockDim` 128 and 256,
times the correct ones alone and compares FKL with the fastest. A `BlockDim` smaller than the recommended block of
a tile gives wrong results and is never chosen. As in the main benchmark there are three comparisons: CPU + GPU,
GPU only and GPU only with the L2 flushed. The files start with `benchmark_figure1_bvf_region_sweep`, with
`Source pixels per region` as the first column, plus `_validation_`, `_nppdx_configs_` (every tile and block tried),
`_summary_` (sizes, coverage, chosen NPPDx configuration, kernels per execution and bytes written) and
`_platform.txt`.

`--region-sweep-validate` only checks the results, without timing. `--region-ncu-target <nppdx|fkl_bvf> <A|B|C|D>
[<tileX>x<tileY> <block>]` runs one execution inside the NVTX range `measure` for Nsight Compute; NPPDx needs the
tile and block, for example the ones chosen in the summary file (block 0 is NPPDx's recommended one).

## Build and run

Requirements: Linux x86_64 or Windows x64 (the platforms of NPPDx), CUDA 13.3 or newer, CMake 3.30, a C++20 host compiler, and
an NVIDIA GPU (sm_75 or newer). FKL and NPPDx are the `fkl` and `nppdx` submodules of this repository. From its root:

    git clone --recursive https://github.com/Libraries-Openly-Fused/bvf-paper-bench.git
    cd bvf-paper-bench
    cmake -S fkl-vs-nppdx -B build-nppdx -DCMAKE_BUILD_TYPE=Release
    cmake --build build-nppdx
    ./build-nppdx/benchmark_figure1_bvf results/my_gpu     # or: ctest --test-dir build-nppdx (writes to build-nppdx/results)

`CUDA_ARCH` (default `native`) selects the CUDA architectures, and `NPPDX_SM` (detected with `nvidia-smi`) the
SM version the NPPDx operations are specialized for, for example `-DNPPDX_SM=890` for compute capability 8.9.
`FKL_DIR` and `NPPDX_DIR` select other FKL and NPPDx checkouts.

## Output files

Following FKL's benchmark convention, each benchmark writes `<benchmark>_<ParArch>.csv` with one row per batch
size: `Batch size`, then `MeanTime`, `TimeVariance`, `MaxTime` and `MinTime` of the first (baseline, NPPDx) and
second (FKL) implementation in milliseconds, `Mean Speedup`, and the `MedianTime`, `P10Time` and `P90Time` of
each. In every iteration both implementations are timed one after the other with CUDA events on one stream,
synchronizing after each. `<benchmark>_gpu_only_<ParArch>.csv` has the same columns for the GPU only benchmark.
Next to them are `_validation_` (errors), `_nppdx_block_sweep_` and `_platform.txt`.

## License

Apache License 2.0, like FKL and NPPDx (see [LICENSE](../LICENSE)).
