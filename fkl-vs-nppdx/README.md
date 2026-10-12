# FKL vs NPPDx

Part of the [BVF paper benchmarks](../README.md) (until October 2026 the separate repository FKL-vs-NPPDx).
Benchmarks comparing the [Fused Kernel Library (FKL)](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary)
and NVIDIA [NPPDx](https://github.com/NVIDIA/NPPDx) on the image preprocessing pipeline of the Backwards Vertical
Fusion paper (Figure 1):

    ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write

One 3840x2160 NV12 frame in device memory, N crops of 32x16 pixels at fixed random positions, NV12 to RGB
(BT.601, limited range), bilinear resize to 64x32, 16 black rows added on top and bottom (64x64), then
`(x * mul - sub) / div` with ImageNet constants (`mul = 1/255`, `sub = (0.485, 0.456, 0.406)`,
`div = (0.229, 0.224, 0.225)`), written as float3 into a tensor of N planes. N is 1, 4, 16, 50 and 100. The frame,
sizes, batch sizes and crop positions are those of every benchmark of this repository, from
[`common/bench_geometry.h`](../common/bench_geometry.h) (see the [workload](../README.md#workload)).

NPPDx is a device-side library: it has no host API and no kernels of its own, so using it always means writing a
CUDA kernel that calls its operations (all of NPPDx's examples do). NPPDx 0.1.2 has no crop, border or type
conversion operation, its affine channel map only takes integer constants, and every output format it writes is
integer (8, 10 or 16 bit, rounded and clamped): the normalized float values of this pipeline cannot leave an NPPDx
kernel through NPPDx. So there are two ways to write the pipeline with NPPDx, and two benchmarks, each comparing it with
the same FKL pipeline:

- **FKL**: one `executeOperations` call with the nine operations (Backwards Vertical Fusion), from the `fkl`
  submodule of this repository (FKL main, which has `AddBorder`). One kernel.
- **NPPDx** (`benchmark_figure1_bvf`, [`benchmarks/nppdx_figure1.h`](benchmarks/nppdx_figure1.h)): one CUDA kernel
  built from NPPDx operations, one thread block per tile of a crop: NV12 ingest of the tile, `FromLimitedLuma` and
  `FromLimitedChroma` (NPPDx's color conversion is full range), `ColorConvert<yuv_bt601, rgb>` and
  `Resize<1, 2, bilinear>` (the J:K ratio of the geometry) on the shared memory tile. AddBorder, Mul, Sub, Div and the
  float3 write are written by hand in the same kernel. One kernel, with hand written CUDA code.
- **NPPDx+NPP** (`benchmark_figure1_nppdx_npp`, [`benchmarks/nppdx_npp_figure1.h`](benchmarks/nppdx_npp_figure1.h)):
  only what NPPDx and NPP provide, with no hand written pixel processing code. NPPDx has no host API, so its operations
  still run in a CUDA kernel written for them, with its launch configuration and the crop and batch given to NPPDx's
  ingest as image pointers, but the kernel only calls NPPDx operations, in the order of the pipeline (the same ones,
  then NPPDx's 8 bit RGB write of the resized tile). NPP does the rest with the function it offers for each step, as
  in the GPU benchmark's NPP pipeline: `nppiCopyConstBorder_8u_C3R` adds the black rows, once per crop (NPP has no
  batched border), then `nppiConvert_8u32f_C3R`, `nppiMulC_32f_C3IR`, `nppiSubC_32f_C3IR` and `nppiDivC_32f_C3IR` run
  on the N planes as one image. N + 5 kernels, with the intermediates in memory, and the resized RGB is rounded to 8
  bits before the normalization.

The two benchmarks share everything but the NPPDx pipeline ([`benchmarks/figure1_common.h`](benchmarks/figure1_common.h)
and [`benchmarks/nppdx_common.h`](benchmarks/nppdx_common.h)), and have the same modes and output files, with their
own names.

## Results on an RTX 4090

Three benchmarks, with the same iterations: 1000 timed after 100 warm-up ones, NPPDx and FKL one after the other in
each, and FKL's mean speedup computed per iteration (NPPDx time / FKL time) and averaged, as in FKL's and
cvGPUSpeedup's benchmarks. NPPDx runs in the configuration chosen for every N (see
[NPPDx configuration](#nppdx-configuration)).

**CPU + GPU**: CUDA events around everything the host does to enqueue the kernel plus its execution. Median
latency in microseconds:

| N | NPPDx (us) | FKL (us) | Speedup |
|---|---|---|---|
| 1 | 5.12 | 3.07 | 1.70 |
| 4 | 5.38 | 3.36 | 1.61 |
| 16 | 6.24 | 4.10 | 1.50 |
| 50 | 7.17 | 7.82 | 0.96 |
| 100 | 8.19 | 11.30 | 0.73 |

**GPU only**: the GPU execution time of each kernel, from CUPTI kernel tracing (no host work). Median in
microseconds:

| N | NPPDx (us) | FKL (us) | Speedup |
|---|---|---|---|
| 1 | 3.62 | 1.41 | 2.58 |
| 4 | 3.71 | 1.50 | 2.47 |
| 16 | 4.58 | 1.66 | 2.78 |
| 50 | 5.50 | 2.66 | 2.06 |
| 100 | 6.59 | 3.78 | 1.74 |

**GPU only, L2 flushed**: like GPU only, but the L2 cache is flushed (a kernel reads twice the L2 size) before every
kernel, as on a GPU that runs other kernels in between, so that the input cannot be assumed to be in L2 (for
NPPDx+NPP, its intermediates between its kernels either). Run with
`benchmark_figure1_bvf <dir> --gpu-only-l2-flushed` after the run without a mode into the same directory, whose NPPDx
configurations it uses. Median in microseconds:

| N | NPPDx (us) | FKL (us) | Speedup |
|---|---|---|---|
| 1 | 4.10 | 1.89 | 2.15 |
| 4 | 4.19 | 2.24 | 1.88 |
| 16 | 5.15 | 2.40 | 2.14 |
| 50 | 6.18 | 3.46 | 1.77 |
| 100 | 7.49 | 4.51 | 1.66 |

FKL's kernel is faster at every N, but end to end NPPDx is faster from N = 50: most of the difference between the
two benchmarks for FKL is host work, building and back-fusing the chain of operations for N crops and launching a
kernel with their parameters (about 1.7 us at N = 1 and 6.7 us at N = 100, `bvf_bench --host-overhead`), while
NPPDx launches a kernel configured at compile time.

The CSV files and the platform description are in [`results/RTX4090`](results/RTX4090): NVIDIA GeForce RTX 4090,
driver 610.43.02, CUDA 13.3, GCC 15.2, AMD Ryzen 9 5950X, Ubuntu 26.04.1, FKL `63782f6`, NPPDx v0.1.2.
GPU clocks were not locked. The NPPDx configuration chosen for every N was the same in two more runs of the
selection.

[`results/RTX4090-20x10`](results/RTX4090-20x10) has the same program built with `-DBENCH_CROP_W=20
-DBENCH_CROP_H=10 -DBENCH_RESIZE_W=64 -DBENCH_RESIZE_H=32` (20x10 crops resized 5:16 to 64x32, the paper's earlier
workload, with crop positions from the same generator), to show what NPPDx's compile-time resize ratio costs it: at
5:16 the only valid tile is the whole 20x10 crop, one thread block per crop.

**DRAM traffic** (Nsight Compute, `dram__bytes_read.sum` and `dram__bytes_write.sum` of one execution, see
`benchmark_figure1_bvf_dram_traffic_GPU_NVIDIA.csv`). With the L2 cache flushed before the kernel, NPPDx reads 35,
70 and 182 KB from DRAM for N = 1, 16 and 100, and FKL 10, 32 and 158 KB. With a warm cache both read nothing from
DRAM, and neither writes to DRAM while the kernel runs: the RTX 4090's 72 MB L2 is write-back and holds the
whole working set (the 12.4 MB NV12 frame and an output of at most 4.9 MB). Collected with
`--ncu-target <nppdx|fkl_bvf> <N> [<tileX>x<tileY> <block>]`, which runs one execution inside the NVTX range
`measure` (NPPDx in the given tile and requested block, here the ones of the configuration file).

## Correctness and sampling conventions

Every run checks both outputs before timing them, and writes the errors to `<benchmark>_validation_GPU_NVIDIA.csv`:

- FKL is compared with FKL's CPU backend running the same chain, with a tolerance of 1e-3.
- NPPDx is compared with the same chain using pixel-center sampling in the resize, with a tolerance of 1e-3.
- NPPDx+NPP is compared with the same chain using pixel-center sampling and the resized RGB rounded to 8 bits as
  NPPDx writes it (+ 0.5, clamped to [0, 255]). A value that falls on a half level can round the other way on the
  GPU, so the tolerance is one 8 bit level after the normalization plus 1e-3 (1 / 255 / 0.224 + 1e-3 = 0.0185); the
  file also counts the values more than 1e-3 away.

The two libraries do not compute the same bilinear resize. FKL (like OpenCV's CUDA `resize`) samples the source at
`dst * scale`, while NPPDx samples at pixel centers, `(dst + 0.5) * scale - 0.5`, clamped to the image. Against
FKL's reference the NPPDx output differs by several units in normalized units for this random content (column
`NPPDx MaxAbsError vs FKL reference` of the validation file). With the 1:2
resize, pixel centers are also what NPP's `nppiResize` computes (`dst * scale - 0.25`). Both
libraries replicate the NV12 chroma and neither clamps the RGB values after the conversion (NPPDx's 8 bit write
clamps them after the resize). Run `<benchmark> <dir> --analyze` to see the comparison against each convention.

The crop positions have even coordinates: NPPDx takes the chroma of a crop at `(x / 2, y / 2)` relative to the
crop origin, and FKL relative to the frame, which only read the same samples for even origins.

## NPPDx configuration

The NPPDx operations and their configurations are in `benchmarks/nppdx_common.h`, derived at compile time from the
geometry; each benchmark chooses its own configuration, the same way. NPPDx tiles are made of 4x2 pixel cells, and its resize maps whole groups of J source pixels, so a tile is a
multiple of lcm(4, J) x lcm(2, J) pixels that splits the crop. The candidate tiles are the whole crop and its
successive halvings (the longer side first) down to 64 pixels: 32x16, 16x16, 16x8 and 8x8 for 32x16 crops. Of those,
the benchmark keeps the ones NPPDx supports on the GPU's SM: `nppdx::is_supported_v` of the ingest and the resize
(their input tile fits in shared memory), the checks NPPDx's resize makes of its output and intermediate tiles, and
the shared memory of all the tiles of the kernel together (for NPPDx+NPP, also the 8 bit RGB write of the output
tile). A larger crop whose whole-crop tile does not fit, for example, is split in tiles instead of failing to compile.

All the operations run in the same kernel, which is launched with the largest of their block sizes (the ingest's own
recommended block covers the tile without the resize halo, and would leave rows of the input tile unloaded). Every
supported tile is tried with NPPDx's recommended configuration and with `BlockDim` 64, 128, 256 and 512, for every N:
the benchmark checks each one (the whole pipeline) against its reference, times its NPPDx kernel alone (for
NPPDx+NPP the NPP calls are the same in every configuration), and compares FKL with the fastest correct one (a `BlockDim` smaller than the recommended block of a tile gives wrong results, and is never chosen; a
configuration this GPU cannot launch is reported and skipped). The chosen configuration of every N drives all three
comparisons (CPU + GPU, GPU only and GPU only with the L2 flushed), the Nsight Compute scripts and the NPPDx
compile-time units (of the single NPPDx kernel). `<benchmark> <dir> --configs <csv>` runs the comparison with the
configurations of a CSV in the format of the configuration file instead of choosing them, for example to re-time other
candidates. Every configuration tried is in `<benchmark>_nppdx_block_sweep_GPU_NVIDIA.csv`, and the chosen ones in
`<benchmark>_nppdx_config_GPU_NVIDIA.csv`.

## Build and run

Requirements: Linux x86_64 or Windows x64 (the platforms of NPPDx), CUDA 13.3 or newer, CMake 3.30, a C++20 host compiler, and
an NVIDIA GPU (sm_75 or newer). FKL and NPPDx are the `fkl` and `nppdx` submodules of this repository. From its root:

    git clone --recursive https://github.com/Libraries-Openly-Fused/bvf-paper-bench.git
    cd bvf-paper-bench
    cmake -S fkl-vs-nppdx -B build-nppdx -DCMAKE_BUILD_TYPE=Release
    cmake --build build-nppdx
    ./build-nppdx/benchmark_figure1_bvf results/my_gpu/nppdx            # NPPDx, one kernel
    ./build-nppdx/benchmark_figure1_nppdx_npp results/my_gpu/nppdx_npp  # NPPDx+NPP
    # or: ctest --test-dir build-nppdx (both, writing to build-nppdx/results)

`CUDA_ARCH` (default `native`) selects the CUDA architectures, and `NPPDX_SM` (detected with `nvidia-smi`) the
SM version the NPPDx operations are specialized for, for example `-DNPPDX_SM=890` for compute capability 8.9.
`FKL_DIR` and `NPPDX_DIR` select other FKL and NPPDx checkouts.

## Output files

Following FKL's benchmark convention, each benchmark writes `<benchmark>_<ParArch>.csv` with one row per batch
size: `Batch size`, then `MeanTime`, `TimeVariance`, `MaxTime` and `MinTime` of the first (baseline, NPPDx or NPPDx+NPP) and
second (FKL) implementation in milliseconds, `Mean Speedup`, and the `MedianTime`, `P10Time` and `P90Time` of
each. In every iteration both implementations are timed one after the other with CUDA events on one stream,
synchronizing after each. `<benchmark>_gpu_only_<ParArch>.csv` has the same columns for the GPU only benchmark.
Next to them are `_validation_` (errors, tolerances and whether each output is within its tolerance), `_nppdx_block_sweep_` (every NPPDx configuration tried: tile, block,
blocks per crop, shared memory, time alone, error, whether it was chosen), `_nppdx_config_` (the chosen configuration
of every N: tile, requested and actual block, blocks per crop, shared memory) and `_platform.txt`.

## License

Apache License 2.0, like FKL and NPPDx (see [LICENSE](../LICENSE)).
