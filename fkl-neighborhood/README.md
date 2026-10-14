# FKL neighborhood experiment

Part of the [BVF paper benchmarks](../README.md): the experiment on neighborhood operations (Experiment 3).

## Pipeline

The [workload](../README.md#workload) of every benchmark of this repository, with a morphology stage after the
resize and without the letterbox of Figure 1:

    ReadNV12 -> YUVtoRGB -> Crop -> Resize -> Morph (K x K) -> Mul -> Sub -> Div -> Write

- **Output:** N planes of 64x32 float3, without the black rows of Figure 1.
- **Morphology:** an erosion (one pass of minimum) or an opening (a minimum, then a maximum).
  - The mask is a K x K rectangle anchored at its center.
  - Every pass replicates the border of the 64x32 plane it reads, including the second pass of the opening, which reads
    the result of the first.
  - The minimum and the maximum are per channel.
- **Shared with `rocm/`:** these definitions are in [`common/bench_neighborhood.h`](../common/bench_neighborhood.h), with
  the sizes of both parts and the host reference.

## FKL's two ways to write a neighborhood operation

**As BVF operations.** [`common/bench_neighborhood_ops.h`](../common/bench_neighborhood_ops.h) has `ErodeBVF<K>` and
`DilateBVF<K>`:
- They are ReadBack operations, like `Crop` and `Resize`, so the whole pipeline is one `executeOperations` call.
- Every output value evaluates the chain before the morphology K^2 times per pass, so K^4 times for the opening.
- Every evaluation of the bilinear resize reads four values of the color conversion.
- That is 4 K^2 (erosion) or 4 K^4 (opening) NV12 reads and conversions per output value.

**With FKL's `MorphologyDPP`.** A Data Parallel Pattern runs every pass in one kernel:
- **Batching:** one grid z plane per region (FusedKernelLibrary PR #393).
- **Input:** the DPP reads the regions through the BVF chain `ReadNV12 -> YUVtoRGB -> Crop -> Resize`, fused into its
  reads.
- **Tiling:** it stages a tile of TW x TH output values, plus the halo of its passes, in shared memory, so every resized
  value of the tile and halo is computed once.
- **Output:** it writes through `Mul -> Sub -> Div -> TensorWrite`, fused into its write. No intermediate is in memory.
- **Tile choice per batch size:** the program compiles the DPP for every tile of `DPP_TILES` (16x8, 32x8, 32x16). It
  validates each one and times it alone (latency, on the regions of every iteration), then uses the fastest correct one
  for that batch size.
  - The main run writes the choice to `fkl_dpp_tiles.csv`, and the timings and errors of every tile to
    `fkl_dpp_tile_sweep.csv`.
  - The other modes read `fkl_dpp_tiles.csv`, so run the main benchmark first.

## Programs

### Part (a): the BVF operations against the DPP, FKL only

`benchmark_fkl_neighborhood_k<K>_p<passes>` is built for every K in {3, 5, 7} and every number of passes: 1 (erode) or
2 (open). It runs N = 1 and 16.

| Implementation | What runs |
|---|---|
| `fkl_bvf_ops` | one `executeOperations` call, one kernel: `ReadYUV -> YUVToRGB -> Crop -> Resize -> ErodeBVF<K> [-> DilateBVF<K>] -> Mul -> Sub -> Div -> TensorWrite` |
| `fkl_dpp` | one `executeErode` / `executeOpen` call, one kernel: `MorphologyDPP<float3, TW, TH, K, K>` |

### Part (b): the DPP against the libraries

These programs use K = 5 and every batch size. There is one program per operation, `_erode` and `_open`.

| Program | Implementation | What runs | Kernels (erode / open) |
|---|---|---|---|
| `benchmark_npp_neighborhood_*` | `npp` | `fkl-vs-npp`'s pipeline up to its batched 32f resize (NV12->RGB of the frame, per region `nppiConvert_8u32f_C3R`, one `nppiResizeBatch_32f_C3R_Advanced` into 64x32 float3 planes); then per region `nppiFilterMinBorder_32f_C3R` (erode), or `nppiFilterMinBorder` into a plane buffer then `nppiFilterMaxBorder_32f_C3R` (open), with `oSrcSize` the plane and `NPP_BORDER_REPLICATE`, into its output plane; then `nppiMulC`, `nppiSubC`, `nppiDivC` on the whole output | 2N + 5 / 3N + 5 |
| | `npp_morph` | the same with `nppiErodeBorder_32f_C3R` (K x K mask of ones) / `nppiMorphOpenBorder_32f_C3R` (two kernels per call, with its scratch buffer) | 2N + 5 / 3N + 5 |
| `benchmark_opencv_cuda_neighborhood_*` | `opencv_cuda` | OpenCV CUDA's morphology filters take no 3 channel image, so `fkl-vs-opencv-cuda`'s pipeline works on 4 channels: NPP NV12->RGB of the frame, `cv::cuda::cvtColor` RGB2RGBA of the frame, then per region `convertTo` 32FC4, `resize`, and the `createMorphologyFilter` (`MORPH_ERODE` / `MORPH_OPEN`, K x K ones) filter into its plane; then `multiply`, `subtract`, `divide` on the 4 channel output and one `cvtColor` RGBA2RGB into the float3 output | 4N + 6 / 6N + 6 |
| | `opencv_cuda_rgba_per_region` | the same with `cvtColor` RGB2RGBA per region instead of on the frame | 5N + 5 / 7N + 5 |
| `benchmark_nppdx_npp_neighborhood_*` | `nppdx_npp` | without hand written pixel processing code: `fkl-vs-nppdx`'s NPPDx kernel of NPPDx+NPP (NV12 ingest, ColorConvert, Resize, 8 bit RGB write; one launch for the N regions), then NPP, since NPPDx 0.1.2 has no morphology: per region `nppiErodeBorder_8u_C3R` (erode), or ErodeBorder into a plane buffer then `nppiDilateBorder_8u_C3R` (open); then `nppiConvert_8u32f_C3R`, `nppiMulC`, `nppiSubC`, `nppiDivC` on all planes. Every NPPDx tile and block is validated and its kernel timed alone, and the fastest correct one is used (`nppdx_npp_configs.csv`, `nppdx_npp_config_sweep.csv`). Built for one region size, like `fkl-vs-nppdx` | N + 5 / 2N + 5 |
| every program | `fkl_dpp` | one kernel (see above) | 1 / 1 |

### Notes on the libraries

- **The morphology functions take two kernels per call**, so the L2 flush cannot sit between them:
  - OpenCV's `apply` copies the plane with `copyMakeBorder` (`BORDER_REFLECT_101`), then runs an NPP kernel, for every
    pass.
  - NPP's `nppiMorphOpenBorder` runs an erosion and a dilation kernel.
  - With the L2 flushed, the flush runs before every library call, not between the kernels of one call.
- **The border semantics agree.** For a centered rectangular mask, `BORDER_REFLECT_101` gives the same minimum and
  maximum as replicating the border.
- **8-bit NPP morphology:** NPP's 8 bit `nppiFilterMin/MaxBorder` take an extra path (three kernels) when the
  destination is not 12-byte aligned. The NPPDx+NPP programs use `nppiErodeBorder` / `nppiDilateBorder`, which have no
  such path.

## Validation

Every implementation is checked against the reference of its conventions:
- **The base:** the Figure 1 references of [`common/bench_gpu.h`](../common/bench_gpu.h) without the letterbox,
  normalized. For NPPDx+NPP, fkl-vs-nppdx's pixel-center 8 bit one.
- **Then the morphology:** `hostMorph` of `common/bench_neighborhood.h`, applied on the host after the normalization.
  This is exact, because the normalization is non-decreasing in every channel and so commutes with the minimum and the
  maximum.

The tolerances are those of every GPU benchmark: 1e-3 for FKL, one level of 8 bit RGB for the 8 bit pipelines.

`test_neighborhood_ops` checks `ErodeBVF` and `DilateBVF` against the host reference, on FKL's CPU backend and on the
GPU, for every K and number of passes:
- on a plane read directly: bit-identical;
- on a batch of three regions of different sizes, cropped and resized: within 1e-5, relative.

## Build and run

Needs FusedKernelLibrary with the batched `MorphologyDPP` (PR #393; the `fkl` submodule points to it). With `OpenCV_DIR`
for the OpenCV programs (or `-DBENCH_OPENCV=OFF`):

    cmake -S fkl-neighborhood -B build-neighborhood -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<OpenCV>
    cmake --build build-neighborhood -j
    ./build-neighborhood/test_neighborhood_ops
    ./build-neighborhood/benchmark_npp_neighborhood_open ../results/fkl-neighborhood/npp_neighborhood_open
    ./build-neighborhood/benchmark_npp_neighborhood_open ../results/fkl-neighborhood/npp_neighborhood_open \
        --gpu-only-l2-flushed

The NPPDx+NPP programs are built for one region size. With `BENCH_NPPDX=ON`, this folder builds only them:

    cmake -S fkl-neighborhood -B build-neighborhood-nppdx-32x16 -DCMAKE_BUILD_TYPE=Release -DBENCH_NPPDX=ON \
        -DBENCH_REGIONS=32x16
    cmake --build build-neighborhood-nppdx-32x16 -j
    ./build-neighborhood-nppdx-32x16/benchmark_nppdx_npp_neighborhood_open \
        ../results/fkl-neighborhood/nppdx_npp_neighborhood_open/32x16

`scripts/run_all.sh` (step `neighborhood`) builds and runs all of them, each program into
`<results>/fkl-neighborhood/<program>`:
- the NPPDx+NPP ones into `<program>/<size>`, for every size of `NPPDX_SIZES`;
- the main run first, then the L2-flushed one.

The timings, modes, output files and validation rules are those of every GPU benchmark on
[`common/bench_gpu.h`](../common/bench_gpu.h), described in the [main README](../README.md#the-gpu-benchmarks). The
FKL implementation is named `fkl_dpp` in the output files (`<baseline>_vs_fkl_dpp_*.csv`).
