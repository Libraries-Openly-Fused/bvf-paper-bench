# FKL neighborhood experiment

Part of the [BVF paper benchmarks](../README.md): the experiment on neighborhood operations (Experiment 3).

## Pipeline

Fixed-size crops of the [frame](../README.md#workload) of every benchmark of this repository, blurred and normalized:

    ReadNV12 -> YUVtoRGB -> Crop (64x32) -> Box (K x K) -> Mul -> Sub -> Div -> Write

- **Why this pipeline:** it is one that both FKL and NPPDx fuse into a single kernel.
  - FKL's `MorphologyDPP` reads through a BVF chain that ends at the crop and writes through the pointwise tail.
  - NPPDx has the box (`BoxBlur`) between its color conversion and its write.
  - A resize between the crop and the box would break both: NPPDx's `Resize` writes no halo, so no area step can
    follow it in one kernel, and a DPP's output must be a write, so it cannot feed a resize.
- **Crops:** N crops of 64x32 per iteration, at uniform random even positions at least 8 pixels from the frame edges, new
  every iteration and the same for every implementation, program and machine (`cropRegionSets`: `RegionSets` of
  [`common/bench_geometry.h`](../common/bench_geometry.h) with a fixed size, from the same seed). No resize and no added
  rows.
- **Output:** N planes of 64x32 float3.
- **Box:** the mean of the K x K window centered on the value, per channel, replicating the edge of the crop. With two
  passes, the second one blurs the result of the first, again replicating the edge of the crop.
- **Shared with `rocm/`:** these definitions are in [`common/bench_neighborhood.h`](../common/bench_neighborhood.h), with
  the sizes of both parts and the host references.

## FKL's two ways to write a neighborhood operation

**As a BVF operation.** [`common/bench_neighborhood_ops.h`](../common/bench_neighborhood_ops.h) has `BoxBVF<K>`:
- It is a ReadBack operation, like `Crop` and `Resize`, so the whole pipeline is one `executeOperations` call.
- Every output value evaluates the chain before the box (an NV12 read and a color conversion) K^2 times per pass, so K^4
  times for two passes.

**With FKL's `MorphologyDPP`.** `executeBoxBlur` (FusedKernelLibrary PR #393) runs the box as a morphology: every pass
sums its window with an `Add` reducer and multiplies the sum by 1 / K^2 (the pass's finalizer), so every pass is a mean.
One kernel:
- **Batching:** one grid z plane per crop (PR #393).
- **Input:** the DPP reads the crops through the BVF chain `ReadNV12 -> YUVtoRGB -> Crop`, fused into its reads.
- **Tiling:** it stages a tile of TW x TH output values, plus the halo of its passes, in shared memory, so every crop
  value of the tile and halo is computed once.
- **Output:** it writes through `Mul -> Sub -> Div -> TensorWrite`, fused into its write. No intermediate is in memory.
- **Tile choice per batch size:** the program compiles the DPP for every tile of `DPP_TILES` (16x8, 32x8, 32x16). It
  validates each one and times it alone (latency, on the regions of every iteration), then uses the fastest correct one
  for that batch size.
  - The main run writes the choice to `fkl_dpp_tiles.csv`, and the timings and errors of every tile to
    `fkl_dpp_tile_sweep.csv`.
  - The other modes read `fkl_dpp_tiles.csv`, so run the main benchmark first.

## Programs

### Part (a): the BVF operation against the DPP, FKL only

`benchmark_fkl_neighborhood_k<K>_p<passes>` is built for every K in {3, 5, 7} and 1 or 2 passes. It runs N = 1 and 16.
`benchmark_fkl_neighborhood_k<K>_p<passes>_resize` is the same with the box after Figure 1's resize: the regions of
[`common/bench_geometry.h`](../common/bench_geometry.h) (20x10, 32x16 and 48x24 by turns), cropped and resized to 64x32,
then the box, replicating the edge of the resized region.

| Implementation | What runs | Chain evaluations per output value |
|---|---|---|
| `fkl_bvf_ops` | one `executeOperations` call, one kernel: `ReadYUV -> YUVToRGB -> Crop [-> Resize] -> BoxBVF<K> [-> BoxBVF<K>] -> Mul -> Sub -> Div -> TensorWrite` | K^2 or K^4 NV12 reads and conversions; with the resize 4 K^2 or 4 K^4 (four per bilinear sample) |
| `fkl_dpp` | one `executeBoxBlur<passes>` call, one kernel: `MorphologyDPP<float3, TW, TH, K, K>` | one per value of a tile and its halo |

### Part (b): the DPP against the libraries

These programs use a 5x5 box, one pass, and every batch size.

| Program | Implementation | What runs | Kernels |
|---|---|---|---|
| `benchmark_npp_neighborhood` | `npp` | NV12->RGB of the frame (8 bit); per crop `nppiFilterBoxBorder_8u_C3R` from the crop of the frame (`oSrcSize` the crop, `NPP_BORDER_REPLICATE`) into its packed 8 bit plane, which rounds down; one `nppiConvert_8u32f_C3R` of the N planes; `nppiMulC`, `nppiSubC`, `nppiDivC` on the whole output | N + 5 |
| | `npp_32f` | NV12->RGB; per crop `nppiConvert_8u32f_C3R` of the crop, then `nppiFilterBoxBorder_32f_C3R` into its output plane; the three passes | 2N + 4 |
| | `npp_32f_frame` | NV12->RGB; one `nppiConvert_8u32f_C3R` of the whole frame; per crop `nppiFilterBoxBorder_32f_C3R` from the crop of the float frame into its output plane; the three passes | N + 5 |
| `benchmark_opencv_cuda_neighborhood` | `opencv_cuda` | NPP NV12->RGB of the frame (OpenCV CUDA has none); per crop `convertTo` 32FC3 of the crop, then `createSeparableLinearFilter` (`CV_32FC3`, 5 taps of 1/5 for the rows and the columns, `BORDER_REPLICATE`; a row and a column kernel) into its plane; `multiply`, `subtract`, `divide` on the float3 output | 3N + 4 |
| | `opencv_cuda_rgba` | NPP NV12->RGB; `cvtColor` RGB2RGBA of the frame; per crop `convertTo` 32FC4, then `createLinearFilter` (`CV_32FC4`, 5x5 taps of 1/25, `BORDER_REPLICATE`; one kernel) into its plane of a 4 channel output; `multiply`, `subtract`, `divide` on it; `cvtColor` RGBA2RGB into the float3 output | 2N + 6 |
| `benchmark_nppdx_neighborhood` | `nppdx` | one kernel ([`src/nppdx_box.h`](src/nppdx_box.h)): NPPDx's NV12 ingest of the crop (the crop is the image, so the halo replicates its edge), `FromLimitedLuma` / `FromLimitedChroma`, `ColorConvert` BT.601 and `BoxBlur<5, 5>`, then the normalization and the float3 write written by hand, as `fkl-vs-nppdx`'s NPPDx kernel (NPPDx has no float output); one launch for the N crops | 1 |
| every program | `fkl_dpp` | one kernel (see above) | 1 |

**NPPDx configurations:** every tile of the crop that NPPDx supports (64x32, then halvings of
the longer side down to 8x8, in 4x2 pixel cells) runs with NPPDx's recommended block and with `BlockDim` 64, 128, 256
and 512. Each one is validated and its kernel timed alone for every batch size, and the fastest correct one is used
(`nppdx_configs.csv`, `nppdx_config_sweep.csv`). NPPDx's kernel fuses the whole pipeline, so there is no NPPDx+NPP
program in this experiment.

### Notes on the libraries

- **NPP has no batched box filter**, so it makes one call per crop. `nppiFilterBoxBorder` accepts only
  `NPP_BORDER_REPLICATE`, and is one kernel per call at any offset, 8 bit and float. The non-border `nppiFilterBox`
  reads the pixels around its ROI, so it is not used.
- **OpenCV CUDA's `createBoxFilter` takes 8UC1, 8UC4 and 32FC1 only**, so the box is a linear filter: the separable one
  takes 3 channels, the 2D one 1 or 4. OpenCV's filters build their border from the image they get (a view of the
  crop), so `BORDER_REPLICATE` replicates the edge of the crop.
- **Multi-kernel calls:** the separable filter runs a row and a column kernel in one call. With the L2 flushed, the flush
  runs before every library call, not between the kernels of one call.

## Validation

Every reference blurs the RGB values and then normalizes them, as every pipeline does: a box does not commute exactly
with the normalization, which rounds. The crops come from FKL's CPU backend (`ReadNV12 -> YUVtoRGB -> Crop`). The box
is computed in double and rounded as the implementation rounds it (`boxReference` of
[`common/bench_neighborhood.h`](../common/bench_neighborhood.h)).

| Reference | Crops | Box | Implementations |
|---|---|---|---|
| `fkl_cpu_float` | float (with the resize for the `_resize` programs) | float | `fkl_dpp`, `fkl_bvf_ops`, `nppdx` |
| `fkl_cpu_8bit_rgb` | the RGB rounded to 8 bits after the conversion (the 8 bit frame of NPP's conversion) | float | `npp_32f`, `npp_32f_frame`, `opencv_cuda`, `opencv_cuda_rgba` |
| `fkl_cpu_8bit_rgb_floor_box` | the same 8 bit RGB | rounded down, floor(sum / 25) | `npp` |

The tolerances are those of every GPU benchmark: 1e-3 against the float reference, and one level of 8 bit RGB against
the others.

`test_neighborhood_ops` checks `BoxBVF` and `MorphologyDPP`'s box blur against the host box for every K and number of
passes:
- `BoxBVF` on FKL's CPU backend and on the GPU;
- `executeBoxBlur` on the GPU, and `MorphologyDPP<ParArch::CPU>` with its passes;
- on a plane read directly and on a batch of 64x32 crops of a larger image, one at its corner;
- within 1e-5, relative.

## Build and run

Needs FusedKernelLibrary with the batched `MorphologyDPP` and `executeBoxBlur` (PR #393; the `fkl` submodule points to
it), and NPPDx (the `nppdx` submodule; CMake 3.30 or newer, or `-DBENCH_NPPDX=OFF`). With `OpenCV_DIR` for the OpenCV
program (or `-DBENCH_OPENCV=OFF`):

    cmake -S fkl-neighborhood -B build-neighborhood -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<OpenCV>
    cmake --build build-neighborhood -j
    ./build-neighborhood/test_neighborhood_ops
    ./build-neighborhood/benchmark_npp_neighborhood ../results/fkl-neighborhood/npp_neighborhood
    ./build-neighborhood/benchmark_npp_neighborhood ../results/fkl-neighborhood/npp_neighborhood --gpu-only-l2-flushed

`scripts/run_all.sh` (step `neighborhood`) builds and runs all of them, each program into
`<results>/fkl-neighborhood/<program>`: the main run first, then the L2-flushed one.

The timings, modes, output files and validation rules are those of every GPU benchmark on
[`common/bench_gpu.h`](../common/bench_gpu.h), described in the [main README](../README.md#the-gpu-benchmarks). The
FKL implementation is named `fkl_dpp` in the output files (`<baseline>_vs_fkl_dpp_*.csv`).
