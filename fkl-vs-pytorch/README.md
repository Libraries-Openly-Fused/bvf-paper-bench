# FKL vs PyTorch

Part of the [BVF paper benchmarks](../README.md). This benchmark runs the Figure 1 pipeline in **eager PyTorch** and in
FKL with BVF, on the CPU and on the GPU, with the [workload](../README.md#workload) of every benchmark of this
repository. PyTorch is used through LibTorch's C++ API: there is no Python in the timed path, and no TorchScript,
`torch.compile` or CUDA graphs. One source builds against LibTorch for CUDA and for ROCm.

| Program | Compares | Harness |
|---|---|---|
| `benchmark_pytorch_cpu` | PyTorch's CPU kernels with FKL's CPU backends, on 1 thread and on as many threads as FKL's thread pool has workers | the CPU harness ([`common/bench_cpu.h`](../common/bench_cpu.h), [the CPU benchmarks](../README.md#the-cpu-benchmarks)) |
| `benchmark_pytorch_gpu` | PyTorch on the GPU (CUDA, or ROCm) with FKL BVF, one kernel | its own (`src/benchmark_pytorch_gpu.cpp`), with the protocol of the GPU benchmarks ([`common/bench_gpu.h`](../common/bench_gpu.h)) |

## PyTorch

The paper uses **torch 2.14.1**, the latest stable release (2026-09-30, git `5c48869`). Its CUDA 13 builds are cu130
(the default wheel on PyPI) and cu132. The paper uses **2.14.1+cu132**, the one closest to the CUDA 13.3 toolkit of the
paper's machine, from the index `https://download.pytorch.org/whl/cu132`:

    python3 -m venv venv-torch
    venv-torch/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132

For a CPU-only build, use the index `.../whl/cpu`; for ROCm, the wheel of the ROCm version. The programs record the
wheel's full version, CUDA or HIP version and git commit (`torch/version.py`) in their platform files. On the paper's
machine, the CUDA runtime library both libraries load is the system's (CUDA 13.3), which LibTorch's CMake links.

## The PyTorch pipeline

[`src/pytorch_pipeline.h`](src/pytorch_pipeline.h) is the same code on the CPU and on the GPU, under
`c10::InferenceMode`. The regions are host integers, so nothing waits for the device. Per iteration:

1. the 16 border rows on top and bottom of every plane of the output tensor are set to 0 (two `zero_` on strided views
   of all the planes);
2. for each region, with views of the region's luma and chroma planes (slices, no copy):
   - **chroma:** multiplied by FKL's chroma coefficients, once per chroma sample, with every offset of the conversion
     folded into the bias, in one of two formulations: `addmm` on the chroma converted to float (MKL on the CPU, a
     conversion and cuBLASLt's split-K GEMM on the GPU: three kernels), or two `addcmul` on the 8-bit U and V views
     (two kernels on the GPU);
   - **luma:** the 8-bit luma view added with its coefficient, each chroma sample broadcast to its 2x2 luma pixels
     (`add`, which converts the luma to float: in its kernel on the GPU, through a float copy that TensorIterator
     makes first on the CPU). Together this is float RGB, with FKL's BT.601 limited range matrix (taken from FKL), no
     clamping;
   - **resize:** bilinear resize to 64x32 (`torch::nn::functional::interpolate`, `align_corners=false`), on one of two
     views of the h x w x 3 crop: permuted then unsqueezed, which PyTorch's layout check treats as NCHW (the output is
     NCHW), or unsqueezed then permuted, which it treats as channels-last (the output is channels-last);
   - **copy:** into the 32 rows of its output plane between the borders: a transposing copy from NCHW, a straight one
     from channels-last;
3. `mul_`, `sub_` and `div_` in place on the whole N x 64 x 64 x 3 output tensor, FKL's layout.

**Batched, without a loop over the regions.** Step 2 can also run on all regions at once (`ResizeInput::Batched`,
`Pipeline::runBatched`):
- **parameters:** per region, the index of its window's first value in the luma and in the chroma plane and its grid
  coordinates, computed on the host and, on the GPU, uploaded in one copy from pinned memory;
- **gather:** the 48 x 24 window at every region's origin (the largest region size: the geometry draws every position
  for that box, so it is inside the frame) with one `take` per plane: N x 24 x 48 luma and N x 12 x 24 x 2 chroma;
- **conversion:** the chroma terms (`addmm` or two `addcmul`) and the broadcast `add` on the whole batch;
- **resize:** one `grid_sample` (bilinear, `align_corners=false`) on the N windows, with each region's pixel-center
  coordinates, `(dst + 0.5) * (in / out) - 0.5` clamped to the region's own first and last pixel, so that the
  window's pixels outside the region get no weight: `interpolate`'s sampling, to float rounding;
- **copy:** one transposing copy into the rows of all the output planes between the borders.

**Variants, chosen by a sweep.** The two formulations of the chroma terms, with the two views for the per-region resize
or batched, make six variants of PyTorch's pipeline. Which is fastest can depend on the device, the thread count and the
batch size. So before every comparison the program runs PyTorch's complete loop with each variant (on that thread count)
and compares FKL with the fastest by median, as FKL vs NPPDx chooses NPPDx's configuration: `pytorch_variant_sweep.csv`
has the sweep and the choice, and the validation checks every variant.

On the GPU, in the per-region variants, a region runs the chroma terms (two `addcmul` kernels, or `addmm`'s conversion
and two cuBLASLt kernels), the add and the upsample kernels, and the copy: a kernel from the NCHW view, a device to
device memcpy from the channels-last one (both contiguous). Each iteration adds 2 fills and the 3 element-wise kernels
(with `addcmul` and NCHW: 5 N + 5 kernels per iteration). The batched variants run the same operators once, whatever
N: the parameters' memcpy, 2 fills, 2 index additions and 2 `take` kernels, the chroma terms, the add, the grid's
stack, `grid_sample`, the copy and the 3 element-wise kernels (with `addcmul`: 15 kernels and 1 memcpy per
iteration; with `addmm` the conversion and cuBLASLt's GEMM, whose algorithm depends on N: 2 or 3 kernels, at some N a
memset as well). `kernels.csv` has the kernel, memcpy and memset counts of the variant of the comparison.

**Sampling convention.** `interpolate` samples at pixel centers, `src = (dst + 0.5) * (in / out) - 0.5` clamped at 0.
FKL samples at `src = dst * (in / out)` (fused_kernel's `resize.h`), and `align_corners=true` is a third convention.
`interpolate` has no option for FKL's, so the outputs differ by up to about 5 in normalized units. Each pipeline is
checked against a reference of its own convention ([`common/bench_reference.h`](../common/bench_reference.h): FKL's,
and pixel centers for PyTorch), and their difference is reported. The work is the same.

## What PyTorch is forced to do that FKL does not

- **A host loop over the regions, or windows of the largest size.** The regions have different sizes, and
  `interpolate` takes one input size per call. In the per-region variants every region is its own sequence of operator
  calls: 5 or 6 device operations per region on the GPU (by variant: 4 to 6 kernels, plus a device to device memcpy in
  the channels-last variants), and about as many operator calls on the CPU. The batched variants avoid the loop with
  one shape for all regions: they gather and convert the 48 x 24 window of every region (1152 pixels, against 200,
  512 or 1152 for the 20x10, 32x16 and 48x24 regions), resample it with a grid of N x 32 x 64 coordinate pairs, and
  upload the regions' parameters every iteration. FKL processes the N regions, each with its own size, in one call and
  one kernel.
- **No code for this GPU's operators in the wheel.** The cu132 wheel's build flags (`pytorch_build_information.txt`)
  give SASS for sm_75, sm_80, sm_86, sm_90, sm_100 and sm_120; its only sm_89 code is a set of CUTLASS FP8 GEMM kernels
  (`cuobjdump -arch sm_89` of `libtorch_cuda.so`). So PyTorch's own kernels in this pipeline run as sm_86 code on the
  RTX 4090 (cuBLASLt's GEMM in the `addmm` variants is cuBLAS's code). FKL's kernel is compiled for the GPU it runs
  on.
- **Intermediates in memory.** Per region: the chroma terms, the h x w x 3 float RGB crop (in the batched variants the
  48 x 24 window, after the gathered 8-bit windows) and the resized 3 x 32 x 64 image; in the batched variants also the
  grid; with `addmm` also the float chroma (on both devices), and on the CPU the float copies TensorIterator makes of
  the 8-bit inputs (the luma, and with `addcmul` the U and V views). The output is written, then rewritten three times
  by `mul_`, `sub_` and `div_`, plus the two border fills. FKL keeps everything in registers and writes each output
  value once.
- **Allocation and dispatch.** Every operator allocates its output (PyTorch's caching allocator on the GPU, its CPU
  allocator on the CPU), and goes through the dispatcher and TensorIterator's setup. FKL's `build()` calls and fusion
  are compile-time types plus a few parameters.
- **Kernel launches, one stream, no overlap (GPU).** Each iteration is about 5 N to 6 N dependent kernels and memcpys
  on one stream in the per-region variants, and about 15 in the batched ones (`kernels.csv`). The
  GPU time (`gpu_time.csv`) is a fraction of the iteration's time (`latency.csv`); the rest is host work and launch
  gaps.
- **Thread count is per operator (CPU).** PyTorch splits each of its own operators into chunks of at least its grain
  size (32768 elements), at most one per thread. On 32 threads, in the per-region variants, its per-region operators are
  one chunk each and run on one thread; only the border fills (from N = 16) and the last three operators (from N = 4)
  are split: at N = 50 into 5 chunks per fill and 19 per operator, at N = 100 into 10 and 32. The batched variants'
  operators work on all regions at once and are split the same way by their size. With `addmm`, MKL's GEMM decides its
  own threading (`at::set_num_threads` sets MKL's thread count too). FKL's thread pool splits the fused loop into one
  chunk per output row of each region.

## What PyTorch gets that FKL does not

- **Each source pixel converted once, and chroma once per chroma sample.** PyTorch converts each source pixel of the
  crop once, and computes the chroma terms once per chroma sample (a quarter of the pixels). FKL's backwards fusion
  converts the 4 source pixels of every output pixel: about 8192 conversions per region against at most 1152.
- **Vendor and vectorized kernels.** MKL for the conversion's GEMM on the CPU when `addmm` is chosen; PyTorch's
  vectorized CPU kernels (AVX2 on the paper's CPU, runtime dispatch); its tuned CUDA kernels.
- **The arithmetic folded, and the fastest variant per comparison.** The offsets of the conversion are folded into
  the bias, and the luma term into one broadcast add; PyTorch's pipeline takes the variant that was fastest in the
  sweep before each comparison, per-region or batched (FKL's pipeline is its generic operations).
- **Few operators for an eager pipeline.** Region views without copies, two fills per iteration for every border
  instead of a pad per region, and in-place tail operators.
- **Fast paths.** `InferenceMode` (no autograd bookkeeping) and the caching allocator, which serves the temporaries
  from its pool: `kernels.csv` records the device memory segments it allocated during each implementation's timed
  iterations (0: no `cudaMalloc` there).
- **Its own idle behaviour.** Each library runs its loop alone. On 32 threads PyTorch keeps libgomp's default spin:
  its workers, still spinning from its previous run, start every run warm.

## CPU program

`benchmark_pytorch_cpu`, on the harness of every CPU benchmark:

- **Sequential loops.** For every batch size, PyTorch's complete loop (100 warm-up and 1000 timed iterations, a 1 ms
  busy-waited idle gap before every run) runs first, then FKL's on the same regions. The two libraries never run at
  the same time.
- **Paired speedups.** The speedup is the mean over the timed iterations of (PyTorch time / FKL time) of the same
  iteration.
- **Variants.** Before each comparison, PyTorch's complete loop with each variant on that thread count; the comparison
  uses the fastest (`pytorch_variant_sweep.csv`).

| Comparison | Files |
|---|---|
| PyTorch on 1 thread (`at::set_num_threads(1)`) against FKL on one thread | `benchmark_cpu_pytorch_1thread_vs_fkl_bvf_CPU.csv` and its `_per_iteration.csv` |
| PyTorch on 32 threads (FKL's pool size) against FKL's thread pool | `benchmark_cpu_pytorch_32threads_vs_fkl_bvf_CPU_MULTITHREAD.csv` and its `_per_iteration.csv` |

It also writes `pytorch_variant_sweep.csv`, `cpu_validation.csv` (one row per validated iteration and variant),
`regions.csv`, `pytorch_build_information.txt` and `cpu_platform.txt`.
`cpu_platform.txt` records how long libgomp's and FKL's workers stay busy after a parallel run, measured at the start;
`benchmark_pytorch_cpu --omp-spin` prints only that. The validation checks:

- PyTorch on 1 thread against its pixel-center reference;
- PyTorch on 32 threads against PyTorch on 1 thread;
- FKL against its reference, and FKL's pool against FKL;
- PyTorch against FKL (the convention difference, reported only).

All but the last must be within 1e-3, or the program exits with an error after writing its files.

### Why the libraries are not interleaved

The first version of the harness alternated the two libraries within every iteration. With PyTorch on 32 threads that
cannot be fair to both. The official wheels run PyTorch's intra-op parallelism on OpenMP (libgomp), whose workers spin
for `GOMP_SPINCOUNT` iterations after a parallel region before they sleep: several milliseconds with the default 300000
on the paper's Ryzen 9 5950X (`cpu_platform.txt` records how long they stay busy per run), far longer than the 1 ms
idle gap and than FKL's 100 us. `GOMP_SPINCOUNT=4000` matches FKL's spin.

Measured interleaved on that machine in development runs (400 iterations; their files are not kept, and these are not
the paper's numbers; medians, us, N = 1 / 4 / 16 / 50 / 100):

| Interleaved | PyTorch on 32 threads | FKL CPU_MULTITHREAD in the same loop | FKL alone |
|---|---|---|---|
| A: `GOMP_SPINCOUNT=4000` (matched spin) | 196 / 907 / 1430 / 4073 / 9405 | 39 / 48 / 79 / 178 / 315 | 41 / 51 / 87 / 197 / 345 |
| B: default spin | 118 / 343 / 1143 / 3175 / 6182 | 54 / 166 / 256 / 884 / 1236 | 41 / 51 / 88 / 198 / 347 |
| C: default spin, 11 ms gap (about twice libgomp's spin as measured then) | 145 / 949 / 1786 / 3298 / 15736 | — | 39 / 85 / 238 / 492 / 596 |

- **B: PyTorch's workers disturb FKL.** PyTorch's spinning workers are still on the cores when FKL runs: FKL is up to
  4.5x slower than alone (N = 50).
- **A: matching the spin costs PyTorch.** With FKL's spin, PyTorch is 1.25 to 2.64x slower than with its default. One
  PyTorch run has long serial stretches (the per-region operators, below the grain size) between its parallel
  regions, and its workers fall asleep between them.
- **C: a longer gap slows both.**

Running each library's loop alone, with its own defaults, avoids all three. That is now the protocol of every CPU and
GPU benchmark of this repository, a rule of the paper's author.

### An observation on FKL's CPU backend

In development runs (their files are not kept, and they are not the paper's numbers), FKL's single-thread CPU time
depended on the binary it was linked into: in a binary with LibTorch, FKL was much slower than in fkl-vs-opencv-cpu's
when the timing thread slept through the gap, and some batch sizes stayed slower when it busy-waited. FKL's CPU loop
calls `InterpolateComplete::exec` out of line for every output pixel, one function shared by every batch size and
backend, and its speed depended on where that code landed (an observation without a proof: the paper's machine gives
no access to the performance counters). This led to the shared, separately compiled, page-aligned FKL code of every CPU
benchmark (common/bench_fkl_cpu.cpp); the [protocol change](../fkl-vs-opencv-cpu/README.md#protocol-change) of FKL vs
OpenCV on the CPU explains the change.

## GPU program

`benchmark_pytorch_gpu` follows the protocol of the GPU benchmarks:

- **One stream.** Both libraries run on one stream of PyTorch's pool, made current; FKL gets its handle.
- **Validation.** Before the timing, on the regions of the first, every 50th and the last timed iteration (1e-3):
  PyTorch against its pixel-center reference, FKL BVF against FKL's chain on its CPU backend. The program fails if
  either is out of tolerance.
- **Variants.** For every batch size, PyTorch's latency loop with each variant first; the comparison, and the GPU time
  pass, use the fastest (`pytorch_variant_sweep.csv`; `kernels.csv` names the one used).
- **Latency.** For every batch size, PyTorch's complete loop (100 warm-up and 1000 timed iterations) runs first, then
  FKL BVF's on the same regions, so the two never run at the same time. Every run is between two device events
  (`c10::Event`, all host enqueue work included) and followed by a synchronization. FKL's speedup is computed per
  iteration (identical inputs) and averaged.
- **GPU time only.** After every batch's latency (so that no latency is measured after a profiler session, as in
  `common/bench_gpu.h`), a second pass with the same loops, PyTorch's then FKL BVF's. The timed iterations are traced
  with PyTorch's profiler (Kineto: CUPTI on CUDA, the ROCm tracer on ROCm), in chunks of 100 iterations, with an empty
  marker kernel after every execution to split the trace. The GPU time of an execution is the sum of its device
  activities: kernels, and memcpys and memsets when there are any (PyTorch's copy into the output plane is a memcpy
  in the channels-last variants, a kernel otherwise; its fills are kernels). The
  span runs from the first start to the last end in the trace: it is measured under the profiler, so it includes
  the profiler's cost per launch and can exceed the latency.
- **Counts.** Kernels, memcpys and memsets of one execution (every traced execution must have the same counts, or the
  program fails), and the device memory segments PyTorch's caching allocator allocated during each implementation's
  timed latency iterations.
- **The GPU.** `platform.txt` describes the device the program ran on (CUDA device 0, the first of
  `CUDA_VISIBLE_DEVICES`), and asks nvidia-smi about that device only, by its PCI bus ID.

Output files: `latency.csv`, `gpu_time.csv`, `pytorch_vs_fkl_bvf_<GPU_NVIDIA|GPU_AMD>.csv` and
`pytorch_vs_fkl_bvf_gpu_only_<...>.csv` (FKL's two-execution format) with their `_per_iteration.csv`,
`pytorch_variant_sweep.csv`, `validation.csv`, `kernels.csv`, `regions.csv`, `platform.txt` and
`pytorch_build_information.txt`.

**Two compilations.** FKL's side ([`src/fkl_gpu.cu`](src/fkl_gpu.cu)) is compiled separately from PyTorch's, as
CUDA by nvcc or as HIP. It gets the nvcc options of every GPU benchmark ([`common/bench_gpu.cmake`](../common/bench_gpu.cmake)),
not the ones LibTorch's CMake adds. On the paper's machine, FKL BVF's kernel is SASS-identical (sm_89, every batch
size) to `fkl_bvf` of `fkl-vs-npp`.

## Build and run

LibTorch's CMake files are the wheel's `torch.utils.cmake_prefix_path`. A CUDA wheel needs the CUDA toolkit and builds
both programs. LibTorch's CMake would give nvcc the C++ compiler as its host compiler when that is clang, so the
project enables CUDA first, with nvcc's default host compiler or `CMAKE_CUDA_HOST_COMPILER`:

    cmake -S fkl-vs-pytorch -B build-pytorch -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ \
          -DCMAKE_PREFIX_PATH=$(venv-torch/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)')
    cmake --build build-pytorch -j
    ./build-pytorch/benchmark_pytorch_cpu ../results/fkl-vs-pytorch/cpu
    ./build-pytorch/benchmark_pytorch_gpu ../results/fkl-vs-pytorch/gpu

`BENCH_PYTORCH_GPU=OFF` builds the CPU program only. `CUDA_ARCH` (default `native`) is the CUDA architecture of FKL's
code; `HIP_ARCH` (default `gfx1200`) is its HIP architecture with a ROCm wheel.

**ROCm.** The same commands build against a ROCm wheel's LibTorch, with `ROCM_PATH` set to the ROCm installation (for
example `/opt/rocm/core-10.1`). LibTorch's CMake enables HIP with its own compiler and architectures, and the project
then sets `HIP_ARCH` on FKL's side only. PyTorch's device is still `torch::kCUDA`, and its streams and guards keep
their `c10::cuda` names in the hipified headers.
