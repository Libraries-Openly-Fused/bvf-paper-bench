# BVF paper benchmark (local)

Latency, kernel counts and correctness of the Figure 1 pipeline of the Backwards Vertical Fusion paper
(ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write) on one 3840x2160
NV12 frame with N in {1, 4, 16, 50, 100} crops of 20x10, for OpenCV CUDA, NPP, FKL VF+HF (4 kernels)
and FKL BVF (1 kernel). Details of each implementation are at the top of `src/bvf_bench.cu`.

Build against the FKL checkout on branch `bvf-paper-bench` (for AddBorder) and the local OpenCV 4.14 build:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=../builds/opencv
    cmake --build build -j
    ./build/bvf_bench ../results     # writes the files listed below
    ./build/bvf_types                # C++ type of the fused read operation for N = 100

OpenCV, NPP and FKL VF+HF write the same intermediates (8 bit RGB frame, then float3 crops, resized crops and
output); FKL BVF keeps float3 in registers from the color conversion to the write. In every one of the 1000 timed
iterations (after 100 warm-up ones) the four implementations run one after the other, and FKL BVF's speedup over
each baseline is computed per iteration and averaged, as in FKL's two-execution benchmarks.

Output files: `latency.csv` (median, p10, p90 and mean in us, kernels launched, FKL BVF mean speedup),
`<baseline>_vs_fkl_bvf_GPU_NVIDIA.csv` (FKL's two-execution format, times in ms), `kernel_breakdown.csv` (time of
each kernel or stage of the multi-kernel pipelines, plus VF+HF with a float3 frame), `validation.csv`,
`kernels.csv` and `platform.txt`.
