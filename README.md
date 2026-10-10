# BVF paper benchmark (local)

Latency, kernel counts and correctness of the Figure 1 pipeline of the Backwards Vertical Fusion paper
(ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write) on one 3840x2160
NV12 frame with N in {1, 4, 16, 50, 100} crops of 20x10, for OpenCV CUDA, NPP, FKL VF+HF (4 kernels)
and FKL BVF (1 kernel). Details of each implementation are at the top of `src/bvf_bench.cu`.

Build against the FKL checkout on branch `bvf-paper-bench` (for AddBorder) and the local OpenCV 4.14 build:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=../builds/opencv
    cmake --build build -j
    ./build/bvf_bench ../results     # latency.csv, validation.csv, kernels.csv, platform.txt
    ./build/bvf_types                # C++ type of the fused read operation for N = 100
