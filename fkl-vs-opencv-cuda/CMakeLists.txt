cmake_minimum_required(VERSION 3.28)
# FKL vs OpenCV CUDA on the Figure 1 pipeline (see README.md). Needs OpenCV 4.14 with CUDA and the opencv_contrib
# modules cudaarithm and cudawarping, and NPP for the NV12 conversion (OpenCV CUDA has none):
#   cmake -S fkl-vs-opencv-cuda -B build-opencv-cuda -DOpenCV_DIR=<OpenCV build or install>
project(fkl_vs_opencv_cuda LANGUAGES CXX CUDA)
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_gpu.cmake)

# OpenCVConfig.cmake still calls find_package(CUDA), which needs the legacy FindCUDA module
cmake_policy(SET CMP0146 OLD)
find_package(OpenCV 4.14 REQUIRED COMPONENTS core cudaarithm cudawarping)

add_bench_target(benchmark_opencv_cuda src/benchmark_opencv_cuda.cu)
target_link_libraries(benchmark_opencv_cuda PRIVATE CUDA::nppc CUDA::nppicc opencv_core opencv_cudaarithm
                                                    opencv_cudawarping)
